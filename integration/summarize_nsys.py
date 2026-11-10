"""Summarize per-GPU denoising occupancy from nsys SQLite exports.

No CPU NVTX duration is treated as GPU execution. Communication is NCCL kernel
intervals plus paired opening/closing GPU UlyssesPcieBarrier envelopes. These
include transfer, synchronization and host progress gaps, NOT pure wire time.
"""
import argparse
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from contextlib import closing
import hashlib
import itertools
import json
from pathlib import Path
import re
import sqlite3

ATTENTION = ('kernel_cake_sage_block_sparse_attention', 'pytorch_flash::flash_fwd_kernel', '_pool_prequantized',
             '_score_kernel', '_topk_kernel', '_expand_mixed_rows',
             '_pack_cake_operands', '_write_cake_output')
BITS = {'attention': 1, 'gemm': 2, 'communication': 4, 'other_gpu': 8}
LABELS = ('attention_only', 'gemm_only', 'communication_only',
          'communication_compute_overlap', 'other_gpu_only', 'unattributed',
          'attention_gemm_overlap')
PHASES = ('dense', 'sparse', 'sparse_compute', 'cache_hits')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def merge(intervals):
    result = []
    for a, b in sorted(intervals):
        if b <= a:
            continue
        if result and a <= result[-1][1]:
            result[-1] = (result[-1][0], max(b, result[-1][1]))
        else:
            result.append((a, b))
    return result


def partition(groups, begin, end):
    require(end > begin, 'Expected a positive denoising interval')
    events = [(begin, 0, 0), (end, 0, 0)]
    for name, intervals in groups.items():
        for a, b in merge((max(begin, a), min(end, b)) for a, b in intervals):
            events += [(a, BITS[name], 1), (b, BITS[name], -1)]
    active = Counter()
    durations = Counter({k: 0 for k in LABELS})
    raw = Counter()
    prev = begin
    for t, entries in itertools.groupby(sorted(events), key=lambda r: r[0]):
        dt = t - prev
        a, g, c, o = (active[BITS[n]] > 0 for n in BITS)
        for name in BITS:
            if active[BITS[name]]:
                raw[name] += dt
        if c and (a or g):
            label = 'communication_compute_overlap'
        elif c:
            label = 'communication_only'
        elif a and g:
            label = 'attention_gemm_overlap'
        elif a:
            label = 'attention_only'
        elif g:
            label = 'gemm_only'
        elif o:
            label = 'other_gpu_only'
        else:
            label = 'unattributed'
        durations[label] += dt
        for _, bit, delta in entries:
            active[bit] += delta
            require(active[bit] >= 0, 'Invalid interval balance')
        prev = t
    require(sum(durations.values()) == end - begin, 'Partition does not cover denoising')
    return dict(durations), dict(raw)


def classify(name):
    if 'nccl' in name.lower():
        return 'communication'
    if 'UlyssesPcieBarrier' in name:
        return 'communication'
    if any(s in name for s in ATTENTION):
        return 'attention'
    lower = name.lower()
    if 'gemm' in lower or 'cublas' in lower or 'matmul' in lower:
        return 'gemm'
    return 'other_gpu'


def load_phase_config(path, run_root, expected_steps):
    logs = list((run_root / path.stem).glob('*.log'))
    texts = [p.read_text() for p in logs if p.name in ('nccl.log', 'rdma.log', 'pcie.log')]
    require(len(texts) == 1, f'Expected one model log in {run_root / path.stem}')
    text = texts[0]
    args = re.search(r'server_args: (\{[^\n\r]+\})', text)
    require(args is not None, 'Missing server_args for dense/sparse schedule')
    config = json.loads(args.group(1))['attention_backend_config']
    require(config['sparsity'] > 0 and config['skip_first_layers'] == 0,
            'Phase analysis expects the H3 sparse schedule with skip_first_layers=0')
    require(text.count('H3_BENCHMARK_PHASE measured_begin') == 1
            and text.count('H3_BENCHMARK_PHASE measured_end') == 1,
            'Expected one measured request in the model log')
    measured = text.split('H3_BENCHMARK_PHASE measured_begin', 1)[1].split(
        'H3_BENCHMARK_PHASE measured_end', 1)[0]
    records = [json.loads(s) for s in re.findall(r'H3_CACHE_DIT_STATS (\{[^\n\r]+\})', measured)]
    require(bool(records) and all(r == records[0] for r in records),
            'Missing or inconsistent measured Cache-DiT records')
    record = records[0]
    hits = record['cached_steps']
    require(record['total_steps'] == expected_steps and record['step_index_base'] == 0
            and len(set(hits)) == len(hits) == record['cache_hits']
            and set(hits) <= set(range(expected_steps)), 'Invalid Cache-DiT step indices')
    cutoff = config['skip_first_steps']
    require(0 <= cutoff <= expected_steps, 'Invalid dense-step cutoff')
    return dict(dense_steps=cutoff, cached_steps=hits,
                log=str(next(p for p in logs if p.name in ('nccl.log', 'rdma.log', 'pcie.log'))))


def phase_intervals(groups, begin, end, steps, config):
    """Split a fixed GPU timeline at step NVTX starts; retain every nanosecond."""
    merged = {k: merge(v) for k, v in groups.items()}
    starts = {k: [a for a, b in v] for k, v in merged.items()}
    ends = {k: [b for a, b in v] for k, v in merged.items()}
    result = {name: dict(denominator_ns=0, partition_ns=Counter(),
                         inclusive_union_ns=Counter(), step_indices=[]) for name in PHASES}
    ordered = sorted(steps)
    for position, (index, a, b) in enumerate(ordered):
        left = begin if position == 0 else a
        right = ordered[position+1][1] if position+1 < len(ordered) else end
        require(left < right and a <= b <= right, 'Overlapping or unordered step NVTX ranges')
        selected = {k: v[bisect_right(ends[k], left):bisect_left(starts[k], right)]
                    for k, v in merged.items()}
        parts, raw = partition(selected, left, right)
        labels = ['dense' if index < config['dense_steps'] else 'sparse']
        if index in config['cached_steps']:
            labels.append('cache_hits')
        elif index >= config['dense_steps']:
            labels.append('sparse_compute')
        for label in labels:
            row = result[label]
            row['denominator_ns'] += right-left
            row['partition_ns'].update(parts)
            row['inclusive_union_ns'].update(raw)
            row['step_indices'].append(index)
    for row in result.values():
        row['partition_ns'] = dict(row['partition_ns'])
        row['inclusive_union_ns'] = dict(row['inclusive_union_ns'])
    return result


def analyze(path, expected_steps=49, run_root=None):
    config = load_phase_config(path, run_root, expected_steps) if run_root else None
    with closing(sqlite3.connect(f'file:{path.resolve()}?mode=ro', uri=True)) as con:
        return analyze_connection(con, path, expected_steps, config)


def analyze_connection(con, path, expected_steps, phase_config=None):
    tables = {name for (name,) in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    strings = dict(con.execute('SELECT id,value FROM StringIds'))
    loops = con.execute("""SELECT n.start,n.end,n.globalTid FROM NVTX_EVENTS n
        LEFT JOIN StringIds s ON n.textId=s.id
        WHERE coalesce(n.text,s.value)='denoising_loop'""").fetchall()
    require(bool(loops), 'No denoising_loop NVTX ranges found')
    require(all(a is not None and b is not None and b > a and tid is not None
                for a, b, tid in loops), 'Incomplete denoising_loop NVTX ranges')
    # Nsight globalTid encodes the process in the high bits and thread in low 24.
    by_pid = {tid & ~((1 << 24) - 1): (a, b) for a, b, tid in loops}
    require(len(by_pid) == len(loops), 'Expected one measured denoising loop per worker process')
    gpu_processes = [(gpu, pid) for gpu, pid in con.execute(
        'SELECT DISTINCT deviceId,globalPid FROM CUPTI_ACTIVITY_KIND_KERNEL') if pid in by_pid]
    pid_by_gpu = dict(gpu_processes)
    require(len(gpu_processes) == len(pid_by_gpu) == len(by_pid)
            and set(pid_by_gpu.values()) == set(by_pid),
            'Expected a one-to-one mapping between worker processes and GPUs')
    steps = 0
    step_ranges = {}
    for pid, (begin, end) in by_pid.items():
        ranges = [(int(name.removeprefix('denoising_step_')), a, b) for name, a, b in con.execute(
            """SELECT coalesce(n.text,s.value),n.start,n.end FROM NVTX_EVENTS n LEFT JOIN StringIds s
            ON n.textId=s.id WHERE coalesce(n.text,s.value) GLOB 'denoising_step_[0-9]*'
            AND (n.globalTid & -16777216)=? AND n.start>=? AND n.end<=?""", (pid, begin, end))]
        indices = [index for index, a, b in ranges]
        require(sorted(indices) == list(range(expected_steps)),
                f'Worker {pid}: expected steps 0..{expected_steps-1}, got {indices}')
        steps += len(indices)
        step_ranges[pid] = ranges
    gpu_rows, kernels_table = [], []
    for gpu, pid in sorted(pid_by_gpu.items()):
        begin, end = by_pid[pid]
        groups = defaultdict(list)
        barriers = defaultdict(list)
        totals = defaultdict(lambda: [0, 0])
        count = 0
        for a, b, stream, name_id, short_id in con.execute('''SELECT start,end,streamId,
            demangledName,shortName FROM CUPTI_ACTIVITY_KIND_KERNEL
            WHERE deviceId=? AND globalPid=? AND start<? AND end>?
            ORDER BY start''', (gpu, pid, end, begin)):
            a, b = max(a, begin), min(b, end)
            name = strings[name_id]
            cat = classify(name)
            groups[cat].append((a, b))
            totals[(cat, name_id, short_id)][0] += 1
            totals[(cat, name_id, short_id)][1] += b - a
            if 'UlyssesPcieBarrier' in name:
                barriers[stream].append((a, b))
            count += 1
        pairs = 0
        for stream, spans in barriers.items():
            require(len(spans) % 2 == 0,
                    f'GPU {gpu}, stream {stream}: unpaired Ulysses barriers ({len(spans)})')
            for first, last in zip(spans[::2], spans[1::2]):
                require(first[1] <= last[0], 'Overlapping barriers in the same stream')
                groups['communication'].append((first[0], last[1]))
                pairs += 1
        copies = 0
        for table in ('CUPTI_ACTIVITY_KIND_MEMCPY', 'CUPTI_ACTIVITY_KIND_MEMSET'):
            # nsys omits event tables when the trace contains no such events.
            if table not in tables:
                continue
            for a, b in con.execute(f'''SELECT start,end FROM {table}
                    WHERE deviceId=? AND start<? AND end>?''', (gpu, end, begin)):
                groups['other_gpu'].append((max(a, begin), min(b, end)))
                if table == 'CUPTI_ACTIVITY_KIND_MEMCPY':
                    copies += 1
        parts, raw = partition(groups, begin, end)
        gpu_rows.append(dict(gpu=gpu, process=pid, begin_ns=begin, end_ns=end,
                             denominator_ns=end-begin, partition_ns=parts,
                             inclusive_union_ns=raw, kernels=count, memcpy=copies,
                             exchange_barrier_pairs=pairs))
        if phase_config is not None:
            phases = phase_intervals(groups, begin, end, step_ranges[pid], phase_config)
            require(phases['dense']['denominator_ns'] + phases['sparse']['denominator_ns']
                    == end-begin, 'Phases do not cover denoising')
            for key in LABELS:
                require(phases['dense']['partition_ns'].get(key, 0)
                        + phases['sparse']['partition_ns'].get(key, 0) == parts[key],
                        f'Phase/overall partition mismatch: {key}')
            gpu_rows[-1]['phases'] = phases
        if gpu == min(pid_by_gpu):
            for (cat, name_id, short_id), (n, duration) in sorted(totals.items(), key=lambda r: -r[1][1]):
                kernels_table.append(dict(category=cat, count=n, sum_s=duration / 1e9,
                    short_name=strings[short_id], demangled_name=strings[name_id]))
    denom = sum(r['denominator_ns'] for r in gpu_rows)
    sums = {k: sum(r['partition_ns'].get(k, 0) for r in gpu_rows) for k in LABELS}
    count = len(gpu_rows)
    pair_counts = {r['exchange_barrier_pairs'] for r in gpu_rows}
    require(len(pair_counts) == 1, 'Inconsistent exchange barrier counts across GPUs')
    result = dict(sqlite=str(path.resolve()), gpu_count=count, step_ranges=steps,
        method=dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    attention_name_patterns=ATTENTION,
                    communication='NCCL GPU kernel union and paired Ulysses GPU barrier envelopes',
                    communication_includes_waiting=True,
                    gpu_time_is_execution_interval_not_sm_utilization=True),
        denominator='sum of per-GPU denoising_loop wall durations',
        mean_denoise_s=denom / (count * 1e9),
        partition={k: dict(mean_s=v / (count * 1e9), percent=v / denom * 100) for k, v in sums.items()},
        inclusive={k: dict(mean_s=sum(r['inclusive_union_ns'].get(k, 0) for r in gpu_rows)/(count * 1e9),
                          percent=sum(r['inclusive_union_ns'].get(k, 0) for r in gpu_rows)/denom*100)
                   for k in BITS},
        gpu_rows=gpu_rows, representative_gpu=min(pid_by_gpu), representative_gpu_kernels=kernels_table)
    require(abs(sum(v['percent'] for v in result['partition'].values()) - 100) < 1e-8,
            'Percentages do not sum to 100')
    if phase_config is not None:
        result['phase_config'] = phase_config
        result['phases'] = {}
        for name in PHASES:
            phase_rows = [r['phases'][name] for r in gpu_rows]
            duration = sum(r['denominator_ns'] for r in phase_rows)
            result['phases'][name] = dict(
                step_indices=phase_rows[0]['step_indices'],
                mean_s=duration/(count*1e9), denoise_fraction_percent=duration/denom*100,
                partition={k: dict(mean_s=sum(r['partition_ns'].get(k, 0) for r in phase_rows)/(count*1e9),
                    percent=sum(r['partition_ns'].get(k, 0) for r in phase_rows)/duration*100 if duration else 0)
                    for k in LABELS})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sqlite', type=Path, nargs='+', help='One or more nsys SQLite exports')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--expected-steps', type=int, default=49)
    parser.add_argument('--run-root', type=Path,
                        help='profile-all directory; enables dense/sparse/cache breakdown from model logs')
    args = parser.parse_args()
    if args.expected_steps < 1:
        parser.error('--expected-steps must be positive')
    if len({p.stem for p in args.sqlite}) != len(args.sqlite):
        parser.error('Input filenames must have unique stems to avoid overwriting results')
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in args.sqlite:
        result = analyze(path, args.expected_steps, args.run_root)
        (args.output / f'{path.stem}.json').write_text(json.dumps(result, indent=2)+'\n')
        rows.append(dict(name=path.stem, gpu_count=result['gpu_count'],
                         mean_denoise_s=result['mean_denoise_s'], partition=result['partition']))
        if 'phases' in result:
            rows[-1]['phases'] = result['phases']
        print(path.stem, 'denoise_s=', round(result['mean_denoise_s'], 3), flush=True)
        for k, v in result['partition'].items():
            print(' ', k, round(v['mean_s'], 3), round(v['percent'], 2), flush=True)
    (args.output / 'summary.json').write_text(json.dumps(rows, indent=2)+'\n')
    lines = ['| Report | GPUs | Denoise s | ' + ' | '.join(LABELS) + ' |',
             '| --- | ---: | ---: | ' + ' | '.join(['---:'] * len(LABELS)) + ' |']
    for row in rows:
        cells = [f"{row['partition'][k]['mean_s']:.3f} s ({row['partition'][k]['percent']:.2f}%)" for k in LABELS]
        lines.append(f"| {row['name']} | {row['gpu_count']} | {row['mean_denoise_s']:.3f} | "
                     + ' | '.join(cells) + ' |')
    (args.output / 'summary.md').write_text('\n'.join(lines)+'\n')
    if args.run_root:
        lines = ['| Report | Phase | Steps | Seconds | % of denoise | ' + ' | '.join(LABELS) + ' |',
                 '| --- | --- | ---: | ---: | ---: | ' + ' | '.join(['---:'] * len(LABELS)) + ' |']
        for row in rows:
            for name, phase in row['phases'].items():
                cells = [f"{phase['partition'][k]['mean_s']:.3f} s ({phase['partition'][k]['percent']:.2f}%)" for k in LABELS]
                lines.append(f"| {row['name']} | {name} | {len(phase['step_indices'])} | "
                             f"{phase['mean_s']:.3f} | {phase['denoise_fraction_percent']:.2f}% | "
                             + ' | '.join(cells) + ' |')
        (args.output / 'phases.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
