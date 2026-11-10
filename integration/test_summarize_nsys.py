"""Interval accounting and trace-integrity checks without CUDA or nsys."""
import sqlite3

import pytest

from integration.summarize_nsys import analyze, classify, partition


def trace(path, gpus=1):
    con = sqlite3.connect(path)
    con.executescript('''
        CREATE TABLE StringIds(id INTEGER, value TEXT);
        CREATE TABLE NVTX_EVENTS(start INTEGER, end INTEGER, globalTid INTEGER,
                                 text TEXT, textId INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER, end INTEGER,
            streamId INTEGER, demangledName INTEGER, shortName INTEGER,
            deviceId INTEGER, globalPid INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(start INTEGER, end INTEGER, deviceId INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_MEMSET(start INTEGER, end INTEGER, deviceId INTEGER);
    ''')
    con.executemany('INSERT INTO StringIds VALUES (?,?)', [
        (1, 'kernel_cake_sage_block_sparse_attention_test'),
        (2, 'cutlass::gemm::GemmUniversal'), (3, 'UlyssesPcieBarrier'),
        (4, 'per_token_quant_fp8_warp_kernel')])
    for gpu in range(gpus):
        pid = (100 + gpu) << 24
        con.executemany('INSERT INTO NVTX_EVENTS VALUES (?,?,?,?,NULL)', [
            (0, 100, pid+1, 'denoising_loop'),
            (0, 50, pid+1, 'denoising_step_0'),
            (50, 100, pid+1, 'denoising_step_1')])
        for a, b, stream, name in [(0, 30, 7, 1), (30, 50, 7, 2),
                                    (50, 51, 7, 3), (79, 80, 7, 3),
                                    (90, 95, 7, 4)]:
            con.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?,?,?)',
                        (a, b, stream, name, name, gpu, pid))
        # Copy is already inside the communication envelope: do not count twice.
        con.execute('INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (60,70,?)', (gpu,))
    con.commit()
    return con


def test_partition_clipping_union_and_overlap():
    parts, inclusive = partition({
        'attention': [(-5, 6), (1, 4)],
        'gemm': [(10, 20)],
        'communication': [(4, 9), (5, 8)],
        'other_gpu': [(5, 7)]}, 0, 10)
    assert parts['attention_only'] == 4
    assert parts['communication_only'] == 3
    assert parts['communication_compute_overlap'] == 2
    assert parts['unattributed'] == 1
    assert sum(parts.values()) == 10
    assert inclusive == {'attention': 6, 'communication': 5, 'other_gpu': 2}


def test_three_way_overlap_is_not_counted_twice():
    parts, _ = partition({name: [(0, 10)] for name in
                          ('attention', 'gemm', 'communication', 'other_gpu')}, 0, 10)
    assert parts['communication_compute_overlap'] == 10
    assert sum(parts.values()) == 10


@pytest.mark.parametrize('gpus', [1, 2, 4, 8])
def test_dynamic_gpu_count_and_exchange_envelope(tmp_path, gpus):
    path = tmp_path/'trace.sqlite'
    trace(path, gpus).close()
    result = analyze(path, expected_steps=2)
    assert result['gpu_count'] == gpus
    assert result['step_ranges'] == gpus * 2
    assert result['partition']['attention_only']['percent'] == 30
    assert result['partition']['gemm_only']['percent'] == 20
    assert result['partition']['communication_only']['percent'] == 30
    assert result['partition']['other_gpu_only']['percent'] == 5
    assert result['partition']['unattributed']['percent'] == 15
    assert all(row['exchange_barrier_pairs'] == 1 for row in result['gpu_rows'])
    assert sum(x['percent'] for x in result['partition'].values()) == 100


def test_no_barriers_is_valid_for_local_or_nccl(tmp_path):
    path = tmp_path/'trace.sqlite'
    with trace(path) as con:
        con.execute('DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE demangledName=3')
    result = analyze(path, expected_steps=2)
    assert result['gpu_rows'][0]['exchange_barrier_pairs'] == 0
    assert result['partition']['communication_only']['percent'] == 0


def test_optional_copy_tables_and_nonzero_gpu(tmp_path):
    path = tmp_path/'trace.sqlite'
    with trace(path) as con:
        con.execute('DROP TABLE CUPTI_ACTIVITY_KIND_MEMCPY')
        con.execute('DROP TABLE CUPTI_ACTIVITY_KIND_MEMSET')
        con.execute('UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET deviceId=4')
    result = analyze(path, expected_steps=2)
    assert result['representative_gpu'] == 4
    assert result['representative_gpu_kernels']
    assert result['gpu_rows'][0]['memcpy'] == 0


def test_inconsistent_barrier_counts_fail(tmp_path):
    path = tmp_path/'trace.sqlite'
    with trace(path, gpus=2) as con:
        con.execute('DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE deviceId=1 AND demangledName=3')
    with pytest.raises(ValueError, match='Inconsistent exchange barrier counts'):
        analyze(path, expected_steps=2)


def test_incomplete_steps_fail(tmp_path):
    path = tmp_path/'trace.sqlite'
    with trace(path) as con:
        con.execute("DELETE FROM NVTX_EVENTS WHERE text='denoising_step_1'")
    with pytest.raises(ValueError, match='expected steps'):
        analyze(path, expected_steps=2)


def test_unpaired_barriers_fail(tmp_path):
    path = tmp_path/'trace.sqlite'
    with trace(path) as con:
        con.execute('DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE start=79')
    with pytest.raises(ValueError, match='unpaired Ulysses barriers'):
        analyze(path, expected_steps=2)


def test_repeated_capture_fails(tmp_path):
    path = tmp_path/'trace.sqlite'
    with trace(path) as con:
        con.execute("INSERT INTO NVTX_EVENTS VALUES (100,200,?, 'denoising_loop',NULL)",
                    ((100<<24)+1,))
    with pytest.raises(ValueError, match='one measured denoising loop'):
        analyze(path, expected_steps=2)


def test_process_mapping_fails(tmp_path):
    path = tmp_path/'trace.sqlite'
    with trace(path) as con:
        con.execute('UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET deviceId=1 WHERE start=90')
    with pytest.raises(ValueError, match='one-to-one mapping'):
        analyze(path, expected_steps=2)


@pytest.mark.parametrize('name,expected', [
    ('ncclDevKernel_SendRecv', 'communication'),
    ('UlyssesPcieBarrier', 'communication'),
    ('_pack_cake_operands', 'attention'),
    ('void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<...>>', 'attention'),
    ('void cutlass::device_kernel<cutlass::gemm::kernel::GemmUniversal<...>>', 'gemm'),
    ('QuantInt8GroupScalePackKernel', 'other_gpu')])
def test_full_kernel_names(name, expected):
    assert classify(name) == expected


def test_dense_sparse_cache_partition_and_reconciliation(tmp_path):
    import json
    path = tmp_path/'trace.sqlite'
    trace(path).close()
    run_root = tmp_path/'runs'
    case = run_root/'trace'
    case.mkdir(parents=True)
    args = {'attention_backend_config': {'sparsity': .75, 'skip_first_layers': 0,
                                         'skip_first_steps': 1}}
    stats = dict(total_steps=2, step_index_base=0, cached_steps=[1], cache_hits=1)
    (case/'nccl.log').write_text('server_args: '+json.dumps(args)+'\n'
        'H3_BENCHMARK_PHASE measured_begin\nH3_CACHE_DIT_STATS '+json.dumps(stats)+'\n'
        'H3_BENCHMARK_PHASE measured_end\n')
    result = analyze(path, expected_steps=2, run_root=run_root)
    dense, sparse = result['phases']['dense'], result['phases']['sparse']
    assert dense['denoise_fraction_percent'] == sparse['denoise_fraction_percent'] == 50
    assert dense['partition']['attention_only']['percent'] == 60
    assert dense['partition']['gemm_only']['percent'] == 40
    assert sparse['partition']['communication_only']['percent'] == 60
    assert result['phases']['sparse_compute']['mean_s'] == 0
    assert result['phases']['cache_hits']['step_indices'] == [1]
    for key, total in result['partition'].items():
        assert dense['partition'][key]['mean_s'] + sparse['partition'][key]['mean_s'] == pytest.approx(total['mean_s'])


def test_step_gaps_and_tail_are_retained():
    from integration.summarize_nsys import phase_intervals
    groups = {'attention': [(0, 100)]}
    phases = phase_intervals(groups, 0, 100, [(0, 5, 40), (1, 60, 90)],
                              dict(dense_steps=1, cached_steps=[]))
    assert phases['dense']['denominator_ns'] == 60
    assert phases['sparse']['denominator_ns'] == 40
    assert phases['sparse_compute']['denominator_ns'] == 40
    assert phases['cache_hits']['denominator_ns'] == 0
