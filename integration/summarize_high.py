"""Summarize native high runs using Cache-DiT's internal per-step records."""

import argparse
import hashlib
import json
import re
import statistics
from pathlib import Path


def summarize(root, mode, num_gpus=8):
    perf = json.loads((root / f"{mode}-perf.json").read_text())
    log = (root / f"{mode}.log").read_text()
    protocol_path = root / f"{mode}-perf-protocol.json"
    protocol = json.loads(protocol_path.read_text()) if protocol_path.exists() else None
    stats_log = log
    if "H3_BENCHMARK_PHASE" in log:
        assert protocol and protocol["same_generator"]
        assert protocol["warmup_requests"] == protocol["measured_requests"] == 1
        assert len(protocol["worker_pids"]) == num_gpus
        assert protocol["measured_request_id"] == perf["request_id"]
        warm_perf = json.loads((root / f"{mode}-perf-warmup.json").read_text())
        assert warm_perf["request_id"] == protocol["warmup_request_id"]
        assert warm_perf["request_id"] != perf["request_id"]
        assert len(warm_perf["denoise_steps_ms"]) == 49
        assert log.count("H3_BENCHMARK_PHASE measured_begin") == 1
        stats_log = log.split("H3_BENCHMARK_PHASE measured_begin", 1)[1].split(
            "H3_BENCHMARK_PHASE measured_end", 1
        )[0]
    records = [
        json.loads(s)
        for s in re.findall(r"H3_CACHE_DIT_STATS (\{[^\n\r]+\})", stats_log)
    ]
    if not records:
        raise ValueError(
            f"{mode}: missing internal Cache-DiT stats; rerun with updated SGLang"
        )
    # Each UP rank reports the same logical steps. Never sum rank/block copies.
    first = records[0]
    assert all(r == first for r in records), (
        "Cache-DiT records differ across ranks/groups"
    )
    hits = set(first["cached_steps"])
    steps = perf["denoise_steps_ms"]
    assert first["step_index_base"] == 0
    assert first["total_steps"] == len(steps) == 49
    assert len(hits) == first["cache_hits"]
    assert hits <= {s["step"] for s in steps}
    misses = [s["duration_ms"] for s in steps if s["step"] not in hits]
    # Native H3 increments attention metadata before forward: step 0..9 dense,
    # step >=10 sparse. Use >=11 as the conservative established steady window.
    steady = [
        s["duration_ms"] for s in steps if s["step"] >= 11 and s["step"] not in hits
    ]
    stages = {s["name"]: s["duration_ms"] / 1000 for s in perf["steps"]}
    route = {"nccl": "nccl", "rdma": "rdma", "pcie": "p2p"}[mode]
    assert "DBCache_F1B0_W4I1M0MC1_R0.04_N49_CFG0" in log
    assert "quality=high" in log and "fp8=False" in log
    if mode != "nccl":
        assert f"transport={route}, world_size={num_gpus}" in log
    return {
        "mode": mode,
        "benchmark_protocol": protocol,
        "measured_request_index": 2 if protocol else 1,
        "actual_route": route,
        "total_request_s": perf["total_duration_ms"] / 1000,
        "denoise_s": stages["MiniMaxH3DenoisingStage"],
        "decode_s": stages["MiniMaxH3DecodingStage"],
        "cache_hits": len(hits),
        "cache_misses": len(misses),
        "steps": len(steps),
        "cache_hit_rate": len(hits) / len(steps),
        "cached_step_indices": sorted(hits),
        "cache_stats_records": len(records),
        "cache_stats_source": "cache_dit.summary(adapter, logging=False); deduplicated UP ranks",
        "cache_miss_step_median_ms": statistics.median(misses),
        "cache_miss_step_mean_ms": statistics.mean(misses),
        "steady_cache_miss_step_count": len(steady),
        "steady_cache_miss_step_median_ms": statistics.median(steady),
        "steady_cache_miss_step_mean_ms": statistics.mean(steady),
        "sparse_computed_step_median_ms": statistics.median(steady),
        "steady_window": "zero-based step 11..48, excluding exact internal cached_steps",
        "runtime_peak_mib": perf["memory_checkpoints"]["runtime_peak"][
            "peak_allocated_mb"
        ],
        "video_sha256": hashlib.sha256(
            next(
                p
                for p in (root / f"{mode}.mp4", root / "videos" / f"{mode}.mp4")
                if p.exists()
            ).read_bytes()
        ).hexdigest(),
        "quality": "high",
        "cache_dit": [4, 0.04, 1],
        "vae": "bf16",
        "stages_s": stages,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    root = parser.parse_args().directory
    rows = [
        summarize(root, mode)
        for mode in ("nccl", "rdma", "pcie")
        if (root / f"{mode}-perf.json").exists()
    ]
    (root / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2))
