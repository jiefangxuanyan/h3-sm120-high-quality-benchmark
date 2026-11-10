"""Run a complete warmup and measured request on one resident DiffGenerator.

Accepts the same options as `sglang generate`. The regular output/perf paths
belong to request 2. Request 1 uses a separate *-warmup path.
"""

import copy
import json
import os
import time
from pathlib import Path

from sglang.multimodal_gen.configs.sample.sampling_params import generate_request_id
from sglang.multimodal_gen.runtime.entrypoints.cli import generate as cli
from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import DiffGenerator
from sglang.multimodal_gen.runtime.utils.argparse import FlexibleArgumentParser


def suffixed(path, suffix):
    p = Path(path)
    return str(p.with_name(p.stem + suffix + p.suffix))


def main():
    parser = cli.add_multimodal_gen_generate_args(FlexibleArgumentParser())
    args, unknown = parser.parse_known_args()
    if not args.perf_dump_path or not args.output_file_path:
        parser.error("--perf-dump-path and --output-file-path are required")

    generators = []

    class WarmedGenerator(DiffGenerator):
        def generate(self, sampling_params_kwargs=None, **kwargs):
            generators.append(self)
            warmup = copy.deepcopy(sampling_params_kwargs)
            warmup["request_id"] = generate_request_id()
            warmup["output_file_name"] = suffixed(warmup["output_file_name"], "-warmup")
            warm_args = copy.copy(args)
            warm_args.perf_dump_path = suffixed(args.perf_dump_path, "-warmup")
            worker_pids = [p.pid for p in self.local_scheduler_process or []]
            print("H3_BENCHMARK_PHASE warmup_begin", flush=True)
            start = time.perf_counter()
            warm_result = super().generate(sampling_params_kwargs=warmup, **kwargs)
            warm_wall_s = time.perf_counter() - start
            if not warm_result:
                raise RuntimeError("Full warmup request returned no output")
            cli.maybe_dump_performance(
                warm_args, self.server_args, warmup.get("prompt"), warm_result
            )
            print("H3_BENCHMARK_PHASE warmup_end", flush=True)
            measured = copy.deepcopy(sampling_params_kwargs)
            measured["request_id"] = generate_request_id()
            print("H3_BENCHMARK_PHASE measured_begin", flush=True)
            capture = os.environ.get("SGLANG_H3_NSYS_CAPTURE") == "1"
            if capture:
                import torch

                torch.cuda.nvtx.range_push("H3_MEASURED_REQUEST")
            try:
                start = time.perf_counter()
                result = super().generate(sampling_params_kwargs=measured, **kwargs)
                measured_wall_s = time.perf_counter() - start
            finally:
                if capture:
                    # End collection after timing; nsys performs its own final
                    # buffer flush and report processing outside generate().
                    torch.cuda.nvtx.range_pop()
            if not result:
                raise RuntimeError("Measured request returned no output")
            assert worker_pids == [p.pid for p in self.local_scheduler_process or []]
            print("H3_BENCHMARK_PHASE measured_end", flush=True)
            protocol = {
                "protocol": "one full warmup request, then one measured request",
                "same_generator": True,
                "parent_pid": os.getpid(),
                "worker_pids": worker_pids,
                "warmup_requests": 1,
                "measured_requests": 1,
                "same_sampling_except_request_id_and_output": True,
                "warmup_request_id": warmup["request_id"],
                "measured_request_id": measured["request_id"],
                "warmup_generate_wall_s": warm_wall_s,
                "measured_generate_wall_s": measured_wall_s,
                "warmup_perf": warm_args.perf_dump_path,
                "measured_perf": args.perf_dump_path,
            }
            Path(suffixed(args.perf_dump_path, "-protocol")).write_text(
                json.dumps(protocol, indent=2) + "\n"
            )
            return result

    # Reuse SGLang's exact CLI config parsing and performance export. Only the
    # generator class changes; both calls use the same resident worker set.
    cli.DiffGenerator = WarmedGenerator
    try:
        cli.generate_cmd(args, unknown)
    finally:
        try:
            for generator in generators:
                generator.shutdown()
        finally:
            cli.DiffGenerator = DiffGenerator


if __name__ == "__main__":
    main()
