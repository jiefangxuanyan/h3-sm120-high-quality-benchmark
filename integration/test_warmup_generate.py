"""Check the real warmup wrapper's lifecycle without loading a model or CUDA."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


@pytest.mark.parametrize("fail_measured", [False, True])
def test_two_requests_then_shutdown_after_cli_export_even_on_failure(
    tmp_path, monkeypatch, fail_measured
):
    events = []
    ids = iter(["warm", "measured"])

    class Generator:
        def __init__(self):
            self.local_scheduler_process = [
                SimpleNamespace(pid=11),
                SimpleNamespace(pid=12),
            ]
            self.server_args = object()
            self.calls = 0

        def generate(self, sampling_params_kwargs=None, **kwargs):
            self.calls += 1
            events.append("generate" + str(self.calls))
            if self.calls == 2 and fail_measured:
                raise RuntimeError("measured failure")
            return [dict(request_id=sampling_params_kwargs["request_id"])]

        def shutdown(self):
            events.append("shutdown")

    def module(name, **attrs):
        m = ModuleType(name)
        m.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, m)
        return m

    sample = module(
        "sglang.multimodal_gen.configs.sample.sampling_params",
        generate_request_id=lambda: next(ids),
    )
    assert sample.generate_request_id
    cli = ModuleType("generate")

    def add_args(parser):
        parser.add_argument("--perf-dump-path", required=True)
        parser.add_argument("--output-file-path", required=True)
        return parser

    cli.add_multimodal_gen_generate_args = add_args
    cli.maybe_dump_performance = lambda *args: events.append("export")
    cli.DiffGenerator = Generator

    def generate_cmd(args, unknown):
        g = cli.DiffGenerator()
        params = dict(
            request_id="original", output_file_name=str(tmp_path / "video.mp4")
        )
        result = g.generate(sampling_params_kwargs=params)
        assert params["request_id"] == "original"
        cli.maybe_dump_performance(args, g.server_args, "prompt", result)

    cli.generate_cmd = generate_cmd
    module("sglang.multimodal_gen.runtime.entrypoints.cli", generate=cli)
    module(
        "sglang.multimodal_gen.runtime.entrypoints.diffusion_generator",
        DiffGenerator=Generator,
    )
    module(
        "sglang.multimodal_gen.runtime.utils.argparse",
        FlexibleArgumentParser=argparse.ArgumentParser,
    )
    monkeypatch.delenv("SGLANG_H3_NSYS_CAPTURE", raising=False)
    perf = tmp_path / "test-perf.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "warmup_generate",
            "--perf-dump-path",
            str(perf),
            "--output-file-path",
            str(tmp_path / "video.mp4"),
        ],
    )
    p = Path(__file__).with_name("warmup_generate.py")
    spec = importlib.util.spec_from_file_location("customer_warmup_test", p)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    if fail_measured:
        with pytest.raises(RuntimeError, match="measured failure"):
            runner.main()
        assert events == ["generate1", "export", "generate2", "shutdown"]
    else:
        runner.main()
        assert events == ["generate1", "export", "generate2", "export", "shutdown"]
        protocol = json.loads((tmp_path / "test-perf-protocol.json").read_text())
        assert protocol["same_generator"] and protocol["worker_pids"] == [11, 12]
        assert (
            protocol["warmup_request_id"] == "warm"
            and protocol["measured_request_id"] == "measured"
        )
    assert cli.DiffGenerator is Generator
