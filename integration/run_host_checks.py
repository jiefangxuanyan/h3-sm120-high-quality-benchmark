"""Run Ulysses/SSA host tests without importing CUDA-only package initializers.

Real module source, Torch CPU tensors, and upstream tests execute normally.
Only logging/trace/JIT import boundaries are stubbed; no CUDA result is claimed.
Run with Python >=3.10 and CPU torch + pytest installed.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import types

ROOT = Path(__file__).resolve().parents[1]
FI = ROOT / "flashinfer"
SG = Path(os.environ.get("SGLANG_SRC", str(ROOT / "sglang-all"))) / "python"
sys.path[:0] = [str(FI), str(SG)]


def module(name, path=None, **attrs):
    obj = types.ModuleType(name)
    if path is not None:
        obj.__path__ = [str(path)]
    obj.__dict__.update(attrs)
    sys.modules[name] = obj
    if "." in name:
        parent, attr = name.rsplit(".", 1)
        if parent in sys.modules:
            setattr(sys.modules[parent], attr, obj)
    return obj


def identity(fn=None, **kwargs):
    return fn if callable(fn) else lambda f: f


def unavailable(*args, **kwargs):
    raise AssertionError("A host check unexpectedly tried to JIT a CUDA kernel")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    parent, attr = name.rsplit(".", 1)
    setattr(sys.modules[parent], attr, obj)
    return obj


module("flashinfer", FI / "flashinfer")
module(
    "flashinfer.api_logging",
    flashinfer_api=identity,
    experimental_auto_backends_allowed=lambda: (
        os.environ.get("FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS") == "1"
    ),
    warn_experimental_backend_once=lambda *a, **kw: None,
)
module("flashinfer.jit", FI / "flashinfer/jit")
module(
    "flashinfer.jit.comm",
    **{
        name: unavailable
        for name in [
            "gen_ulysses_a2a_module",
            "gen_ulysses_pcie_module",
            "gen_ulysses_lowp_module",
            "gen_ulysses_lowp_sm90_module",
        ]
    },
)
module("flashinfer.trace", FI / "flashinfer/trace")
module("flashinfer.trace.templates", FI / "flashinfer/trace/templates")
trace_names = [
    "ulysses_exchange_chunks_trace",
    "ulysses_gather_heads_trace",
    "ulysses_scatter_heads_trace",
    "ulysses_scatter_qkv_trace_dispatch",
    "ulysses_lowp_k_grouped_amax_trace",
    "ulysses_lowp_k_sum_v_amax_trace",
    "ulysses_lowp_q_grouped_amax_trace",
    "ulysses_lowp_quant_qkv_pack_trace",
    "ulysses_lowp_unpack_for_sage_trace",
]
module("flashinfer.trace.templates.comm", **dict.fromkeys(trace_names))
module(
    "flashinfer.utils",
    register_custom_op=identity,
    device_support_pdl=lambda device: False,
)
comm = module("flashinfer.comm", FI / "flashinfer/comm")
load("flashinfer.comm.ulysses_topology", FI / "flashinfer/comm/ulysses_topology.py")
ulysses = load("flashinfer.comm.ulysses", FI / "flashinfer/comm/ulysses.py")
for name in [
    "UlyssesCommunicator",
    "UlyssesWorkspace",
    "UlyssesQKV",
    "UlyssesQKVWorkspace",
    "init_ulysses_a2a",
]:
    setattr(comm, name, getattr(ulysses, name))
topology = sys.modules["flashinfer.comm.ulysses_topology"]
for name in [
    "resolve_ulysses_backend",
    "UlyssesBackendError",
    "UlyssesBackendDecision",
    "UlyssesRankTopology",
    "probe_ulysses_rank_topology",
]:
    setattr(comm, name, getattr(topology, name))
load("flashinfer.comm._ulysses_lowp", FI / "flashinfer/comm/_ulysses_lowp.py")

module("flashinfer.comm.vllm_ar", meta_size=unavailable)
module(
    "flashinfer.comm.cuda_ipc",
    create_shared_buffer=unavailable,
    cudart=types.SimpleNamespace(cudaMalloc=unavailable),
)

# Import only the SGLang modules under test, without starting its runtime.
parts = [
    "sglang",
    "sglang.multimodal_gen",
    "sglang.multimodal_gen.runtime",
    "sglang.multimodal_gen.runtime.distributed",
    "sglang.multimodal_gen.runtime.distributed.device_communicators",
    "sglang.multimodal_gen.runtime.utils",
    "sglang.multimodal_gen.runtime.layers",
    "sglang.multimodal_gen.runtime.layers.attention",
    "sglang.multimodal_gen.runtime.layers.attention.backends",
]
for name in parts:
    module(name, SG / name.replace(".", "/"))
from contextlib import nullcontext

logger = types.SimpleNamespace(
    **{
        x: lambda *a, **kw: None
        for x in ["info", "info_once", "warning", "error", "exception"]
    }
)
module(
    "sglang.multimodal_gen.runtime.utils.logging_utils", init_logger=lambda name: logger
)
module(
    "sglang.multimodal_gen.runtime.utils.nvtx_pytorch_hooks",
    maybe_nvtx_range=lambda *a: nullcontext(),
    ulysses_nvtx_enabled=lambda: False,
)
transport_name = "sglang.multimodal_gen.runtime.distributed.device_communicators.flashinfer_ulysses_a2a"
load(transport_name, SG / transport_name.replace(".", "/").__add__(".py"))
backends = "sglang.multimodal_gen.runtime.layers.attention.backends"
module(
    backends + ".attention_backend", AttentionBackend=type("AttentionBackend", (), {})
)
module(
    backends + ".subblock_sparse",
    SubBlockRouter=None,
    load_bsa_attn_sm120_blk64_sage_fwd=None,
)
module(
    backends + ".subblock_sparse_attn_sm120_sage",
    SubBlockSparseSageAttentionSM120Impl=object,
    SubBlockSparseSageAttentionSM120Backend=type("SageBackend", (), {}),
)
module(backends + ".subblock_sparse_attn_sm120", SubBlockSparseScheduleSM120=object)
module(backends + ".subblock_sparse_attn", SubBlockSparseSchedule=object)
module(
    "sglang.multimodal_gen.runtime.managers",
    SG / "sglang/multimodal_gen/runtime/managers",
)
module(
    "sglang.multimodal_gen.runtime.managers.forward_context", get_forward_context=None
)
module("sglang.multimodal_gen.runtime.platforms", AttentionBackendEnum=object)
module(backends + ".sage_sm120_ulysses_lowp_a2a_v2g", check_v2g_route=unavailable)
module(backends + ".sage_ulysses_route", check_v2g_route=unavailable)
module(
    "sglang.multimodal_gen.runtime.distributed.parallel_state", get_sp_group=unavailable
)
adapter_name = backends + ".subblock_sparse_attn_sm120_sage_a2a"
load(adapter_name, SG / (adapter_name.replace(".", "/") + ".py"))

import pytest
import shutil

if sys.platform == "darwin":
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0")

# Copy test modules to avoid CUDA-wide parent conftest imports. Fixture source
# comes from the upstream conftest, not a reimplementation.
with tempfile.TemporaryDirectory(prefix="h3-host-tests-") as temp:
    target = Path(temp)
    fixtures = (FI / "tests/comm/conftest.py").read_text()
    fixtures = fixtures[: fixtures.index('"""\nShared test utilities')]
    (target / "conftest.py").write_text(fixtures)
    names = [
        "test_ulysses_communicator.py",
        "test_ulysses_topology.py",
        "test_ulysses_lowp.py",
        "test_ulysses_lowp_boundary.py",
    ]
    for name in names:
        shutil.copyfile(FI / "tests/comm" / name, target / name)
    for name in ["test_flashinfer_ulysses_a2a.py", "test_cake_sage_a2a_compute.py"]:
        shutil.copyfile(SG / "sglang/multimodal_gen/test/unit" / name, target / name)
    raise SystemExit(
        pytest.main(
            [
                str(target),
                "--confcutdir=" + temp,
                "-q",
                "--tb=short",
                "-k",
                "not resolve_2rank",
            ]
        )
    )
