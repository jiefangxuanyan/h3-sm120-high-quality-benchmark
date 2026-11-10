"""CPU contract checks of the actual high deployment and cache builder methods."""

import ast
import os
import sys
import types
import warnings
from pathlib import Path
from types import SimpleNamespace as NS

root = Path(
    os.environ.get(
        "SGLANG_SRC", str(Path(__file__).resolve().parents[1] / "sglang-all")
    )
)
config = root / "python/sglang/multimodal_gen/configs/pipeline_configs/minimax_h3.py"
stage = (
    root
    / "python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/denoising.py"
)
platform = NS(
    is_cuda=lambda: True,
    get_device_name=lambda: "NVIDIA Graphics Device",
    get_device_capability=lambda: NS(to_int=lambda: 120),
    is_sm120=lambda: True,
)
namespace = dict(
    warnings=warnings,
    current_platform=platform,
    envs=NS(SGLANG_DIFFUSION_MINIMAX_H3_VAE_OUTPUT_PROJECTION_FP16=False),
)
methods = []
config_class = next(n for n in ast.parse(config.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "MiniMaxH3PipelineConfig")
for node in config_class.body:
    if isinstance(node, ast.FunctionDef) and node.name in (
        "_server_arg_value",
        "validate_quality_deployment",
    ):
        methods.append(ast.unparse(node))
exec(
    "from __future__ import annotations\nclass Config:\n"
    + "\n".join("    " + line for method in methods for line in method.splitlines()),
    namespace,
)
c = namespace["Config"]()
c.text_encoder_configs = [NS(quant_config=None)]
base = dict(
    attention_backend=None,
    backend="sglang",
    component_attention_backends={
        "transformer": "subblock_sparse_attn_sm120_sage_a2a",
        "video_vae": "torch_cudnn_sdpa",
    },
    enable_breakable_cuda_graph=False,
    enable_torch_compile=False,
    is_dit_layerwise_offload_selected=False,
    model_variant="fl2va",
    num_gpus=8,
    performance_mode="speed",
    quantization="fp8",
    transformer_weights_path=None,
    component_quantizations={"video_vae": "fp8"},
    regional_compile=False,
    ring_degree=1,
    sp_degree=8,
    tp_size=1,
    ulysses_degree=8,
    use_fsdp_inference=False,
)
count = 0
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for backend in (
        "subblock_sparse_attn_sm120_sage",
        "subblock_sparse_attn_sm120_sage_a2a",
    ):
        for vq in (None, "fp8"):
            args = {
                **base,
                "component_attention_backends": {
                    **base["component_attention_backends"],
                    "transformer": backend,
                },
                "component_quantizations": {} if vq is None else {"video_vae": vq},
            }
            c.validate_quality_deployment(NS(**args))
            count += 1
    for key, value in [
        ("tp_size", 2),
        ("ring_degree", 2),
        ("num_gpus", 3),
        ("quantization", None),
        ("transformer_weights_path", "other"),
        ("model_variant", "ref2va"),
    ]:
        try:
            c.validate_quality_deployment(NS(**{**base, key: value}))
        except ValueError:
            count += 1
        else:
            raise AssertionError((key, value))
platform.get_device_name = lambda: "NVIDIA H200"
platform.get_device_capability = lambda: NS(to_int=lambda: 90)
old = {
    **base,
    "num_gpus": 4,
    "sp_degree": 4,
    "ulysses_degree": 4,
    "quantization": None,
    "component_attention_backends": {},
    "component_quantizations": {},
}
c.validate_quality_deployment(NS(**old))
count += 1
module = types.ModuleType("sglang.multimodal_gen.runtime.platforms")
module.current_platform = platform
sys.modules[module.__name__] = module
node = next(
    n
    for n in ast.walk(ast.parse(stage.read_text()))
    if isinstance(n, ast.FunctionDef) and n.name == "_build_cache_dit_config"
)
ns = dict(
    MINIMAX_H3_HIGH_QUALITY_CACHE_DIT_CONFIG=(4, 0.04, 1),
    CacheDitConfig=lambda **kwargs: NS(**kwargs),
)
exec("from __future__ import annotations\n" + ast.unparse(node), ns)
for sm120, overrides, expected in [
    (True, {}, (4, 0.04, 1)),
    (
        True,
        {
            "max_warmup_steps": 4,
            "residual_diff_threshold": 0.08,
            "max_continuous_cached_steps": 3,
        },
        (4, 0.04, 1),
    ),
    (False, {}, (4, 0.04, 1)),
]:
    platform.is_sm120 = lambda: sm120
    obj = NS(_minimax_h3_quality="high", _cache_dit_request_overrides=overrides)
    result = ns[node.name](obj, 49, None, "dynamic")
    assert (
        result.max_warmup_steps,
        result.residual_diff_threshold,
        result.max_continuous_cached_steps,
    ) == expected
    assert (result.Fn_compute_blocks, result.Bn_compute_blocks) == (1, 0)
    count += 1
print(count, "high/Cache-DiT contract cases passed")
