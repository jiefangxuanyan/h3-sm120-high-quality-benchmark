"""Isolated CPU check of the model's checkpoint-vs-runtime quantization guard."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace

root = Path(
    os.environ.get(
        "SGLANG_SRC", str(Path(__file__).resolve().parents[1] / "sglang-all")
    )
)
path = root / "python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py"
node = next(
    n
    for n in ast.parse(path.read_text()).body
    if isinstance(n, ast.FunctionDef) and n.name == "_h3_checkpoint_is_quantized"
)
namespace = {}
exec("from __future__ import annotations\n" + ast.unparse(node), namespace)
check = namespace[node.name]
assert not check(None)
cases = 1
for name, flag in [
    ("fp8", "is_checkpoint_fp8_serialized"),
    ("mxfp8", "is_checkpoint_fp8_serialized"),
    ("mxfp4", "is_checkpoint_mxfp4_serialized"),
    ("mxfp4_npu", "is_checkpoint_mxfp4_npu_serialized"),
]:
    for serialized in (False, True):
        config = SimpleNamespace(get_name=lambda: name, **{flag: serialized})
        assert check(config) is serialized
        cases += 1
    assert check(SimpleNamespace(get_name=lambda: name))
    cases += 1
assert check(SimpleNamespace(get_name=lambda: "unknown"))
print(f"{cases + 1} checkpoint guard cases passed")
