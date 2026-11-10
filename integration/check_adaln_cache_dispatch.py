"""Exercise real H3 block forward/cache methods through a shared-kwargs wrapper.

CUDA imports are isolated; modulation uses Torch CPU reference arithmetic.
The regression must select distinct per-layer AdaLN rows under Cache-DiT.
"""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
import os
import torch

root = Path(os.environ.get('SGLANG_SRC', Path(__file__).resolve().parents[1] / 'sglang-all'))
base = root / 'python/sglang/multimodal_gen/runtime/models/dits'

def methods(path, cls, names, namespace):
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == cls)
    source = '\n\n'.join(ast.unparse(n) for n in node.body if isinstance(n, ast.FunctionDef) and n.name in names)
    assert len([n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in names]) == len(names)
    exec('from __future__ import annotations\n' + source, namespace)

ns = dict(torch=torch)
methods(base / 'minimax_h3_adaln_cache.py', 'MiniMaxH3AdalnCache', {'begin_step', 'block_for_current_step', 'block_all'}, ns)
class Cache:
    begin_step = ns['begin_step']
    block_for_current_step = ns['block_for_current_step']
    block_all = ns['block_all']
    _step_block_params = None
    num_layers, hidden_size = 3, 2
    block_params = torch.arange(2 * 2 * 3 * 6 * 2, dtype=torch.float32).reshape(2, 2, 3, 12) / 100
cache = Cache()
try:
    cache.block_for_current_step(0)
except ValueError:
    pass
else:
    raise AssertionError('Uninitialized step must fail')

ns = dict(torch=torch, _BF16_DTYPE=torch.float32,
    _accepts_mxfp8_input=lambda _: False,
    _modulate_scale_shift=lambda x, shift, scale, ids, **kw: x * (1 + scale[ids]) + shift[ids],
    _modulate_gate=lambda x, gate, h, ids, **kw: x + gate[ids] * h)
methods(base / 'minimax_h3.py', 'MiniMaxH3DiTBlock', {'forward'}, ns)
forward = ns['forward']
blocks = [NS(adaln_proj=None, _adaln_cache_ref=(cache,), adaln_layer_index=i,
             norm1=lambda x: x, norm2=lambda x: x, preserve_input_for_cache_dit=True,
             attn=NS(qkv_proj=None), mlp=NS(fc1=None)) for i in range(3)]
class Identity:
    qkv_proj = fc1 = None
    def __call__(self, x, **kwargs): return x
for b in blocks:
    b.attn = b.mlp = Identity()
kwargs = dict(adaln_input=torch.zeros(2, 1), combined_indices=torch.tensor([0, 1, 0, 1]),
              rope_cache=None, cu_seqlens=None, max_seqlen=4)
for slot in (0, 1, 0):
    cache.begin_step(cache_plan_index=torch.tensor(slot), num_timesteps=2)
    expected = cache.block_all(cache_plan_index=torch.tensor(slot), num_timesteps=2)
    x = torch.ones(4, 2)
    oracle = x.clone()
    # Cache-DiT semantics: forward identical kwargs to each real block.
    for i, b in enumerate(blocks):
        x = forward(b, x, **kwargs)
        oracle = forward(b, oracle, adaln_params=expected[i], **kwargs)
    torch.testing.assert_close(x, oracle, rtol=0, atol=0)
    wrong = torch.ones(4, 2)
    for b in blocks:
        wrong = forward(b, wrong, adaln_params=expected[0], **kwargs)
    assert not torch.equal(x, wrong), 'Oracle must detect the old layer-0 bug'
    assert '_adaln_cache_ref' not in cache.__dict__
print('AdaLN shared-kwargs dispatch: 3 plans, 3 distinct layers, exact CPU oracle passed')
