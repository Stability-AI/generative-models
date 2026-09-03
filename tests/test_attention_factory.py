import torch

import sgm.modules.attention as attention_module
from sgm.modules.diffusionmodules.model import (
    MemoryEfficientCrossAttentionWrapper,
    make_attn,
)


class _ReferenceOps:
    @staticmethod
    def memory_efficient_attention(q, k, v, attn_bias=None, op=None):
        scores = q @ k.transpose(-2, -1) / q.shape[-1] ** 0.5
        return torch.softmax(scores, dim=-1) @ v


class _ReferenceXformers:
    __version__ = "0.0.21"
    ops = _ReferenceOps()


def test_make_attn_selects_cross_attention_without_mutating_kwargs():
    kwargs = {"context_dim": 4, "heads": 1, "dim_head": 4}
    layer = make_attn(4, attn_type="memory-efficient-cross-attn", attn_kwargs=kwargs)

    assert isinstance(layer, MemoryEfficientCrossAttentionWrapper)
    assert kwargs == {"context_dim": 4, "heads": 1, "dim_head": 4}


def test_cross_attention_wrapper_preserves_spatial_residual(monkeypatch):
    monkeypatch.setattr(attention_module, "xformers", _ReferenceXformers, raising=False)
    layer = MemoryEfficientCrossAttentionWrapper(
        query_dim=4, context_dim=4, heads=1, dim_head=4
    )

    with torch.no_grad():
        identity = torch.eye(4)
        layer.to_q.weight.copy_(identity)
        layer.to_k.weight.copy_(identity)
        layer.to_v.weight.copy_(identity)
        layer.to_out[0].weight.copy_(identity)
        layer.to_out[0].bias.zero_()

    x = torch.arange(32, dtype=torch.float32).reshape(2, 4, 2, 2) / 10
    context = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) / 10
    output = layer(x, context=context)

    query = x.flatten(2).transpose(1, 2)
    scores = query @ context.transpose(-2, -1) / 4**0.5
    expected_attention = torch.softmax(scores, dim=-1) @ context
    expected = x + expected_attention.transpose(1, 2).reshape_as(x)

    assert output.shape == x.shape
    torch.testing.assert_close(output, expected)
