import os
import pytest
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

_EXT_DIR = os.path.join(os.path.dirname(__file__), "..", "csrc")
_FLAGS   = ["-O3", "-march=native", "-ffast-math"]

ext = load(
    name="fast_attention_test",
    sources=[os.path.join(_EXT_DIR, "attention.cpp")],
    extra_cflags=_FLAGS,
    verbose=False,
)


def _ref_attention(Q, K, V, mask=None):
    scale  = Q.size(-1) ** -0.5
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, V)


@pytest.mark.parametrize("B,H,S,D", [(1, 1, 4, 8), (2, 4, 16, 32), (4, 8, 32, 64)])
def test_forward_correctness(B, H, S, D):
    Q = torch.randn(B, H, S, D)
    K = torch.randn(B, H, S, D)
    V = torch.randn(B, H, S, D)

    got = ext.forward(Q, K, V, None)
    ref = _ref_attention(Q, K, V)

    torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4)


def test_with_causal_mask():
    B, H, S, D = 2, 2, 8, 16
    Q = torch.randn(B, H, S, D)
    K = torch.randn(B, H, S, D)
    V = torch.randn(B, H, S, D)

    # Upper-triangular causal mask (True = masked out)
    mask = torch.ones(S, S, dtype=torch.bool).triu(diagonal=1)

    got = ext.forward(Q, K, V, mask)
    ref = _ref_attention(Q, K, V, mask.unsqueeze(0).unsqueeze(0))

    torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4)


def test_batch_size_one():
    Q = torch.randn(1, 1, 4, 8)
    K = torch.randn(1, 1, 4, 8)
    V = torch.randn(1, 1, 4, 8)
    out = ext.forward(Q, K, V, None)
    assert out.shape == (1, 1, 4, 8)


def test_large_tensor():
    B, H, S, D = 2, 4, 128, 64
    Q = torch.randn(B, H, S, D)
    K = torch.randn(B, H, S, D)
    V = torch.randn(B, H, S, D)
    out = ext.forward(Q, K, V, None)
    assert out.shape == (B, H, S, D)


def test_gradcheck():
    B, H, S, D = 1, 1, 4, 4
    Q = torch.randn(B, H, S, D, dtype=torch.float64, requires_grad=True)
    K = torch.randn(B, H, S, D, dtype=torch.float64, requires_grad=True)
    V = torch.randn(B, H, S, D, dtype=torch.float64, requires_grad=True)

    assert torch.autograd.gradcheck(_ref_attention, (Q, K, V), eps=1e-6, atol=1e-4)


def test_wrong_dim_raises():
    Q = torch.randn(2, 4, 8)   # 3D, should be 4D
    K = torch.randn(2, 4, 8)
    V = torch.randn(2, 4, 8)
    with pytest.raises(RuntimeError):
        ext.forward(Q, K, V, None)


def test_shape_mismatch_raises():
    Q = torch.randn(2, 4, 8, 16)
    K = torch.randn(2, 4, 8, 32)  # head_dim mismatch
    V = torch.randn(2, 4, 8, 16)
    with pytest.raises(RuntimeError):
        ext.forward(Q, K, V, None)
