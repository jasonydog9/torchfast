import math
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.utils.cpp_extension import load
import os

_EXT_DIR = os.path.join(os.path.dirname(__file__), "..", "csrc")
_FLAGS   = ["-O3", "-march=native", "-ffast-math"]


def _load(name, src):
    return load(
        name=name,
        sources=[os.path.join(_EXT_DIR, src)],
        extra_cflags=_FLAGS,
        verbose=False,
    )


_fused_linear_ext  = None
_norm_ext          = None
_attention_ext     = None
_focal_ext         = None


def _get_fused_linear():
    global _fused_linear_ext
    if _fused_linear_ext is None:
        _fused_linear_ext = _load("fused_linear_act", "fused_linear_act.cpp")
    return _fused_linear_ext


def _get_norm():
    global _norm_ext
    if _norm_ext is None:
        _norm_ext = _load("fast_normalization", "normalization.cpp")
    return _norm_ext


def _get_attention():
    global _attention_ext
    if _attention_ext is None:
        _attention_ext = _load("fast_attention", "attention.cpp")
    return _attention_ext


def _get_focal():
    global _focal_ext
    if _focal_ext is None:
        _focal_ext = _load("focal_loss", "losses.cpp")
    return _focal_ext


# ---------------------------------------------------------------------------
# FusedLinear
# ---------------------------------------------------------------------------

class _FusedLinearActFn(Function):
    @staticmethod
    def forward(ctx, input, weight, bias, activation):
        ext = _get_fused_linear()
        # C++ forward now returns (out, pre_act).
        # pre_act = X @ W.T + b  BEFORE activation — saved here so backward
        # never has to recompute the BLAS call.  Costs one extra [N,M] clone
        # in forward; saves a full BLAS GEMM (~3-4ms) per backward call.
        out, pre_act = ext.forward(input, weight, bias, activation)
        ctx.save_for_backward(input, weight, pre_act)
        ctx.activation = activation
        return out

    @staticmethod
    def backward(ctx, grad_out):
        input, weight, pre_act = ctx.saved_tensors
        ext = _get_fused_linear()
        # New backward signature: (grad_out, pre_act, input, weight, act)
        # pre_act replaces the old (input, weight, bias) recompute path.
        g_in, g_w, g_b = ext.backward(
            grad_out.contiguous(), pre_act, input, weight, ctx.activation
        )
        return g_in, g_w, g_b, None


class FusedLinear(nn.Module):
    """Drop-in for nn.Linear + nn.GELU/ReLU/SiLU fused into one kernel pass."""

    def __init__(self, in_features: int, out_features: int, activation: str = "gelu"):
        super().__init__()
        assert activation in ("gelu", "relu", "silu")
        self.in_features  = in_features
        self.out_features = out_features
        self.activation   = activation
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(out_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _FusedLinearActFn.apply(
            x.contiguous().float(),
            self.weight.contiguous().float(),
            self.bias.contiguous().float(),
            self.activation,
        )

    def extra_repr(self):
        return f"in={self.in_features}, out={self.out_features}, act={self.activation}"


# ---------------------------------------------------------------------------
# FastLayerNorm
# ---------------------------------------------------------------------------

class _LayerNormFn(Function):
    @staticmethod
    def forward(ctx, input, weight, bias, eps):
        ext = _get_norm()
        out, mean, rstd = ext.forward(input, weight, bias, eps)
        ctx.save_for_backward(input, weight, mean, rstd)
        ctx.eps = eps
        return out

    @staticmethod
    def backward(ctx, grad_out):
        input, weight, mean, rstd = ctx.saved_tensors
        ext = _get_norm()
        g_in, g_w, g_b = ext.backward(
            grad_out.contiguous(), input, weight, mean, rstd, ctx.eps
        )
        return g_in, g_w, g_b, None


class FastLayerNorm(nn.Module):
    """Drop-in for nn.LayerNorm (1-D feature axis)."""

    def __init__(self, features: int, eps: float = 1e-5):
        super().__init__()
        self.features = features
        self.eps      = eps
        self.weight   = nn.Parameter(torch.ones(features))
        self.bias     = nn.Parameter(torch.zeros(features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x2d = x.reshape(-1, self.features).contiguous().float()
        out = _LayerNormFn.apply(
            x2d,
            self.weight.contiguous().float(),
            self.bias.contiguous().float(),
            self.eps,
        )
        return out.reshape(shape)

    def extra_repr(self):
        return f"features={self.features}, eps={self.eps}"


# ---------------------------------------------------------------------------
# FastAttention
# ---------------------------------------------------------------------------

class FastAttention(nn.Module):
    """
    Drop-in for nn.MultiheadAttention (self-attention, same Q/K/V source).
    Uses scaled dot-product attention kernel (CPU).
    """

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim  = embed_dim
        self.num_heads  = num_heads
        self.head_dim   = embed_dim // num_heads
        self.q_proj     = nn.Linear(embed_dim, embed_dim)
        self.k_proj     = nn.Linear(embed_dim, embed_dim)
        self.v_proj     = nn.Linear(embed_dim, embed_dim)
        self.out_proj   = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        query: torch.Tensor,          # [B, S, E]
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        B, S, E = query.shape
        H, D    = self.num_heads, self.head_dim

        def _project_split(proj, x):
            return proj(x).reshape(B, S, H, D).permute(0, 2, 1, 3).contiguous().float()

        Q = _project_split(self.q_proj, query)
        K = _project_split(self.k_proj, key)
        V = _project_split(self.v_proj, value)

        ext = _get_attention()
        out = ext.forward(Q, K, V, attn_mask)  # [B, H, S, D]
        out = out.permute(0, 2, 1, 3).reshape(B, S, E)
        return self.out_proj(out)

    def extra_repr(self):
        return f"embed_dim={self.embed_dim}, num_heads={self.num_heads}"


# ---------------------------------------------------------------------------
# FocalLoss
# ---------------------------------------------------------------------------

class _FocalLossFn(Function):
    @staticmethod
    def forward(ctx, logits, targets, alpha, gamma):
        ext = _get_focal()
        loss = ext.forward(logits, targets, alpha, gamma)
        ctx.save_for_backward(logits, targets)
        ctx.alpha = alpha
        ctx.gamma = gamma
        return loss

    @staticmethod
    def backward(ctx, grad_out):
        logits, targets = ctx.saved_tensors
        ext = _get_focal()
        g = ext.backward(grad_out.contiguous(), logits, targets, ctx.alpha, ctx.gamma)
        return g, None, None, None


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.
    Drop-in for nn.CrossEntropyLoss with class-imbalance handling.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return _FocalLossFn.apply(
            logits.contiguous().float(),
            targets.contiguous().long(),
            self.alpha,
            self.gamma,
        )

    def extra_repr(self):
        return f"alpha={self.alpha}, gamma={self.gamma}"
