"""Models: the LayerNorm-free transformer of spec §6 and the App. E linear models.

Parameter names and shapes match the authors' released checkpoints, so
``load_state_dict(..., strict=True)`` works on them directly.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

ACTS = {"ReLU": F.relu, "GeLU": F.gelu, "Tanh": torch.tanh}


def _randn(*shape: int, scale: float, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    return torch.randn(*shape, generator=generator) * scale


class Embed(nn.Module):
    """Token embedding table, stored transposed as (d_model, d_vocab)."""

    def __init__(self, d_vocab: int, d_model: int, generator=None):
        super().__init__()
        self.W_E = nn.Parameter(_randn(d_model, d_vocab, scale=1 / math.sqrt(d_model), generator=generator))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.W_E.T[tokens]


class Unembed(nn.Module):
    def __init__(self, d_vocab: int, d_model: int, generator=None):
        super().__init__()
        self.W_U = nn.Parameter(_randn(d_model, d_vocab, scale=1 / math.sqrt(d_vocab), generator=generator))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.W_U


class PosEmbed(nn.Module):
    def __init__(self, n_ctx: int, d_model: int, generator=None):
        super().__init__()
        self.W_pos = nn.Parameter(_randn(n_ctx, d_model, scale=1 / math.sqrt(d_model), generator=generator))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.W_pos[: x.shape[-2]]


class Attention(nn.Module):
    """Bidirectional attention whose post-softmax matrix is blended with all-ones.

    ``A = attn_coeff * softmax(scores) + (1 - attn_coeff)`` -- the constant is added to
    every entry (J = ones, not ones / n), so rows sum to 2 - attn_coeff at n_ctx = 2.
    The causal ``mask`` buffer is created so checkpoints load, but never applied.
    """

    def __init__(self, d_model: int, n_heads: int, d_head: int, n_ctx: int, attn_coeff: float, generator=None):
        super().__init__()
        s = 1 / math.sqrt(d_model)
        self.W_Q = nn.Parameter(_randn(n_heads, d_head, d_model, scale=s, generator=generator))
        self.W_K = nn.Parameter(_randn(n_heads, d_head, d_model, scale=s, generator=generator))
        self.W_V = nn.Parameter(_randn(n_heads, d_head, d_model, scale=s, generator=generator))
        self.W_O = nn.Parameter(_randn(d_model, n_heads * d_head, scale=s, generator=generator))
        self.register_buffer("mask", torch.tril(torch.ones(n_ctx, n_ctx)))
        self.d_head = d_head
        self.attn_coeff = attn_coeff

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = torch.einsum("hkd,bpd->bhpk", self.W_Q, x)
        k = torch.einsum("hkd,bpd->bhpk", self.W_K, x)
        v = torch.einsum("hkd,bpd->bhpk", self.W_V, x)
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.d_head)
        attn = F.softmax(scores, dim=-1)
        attn = attn * self.attn_coeff + (1 - self.attn_coeff)
        z = attn @ v  # (b, h, q, d_head)
        b, _, n, _ = z.shape
        z_flat = z.permute(0, 2, 1, 3).reshape(b, n, -1)  # head-major concatenation
        return z_flat @ self.W_O.T


class MLP(nn.Module):
    def __init__(self, d_model: int, d_mlp: int, act_type: str, generator=None):
        super().__init__()
        self.W_in = nn.Parameter(_randn(d_mlp, d_model, scale=1 / math.sqrt(d_mlp), generator=generator))
        self.b_in = nn.Parameter(torch.zeros(d_mlp))
        self.W_out = nn.Parameter(_randn(d_model, d_mlp, scale=1 / math.sqrt(d_model), generator=generator))
        self.b_out = nn.Parameter(torch.zeros(d_model))
        self.act_type = act_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = ACTS[self.act_type](x @ self.W_in.T + self.b_in)
        return h @ self.W_out.T + self.b_out


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_head, n_ctx, act_type, attn_coeff, generator=None):
        super().__init__()
        self.attn = Attention(d_model, n_heads, d_head, n_ctx, attn_coeff, generator=generator)
        self.mlp = MLP(d_model, 4 * d_model, act_type, generator=generator)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        return x + self.mlp(x)


class Transformer(nn.Module):
    """One- to four-layer transformer, no LayerNorm, read at the last position."""

    def __init__(
        self,
        n_layers: int = 1,
        d_vocab: int = 59,
        d_model: int = 128,
        n_heads: int = 4,
        d_head: Optional[int] = None,
        n_ctx: int = 2,
        act_type: str = "ReLU",
        attn_coeff: float = 0.0,
        seed: Optional[int] = None,
    ):
        super().__init__()
        assert 0.0 <= attn_coeff <= 1.0
        assert act_type in ACTS
        d_head = d_model // n_heads if d_head is None else d_head
        gen = torch.Generator().manual_seed(int(seed)) if seed is not None else None

        self.cfg = dict(
            n_layers=n_layers, d_vocab=d_vocab, d_model=d_model, n_heads=n_heads,
            d_head=d_head, n_ctx=n_ctx, act_type=act_type, attn_coeff=attn_coeff,
        )
        self.embed = Embed(d_vocab, d_model, generator=gen)
        self.pos_embed = PosEmbed(n_ctx, d_model, generator=gen)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_head, n_ctx, act_type, attn_coeff, generator=gen)
             for _ in range(n_layers)]
        )
        self.unembed = Unembed(d_vocab, d_model, generator=gen)

    @property
    def attn_coeff(self) -> float:
        return self.blocks[0].attn.attn_coeff

    def set_attn_coeff(self, alpha: float) -> None:
        for blk in self.blocks:
            blk.attn.attn_coeff = alpha

    def forward_from_embeddings(self, e: torch.Tensor) -> torch.Tensor:
        """Run the model from token embeddings (B, n, d), before W_pos is added."""
        x = self.pos_embed(e)
        for blk in self.blocks:
            x = blk(x)
        return self.unembed(x)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Logits at every position: (B, n_ctx, d_vocab)."""
        return self.forward_from_embeddings(self.embed(tokens))

    def final_logits(self, tokens: torch.Tensor) -> torch.Tensor:
        """Logits at the last position only: (B, d_vocab)."""
        return self.forward(tokens)[:, -1, :]

    def parameters_norm(self) -> float:
        return sum(torch.sum(p * p).item() for p in self.parameters()) ** 0.5


def from_config(cfg: dict, seed: Optional[int] = None) -> Transformer:
    """Build a model from an authors'-style config dict (``code/save/config_*.json``)."""
    d_vocab = cfg["C"] * (2 if cfg.get("diff_vocab") else 1) + (1 if cfg.get("eqn_sign") else 0)
    n_ctx = 2 + (1 if cfg.get("eqn_sign") else 0)
    d_model = cfg["d_model"]
    n_heads = cfg["n_heads"]
    return Transformer(
        n_layers=cfg.get("n_layers", 1),
        d_vocab=d_vocab,
        d_model=d_model,
        n_heads=n_heads,
        d_head=cfg.get("d_head", d_model // n_heads),
        n_ctx=n_ctx,
        act_type=cfg.get("act_fn", "ReLU"),
        attn_coeff=cfg["attn_coeff"],
        seed=seed,
    )


# --------------------------------------------------------------------------------------
# Linear models (App. E, spec §6). Width 256. Parameter names match the authors'
# released linear checkpoints (embed / embed1 / embed2 / unembed / l1 / l2).
# --------------------------------------------------------------------------------------

D_HIDDEN = 256


def _emb(num: int, dim: int, scale_dim: int) -> nn.Embedding:
    emb = nn.Embedding(num, dim)
    with torch.no_grad():
        emb.weight.div_(math.sqrt(scale_dim))
    return emb


class _LinearBase(nn.Module):
    """Shared plumbing: a (p, 256) unembedding table, logits = h @ unembed.weight.T."""

    def __init__(self, p: int = 59, d_hidden: int = D_HIDDEN, seed: Optional[int] = None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(int(seed))
        self.p = p
        self.d_hidden = d_hidden

    def hidden(self, tokens: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.hidden(tokens) @ self.unembed.weight.T


class LinearA(_LinearBase):
    """alpha / MyModelA: U . ReLU(l1(e_a + e_b)), one 256-d table."""

    def __init__(self, p: int = 59, d_hidden: int = D_HIDDEN, seed=None):
        super().__init__(p, d_hidden, seed)
        self.embed = _emb(p, d_hidden, d_hidden // 2)
        self.unembed = _emb(p, d_hidden, d_hidden)
        self.l1 = nn.Linear(d_hidden, d_hidden)

    def hidden(self, tokens):
        e = self.embed(tokens)
        return F.relu(self.l1(e[:, 0] + e[:, 1]))


class LinearX(_LinearBase):
    """alpha' / MyModelX: U . ReLU(l1(e1_a + e2_b)), two 256-d tables."""

    def __init__(self, p: int = 59, d_hidden: int = D_HIDDEN, seed=None):
        super().__init__(p, d_hidden, seed)
        self.embed1 = _emb(p, d_hidden, d_hidden // 2)
        self.embed2 = _emb(p, d_hidden, d_hidden // 2)
        self.unembed = _emb(p, d_hidden, d_hidden)
        self.l1 = nn.Linear(d_hidden, d_hidden)

    def hidden(self, tokens):
        return F.relu(self.l1(self.embed1(tokens[:, 0]) + self.embed2(tokens[:, 1])))


class LinearB(_LinearBase):
    """beta / MyModelB: U . ReLU(l2 . ReLU(l1(e_a + e_b)))."""

    def __init__(self, p: int = 59, d_hidden: int = D_HIDDEN, seed=None):
        super().__init__(p, d_hidden, seed)
        self.embed = _emb(p, d_hidden, d_hidden // 2)
        self.unembed = _emb(p, d_hidden, d_hidden)
        self.l1 = nn.Linear(d_hidden, d_hidden)
        self.l2 = nn.Linear(d_hidden, d_hidden)

    def hidden(self, tokens, second_relu: bool = True):
        e = self.embed(tokens)
        h = self.l2(F.relu(self.l1(e[:, 0] + e[:, 1])))
        return F.relu(h) if second_relu else h

    def forward(self, tokens, second_relu: bool = True):
        """``second_relu=False`` drops the second ReLU, as App. L / target T17 requires."""
        return self.hidden(tokens, second_relu=second_relu) @ self.unembed.weight.T


class LinearC(_LinearBase):
    """gamma / MyModelC: U . ReLU(l2 . ReLU(l1(e_a) + l1(e_b))); l1's bias is added twice."""

    def __init__(self, p: int = 59, d_hidden: int = D_HIDDEN, seed=None):
        super().__init__(p, d_hidden, seed)
        self.embed = _emb(p, d_hidden, d_hidden // 2)
        self.unembed = _emb(p, d_hidden, d_hidden)
        self.l1 = nn.Linear(d_hidden, d_hidden)
        self.l2 = nn.Linear(d_hidden, d_hidden)

    def hidden(self, tokens):
        e = self.embed(tokens)
        h = F.relu(self.l1(e[:, 0]) + self.l1(e[:, 1]))
        return F.relu(self.l2(h))


class LinearD(_LinearBase):
    """delta / MyModelD: U . ReLU(l1([e_a ; e_b])), one 128-d table concatenated to 256."""

    def __init__(self, p: int = 59, d_hidden: int = D_HIDDEN, seed=None):
        super().__init__(p, d_hidden, seed)
        self.embed = _emb(p, d_hidden // 2, d_hidden // 2)
        self.unembed = _emb(p, d_hidden, d_hidden)
        self.l1 = nn.Linear(d_hidden, d_hidden)

    def hidden(self, tokens):
        e = self.embed(tokens)
        return F.relu(self.l1(torch.cat([e[:, 0], e[:, 1]], dim=1)))


LINEAR_MODELS = {
    "alpha": LinearA, "A": LinearA,
    "alpha_prime": LinearX, "X": LinearX,
    "beta": LinearB, "B": LinearB,
    "gamma": LinearC, "C": LinearC,
    "delta": LinearD, "D": LinearD,
}


def linear_from_config(cfg: dict, seed: Optional[int] = None) -> _LinearBase:
    """Build a linear model from an authors'-style config (``model_type`` A/X/B/C/D)."""
    cls = LINEAR_MODELS[cfg["model_type"]]
    return cls(p=cfg.get("n_vocab", cfg["C"]), d_hidden=cfg.get("d_hidden", D_HIDDEN), seed=seed)
