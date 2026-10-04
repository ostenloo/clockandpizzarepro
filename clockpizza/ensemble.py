"""Stacked-parameter ensemble models (spec §11).

Many runs train in one process by giving every parameter a leading member axis E.
Members share width, depth and variant; they differ in seed, attention rate and
train/validation split. One AdamW over the stacked tensors equals E separate
optimizers because AdamW is elementwise, and the total loss is the *sum* over members
of each member's mean train loss, so each member's gradient equals its solo gradient.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import ACTS, Transformer


class EnsembleTransformer(nn.Module):
    """E independent transformers sharing one set of stacked parameter tensors.

    Built by stacking E solo models, so member e is bitwise identical at
    initialization to ``Transformer(..., seed=seeds[e])``.
    """

    def __init__(self, members: Sequence[Transformer], attn_coeffs: Sequence[float]):
        super().__init__()
        assert len(members) == len(attn_coeffs) and members
        cfg = dict(members[0].cfg)
        for m in members[1:]:
            shared = {k: v for k, v in m.cfg.items() if k != "attn_coeff"}
            assert shared == {k: v for k, v in cfg.items() if k != "attn_coeff"}, \
                "ensemble members must share width, depth and variant"
        self.cfg = cfg
        self.n_members = len(members)
        self.n_layers = cfg["n_layers"]

        def stack(path):
            return nn.Parameter(torch.stack([path(m).detach().clone() for m in members]))

        self.W_E = stack(lambda m: m.embed.W_E)
        self.W_pos = stack(lambda m: m.pos_embed.W_pos)
        self.W_U = stack(lambda m: m.unembed.W_U)
        self.W_Q, self.W_K, self.W_V, self.W_O = (nn.ParameterList() for _ in range(4))
        self.W_in, self.b_in, self.W_out, self.b_out = (nn.ParameterList() for _ in range(4))
        for i in range(self.n_layers):
            self.W_Q.append(stack(lambda m, i=i: m.blocks[i].attn.W_Q))
            self.W_K.append(stack(lambda m, i=i: m.blocks[i].attn.W_K))
            self.W_V.append(stack(lambda m, i=i: m.blocks[i].attn.W_V))
            self.W_O.append(stack(lambda m, i=i: m.blocks[i].attn.W_O))
            self.W_in.append(stack(lambda m, i=i: m.blocks[i].mlp.W_in))
            self.b_in.append(stack(lambda m, i=i: m.blocks[i].mlp.b_in))
            self.W_out.append(stack(lambda m, i=i: m.blocks[i].mlp.W_out))
            self.b_out.append(stack(lambda m, i=i: m.blocks[i].mlp.b_out))

        self.register_buffer("alpha", torch.tensor(list(attn_coeffs), dtype=torch.float32))

    @classmethod
    def from_seeds(cls, seeds: Sequence[int], attn_coeffs: Sequence[float], **model_kwargs):
        members = [Transformer(attn_coeff=a, seed=s, **model_kwargs)
                   for s, a in zip(seeds, attn_coeffs)]
        return cls(members, attn_coeffs)

    # ------------------------------------------------------------------------ forward

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Logits at the last position for every member: (E, B, d_vocab)."""
        d_head = self.cfg["d_head"]
        act = ACTS[self.cfg["act_type"]]
        n = tokens.shape[-1]
        # (E, V, d) indexed by (B, n) -> (E, B, n, d)
        x = self.W_E.transpose(1, 2)[:, tokens] + self.W_pos[:, None, :n, :]
        # the member axis must be first for the blend to broadcast per member
        alpha = self.alpha.view(-1, 1, 1, 1, 1)

        for i in range(self.n_layers):
            q = torch.einsum("ehkd,ebpd->ebhpk", self.W_Q[i], x)
            k = torch.einsum("ehkd,ebpd->ebhpk", self.W_K[i], x)
            v = torch.einsum("ehkd,ebpd->ebhpk", self.W_V[i], x)
            attn = F.softmax(q @ k.transpose(-1, -2) / math.sqrt(d_head), dim=-1)
            attn = attn * alpha + (1 - alpha)
            z = attn @ v  # (E, B, h, n, d_head)
            e_, b_, h_, n_, _ = z.shape
            z_flat = z.permute(0, 1, 3, 2, 4).reshape(e_, b_, n_, h_ * d_head)
            x = x + torch.einsum("edf,ebnf->ebnd", self.W_O[i], z_flat)
            hidden = act(torch.einsum("emd,ebnd->ebnm", self.W_in[i], x) + self.b_in[i][:, None, None, :])
            x = x + torch.einsum("edm,ebnm->ebnd", self.W_out[i], hidden) + self.b_out[i][:, None, None, :]

        return torch.einsum("edv,ebd->ebv", self.W_U, x[:, :, -1, :])

    # ------------------------------------------------------------------------- export

    def export_member(self, e: int, attn_coeff: float | None = None) -> Transformer:
        """Member ``e`` as a standalone model in the §6 layout, so every analysis
        runs on single models."""
        cfg = dict(self.cfg)
        cfg["attn_coeff"] = float(self.alpha[e]) if attn_coeff is None else attn_coeff
        out = Transformer(
            n_layers=cfg["n_layers"], d_vocab=cfg["d_vocab"], d_model=cfg["d_model"],
            n_heads=cfg["n_heads"], d_head=cfg["d_head"], n_ctx=cfg["n_ctx"],
            act_type=cfg["act_type"], attn_coeff=cfg["attn_coeff"],
        )
        with torch.no_grad():
            out.embed.W_E.copy_(self.W_E[e])
            out.pos_embed.W_pos.copy_(self.W_pos[e])
            out.unembed.W_U.copy_(self.W_U[e])
            for i, blk in enumerate(out.blocks):
                blk.attn.W_Q.copy_(self.W_Q[i][e])
                blk.attn.W_K.copy_(self.W_K[i][e])
                blk.attn.W_V.copy_(self.W_V[i][e])
                blk.attn.W_O.copy_(self.W_O[i][e])
                blk.mlp.W_in.copy_(self.W_in[i][e])
                blk.mlp.b_in.copy_(self.b_in[i][e])
                blk.mlp.W_out.copy_(self.W_out[i][e])
                blk.mlp.b_out.copy_(self.b_out[i][e])
        return out.eval()


def ensemble_loss(logits: torch.Tensor, labels: torch.Tensor, train_mask: torch.Tensor
                  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Sum over members of each member's mean train cross-entropy (float64 log_softmax).

    ``logits`` is (E, B, V), ``labels`` is (B,), ``train_mask`` is (E, B) boolean -- one
    train/validation split per member. Returns (total loss, per-member losses).
    """
    logprobs = F.log_softmax(logits.to(torch.float64), dim=-1)
    lp = logprobs.gather(-1, labels.view(1, -1, 1).expand(logits.shape[0], -1, 1)).squeeze(-1)
    mask = train_mask.to(lp.dtype)
    per_member = -(lp * mask).sum(dim=1) / mask.sum(dim=1)
    return per_member.sum(), per_member
