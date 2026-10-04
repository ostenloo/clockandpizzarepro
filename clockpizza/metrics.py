"""Metrics: correct-logit matrix, distance irrelevance, gradient symmetricity,
circularity, accuracy and formula FVE (spec §8).
"""

from __future__ import annotations

import itertools
import random
from typing import Iterable, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

P = 59
CIRCULAR_THRESHOLD = 0.995


# ----------------------------------------------------------------------------- accuracy

def accuracy(logits: torch.Tensor, labels: torch.Tensor, p: int = P) -> float:
    """Top-1 accuracy over the first ``p`` logits."""
    return (logits[:, :p].argmax(dim=-1) == labels).float().mean().item()


# ------------------------------------------------------------- correct-logit matrix / DI

def _pair_grid(p: int) -> tuple[torch.Tensor, torch.Tensor]:
    idx = torch.arange(p * p)
    return idx // p, idx % p


def correct_logit_matrix(logits: torch.Tensor, p: int = P) -> torch.Tensor:
    """M[(a + b) % p, (a - b) % p] = logit of the correct class for pair (a, b).

    ``logits`` holds the last-position logits for all p^2 pairs in row-major order.
    """
    a, b = _pair_grid(p)
    vals = logits[torch.arange(p * p), (a + b) % p]
    M = torch.zeros(p, p, dtype=vals.dtype)
    M[(a + b) % p, (a - b) % p] = vals
    return M


def top_wrong_logit_matrix(logits: torch.Tensor, p: int = P) -> torch.Tensor:
    """Same layout, but holding the largest *incorrect* logit per pair."""
    a, b = _pair_grid(p)
    lg = logits[:, :p].clone()
    lg[torch.arange(p * p), (a + b) % p] = -float("inf")
    vals = lg.max(dim=-1).values
    M = torch.zeros(p, p, dtype=vals.dtype)
    M[(a + b) % p, (a - b) % p] = vals
    return M


def distance_irrelevance(M: torch.Tensor) -> float:
    """Def. 4.2: mean over columns of the within-column std, over the std of all entries.

    Columns are indexed by (a - b) % p, rows by (a + b) % p; population std (ddof = 0).
    """
    M = M.double()
    return (M.std(dim=0, unbiased=False).mean() / M.std(unbiased=False)).item()


def distance_irrelevance_from_logits(logits: torch.Tensor, p: int = P, variant: str = "correct") -> float:
    """``variant='correct'`` for Def. 4.2; ``'top_wrong'`` for the released run table's column."""
    M = correct_logit_matrix(logits, p) if variant == "correct" else top_wrong_logit_matrix(logits, p)
    return distance_irrelevance(M)


# ------------------------------------------------------------------ gradient symmetricity

def gs_triples(p: int = P, n: int = 100, seed: int = 42) -> list[tuple[int, int, int]]:
    """All (a, b, c) in Z_p^3 lexicographically, shuffled with ``random.Random(seed)``, first n."""
    triples = list(itertools.product(range(p), repeat=3))
    random.Random(seed).shuffle(triples)
    return triples[:n]


def gradient_symmetricity(
    model,
    p: int = P,
    triples: Optional[Sequence[tuple[int, int, int]]] = None,
    diff_vocab: bool = False,
    eqn_sign: bool = False,
    device: Optional[torch.device] = None,
) -> float:
    """Def. 4.1: mean cosine similarity of the two input-embedding gradients.

    The two token embeddings are separate leaves (taken before W_pos is added), so
    a = b triples still give two distinct gradients.
    """
    triples = gs_triples(p) if triples is None else triples
    device = next(model.parameters()).device if device is None else device

    a = torch.tensor([t[0] for t in triples], dtype=torch.long, device=device)
    b = torch.tensor([t[1] for t in triples], dtype=torch.long, device=device)
    c = torch.tensor([t[2] for t in triples], dtype=torch.long, device=device)

    vocab = 2 * p if diff_vocab else p
    cols = [a, b + p if diff_vocab else b]
    if eqn_sign:
        cols.append(torch.full_like(a, (vocab + 1) - 1))
    tokens = torch.stack(cols, dim=1)

    e = model.embed(tokens).detach().clone().requires_grad_(True)
    logits = model.forward_from_embeddings(e)[:, -1, :]
    target = logits[torch.arange(len(triples), device=device), c].sum()
    (grad,) = torch.autograd.grad(target, e)

    cos = F.cosine_similarity(grad[:, 0].double(), grad[:, 1].double(), dim=-1)
    return cos.mean().item()


# -------------------------------------------------------------------------- circularity

def circle_score(v: np.ndarray, p: int = P) -> float:
    """Def. B.1 per-component score: 2 |DFT_k(v)|^2 / (p sum v^2), maximized over k, clipped to [0, 1]."""
    v = np.asarray(v, dtype=np.float64)
    denom = p * float(np.sum(v * v))
    if denom == 0.0:
        return 0.0
    power = 2.0 * np.abs(np.fft.fft(v)[1:p]) ** 2 / denom
    return float(min(1.0, max(0.0, power.max())))


def pc_projections(W: torch.Tensor | np.ndarray, n_pcs: int = 4) -> np.ndarray:
    """Centered PCA of the token embeddings (rows = tokens); returns (n_tokens, n_pcs)."""
    W = torch.as_tensor(np.asarray(W), dtype=torch.float64)
    Wc = W - W.mean(dim=0, keepdim=True)
    _, _, Vt = torch.linalg.svd(Wc, full_matrices=False)
    return (Wc @ Vt.T)[:, :n_pcs].numpy()


def circularity(W_num: torch.Tensor | np.ndarray, p: int = P, n_pcs: int = 4) -> float:
    """Def. B.1: mean circle score of the first ``n_pcs`` PCs of the p number-token embeddings."""
    proj = pc_projections(W_num, n_pcs=n_pcs)
    return float(np.mean([circle_score(proj[:, l], p) for l in range(n_pcs)]))


def model_circularity(model, p: int = P, n_pcs: int = 4) -> float:
    """Circularity of a transformer's number-token embeddings (rows of W_E.T for tokens 0..p-1)."""
    W = model.embed.W_E.T[:p].detach().cpu()
    return circularity(W, p=p, n_pcs=n_pcs)


def is_circular(circ: float) -> bool:
    return circ >= CIRCULAR_THRESHOLD


# ---------------------------------------------------------------------------------- FVE

def fve(y: np.ndarray | torch.Tensor, f: np.ndarray | torch.Tensor) -> float:
    """Fraction of variance explained, after standardizing both arrays to mean 0, std 1.

    Equals ``sklearn.metrics.explained_variance_score(standardize(y), standardize(f))``.
    """
    y = _standardize(y)
    f = _standardize(f)
    return float(1.0 - np.var(y - f) / np.var(y))


def _standardize(x: np.ndarray | torch.Tensor) -> np.ndarray:
    x = np.asarray(torch.as_tensor(x).detach().cpu().numpy(), dtype=np.float64).ravel()
    return (x - x.mean()) / x.std()


# ----------------------------------------------------------------------------- labelling

def label(gs: float, di: float) -> str:
    """Our summary convention (spec §8): Pizza / Clock / mixed."""
    if di < 0.4 and gs > 0.98:
        return "pizza"
    if di >= 0.4 and gs <= 0.98:
        return "clock"
    return "mixed"
