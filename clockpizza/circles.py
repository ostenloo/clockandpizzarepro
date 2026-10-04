"""Circle isolation, frequency identification, formula fits and accompanying
pizzas (spec §9).

Circles are found by *frequency*, never by hardcoded PC index: Fig. 4 and Fig. 13 of
the paper number PCs differently.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, replace
from typing import Optional, Sequence

import numpy as np
import torch

from .metrics import P, circle_score, fve

TWO_PI = 2 * math.pi


# ------------------------------------------------------------------------ PCA of W_E

@dataclass
class EmbeddingPCA:
    mu: torch.Tensor  # (d,) token mean
    proj: torch.Tensor  # (p, r) projections onto PCs
    Vt: torch.Tensor  # (r, d) PC directions, rows in descending variance order

    def scores(self, n_pcs: Optional[int] = None, p: int = P) -> np.ndarray:
        n = self.proj.shape[1] if n_pcs is None else n_pcs
        return np.array([circle_score(self.proj[:, i].numpy(), p) for i in range(n)])


def token_rows(model_or_W, p: int = P) -> torch.Tensor:
    """The p number-token embeddings as rows, from a model or a raw (d, vocab) W_E."""
    if hasattr(model_or_W, "token_embeddings"):
        return model_or_W.token_embeddings(p).detach().cpu().double()
    return model_or_W.T[:p].detach().cpu().double()


def embedding_pca(W_E: torch.Tensor, p: int = P) -> EmbeddingPCA:
    """Centered PCA of the p number-token embeddings.

    Accepts a model or a raw (d_model, d_vocab) W_E.
    """
    W = token_rows(W_E, p)
    mu = W.mean(dim=0)
    _, _, Vt = torch.linalg.svd(W - mu, full_matrices=False)
    return EmbeddingPCA(mu=mu, proj=(W - mu) @ Vt.T, Vt=Vt)


# ------------------------------------------------------------- frequencies and spacings

def fold(k: int, p: int = P) -> int:
    """Fold a frequency or spacing to the range 1..p//2."""
    k %= p
    return min(k, p - k)


def inv_mod(x: int, p: int = P) -> int:
    return pow(int(x) % p, p - 2, p)


def delta_from_k(k: int, p: int = P) -> int:
    """Token-id step between neighbouring points on a circle: delta = k^-1 mod p, folded."""
    return fold(inv_mod(k, p), p)


def k_from_delta(delta: int, p: int = P) -> int:
    return fold(inv_mod(delta, p), p)


def accompanying_k(k: int, p: int = P) -> int:
    """Frequency of the accompanying circle: 2k mod p, folded."""
    return fold(2 * k, p)


def accompanying_delta(delta: int, p: int = P) -> int:
    """Spacing of the accompanying circle: delta * 2^-1 mod p, folded."""
    return fold(delta * inv_mod(2, p), p)


def dominant_freq(v: np.ndarray | torch.Tensor, p: int = P) -> int:
    """Dominant DFT frequency of a PC's p projections, folded to 1..p//2."""
    v = np.asarray(torch.as_tensor(v).detach().cpu().numpy(), dtype=np.float64).ravel()
    mags = np.abs(np.fft.fft(v)[1:p])
    return fold(int(np.argmax(mags)) + 1, p)


@dataclass
class Circle:
    k: int  # frequency, folded to 1..p//2
    delta: int  # token-id spacing, k^-1 mod p folded
    pcs: tuple[int, int]  # the PC pair carrying it (0-indexed)
    scores: tuple[float, float]  # each PC's circle score c_l
    accompanies: Optional[int] = None  # frequency k of the main circle this one accompanies

    @property
    def w(self) -> float:
        return TWO_PI * self.k / P

    def __repr__(self) -> str:
        i, j = self.pcs
        tag = f", accompanies k={self.accompanies}" if self.accompanies is not None else ""
        return (f"Circle(k={self.k}, delta={self.delta}, PCs={i + 1},{j + 1}, "
                f"c={self.scores[0]:.3f},{self.scores[1]:.3f}{tag})")


def find_circles(
    W_E,
    p: int = P,
    n_pcs: int = 20,
    min_score: float = 0.9,
    pca: Optional[EmbeddingPCA] = None,
) -> list[Circle]:
    """Pair up PCs that share a dominant frequency and both score at least ``min_score``.

    Returns circles in PC order, so the leading ones are the highest-variance circles.
    """
    pca = embedding_pca(W_E, p) if pca is None else pca
    n_pcs = min(n_pcs, pca.proj.shape[1])
    scores = pca.scores(n_pcs, p)
    freqs = [dominant_freq(pca.proj[:, i], p) for i in range(n_pcs)]

    circles: list[Circle] = []
    used: set[int] = set()
    for i in range(n_pcs):
        if i in used or scores[i] < min_score:
            continue
        for j in range(i + 1, n_pcs):
            if j in used or scores[j] < min_score or freqs[j] != freqs[i]:
                continue
            circles.append(Circle(k=freqs[i], delta=delta_from_k(freqs[i], p), pcs=(i, j),
                                  scores=(float(scores[i]), float(scores[j]))))
            used.update({i, j})
            break
    return circles


def accompanying_candidates(circles: Sequence[Circle], p: int = P) -> list[tuple[int, int]]:
    """Frequency-rule candidates: (index of the later circle, index of the main circle)
    whenever the later circle's k equals 2k of an earlier one, folded.

    The rule alone is not sufficient: folding makes it fire on coincidences. Released
    Model B's fourth main circle has k = 17 = fold(2 * 21), but its A-formula fit is
    -75% rather than ~97%, so :func:`classify_circles` confirms candidates by fit.
    """
    out: list[tuple[int, int]] = []
    claimed: set[int] = set()
    for j, c in enumerate(circles):
        for i, m in enumerate(circles[:j]):
            if i in claimed or any(i == mi for _, mi in out):
                continue
            if accompanying_k(m.k, p) == c.k:
                out.append((j, i))
                claimed.add(j)
                break
    return out


def classify_circles(
    circles: Sequence[Circle],
    model=None,
    dataset=None,
    pca: Optional[EmbeddingPCA] = None,
    fve_threshold: float = 0.9,
    p: int = P,
) -> tuple[list[Circle], list[Circle]]:
    """Split circles into (main, accompanying).

    A circle is accompanying when its frequency is 2k of an earlier circle *and* the
    A formula at that main circle's w_k explains at least ``fve_threshold`` of its
    isolated logits. Without ``model`` and ``dataset`` the frequency rule is used
    alone, which over-reports (see :func:`accompanying_candidates`).
    """
    candidates = dict(accompanying_candidates(circles, p))
    acc_idx: set[int] = set()
    pairing: dict[int, int] = {}

    for j, i in candidates.items():
        main_k = circles[i].k
        if model is not None and dataset is not None:
            iso = isolated_model(model, list(circles[j].pcs), p=p, pca=pca)
            with torch.no_grad():
                logits = iso.final_logits(dataset.inputs)
            if formula_fve(logits, "accompanying", main_k, p) < fve_threshold:
                continue
        acc_idx.add(j)
        pairing[j] = main_k

    main, acc = [], []
    for j, c in enumerate(circles):
        if j in acc_idx:
            acc.append(replace(c, accompanies=pairing[j]))
        else:
            main.append(c)
    return main, acc


# ------------------------------------------------------------------- isolation operator

def isolated_rows(keep: Sequence[int], pca: EmbeddingPCA) -> torch.Tensor:
    """The isolated token embeddings as rows (p, d), with the token mean added back."""
    return pca.mu + pca.proj[:, list(keep)] @ pca.Vt[list(keep)]


def isolate_embedding(W_E: torch.Tensor, keep: Sequence[int], p: int = P,
                      pca: Optional[EmbeddingPCA] = None) -> torch.Tensor:
    """Keep the given PCs of the number-token embeddings and add the token mean back.

    Returns a new W_E (d, d_vocab); non-number tokens, if any, are left untouched.
    """
    pca = embedding_pca(W_E, p) if pca is None else pca
    out = W_E.detach().clone()
    out[:, :p] = isolated_rows(keep, pca).T.to(out.dtype)
    return out


def isolated_model(model, keep: Sequence[int], p: int = P, pca: Optional[EmbeddingPCA] = None):
    """A deep copy of ``model`` with its number-token embeddings isolated to ``keep``.

    Every other weight -- W_pos included -- is unchanged. Works for the transformer
    (whose table is ``embed.W_E``, stored transposed) and for the App. E linear models
    (whose table is an ``nn.Embedding`` with tokens as rows).
    """
    pca = embedding_pca(model, p) if pca is None else pca
    out = copy.deepcopy(model)
    rows = isolated_rows(keep, pca)
    with torch.no_grad():
        if hasattr(out, "embed") and hasattr(out.embed, "W_E"):
            out.embed.W_E[:, :p] = rows.T.to(out.embed.W_E.dtype)
        else:
            out.embed_table.weight[:p] = rows.to(out.embed_table.weight.dtype)
    return out


def circle_pcs(circles: Sequence[Circle]) -> list[int]:
    """Flatten circles to the PC indices they occupy."""
    return [i for c in circles for i in c.pcs]


# ---------------------------------------------------------------------------- formulas

def _abc(p: int = P):
    a = np.arange(p).reshape(p, 1, 1)
    b = np.arange(p).reshape(1, p, 1)
    c = np.arange(p).reshape(1, 1, p)
    return a, b, c


def q_clock(k: int, p: int = P) -> np.ndarray:
    """cos(w_k (a + b - c)) over all p^3 triples."""
    w = TWO_PI * k / p
    a, b, c = _abc(p)
    return np.cos(w * (a + b - c)) * np.ones((p, p, p))


def q_pizza(k: int, p: int = P) -> np.ndarray:
    """|cos(w_k (a - b) / 2)| cos(w_k (a + b - c)) -- the idealized Table 2 formula."""
    w = TWO_PI * k / p
    a, b, c = _abc(p)
    return np.abs(np.cos(w * (a - b) / 2)) * np.cos(w * (a + b - c))


def q_pizza_absdiff(k: int, p: int = P) -> np.ndarray:
    """The released notebook's "Qpizza": App. A's abs(cos) - abs(sin) construction."""
    w = TWO_PI * k / p
    a, b, c = _abc(p)
    s0 = np.cos(w * a) + np.cos(w * b)
    s1 = np.sin(w * a) + np.sin(w * b)
    co, si = np.cos(w * c / 2), np.sin(w * c / 2)
    return np.abs(co * s0 + si * s1) - np.abs(-si * s0 + co * s1)


def q_accompanying(k: int, p: int = P) -> np.ndarray:
    """A_abc for an accompanying circle, with w_k of the main circle it accompanies."""
    w = TWO_PI * k / p
    a, b, c = _abc(p)
    s0 = np.cos(2 * w * a) + np.cos(2 * w * b)
    s1 = np.sin(2 * w * a) + np.sin(2 * w * b)
    return -(np.cos(w * c) * s0 + np.sin(w * c) * s1)


FORMULAS = {"clock": q_clock, "pizza": q_pizza, "pizza_absdiff": q_pizza_absdiff,
            "accompanying": q_accompanying}


def logit_tensor(logits: torch.Tensor, p: int = P) -> np.ndarray:
    """Reshape last-position logits for all p^2 pairs into a (p, p, p) [a, b, c] tensor."""
    return logits[:, :p].detach().cpu().double().numpy().reshape(p, p, p)


def formula_fve(logits: torch.Tensor, formula: str, k: int, p: int = P) -> float:
    """FVE of a formula against the model's logits over all p^3 triples."""
    return fve(logit_tensor(logits, p).ravel(), FORMULAS[formula](k, p).ravel())


# ------------------------------------------------------- isolated heatmaps and accuracy

def logit_heatmap(logits: torch.Tensor, k: int, p: int = P) -> np.ndarray:
    """Correct-logit heatmap: row i is (a - b) = i * delta, column j is (a + b) = j.

    Equivalently rows are ordered by k * (a - b) mod p, which the paper labels "(a - b)/delta".
    """
    from .metrics import correct_logit_matrix

    M = correct_logit_matrix(logits, p).T.double().numpy()  # rows (a - b), cols (a + b)
    delta = delta_from_k(k, p)
    return np.stack([M[(i * delta) % p] for i in range(p)])


def accuracy_per_diff(logits: torch.Tensor, labels: torch.Tensor, p: int = P) -> np.ndarray:
    """Accuracy for each value of (a - b) mod p; index d holds the accuracy at a - b = d."""
    idx = torch.arange(p * p)
    a, b = idx // p, idx % p
    correct = (logits[:, :p].argmax(dim=-1) == labels).double()
    diff = (a - b) % p
    out = np.zeros(p)
    for d in range(p):
        out[d] = correct[diff == d].mean().item()
    return out
