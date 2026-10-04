"""Test 3 (spec §12): metric oracles."""

import math

import numpy as np
import torch
import torch.nn as nn

from clockpizza import metrics as M
from clockpizza.metrics import P

TWO_PI = 2 * math.pi


# ------------------------------------------------------------------------ circularity

def _circle_embeddings(freqs=(1, 7), amps=(1.0, 0.6), noise=0.01, d=16, seed=0):
    """Two pure cosine circles plus 1% Gaussian noise.

    The circles get distinct amplitudes: with equal variance the PCA directions are
    degenerate across circles and the components come out as frequency mixtures.
    """
    g = torch.Generator().manual_seed(seed)
    j = torch.arange(P, dtype=torch.float64)
    cols = []
    for k, amp in zip(freqs, amps):
        w = TWO_PI * k / P
        cols += [amp * torch.cos(w * j), amp * torch.sin(w * j)]
    W = torch.zeros(P, d, dtype=torch.float64)
    W[:, : len(cols)] = torch.stack(cols, dim=1)
    return W + noise * torch.randn(P, d, generator=g, dtype=torch.float64)


def test_circularity_of_pure_circles_is_near_one():
    assert M.circularity(_circle_embeddings()) > 0.999


def test_circularity_of_gaussian_embeddings_is_low():
    g = torch.Generator().manual_seed(0)
    scores = [M.circularity(torch.randn(P, 128, generator=g, dtype=torch.float64)) for _ in range(5)]
    assert max(scores) < 0.3
    assert abs(float(np.mean(scores)) - 0.14) < 0.08


def test_circular_threshold():
    assert M.is_circular(0.995) and not M.is_circular(0.9949)


# ----------------------------------------------------------------- distance irrelevance

def test_di_is_zero_when_logits_depend_only_on_the_difference():
    d = torch.arange(P, dtype=torch.float64)
    M_mat = torch.abs(torch.cos(TWO_PI * d / P / 2)).repeat(P, 1)  # constant down each column
    assert M.distance_irrelevance(M_mat) < 1e-12


def test_di_is_one_when_logits_depend_only_on_the_sum():
    g = torch.Generator().manual_seed(0)
    s = torch.randn(P, 1, generator=g, dtype=torch.float64)
    M_mat = s.repeat(1, P) + 1e-9 * torch.randn(P, P, generator=g, dtype=torch.float64)
    assert abs(M.distance_irrelevance(M_mat) - 1.0) < 1e-6


def test_correct_and_top_wrong_matrices_place_values_correctly():
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(P * P, P, generator=g, dtype=torch.float64)
    Mc = M.correct_logit_matrix(logits)
    Mw = M.top_wrong_logit_matrix(logits)
    for i in (0, 1, 123, 3480):
        a, b = i // P, i % P
        r, c = (a + b) % P, (a - b) % P
        assert Mc[r, c] == logits[i, (a + b) % P]
        row = logits[i].clone()
        row[(a + b) % P] = -float("inf")
        assert Mw[r, c] == row.max()


# ------------------------------------------------------------------ gradient symmetricity

class _SumToy(nn.Module):
    """Logits depend only on e_a + e_b, so both input gradients coincide: GS = 1."""

    def __init__(self, d=8, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.W_E_T = nn.Parameter(torch.randn(P, d))
        self.lin = nn.Linear(d, P)

    def embed_tokens(self, tokens):
        return self.W_E_T[tokens]

    def forward_from_embeddings(self, e):
        out = self.lin(e.sum(dim=1))
        return out.unsqueeze(1).expand(-1, e.shape[1], -1)


class _ClockToy(nn.Module):
    """The bilinear Clock logit of Eq. (1) on unit-circle embeddings."""

    def __init__(self, k=1):
        super().__init__()
        self.w = TWO_PI * k / P
        j = torch.arange(P, dtype=torch.float32)
        self.W_E_T = nn.Parameter(torch.stack([torch.cos(self.w * j), torch.sin(self.w * j)], dim=1))

    def embed_tokens(self, tokens):
        return self.W_E_T[tokens]

    def forward_from_embeddings(self, e):
        x1, y1 = e[:, 0, 0], e[:, 0, 1]
        x2, y2 = e[:, 1, 0], e[:, 1, 1]
        c = torch.arange(P, dtype=e.dtype, device=e.device)
        cos_c, sin_c = torch.cos(self.w * c), torch.sin(self.w * c)
        real = (x1 * x2 - y1 * y2).unsqueeze(1)
        imag = (y1 * x2 + x1 * y2).unsqueeze(1)
        out = real * cos_c + imag * sin_c
        return out.unsqueeze(1).expand(-1, e.shape[1], -1)


def test_gs_triples_are_deterministic_and_lexicographic_before_shuffle():
    t1, t2 = M.gs_triples(), M.gs_triples()
    assert t1 == t2 and len(t1) == 100
    assert all(0 <= x < P for t in t1 for x in t)
    assert len(set(t1)) == 100


def test_gs_is_one_for_a_sum_only_model():
    assert abs(M.gradient_symmetricity(_SumToy()) - 1.0) < 1e-5


def test_gs_of_the_bilinear_clock_logit_matches_its_analytic_value():
    """For the Eq. (1) logit on unit-circle embeddings, GS = mean cos(w_k (a - b))."""
    tr = M.gs_triples()
    a = np.array([t[0] for t in tr], dtype=np.float64)
    b = np.array([t[1] for t in tr], dtype=np.float64)
    for k in (1, 5, 17, 29):
        analytic = float(np.mean(np.cos(TWO_PI * k / P * (a - b))))
        assert abs(M.gradient_symmetricity(_ClockToy(k=k)) - analytic) < 1e-5


def test_gs_of_the_bilinear_clock_logit_is_near_zero():
    """Spec's oracle (about 0.07) is the magnitude averaged over frequency; any single
    frequency is a small signed sampling fluctuation."""
    tr = M.gs_triples()
    a = np.array([t[0] for t in tr], dtype=np.float64)
    b = np.array([t[1] for t in tr], dtype=np.float64)
    mags = [abs(float(np.mean(np.cos(TWO_PI * k / P * (a - b))))) for k in range(1, P)]
    assert abs(float(np.mean(mags)) - 0.07) < 0.02
    assert max(mags) < 0.2
    assert abs(M.gradient_symmetricity(_ClockToy(k=17))) < 0.2


# ---------------------------------------------------------------------------------- FVE

def test_fve_of_a_formula_against_itself_is_one():
    g = torch.Generator().manual_seed(0)
    y = torch.randn(1000, generator=g, dtype=torch.float64).numpy()
    assert abs(M.fve(y, y) - 1.0) < 1e-12
    assert abs(M.fve(y, 3.0 * y + 7.0) - 1.0) < 1e-12  # standardization removes scale/offset


def test_fve_matches_sklearn():
    from sklearn.metrics import explained_variance_score

    g = torch.Generator().manual_seed(0)
    y = torch.randn(500, generator=g, dtype=torch.float64).numpy()
    f = y + 0.5 * torch.randn(500, generator=g, dtype=torch.float64).numpy()
    ys, fs = M._standardize(y), M._standardize(f)
    assert abs(M.fve(y, f) - explained_variance_score(ys, fs)) < 1e-12


# ----------------------------------------------------------------------------- labelling

def test_labels():
    assert M.label(0.99, 0.2) == "pizza"
    assert M.label(0.33, 0.85) == "clock"
    assert M.label(0.99, 0.85) == "mixed"
    assert M.label(0.33, 0.2) == "mixed"
