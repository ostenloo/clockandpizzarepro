"""Test 4 (spec §12): circle detection and the isolation operator."""

import math

import numpy as np
import torch

from clockpizza import circles as C
from clockpizza.data import make_dataset
from clockpizza.metrics import P

TWO_PI = 2 * math.pi


def test_keeping_all_pcs_reproduces_the_original_logits():
    from clockpizza.model import Transformer

    m = Transformer(attn_coeff=0.0, seed=1)
    d = make_dataset()
    with torch.no_grad():
        base = m.final_logits(d.inputs)
    iso = C.isolated_model(m, list(range(P)))
    with torch.no_grad():
        got = iso.final_logits(d.inputs)
    assert torch.allclose(base, got, atol=1e-5)


def test_isolation_adds_the_token_mean_back():
    from clockpizza.model import Transformer

    m = Transformer(attn_coeff=0.0, seed=1)
    pca = C.embedding_pca(m.embed.W_E)
    W_iso = C.isolate_embedding(m.embed.W_E, [0, 1], pca=pca)
    assert torch.allclose(W_iso.T[:P].double().mean(dim=0), pca.mu, atol=1e-6)


def test_isolation_leaves_other_weights_untouched():
    from clockpizza.model import Transformer

    m = Transformer(attn_coeff=0.0, seed=1)
    iso = C.isolated_model(m, [0, 1])
    assert torch.equal(iso.pos_embed.W_pos, m.pos_embed.W_pos)
    assert torch.equal(iso.unembed.W_U, m.unembed.W_U)
    assert torch.equal(iso.blocks[0].attn.W_Q, m.blocks[0].attn.W_Q)
    assert not torch.equal(iso.embed.W_E, m.embed.W_E)


def test_k_to_delta_round_trips_for_every_frequency():
    for k in range(1, P):
        delta = C.delta_from_k(k)
        assert 1 <= delta <= P // 2
        assert C.k_from_delta(delta) == C.fold(k)
        assert (k * delta) % P in (1, P - 1)  # folding may flip the sign


def test_accompanying_frequency_and_spacing():
    # released Model A: main k = 17, 3, 15 (delta 7, 20, 4); accompanying k = 25, 6, 29
    for k, delta, k_acc, delta_acc in [(17, 7, 25, 26), (3, 20, 6, 10), (15, 4, 29, 2)]:
        assert C.delta_from_k(k) == delta
        assert C.accompanying_k(k) == k_acc
        assert C.accompanying_delta(delta) == delta_acc


def _synthetic_circle(k, amp=1.0, d=16, noise=0.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    j = torch.arange(P, dtype=torch.float64)
    w = TWO_PI * k / P
    W = torch.zeros(P, d, dtype=torch.float64)
    W[:, 0] = amp * torch.cos(w * j)
    W[:, 1] = amp * torch.sin(w * j)
    if noise:
        W = W + noise * torch.randn(P, d, generator=g, dtype=torch.float64)
    return W


def test_a_synthetic_circle_is_detected_at_its_frequency():
    for k in (1, 3, 15, 17, 25, 29):
        W = _synthetic_circle(k, noise=0.01, seed=k)
        assert C.dominant_freq(W[:, 0]) == k
        # pass as W_E (d, p)
        found = C.find_circles(W.T.float(), n_pcs=6)
        assert found and found[0].k == k
        assert found[0].delta == C.delta_from_k(k)


def test_find_circles_requires_a_same_frequency_partner():
    """A lone high-scoring PC is not a circle (released Model B has one at c = 0.93)."""
    W = torch.cat([_synthetic_circle(17, amp=1.0), _synthetic_circle(5, amp=0.5)[:, :1]], dim=1)
    g = torch.Generator().manual_seed(0)
    W = W + 0.01 * torch.randn_like(W, generator=g)
    found = C.find_circles(W.T.float(), n_pcs=8)
    assert [c.k for c in found] == [17]


def test_accompanying_candidates_follow_the_doubling_rule():
    W = (_synthetic_circle(17, amp=1.0, seed=1)
         + torch.roll(_synthetic_circle(C.accompanying_k(17), amp=0.5, seed=2), 2, dims=1))
    found = C.find_circles(W.T.float(), n_pcs=8)
    assert [c.k for c in found] == [17, C.accompanying_k(17)]
    assert C.accompanying_candidates(found) == [(1, 0)]
    # without a model to fit, classify_circles falls back to the frequency rule alone
    main, acc = C.classify_circles(found)
    assert [c.k for c in main] == [17]
    assert [c.k for c in acc] == [C.accompanying_k(17)]
    assert acc[0].accompanies == 17


def test_formula_fve_against_itself_is_one():
    for name in ("clock", "pizza", "pizza_absdiff", "accompanying"):
        f = C.FORMULAS[name](17)
        assert f.shape == (P, P, P)
        from clockpizza.metrics import fve
        assert abs(fve(f.ravel(), f.ravel()) - 1.0) < 1e-12


def test_logit_heatmap_orders_rows_by_frequency():
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(P * P, P, generator=g, dtype=torch.float64)
    k = 17
    H = C.logit_heatmap(logits, k)
    assert H.shape == (P, P)
    delta = C.delta_from_k(k)
    # row i, column j holds the correct logit of the pair with a - b = i*delta, a + b = j
    for i, j in [(0, 0), (3, 11), (58, 58)]:
        s, dd = j, (i * delta) % P
        a = ((s + dd) * pow(2, P - 2, P)) % P
        b = (s - a) % P
        assert abs(H[i, j] - logits[a * P + b, (a + b) % P].item()) < 1e-12


def test_accuracy_per_diff():
    d = make_dataset()
    g = torch.Generator().manual_seed(0)
    logits = torch.zeros(P * P, P)
    logits[torch.arange(P * P), d.labels] = 1.0  # all correct
    assert np.allclose(C.accuracy_per_diff(logits, d.labels), 1.0)
    logits = torch.randn(P * P, P, generator=g)
    pd = C.accuracy_per_diff(logits, d.labels)
    assert pd.shape == (P,)
    assert abs(pd.mean() - (logits.argmax(-1) == d.labels).float().mean().item()) < 1e-9
