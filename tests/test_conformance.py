"""Test 5 (spec §12): conformance on the authors' released checkpoints.

Fixtures live in the official repo (``third_party/pizza``); the whole module skips if
it has not been cloned. This test gates M1.
"""

import json
import pathlib

import pytest
import torch

from clockpizza import circles as C
from clockpizza import metrics as M
from clockpizza.data import make_dataset
from clockpizza.metrics import P
from clockpizza.model import from_config, linear_from_config

SAVE = pathlib.Path("third_party/pizza/code/save")
pytestmark = pytest.mark.skipif(not SAVE.is_dir(), reason="official repo not cloned")

TOL = 1e-3


def load(rid: str):
    cfg = json.loads((SAVE / f"config_{rid}.json").read_text())
    build = linear_from_config if "model_type" in cfg else from_config
    model = build(cfg)
    sd = torch.load(SAVE / f"model_{rid}.pt", map_location="cpu")
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model, cfg


def metrics_of(rid: str):
    model, cfg = load(rid)
    data = make_dataset(diff_vocab=bool(cfg.get("diff_vocab")), eqn_sign=bool(cfg.get("eqn_sign")))
    with torch.no_grad():
        logits = model.final_logits(data.inputs)
    return dict(
        accuracy=M.accuracy(logits, data.labels),
        di=M.distance_irrelevance_from_logits(logits),
        di_top_wrong=M.distance_irrelevance_from_logits(logits, variant="top_wrong"),
        gs=M.gradient_symmetricity(model, diff_vocab=bool(cfg.get("diff_vocab")),
                                   eqn_sign=bool(cfg.get("eqn_sign"))),
        circularity=M.model_circularity(model),
    )


# ------------------------------------------------------------------- M1 scalar metrics

@pytest.mark.parametrize(
    "rid,expected",
    [
        ("p99zdpze5l", dict(accuracy=1.0, di=0.1717, gs=0.9937, circularity=0.9978)),  # Model A
        ("l8k1hzciux", dict(accuracy=1.0, di=0.8483, gs=0.3336, circularity=0.9989)),  # Model B
        ("xdgs2cjbtr", dict(di=0.1563, gs=0.9946)),  # d = 1024, alpha = 1
    ],
)
def test_released_checkpoint_metrics(rid, expected):
    got = metrics_of(rid)
    for key, want in expected.items():
        assert abs(got[key] - want) < TOL, f"{rid} {key}: got {got[key]:.4f}, want {want}"


def test_every_released_checkpoint_loads_strict():
    rids = sorted(p.stem.removeprefix("config_") for p in SAVE.glob("config_*.json"))
    assert len(rids) >= 19
    for rid in rids:
        load(rid)  # raises on any key or shape mismatch


# --------------------------------------------------------------- circles of Model A / B

def _circles(rid):
    model, cfg = load(rid)
    pca = C.embedding_pca(model.embed.W_E)
    found = C.find_circles(model.embed.W_E, pca=pca)
    data = make_dataset(diff_vocab=bool(cfg.get("diff_vocab")), eqn_sign=bool(cfg.get("eqn_sign")))
    return model, pca, found, data


def test_model_a_circle_frequencies():
    model, pca, found, data = _circles("p99zdpze5l")
    main, acc = C.classify_circles(found, model, data, pca=pca)
    assert [(c.k, c.delta) for c in main] == [(17, 7), (3, 20), (15, 4)]
    assert [(c.k, c.delta) for c in acc] == [(25, 26), (6, 10), (29, 2)]
    assert [c.pcs for c in main] == [(0, 1), (2, 3), (4, 5)]
    assert [c.pcs for c in acc] == [(6, 7), (8, 9), (10, 11)]


def test_model_b_circle_frequencies():
    model, pca, found, data = _circles("l8k1hzciux")
    main, acc = C.classify_circles(found, model, data, pca=pca)
    assert [(c.k, c.delta) for c in main] == [(13, 9), (16, 11), (21, 14), (17, 7)]
    assert acc == []


# ------------------------------------------------------------------- Tables 2 and 3 FVE

TABLE2 = [  # (k, FVE clock, FVE pizza, FVE abs-difference variant), in percent
    (17, 75.41, 99.18, 98.31),
    (3, 75.62, 99.18, 98.31),
    (15, 75.38, 99.28, 98.41),
]
TABLE3 = [(25, 17, 97.56), (6, 3, 97.23), (29, 15, 97.69)]  # (k_acc, k_main, FVE of A)


def test_table2_fve():
    model, pca, found, data = _circles("p99zdpze5l")
    main, _ = C.classify_circles(found, model, data, pca=pca)
    for circle, (k, clock, pizza, absdiff) in zip(main, TABLE2):
        assert circle.k == k
        iso = C.isolated_model(model, list(circle.pcs), pca=pca)
        with torch.no_grad():
            logits = iso.final_logits(data.inputs)
        assert abs(C.formula_fve(logits, "clock", k) * 100 - clock) < 0.05
        assert abs(C.formula_fve(logits, "pizza", k) * 100 - pizza) < 0.05
        assert abs(C.formula_fve(logits, "pizza_absdiff", k) * 100 - absdiff) < 0.05


def test_table3_fve():
    model, pca, found, data = _circles("p99zdpze5l")
    _, acc = C.classify_circles(found, model, data, pca=pca)
    for circle, (k_acc, k_main, want) in zip(acc, TABLE3):
        assert circle.k == k_acc and circle.accompanies == k_main
        iso = C.isolated_model(model, list(circle.pcs), pca=pca)
        with torch.no_grad():
            logits = iso.final_logits(data.inputs)
        assert abs(C.formula_fve(logits, "accompanying", k_main) * 100 - want) < 0.05


# ------------------------------------------------------------- isolation accuracies (§9)

def _iso_accuracy(model, pca, keep, data):
    iso = C.isolated_model(model, keep, pca=pca)
    with torch.no_grad():
        logits = iso.final_logits(data.inputs)
    return M.accuracy(logits, data.labels) * 100, logits


def test_model_a_isolation_accuracies():
    model, pca, found, data = _circles("p99zdpze5l")
    main, acc = C.classify_circles(found, model, data, pca=pca)

    for circle, want in zip(main, (34.8, 32.8, 26.0)):
        got, _ = _iso_accuracy(model, pca, list(circle.pcs), data)
        assert abs(got - want) < 0.05

    for keep, want in [
        (C.circle_pcs(main), 99.66),
        (C.circle_pcs(main + acc), 100.0),
        (C.circle_pcs(acc), 16.7),
        ([i for i in range(P) if i not in C.circle_pcs(acc)], 99.68),
    ]:
        got, _ = _iso_accuracy(model, pca, keep, data)
        assert abs(got - want) < 0.05


def test_model_a_first_circle_collapses_at_one_difference():
    model, pca, found, data = _circles("p99zdpze5l")
    main, _ = C.classify_circles(found, model, data, pca=pca)
    _, logits = _iso_accuracy(model, pca, list(main[0].pcs), data)
    per_diff = C.accuracy_per_diff(logits, data.labels)
    assert (per_diff == 0).sum() == 1
    assert per_diff.max() <= 0.80


def test_model_b_isolation_accuracies():
    model, pca, _, data = _circles("l8k1hzciux")
    assert abs(_iso_accuracy(model, pca, [0, 1], data)[0] - 13.5) < 0.05
    assert abs(_iso_accuracy(model, pca, [0, 1, 2, 3, 4, 5], data)[0] - 100.0) < 0.05
