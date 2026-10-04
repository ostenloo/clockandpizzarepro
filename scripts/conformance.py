#!/usr/bin/env python
"""M1 conformance: reproduce the released checkpoints' metrics, Tables 2-3 and the
§9 isolation accuracies. Prints a Markdown report to stdout.

Usage: python scripts/conformance.py [--save path/to/code/save]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from clockpizza import circles as C  # noqa: E402
from clockpizza import metrics as M  # noqa: E402
from clockpizza.data import make_dataset  # noqa: E402
from clockpizza.metrics import P  # noqa: E402
from clockpizza.model import from_config, linear_from_config  # noqa: E402

EXPECTED = {
    "p99zdpze5l": dict(accuracy=1.0, di=0.1717, gs=0.9937, circularity=0.9978),
    "l8k1hzciux": dict(accuracy=1.0, di=0.8483, gs=0.3336, circularity=0.9989),
    "xdgs2cjbtr": dict(di=0.1563, gs=0.9946),
}
TABLE2 = {17: (75.41, 99.18, 98.31), 3: (75.62, 99.18, 98.31), 15: (75.38, 99.28, 98.41)}
TABLE3 = {25: 97.56, 6: 97.23, 29: 97.69}
ISO_A = {"main #1": 34.8, "main #2": 32.8, "main #3": 26.0, "PCs 1-6 (main)": 99.66,
         "PCs 1-12 (all)": 100.0, "PCs 7-12 (accompanying)": 16.7, "minus accompanying": 99.68}
ISO_B = {"PCs 1-2": 13.5, "PCs 1-6": 100.0}


def load(save: pathlib.Path, rid: str):
    cfg = json.loads((save / f"config_{rid}.json").read_text())
    model = (linear_from_config if "model_type" in cfg else from_config)(cfg)
    model.load_state_dict(torch.load(save / f"model_{rid}.pt", map_location="cpu"), strict=True)
    model.eval()
    return model, cfg


def mark(got: float, want: float, tol: float) -> str:
    return "pass" if abs(got - want) < tol else f"**FAIL** (want {want})"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", default="third_party/pizza/code/save", type=pathlib.Path)
    args = ap.parse_args()
    save = args.save
    if not save.is_dir():
        print(f"fixtures not found at {save}; clone https://github.com/fjzzq2002/pizza "
              f"into third_party/pizza", file=sys.stderr)
        return 2

    failures = 0
    data = make_dataset()

    print("### Checkpoint metrics (tolerance 0.001)\n")
    print("| run id | accuracy | DI (correct) | DI (top-wrong) | GS | circularity | status |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for rid, expected in EXPECTED.items():
        model, cfg = load(save, rid)
        dv, eq = bool(cfg.get("diff_vocab")), bool(cfg.get("eqn_sign"))
        ds = make_dataset(diff_vocab=dv, eqn_sign=eq)
        with torch.no_grad():
            logits = model.final_logits(ds.inputs)
        got = dict(
            accuracy=M.accuracy(logits, ds.labels),
            di=M.distance_irrelevance_from_logits(logits),
            di_top_wrong=M.distance_irrelevance_from_logits(logits, variant="top_wrong"),
            gs=M.gradient_symmetricity(model, diff_vocab=dv, eqn_sign=eq),
            circularity=M.model_circularity(model),
        )
        bad = [k for k, w in expected.items() if abs(got[k] - w) >= 1e-3]
        failures += len(bad)
        status = "pass" if not bad else "**FAIL**: " + ", ".join(bad)
        print(f"| `{rid}` | {got['accuracy']:.4f} | {got['di']:.4f} | {got['di_top_wrong']:.4f} "
              f"| {got['gs']:.4f} | {got['circularity']:.4f} | {status} |")

    rids = sorted(p.stem.removeprefix("config_") for p in save.glob("config_*.json"))
    loaded = 0
    for rid in rids:
        try:
            load(save, rid)
            loaded += 1
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"\n**FAIL** strict load of `{rid}`: {exc}")
    print(f"\nStrict `load_state_dict` on all released checkpoints: {loaded}/{len(rids)}.\n")

    model, _ = load(save, "p99zdpze5l")
    pca = C.embedding_pca(model.embed.W_E)
    found = C.find_circles(model.embed.W_E, pca=pca)
    main, acc = C.classify_circles(found, model, data, pca=pca)

    def iso_logits(keep):
        iso = C.isolated_model(model, keep, pca=pca)
        with torch.no_grad():
            return iso.final_logits(data.inputs)

    print("### Model A circles, Tables 2 and 3 (tolerance 0.05 points)\n")
    print("| circle | k | delta | PCs | FVE Q_clock | FVE Q_pizza | FVE abs-diff | FVE A | status |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for i, c in enumerate(main + acc, start=1):
        logits = iso_logits(list(c.pcs))
        pcs = f"{c.pcs[0] + 1},{c.pcs[1] + 1}"
        if c.accompanies is None:
            cl, pz, ad = (C.formula_fve(logits, f, c.k) * 100 for f in ("clock", "pizza", "pizza_absdiff"))
            w_cl, w_pz, w_ad = TABLE2[c.k]
            st = [mark(cl, w_cl, 0.05), mark(pz, w_pz, 0.05), mark(ad, w_ad, 0.05)]
            failures += sum(s != "pass" for s in st)
            print(f"| #{i} main | {c.k} | {c.delta} | {pcs} | {cl:.2f}% | {pz:.2f}% | {ad:.2f}% | - "
                  f"| {'pass' if all(s == 'pass' for s in st) else ' '.join(st)} |")
        else:
            a = C.formula_fve(logits, "accompanying", c.accompanies) * 100
            st = mark(a, TABLE3[c.k], 0.05)
            failures += st != "pass"
            print(f"| #{i} acc. of k={c.accompanies} | {c.k} | {c.delta} | {pcs} | - | - | - "
                  f"| {a:.2f}% | {st} |")

    print("\n### Isolation accuracies, all 3,481 pairs (tolerance 0.05 points)\n")
    print("| model | embedding kept | accuracy | released | status |")
    print("| --- | --- | --- | --- | --- |")
    keeps = {f"main #{i + 1}": list(c.pcs) for i, c in enumerate(main)}
    keeps["PCs 1-6 (main)"] = C.circle_pcs(main)
    keeps["PCs 1-12 (all)"] = C.circle_pcs(main + acc)
    keeps["PCs 7-12 (accompanying)"] = C.circle_pcs(acc)
    keeps["minus accompanying"] = [i for i in range(P) if i not in C.circle_pcs(acc)]
    for tag, keep in keeps.items():
        got = M.accuracy(iso_logits(keep), data.labels) * 100
        st = mark(got, ISO_A[tag], 0.05)
        failures += st != "pass"
        print(f"| A (Pizza) | {tag} | {got:.2f}% | {ISO_A[tag]}% | {st} |")

    per_diff = C.accuracy_per_diff(iso_logits(list(main[0].pcs)), data.labels)
    zeros = (per_diff == 0).sum()
    ok = zeros == 1 and per_diff.max() <= 0.80
    failures += not ok
    print(f"| A (Pizza) | main #1, per (a-b) | 0% at {zeros} difference, max {per_diff.max() * 100:.1f}% "
          f"| 0% at one, <=80% | {'pass' if ok else '**FAIL**'} |")

    modelB, _ = load(save, "l8k1hzciux")
    pcaB = C.embedding_pca(modelB.embed.W_E)
    foundB = C.find_circles(modelB.embed.W_E, pca=pcaB)
    mainB, accB = C.classify_circles(foundB, modelB, data, pca=pcaB)
    for tag, keep in {"PCs 1-2": [0, 1], "PCs 1-6": [0, 1, 2, 3, 4, 5]}.items():
        iso = C.isolated_model(modelB, keep, pca=pcaB)
        with torch.no_grad():
            got = M.accuracy(iso.final_logits(data.inputs), data.labels) * 100
        st = mark(got, ISO_B[tag], 0.05)
        failures += st != "pass"
        print(f"| B (Clock) | {tag} | {got:.2f}% | {ISO_B[tag]}% | {st} |")

    print(f"\nModel B circles: {[(c.k, c.delta) for c in mainB]} main, "
          f"{[(c.k, c.delta) for c in accB]} accompanying.")
    print(f"\n**{'M1 conformance passed' if failures == 0 else f'{failures} conformance failures'}.**")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
