#!/usr/bin/env python
"""E1 (spec §10): canonical Pizza vs Clock. 1 layer, d = 128, alpha in {0, 1}, 48 seeds each.

Trains 96 runs on stacked parameters, evaluates T1-T4, then runs every §9 analysis on
the representative circular run per alpha (the one closest to that alpha's median GS
and DI) for T6-T12. Resumable: run ids already in results/runs.jsonl are skipped.

Usage: python scripts/e1.py [--seeds 48] [--chunk 48] [--device cuda]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics as st
import sys
from typing import Any, Optional, Sequence

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from clockpizza import circles as C  # noqa: E402
from clockpizza import metrics as M  # noqa: E402
from clockpizza.data import make_dataset  # noqa: E402
from clockpizza.metrics import P  # noqa: E402
from clockpizza.train import (  # noqa: E402
    RunConfig, append_record, load_registry, load_run, registry_ids, train_ensemble,
)

EXPERIMENT = "E1"


def iqr(xs: Sequence[float]) -> tuple[float, float]:
    xs = sorted(xs)
    if len(xs) < 4:
        return (xs[0], xs[-1]) if xs else (float("nan"), float("nan"))
    q = st.quantiles(xs, n=4)
    return q[0], q[2]


def fmt_dist(xs: Sequence[float]) -> str:
    if not xs:
        return "-"
    lo, hi = iqr(xs)
    return f"{st.median(xs):.4f} ({lo:.4f}-{hi:.4f}, n={len(xs)})"


def representative(runs: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """The circular run closest to the group's median GS and DI.

    Distance is the sum of |x - median| scaled by that metric's IQR, so the two
    metrics contribute comparably regardless of their spread.
    """
    if not runs:
        return None
    gs_med, di_med = st.median(r["gs"] for r in runs), st.median(r["di"] for r in runs)
    gs_lo, gs_hi = iqr([r["gs"] for r in runs])
    di_lo, di_hi = iqr([r["di"] for r in runs])
    gs_scale = max(gs_hi - gs_lo, 1e-6)
    di_scale = max(di_hi - di_lo, 1e-6)
    return min(runs, key=lambda r: abs(r["gs"] - gs_med) / gs_scale
               + abs(r["di"] - di_med) / di_scale)


def train(seeds: int, chunk: int, device: Optional[str], steps: int) -> None:
    done = registry_ids()
    todo = [RunConfig(seed=s, attn_coeff=a, steps=steps, experiment=EXPERIMENT)
            for a in (0.0, 1.0) for s in range(seeds)]
    todo = [c for c in todo if c.run_id not in done]
    if not todo:
        print(f"all {2 * seeds} E1 runs already in the registry\n", flush=True)
        return
    print(f"training {len(todo)} of {2 * seeds} E1 runs in chunks of {chunk}\n", flush=True)
    for i in range(0, len(todo), chunk):
        batch = todo[i:i + chunk]
        print(f"  chunk {i // chunk + 1}: E = {len(batch)} "
              f"(alphas {sorted({c.attn_coeff for c in batch})})", flush=True)
        for rec in train_ensemble(batch, device=device, progress=True):
            append_record(rec)
        print(f"    done ({batch[0].steps} steps)", flush=True)


def analyse(run: dict[str, Any], name: str) -> dict[str, Any]:
    """Run every §9 analysis for one run, print it, and return the quantities targets need."""
    model, _ = load_run(run["run_id"])
    data = make_dataset()
    pca = C.embedding_pca(model.embed.W_E)
    found = C.find_circles(model.embed.W_E, pca=pca)
    main_circles, acc_circles = C.classify_circles(found, model, data, pca=pca)

    def iso_logits(keep):
        m = C.isolated_model(model, keep, pca=pca)
        with torch.no_grad():
            return m.final_logits(data.inputs)

    print(f"\n### {name}: run `{run['run_id']}` (alpha = {run['config']['attn_coeff']}, "
          f"seed {run['seed']})\n")
    print(f"GS {run['gs']:.4f}, DI {run['di']:.4f}, circularity {run['circularity']:.4f}, "
          f"label {run['label']}.\n")
    print("PC circle scores (first 14): " + " ".join(f"{x:.2f}" for x in pca.scores(14)) + "\n")

    print("| circle | k | delta | PCs | accompanies | accuracy alone | FVE clock | FVE pizza "
          "| FVE abs-diff | FVE A |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    out: dict[str, Any] = dict(main=main_circles, acc=acc_circles, circle_accuracy=[],
                              fve_pizza=[], fve_clock=[], fve_absdiff=[], fve_a=[], kept={})
    for i, c in enumerate(main_circles + acc_circles, start=1):
        logits = iso_logits(list(c.pcs))
        a = M.accuracy(logits, data.labels) * 100
        pcs = f"{c.pcs[0] + 1},{c.pcs[1] + 1}"
        if c.accompanies is None:
            cl = C.formula_fve(logits, "clock", c.k) * 100
            pz = C.formula_fve(logits, "pizza", c.k) * 100
            ad = C.formula_fve(logits, "pizza_absdiff", c.k) * 100
            out["circle_accuracy"].append(a)
            out["fve_clock"].append(cl)
            out["fve_pizza"].append(pz)
            out["fve_absdiff"].append(ad)
            print(f"| #{i} main | {c.k} | {c.delta} | {pcs} | - | {a:.2f}% | {cl:.2f}% "
                  f"| {pz:.2f}% | {ad:.2f}% | - |")
        else:
            fa = C.formula_fve(logits, "accompanying", c.accompanies) * 100
            out["fve_a"].append(fa)
            print(f"| #{i} acc. | {c.k} | {c.delta} | {pcs} | k={c.accompanies} | {a:.2f}% "
                  f"| - | - | - | {fa:.2f}% |")

    keeps = {"all main circles": C.circle_pcs(main_circles),
             "all circles": C.circle_pcs(main_circles + acc_circles)}
    if acc_circles:
        keeps["accompanying only"] = C.circle_pcs(acc_circles)
        keeps["full minus accompanying"] = [i for i in range(P)
                                            if i not in C.circle_pcs(acc_circles)]
    if len(C.circle_pcs(main_circles)) >= 6:
        keeps["first three circles (PCs 1-6)"] = C.circle_pcs(main_circles)[:6]
    print("\n| embedding kept | accuracy |")
    print("| --- | --- |")
    for tag, keep in keeps.items():
        out["kept"][tag] = M.accuracy(iso_logits(keep), data.labels) * 100
        print(f"| {tag} | {out['kept'][tag]:.2f}% |")

    if main_circles:
        pd = C.accuracy_per_diff(iso_logits(list(main_circles[0].pcs)), data.labels)
        out["zero_diffs"] = int((pd == 0).sum())
        out["max_per_diff"] = float(pd.max() * 100)
        print(f"\nMain circle #1 per (a - b): {out['zero_diffs']} differences at 0%, "
              f"max {out['max_per_diff']:.1f}%.")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=48)
    ap.add_argument("--chunk", type=int, default=48)
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--device", default=None)
    ap.add_argument("--skip-training", action="store_true")
    args = ap.parse_args()

    if not args.skip_training:
        train(args.seeds, args.chunk, args.device, args.steps)

    runs = [r for r in load_registry() if r["config"]["experiment"] == EXPERIMENT]
    print(f"# E1 -- canonical Pizza vs Clock\n\n{len(runs)} runs in the registry.\n")

    print("## Metric distributions, median (IQR, n)\n")
    print("| alpha | all finished | 100% validation | circular | GS (circular) | "
          "DI correct (circular) | DI top-wrong (circular) | circularity (all) |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    groups: dict[float, list[dict[str, Any]]] = {}
    for alpha in (0.0, 1.0):
        g = [r for r in runs if r["config"]["attn_coeff"] == alpha]
        perfect = [r for r in g if r["val_accuracy"] == 1.0]
        circ = [r for r in perfect if r["circular"]]
        groups[alpha] = circ
        print(f"| {alpha} | {len(g)} | {len(perfect)} | {len(circ)} "
              f"| {fmt_dist([r['gs'] for r in circ])} "
              f"| {fmt_dist([r['di'] for r in circ])} "
              f"| {fmt_dist([r['di_top_wrong'] for r in circ])} "
              f"| {fmt_dist([r['circularity'] for r in g])} |")

    targets: list[tuple[str, str, str, float, str, bool]] = []

    def record(tid, quantity, paper, got, band, ok):
        targets.append((tid, quantity, paper, got, band, ok))

    if groups[0.0]:
        gs0 = st.median(r["gs"] for r in groups[0.0])
        di0 = st.median(r["di"] for r in groups[0.0])
        record("T1", "alpha = 0: GS median", "0.9937", gs0, ">= 0.98", gs0 >= 0.98)
        record("T2", "alpha = 0: DI (correct) median", "0.17", di0, "<= 0.40", di0 <= 0.40)
    if groups[1.0]:
        gs1 = st.median(r["gs"] for r in groups[1.0])
        di1 = st.median(r["di"] for r in groups[1.0])
        record("T3", "alpha = 1: GS median", "0.3336", gs1, "<= 0.75", gs1 <= 0.75)
        record("T4", "alpha = 1: DI (correct) median", "0.85", di1, ">= 0.60", di1 >= 0.60)

    print("\n## Representative runs and §9 analyses")
    reps = {}
    for alpha, label in ((0.0, "Model A analog (Pizza, alpha = 0)"),
                         (1.0, "Model B analog (Clock, alpha = 1)")):
        rep = representative(groups[alpha])
        if rep is None:
            print(f"\n### {label}\n\nNo circular run with 100% validation accuracy.")
            continue
        reps[alpha] = rep
        a = analyse(rep, label)
        kept = a["kept"]

        if alpha == 0.0 and a["main"]:
            record("T6", "isolated circle FVE: pizza vs clock", "99.2% vs 75.4%",
                   min(a["fve_pizza"]), "pizza >= 95 and clock <= 85",
                   min(a["fve_pizza"]) >= 95 and max(a["fve_clock"]) <= 85)
            lo = min(a["circle_accuracy"])
            record("T8", "one circle kept: accuracy", "32.8%", lo,
                   "15-50% with 0% at some a-b", 15 <= lo <= 50 and a["zero_diffs"] >= 1)
            record("T9", "main circles kept: accuracy", "91.4% / 99.7%",
                   kept["all main circles"], ">= 90%", kept["all main circles"] >= 90)
            record("T11", "all circles kept: accuracy", "100%", kept["all circles"],
                   ">= 99.9%", kept["all circles"] >= 99.9)
            if a["acc"]:
                record("T7", "accompanying circles: FVE of A", "97.2-97.7%", min(a["fve_a"]),
                       ">= 90%", min(a["fve_a"]) >= 90)
                record("T10", "accompanying only: accuracy", "16.7%", kept["accompanying only"],
                       "<= 40%", kept["accompanying only"] <= 40)
        elif alpha == 1.0:
            tag = "first three circles (PCs 1-6)" if "first three circles (PCs 1-6)" in kept \
                else "all main circles"
            record("T12", f"Clock, {tag} kept: accuracy", "100%", kept[tag],
                   ">= 99.9%", kept[tag] >= 99.9)

    print("\n## Targets\n")
    print("| ID | quantity | paper | ours | band | verdict |")
    print("| --- | --- | --- | --- | --- | --- |")
    for tid, quantity, paper, got, band, ok in sorted(targets):
        unit = "%" if "accuracy" in quantity or "FVE" in quantity else ""
        print(f"| {tid} | {quantity} | {paper} | {got:.4g}{unit} | {band} "
              f"| {'**pass**' if ok else '**FAIL**'} |")
    failed = [t[0] for t in targets if not t[5]]
    print(f"\n**{'All E1 targets in band' if not failed else 'Out of band: ' + ', '.join(failed)}.**")

    out = pathlib.Path("results/e1_representatives.json")
    out.write_text(json.dumps({str(a): r["run_id"] for a, r in reps.items()}, indent=2) + "\n")
    print(f"\nRepresentative run ids written to `{out}`.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
