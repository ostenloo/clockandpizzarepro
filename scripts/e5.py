#!/usr/bin/env python
"""E5 (spec §10): the five App. E linear models, 200 runs each.

Reproduces Fig. 14-16 and evaluates T18 (the ordering of the distance-irrelevance
medians: delta > alpha, alpha' > beta, gamma). Resumable.

Usage: python scripts/e5.py [--runs-per-model 200] [--skip-training]
"""

from __future__ import annotations

import argparse
import pathlib
import statistics as st
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from clockpizza import plots  # noqa: E402
from clockpizza.summary import (  # noqa: E402
    Targets, by_experiment, circular, fmt_dist, perfect, print_figures,
)
from clockpizza.sweeps import e5_configs, run as run_sweep  # noqa: E402
from clockpizza.train import load_registry, load_run  # noqa: E402

EXPERIMENT = "E5"
ORDER = ["alpha", "alpha_prime", "beta", "gamma", "delta"]
PAPER = {"delta": "0.92-0.98", "alpha": "0.4-0.7", "alpha_prime": "0.4-0.7",
         "beta": "low", "gamma": "low"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-per-model", type=int, default=200)
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--device", default=None)
    ap.add_argument("--skip-training", action="store_true")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    if not args.skip_training:
        run_sweep(e5_configs(args.runs_per_model, steps=args.steps), device=args.device,
                  progress=True)

    runs = by_experiment(load_registry(), EXPERIMENT)
    print(f"# E5 -- linear models\n\n{len(runs)} runs.\n")
    if not runs:
        print("*No E5 runs in the registry yet; nothing to analyse.*")
        return 0

    targets = Targets()
    print("## Metric distributions by model, median (IQR, n)\n")
    print("| model | runs | 100% validation | circular | DI (correct) | GS | circularity "
          "| paper DI |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    medians: dict[str, float] = {}
    for kind in ORDER:
        g = [r for r in runs if r["config"]["model_type"] == kind]
        if not g:
            continue
        ok = perfect(g)
        medians[kind] = st.median(r["di"] for r in g)
        print(f"| {kind} | {len(g)} | {len(ok)} | {len(circular(g))} "
              f"| {fmt_dist([r['di'] for r in g])} | {fmt_dist([r['gs'] for r in g])} "
              f"| {fmt_dist([r['circularity'] for r in g])} | {PAPER.get(kind, '-')} |")

    # ------------------------------------------------------------------------- T18
    have = [k for k in ORDER if k in medians]
    if {"delta", "beta"} <= set(have):
        top = medians["delta"]
        mid = [medians[k] for k in ("alpha", "alpha_prime") if k in medians]
        low = [medians[k] for k in ("beta", "gamma") if k in medians]
        ok18 = bool(mid) and bool(low) and top > max(mid) and min(mid) > max(low)
        targets.record("T18", "linear DI medians: delta > alpha, alpha' > beta, gamma",
                       "delta 0.92-0.98 > alpha, alpha' 0.4-0.7 > beta, gamma",
                       ", ".join(f"{k} {medians[k]:.2f}" for k in have),
                       "same ordering", ok18)

    made = {}
    if not args.no_figures:
        made["fig14"] = plots.fig14_linear_metrics(runs)
        made["fig15"] = plots.fig15_linear_di_hist(runs)
        picks = {}
        for kind in ("beta", "delta"):
            cands = [r for r in circular(perfect(runs)) if r["config"]["model_type"] == kind]
            if cands:
                best = max(cands, key=lambda r: r["circularity"])
                try:
                    picks[kind] = load_run(best["run_id"])[0]
                except FileNotFoundError:
                    pass
        if picks:
            made["fig16"] = plots.fig16_linear_isolation(picks)
        else:
            print("\n*Fig. 16 skipped: no circular beta/delta run with weights on this host.*")
    print_figures(made)
    return targets.print("E5")


if __name__ == "__main__":
    raise SystemExit(main())
