#!/usr/bin/env python
"""E7 (spec §10, App. I): the GeLU, diff_vocab and eqn_sign variants, 500 runs each.

Reproduces Fig. 18-21. No §3 target depends on it; the point is that the Pizza/Clock
split survives each change of setup. Per §10 the two token variants are reported
without the circularity filter. Resumable.

Usage: python scripts/e7.py [--runs 500] [--skip-training]
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from clockpizza import plots  # noqa: E402
from clockpizza.summary import by_experiment, circular, fmt_dist, perfect, print_figures  # noqa: E402
from clockpizza.sweeps import e7_configs, run as run_sweep  # noqa: E402
from clockpizza.train import load_registry, load_run  # noqa: E402

EXPERIMENT = "E7"


def variant_of(r: dict) -> str:
    c = r["config"]
    if c.get("diff_vocab"):
        return "diff_vocab"
    if c.get("eqn_sign"):
        return "eqn_sign"
    return c.get("act_fn", "ReLU")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=500)
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--device", default=None)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--skip-training", action="store_true")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    if not args.skip_training:
        run_sweep(e7_configs(args.runs, steps=args.steps), device=args.device,
                  chunk=args.chunk, progress=True)

    runs = by_experiment(load_registry(), EXPERIMENT)
    print(f"# E7 -- setup variants\n\n{len(runs)} runs.\n")
    if not runs:
        print("*No E7 runs in the registry yet; nothing to analyse.*")
        return 0

    print("## By variant\n")
    print("| variant | runs | 100% validation | circular | GS | DI (correct) "
          "| DI (top-wrong) | boundary alpha* (GS) |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for kind in sorted({variant_of(r) for r in runs}):
        g = [r for r in runs if variant_of(r) == kind]
        ok = perfect(g)
        # §10: no circularity filter for the two token variants
        pool = ok if kind in ("diff_vocab", "eqn_sign") else circular(ok)
        bound = plots.logistic_boundary(
            np.array([r["config"]["attn_coeff"] for r in pool]),
            np.array([r["gs"] > 0.98 for r in pool])) if pool else None
        alpha_star = f"{bound['alpha_star']:.3f}" if bound else "-"
        print(f"| {kind} | {len(g)} | {len(ok)} | {len(circular(ok))} "
              f"| {fmt_dist([r['gs'] for r in pool])} | {fmt_dist([r['di'] for r in pool])} "
              f"| {fmt_dist([r['di_top_wrong'] for r in pool])} | {alpha_star} |")

    made = {}
    if not args.no_figures:
        for kind in sorted({variant_of(r) for r in runs}):
            number = plots.VARIANT_FIGURE.get(kind, 18)
            made[f"fig{number:02d}_{kind}"] = plots.fig_variant_metrics(runs, kind)
        dv = [r for r in perfect(runs) if r["config"].get("diff_vocab")
              and r["di"] < 0.4 and r["gs"] > 0.98]
        if dv:
            try:
                made["fig20"] = plots.fig20_aligned_embeddings(
                    load_run(max(dv, key=lambda r: r["gs"])["run_id"])[0])
            except FileNotFoundError:
                print("\n*Fig. 20 skipped: weights not on this host.*")
        else:
            print("\n*Fig. 20 skipped: no Pizza-like `diff_vocab` run.*")
    print_figures(made)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
