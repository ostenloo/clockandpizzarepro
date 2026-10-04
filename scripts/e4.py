#!/usr/bin/env python
"""E4 (spec §10): 800 runs each at 2, 3 and 4 layers, d = 128, alpha ~ U[0, 1].

Reproduces Fig. 11 and evaluates T15 (deeper nets are less often circular). Resumable.

Usage: python scripts/e4.py [--runs-per-depth 800] [--skip-training]
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from clockpizza import plots  # noqa: E402
from clockpizza.summary import (  # noqa: E402
    Targets, by_experiment, circular, fmt_dist, perfect, print_figures,
)
from clockpizza.sweeps import e4_configs, run as run_sweep  # noqa: E402
from clockpizza.train import load_registry  # noqa: E402

EXPERIMENT = "E4"
PAPER_SHARE = {2: 9.95, 3: 11.55, 4: 6.08}       # §3 T15
RELEASED_SHARE = {2: 12.2, 3: 12.8, 4: 8.5}      # released run table
T5_REFERENCE = 34.31                              # 1-layer circular share


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-per-depth", type=int, default=800)
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--device", default=None)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--skip-training", action="store_true")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    if not args.skip_training:
        run_sweep(e4_configs(args.runs_per_depth, steps=args.steps), device=args.device,
                  chunk=args.chunk, progress=True)

    runs = by_experiment(load_registry(), EXPERIMENT)
    print(f"# E4 -- depth sweep\n\n{len(runs)} runs.\n")
    if not runs:
        print("*No E4 runs in the registry yet; nothing to analyse.*")
        return 0

    targets = Targets()
    print("## Circular share by depth\n")
    print("| layers | runs | 100% validation | circular | share | paper | released table "
          "| GS (circular) | DI (circular) |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    shares = {}
    for depth in sorted({r["config"]["n_layers"] for r in runs}):
        g = [r for r in runs if r["config"]["n_layers"] == depth]
        ok = perfect(g)
        circ = circular(ok)
        share = 100 * len(circ) / len(ok) if ok else float("nan")
        shares[depth] = share
        print(f"| {depth} | {len(g)} | {len(ok)} | {len(circ)} | {share:.2f}% "
              f"| {PAPER_SHARE.get(depth, '-')}% | {RELEASED_SHARE.get(depth, '-')}% "
              f"| {fmt_dist([r['gs'] for r in circ])} | {fmt_dist([r['di'] for r in circ])} |")

    if shares:
        worst = max(shares.values())
        targets.record("T15", "circular share at 2 / 3 / 4 layers",
                       "9.95 / 11.55 / 6.08%",
                       " / ".join(f"{shares[d]:.2f}%" for d in sorted(shares)),
                       f"each below the 1-layer {T5_REFERENCE}%", worst < T5_REFERENCE)

    made = {}
    if not args.no_figures:
        made["fig11"] = plots.fig11_depth(runs)
    print_figures(made)
    return targets.print("E4")


if __name__ == "__main__":
    raise SystemExit(main())
