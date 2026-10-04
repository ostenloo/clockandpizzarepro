#!/usr/bin/env python
"""E3 (spec §10): 1,200 runs, alpha ~ U[0, 1], width on a 24-point log grid in [32, 512).

Reproduces Fig. 7 bottom and Fig. 10 bottom, and evaluates T14 (the phase boundary
rises with width). Resumable.

Usage: python scripts/e3.py [--runs 1200] [--skip-training]
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from clockpizza import plots  # noqa: E402
from clockpizza.summary import Targets, by_experiment, circular, perfect, print_figures  # noqa: E402
from clockpizza.sweeps import e3_configs, run as run_sweep  # noqa: E402
from clockpizza.train import load_registry  # noqa: E402

EXPERIMENT = "E3"
# Released run-table GS boundaries at d = 32, 64, 128, 256, 512 (spec §10).
RELEASED_GS = {32: 0.25, 64: 0.41, 128: 0.57, 256: 0.72, 512: 0.88}
MIN_RISE = 0.3


def nearest(widths, target):
    return min(widths, key=lambda w: abs(np.log2(w) - np.log2(target)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=1200)
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--device", default=None)
    ap.add_argument("--budget-gb", type=float, default=8.0)
    ap.add_argument("--skip-training", action="store_true")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    if not args.skip_training:
        run_sweep(e3_configs(args.runs, steps=args.steps), device=args.device,
                  budget_gb=args.budget_gb, progress=True)

    runs = by_experiment(load_registry(), EXPERIMENT)
    ok_runs = perfect(runs)
    circ = circular(ok_runs)
    print(f"# E3 -- width sweep\n\n{len(runs)} runs; {len(ok_runs)} at 100% validation "
          f"accuracy; {len(circ)} of those circular.\n")
    if not circ:
        print("*No circular E3 runs in the registry yet; nothing to analyse.*")
        return 0

    targets = Targets()

    print("## Phase boundary by width (logistic fit on circular runs)\n")
    print("| d | n circular | alpha* (GS > 0.98) | released | alpha* (DI top-wrong < 0.6) |")
    print("| --- | --- | --- | --- | --- |")
    gs_bounds = plots.boundary_by_group(circ, "gs", 0.98, True)
    di_bounds = plots.boundary_by_group(circ, "di_top_wrong", 0.6, False)
    widths = sorted({r["config"]["d_model"] for r in circ})
    for w in widths:
        n = sum(1 for r in circ if r["config"]["d_model"] == w)
        rel = RELEASED_GS.get(w)
        gs = gs_bounds.get(w)
        di = di_bounds.get(w)
        print(f"| {w} | {n} | {f'{gs:.3f}' if gs is not None else '-'} "
              f"| {rel if rel is not None else '-'} "
              f"| {f'{di:.3f}' if di is not None else '-'} |")

    # ------------------------------------------------------------------------- T14
    if len(gs_bounds) >= 2:
        lo_w, hi_w = nearest(list(gs_bounds), 32), nearest(list(gs_bounds), 512)
        rise = gs_bounds[hi_w] - gs_bounds[lo_w]
        targets.record("T14", f"rise of the GS boundary, d = {lo_w} to {hi_w}",
                       "0.25 -> 0.88 (rise 0.63)",
                       f"{gs_bounds[lo_w]:.2f} -> {gs_bounds[hi_w]:.2f} (rise {rise:.2f})",
                       f"rise >= {MIN_RISE}", rise >= MIN_RISE)

        # the spec also describes the boundary as roughly linear in log width
        xs = np.log2(np.array(sorted(gs_bounds)))
        ys = np.array([gs_bounds[w] for w in sorted(gs_bounds)])
        slope, intercept = np.polyfit(xs, ys, 1)
        r = np.corrcoef(xs, ys)[0, 1]
        print(f"\nGS boundary against log2 d: slope {slope:+.3f} per doubling, "
              f"intercept {intercept:+.3f}, r = {r:.3f} over {len(xs)} widths.")

    made = {}
    if not args.no_figures:
        made["fig07_bottom"] = plots.fig07_width_phase(circ)
        made["fig10_bottom"] = plots.fig10_width_circular_share(ok_runs)
    print_figures(made)
    return targets.print("E3")


if __name__ == "__main__":
    raise SystemExit(main())
