#!/usr/bin/env python
"""E2 (spec §10): 2,400 runs at 1 layer, d = 128, alpha ~ U[0, 1].

Reproduces Fig. 6, Fig. 7 top and Fig. 10 top, and evaluates T5 (circular share) and
T13 (the d = 128 phase boundary). Resumable: run ids already in results/runs.jsonl are
skipped.

Usage: python scripts/e2.py [--runs 2400] [--chunk 128] [--skip-training]
"""

from __future__ import annotations

import argparse
import pathlib
import statistics as st
import sys
from typing import Any, Sequence

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from clockpizza import plots  # noqa: E402
from clockpizza.sweeps import e2_configs, run as run_sweep  # noqa: E402
from clockpizza.summary import fmt_dist, iqr  # noqa: E402
from clockpizza.train import load_registry  # noqa: E402

EXPERIMENT = "E2"

# Released run-table boundaries (spec §10). The DI column there is the top-wrong variant.
RELEASED_GS_BOUNDARY = 0.52
RELEASED_DI_BOUNDARY = 0.59
BOUNDARY_TOLERANCE = 0.15

# Reference distributions from the released run table (spec §10), DI top-wrong.
REFERENCE = [
    ((0.0, 0.3), 284, 0.993, "0.992-0.994", 0.21, "0.17-0.36", 80),
    ((0.3, 0.5), 162, 0.993, "0.992-0.994", 0.24, "0.19-0.34", 81),
    ((0.5, 0.7), 127, 0.84, "0.75-0.94", 0.56, "0.39-0.64", 28),
    ((0.8, 1.0), 159, 0.50, "0.43-0.58", 0.79, "0.75-0.85", 0),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=2400)
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--device", default=None)
    ap.add_argument("--skip-training", action="store_true")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    if not args.skip_training:
        run_sweep(e2_configs(args.runs, steps=args.steps), device=args.device,
                  chunk=args.chunk, progress=True)

    runs = [r for r in load_registry() if r["config"]["experiment"] == EXPERIMENT]
    perfect = [r for r in runs if r["val_accuracy"] == 1.0]
    circular = [r for r in perfect if r["circular"]]

    print(f"# E2 -- attention-rate sweep\n\n{len(runs)} runs; {len(perfect)} at 100% "
          f"validation accuracy; {len(circular)} of those circular.\n")
    if not runs:
        print("*No E2 runs in the registry yet; nothing to analyse.*")
        return 0

    failures: list[str] = []
    targets: list[tuple[str, str, str, str, str, bool]] = []

    # ------------------------------------------------------------------------- T5
    share = 100 * len(circular) / len(perfect) if perfect else float("nan")
    ok = 20 <= share <= 50
    targets.append(("T5", "share of 100%-validation runs that are circular", "34.31%",
                    f"{share:.2f}% ({len(circular)}/{len(perfect)})", "20-50%", ok))

    # ------------------------------------------------------------------------ T13
    alpha = np.array([r["config"]["attn_coeff"] for r in circular])
    print("## Phase boundaries (unregularized logistic fit on circular runs)\n")
    print("| label | alpha* | released | within 0.15 |")
    print("| --- | --- | --- | --- |")
    boundaries: dict[str, float] = {}
    for name, values, thresh, above, released in (
        ("GS > 0.98", [r["gs"] for r in circular], 0.98, True, RELEASED_GS_BOUNDARY),
        ("DI (correct) < 0.6", [r["di"] for r in circular], 0.6, False, None),
        ("DI (top-wrong) < 0.6", [r["di_top_wrong"] for r in circular], 0.6, False,
         RELEASED_DI_BOUNDARY),
    ):
        y = np.array(values)
        bound = plots.logistic_boundary(alpha, (y > thresh) if above else (y < thresh))
        if bound is None:
            print(f"| {name} | (degenerate) | {released} | - |")
            continue
        boundaries[name] = bound["alpha_star"]
        if released is None:
            print(f"| {name} | {bound['alpha_star']:.3f} | - | - |")
        else:
            near = abs(bound["alpha_star"] - released) <= BOUNDARY_TOLERANCE
            print(f"| {name} | {bound['alpha_star']:.3f} | {released} "
                  f"| {'yes' if near else '**no**'} |")

    gs_b = boundaries.get("GS > 0.98")
    di_b = boundaries.get("DI (top-wrong) < 0.6")
    if gs_b is not None and di_b is not None:
        ok13 = (abs(gs_b - RELEASED_GS_BOUNDARY) <= BOUNDARY_TOLERANCE
                and abs(di_b - RELEASED_DI_BOUNDARY) <= BOUNDARY_TOLERANCE)
        targets.append(("T13", "d = 128 phase boundary alpha*", "~0.5 (Fig. 7)",
                        f"GS {gs_b:.2f}, DI {di_b:.2f}",
                        f"within 0.15 of {RELEASED_GS_BOUNDARY} / {RELEASED_DI_BOUNDARY}", ok13))

    # ------------------------------------------------- reference distribution table
    print("\n## Distributions by alpha, against the released run table\n")
    print("| alpha range | n (ours / released) | GS median (IQR) | released | "
          "DI top-wrong median (IQR) | released | DI < 0.4 (ours / released) |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for (lo, hi), n_ref, gs_ref, gs_iqr_ref, di_ref, di_iqr_ref, share_ref in REFERENCE:
        g = [r for r in circular if lo <= r["config"]["attn_coeff"] < hi]
        if not g:
            print(f"| [{lo}, {hi}) | 0 / {n_ref} | - | {gs_ref} | - | {di_ref} | - |")
            continue
        gs_v = [r["gs"] for r in g]
        di_v = [r["di_top_wrong"] for r in g]
        gl, gh = iqr(gs_v)
        dl, dh = iqr(di_v)
        pct = 100 * sum(x < 0.4 for x in di_v) / len(di_v)
        print(f"| [{lo}, {hi}) | {len(g)} / {n_ref} | {st.median(gs_v):.3f} ({gl:.3f}-{gh:.3f}) "
              f"| {gs_ref} ({gs_iqr_ref}) | {st.median(di_v):.2f} ({dl:.2f}-{dh:.2f}) "
              f"| {di_ref} ({di_iqr_ref}) | {pct:.0f}% / {share_ref}% |")

    # ------------------------------------------------------------------- the targets
    print("\n## Targets\n")
    print("| ID | quantity | paper | ours | band | verdict |")
    print("| --- | --- | --- | --- | --- | --- |")
    for tid, quantity, paper, got, band, ok in targets:
        if not ok:
            failures.append(tid)
        print(f"| {tid} | {quantity} | {paper} | {got} | {band} "
              f"| {'**pass**' if ok else '**FAIL**'} |")

    # ----------------------------------------------------------------------- figures
    if not args.no_figures and runs:
        print("\n## Figures\n")
        print("| figure | file |")
        print("| --- | --- |")
        made = {
            "fig06": plots.fig06_di_vs_gs(runs),
            "fig06_topwrong": plots.fig06_di_vs_gs(runs, di_key="di_top_wrong"),
            "fig10_top": plots.fig10_circular_share(perfect, tag="top"),
        }
        if circular:
            made["fig07_top"] = plots.fig07_phase_boundary(circular, tag="top")
        for key, path in sorted(made.items()):
            print(f"| {key} | `{path}` |")

    print(f"\n**{'All E2 targets in band' if not failures else 'Out of band: ' + ', '.join(failures)}.**")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
