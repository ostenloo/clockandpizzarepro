#!/usr/bin/env python
"""M3 gate: measure ensemble throughput and project the cost of E1-E3 (spec §4.4, §11).

Gate: test 6 passes (pytest tests/test_ensemble.py) and the projected cost of E1-E3 is
within the §11 budget of 12 GPU-hours.

Usage: python scripts/m3.py [--steps 200] [--members 1 8 32 64 128]
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from clockpizza.train import RunConfig, set_precision, train_ensemble  # noqa: E402

E1_RUNS, E2_RUNS, E3_RUNS = 96, 2400, 1200  # 48 seeds x 2 alphas; §10 run counts
BUDGET_HOURS = 12.0


def measure(n_members: int, steps: int, d_model: int, device: str) -> dict:
    cfgs = [RunConfig(seed=1000 + i, attn_coeff=i / max(n_members - 1, 1), steps=steps,
                      d_model=d_model, log_every=10 ** 9, experiment="m3-throughput")
            for i in range(n_members)]
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    train_ensemble(cfgs, device=device, save_weights=False)
    if device == "cuda":
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    per_run_20k = wall / steps * 20_000 / n_members
    return dict(members=n_members, wall=wall, per_step=wall / steps,
                per_run_20k=per_run_20k,
                mem_gb=(torch.cuda.max_memory_allocated() / 2**30) if device == "cuda" else 0.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--members", type=int, nargs="+", default=[1, 8, 32, 64, 128])
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    set_precision()

    print(f"### Ensemble throughput (d = {args.d_model}, {args.steps} steps, {device})\n")
    print("| members | wall | s / step | projected s per 20k-step run | peak memory |")
    print("| --- | --- | --- | --- | --- |")
    rows = []
    for n in args.members:
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        try:
            row = measure(n, args.steps, args.d_model, device)
        except torch.cuda.OutOfMemoryError:
            print(f"| {n} | out of memory | - | - | - |")
            torch.cuda.empty_cache()
            continue
        rows.append(row)
        print(f"| {row['members']} | {row['wall']:.1f}s | {row['per_step']:.4f} "
              f"| {row['per_run_20k']:.1f}s | {row['mem_gb']:.1f} GB |")

    best = min(rows, key=lambda r: r["per_run_20k"])
    per_run_h = best["per_run_20k"] / 3600
    print(f"\nBest: E = {best['members']}, {best['per_run_20k']:.1f}s per 20k-step run "
          f"({per_run_h * 3600:.1f}s, {best['mem_gb']:.1f} GB peak).\n")

    print("### Projected cost\n")
    print("| experiment | runs | projected GPU-hours |")
    print("| --- | --- | --- |")
    total = 0.0
    for name, runs, note in [("E1 (48 seeds x 2 alphas, d = 128)", E1_RUNS, ""),
                             ("E2 (d = 128 sweep)", E2_RUNS, ""),
                             ("E3 (width sweep, d = 32..511)", E3_RUNS, "scaled by mean d^2")]:
        # E3 spans 24 log-spaced widths in [32, 512); cost grows about with d^2
        scale = 1.0
        if runs == E3_RUNS:
            widths = [int(round(32 * (511 / 32) ** (i / 23))) for i in range(24)]
            scale = sum(w ** 2 for w in widths) / len(widths) / args.d_model ** 2
        hours = runs * per_run_h * scale
        total += hours
        print(f"| {name} | {runs} | {hours:.2f}{' (' + note + ')' if note else ''} |")
    print(f"| **E1-E3 total** | {E1_RUNS + E2_RUNS + E3_RUNS} | **{total:.2f}** |")

    ok = total <= BUDGET_HOURS
    print(f"\nBudget: {BUDGET_HOURS} GPU-hours for E1-E3. "
          f"**{'Within budget' if ok else 'OVER BUDGET -- ask before proceeding (§2.7)'}.**")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
