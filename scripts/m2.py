#!/usr/bin/env python
"""M2 gate: train 8 seeds each at alpha = 0 and alpha = 1, d = 128 (spec §4.3).

Gate: at least 7 of 8 per alpha reach 100% validation accuracy; wall time per run
recorded. Runs already in results/runs.jsonl are skipped, so this is resumable.

Usage: python scripts/m2.py [--seeds 8] [--steps 20000] [--device cuda]
"""

from __future__ import annotations

import argparse
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from clockpizza.train import (  # noqa: E402
    REGISTRY, RunConfig, append_record, load_registry, registry_ids, train_solo,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--device", default=None)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.0, 1.0])
    ap.add_argument("--experiment", default="M2")
    args = ap.parse_args()

    done = registry_ids()
    configs = [RunConfig(seed=s, attn_coeff=a, steps=args.steps, experiment=args.experiment)
               for a in args.alphas for s in range(args.seeds)]

    for i, cfg in enumerate(configs, start=1):
        if cfg.run_id in done:
            print(f"[{i}/{len(configs)}] skip {cfg.run_id} (alpha={cfg.attn_coeff}, seed={cfg.seed})",
                  flush=True)
            continue
        print(f"[{i}/{len(configs)}] train {cfg.run_id}  alpha={cfg.attn_coeff}  seed={cfg.seed}",
              flush=True)
        rec = train_solo(cfg, device=args.device, progress=True)
        append_record(rec)
        print(f"    val_acc={rec['val_accuracy']:.4f}  GS={rec['gs']:.4f}  DI={rec['di']:.4f}  "
              f"circ={rec['circularity']:.4f}  {rec['label']}  {rec['wall_seconds']:.1f}s", flush=True)

    # ---------------------------------------------------------------------- gate report
    runs = [r for r in load_registry() if r["config"]["experiment"] == args.experiment
            and r["config"]["steps"] == args.steps]
    print("\n### M2 runs\n")
    print("| alpha | seed | val acc | GS | DI (correct) | DI (top-wrong) | circularity | label | wall |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    failures = 0
    evaluated = False
    for alpha in args.alphas:
        group = sorted((r for r in runs if r["config"]["attn_coeff"] == alpha), key=lambda r: r["seed"])
        for r in group:
            print(f"| {alpha} | {r['seed']} | {r['val_accuracy'] * 100:.2f}% | {r['gs']:.4f} "
                  f"| {r['di']:.4f} | {r['di_top_wrong']:.4f} | {r['circularity']:.4f} "
                  f"| {r['label']} | {r['wall_seconds']:.0f}s |")

    print("\n### Gate\n")
    print("| alpha | n | 100% validation | circular | GS median | DI median | wall median |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for alpha in args.alphas:
        group = [r for r in runs if r["config"]["attn_coeff"] == alpha]
        if not group:
            continue
        perfect = [r for r in group if r["val_accuracy"] == 1.0]
        circ = [r for r in group if r["circular"]]
        if len(group) >= 8:
            evaluated = True
            failures += len(perfect) < 7
        print(f"| {alpha} | {len(group)} | {len(perfect)}/{len(group)} | {len(circ)}/{len(group)} "
              f"| {statistics.median(r['gs'] for r in group):.4f} "
              f"| {statistics.median(r['di'] for r in group):.4f} "
              f"| {statistics.median(r['wall_seconds'] for r in group):.0f}s |")

    print(f"\nRegistry: `{REGISTRY}` ({len(load_registry())} runs).")
    if not evaluated:
        print("\n**M2 gate not evaluated** (fewer than 8 runs per alpha).")
        return 0
    print(f"\n**{'M2 gate passed' if failures == 0 else 'M2 gate FAILED'}** "
          f"(>= 7 of 8 runs per alpha at 100% validation accuracy).")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
