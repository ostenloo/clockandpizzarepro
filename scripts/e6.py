#!/usr/bin/env python
"""E6 (spec §10, App. H): d = 1024 at alpha = 1, up to 32 seeds.

Evaluates T16 -- that a Pizza run (DI < 0.4 and GS > 0.98) exists at full attention
once the model is wide enough -- and draws its isolation figure (Fig. 17). Stops at the
first qualifying run unless --all is given. Resumable.

Usage: python scripts/e6.py [--seeds 32] [--skip-training]
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from clockpizza import circles as C  # noqa: E402
from clockpizza import metrics as M  # noqa: E402
from clockpizza import plots  # noqa: E402
from clockpizza.data import make_dataset  # noqa: E402
from clockpizza.summary import Targets, by_experiment, print_figures  # noqa: E402
from clockpizza.sweeps import e6_configs  # noqa: E402
from clockpizza.train import append_record, load_registry, load_run, registry_ids, train_solo  # noqa: E402

EXPERIMENT = "E6"


def is_pizza(r: dict) -> bool:
    return r["di"] < 0.4 and r["gs"] > 0.98


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=32)
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--device", default=None)
    ap.add_argument("--all", action="store_true", help="train every seed, not just until a hit")
    ap.add_argument("--skip-training", action="store_true")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    if not args.skip_training:
        done = registry_ids()
        for cfg in e6_configs(args.seeds, steps=args.steps):
            if cfg.run_id in done:
                continue
            existing = [r for r in by_experiment(load_registry(), EXPERIMENT) if is_pizza(r)]
            if existing and not args.all:
                break
            print(f"training seed {cfg.seed} at d = {cfg.d_model}, alpha = {cfg.attn_coeff}",
                  flush=True)
            rec = train_solo(cfg, device=args.device, progress=True)
            append_record(rec)
            print(f"  DI {rec['di']:.4f}  GS {rec['gs']:.4f}  circ {rec['circularity']:.4f}"
                  f"  {'PIZZA' if is_pizza(rec) else rec['label']}  "
                  f"{rec['wall_seconds']:.0f}s", flush=True)

    runs = by_experiment(load_registry(), EXPERIMENT)
    print(f"# E6 -- d = 1024 at full attention\n\n{len(runs)} runs.\n")
    if not runs:
        print("*No E6 runs in the registry yet; nothing to analyse.*")
        return 0

    print("| seed | val acc | DI | GS | circularity | label | Pizza? |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for r in sorted(runs, key=lambda r: r["seed"]):
        print(f"| {r['seed']} | {r['val_accuracy'] * 100:.2f}% | {r['di']:.4f} "
              f"| {r['gs']:.4f} | {r['circularity']:.4f} | {r['label']} "
              f"| {'**yes**' if is_pizza(r) else 'no'} |")

    hits = [r for r in runs if is_pizza(r)]
    targets = Targets()
    targets.record("T16", "a Pizza run exists at d = 1024, alpha = 1",
                   "DI 0.156, GS 0.995",
                   f"{len(hits)} of {len(runs)} seeds"
                   + (f"; best DI {min(r['di'] for r in hits):.4f}, "
                      f"GS {max(r['gs'] for r in hits):.4f}" if hits else ""),
                   ">= 1 within 32 seeds", bool(hits) and len(runs) <= 32)

    made = {}
    if hits and not args.no_figures:
        best = min(hits, key=lambda r: r["di"])
        try:
            model, _ = load_run(best["run_id"])
        except FileNotFoundError:
            print(f"\n*Fig. 17 skipped: weights for `{best['run_id']}` are not on this host.*")
        else:
            data = make_dataset()
            pca = C.embedding_pca(model)
            main, acc = C.classify_circles(C.find_circles(model, pca=pca), model, data, pca=pca)
            print(f"\n## Circles of the Pizza run `{best['run_id']}` (seed {best['seed']})\n")
            print("| circle | k | delta | PCs | accompanies | accuracy alone |")
            print("| --- | --- | --- | --- | --- | --- |")
            for i, c in enumerate(main + acc, start=1):
                with torch.no_grad():
                    a = M.accuracy(C.isolated_model(model, list(c.pcs), pca=pca)
                                   .final_logits(data.inputs), data.labels) * 100
                print(f"| #{i} | {c.k} | {c.delta} | {c.pcs[0] + 1},{c.pcs[1] + 1} "
                      f"| {c.accompanies or '-'} | {a:.2f}% |")
            if main:
                made["fig17"] = plots.fig04_main_circles(model, main, data, pca)
    print_figures(made)
    return targets.print("E6")


if __name__ == "__main__":
    raise SystemExit(main())
