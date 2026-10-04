#!/usr/bin/env python
"""E8 (spec §10, App. J-K): training dynamics from mid-training checkpoints.

alpha = 1 with checkpoints at steps 90, 210, 300, 510, 840, and alpha = 0 with one at
step 600; 4 seeds each. Reproduces Fig. 22-23 and checks App. K's claim that removing
the accompanying circle at step 600 costs about 1.8 points of accuracy
(99.7% -> 97.9%). Resumable.

Usage: python scripts/e8.py [--seeds 4] [--skip-training]
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
from clockpizza.summary import by_experiment, print_figures  # noqa: E402
from clockpizza.sweeps import e8_configs  # noqa: E402
from clockpizza.train import (  # noqa: E402
    CHECKPOINT_DIR, append_record, load_checkpoint, load_registry, load_run, registry_ids,
    train_solo,
)

EXPERIMENT = "E8"
APP_K_WITH, APP_K_WITHOUT = 99.7, 97.9


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--device", default=None)
    ap.add_argument("--skip-training", action="store_true")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    if not args.skip_training:
        done = registry_ids()
        for cfg in e8_configs(args.seeds, steps=args.steps):
            if cfg.run_id in done:
                continue
            print(f"training seed {cfg.seed}, alpha = {cfg.attn_coeff}, "
                  f"checkpoints at {cfg.checkpoint_steps}", flush=True)
            append_record(train_solo(cfg, device=args.device, progress=True))

    runs = by_experiment(load_registry(), EXPERIMENT)
    print(f"# E8 -- training dynamics\n\n{len(runs)} runs.\n")
    if not runs:
        print("*No E8 runs in the registry yet; nothing to analyse.*")
        return 0

    data = make_dataset()
    made = {}

    # ------------------------------------------- App. K: the accompanying circle at step 600
    pizza_runs = [r for r in runs if r["config"]["attn_coeff"] == 0.0]
    print("## App. K -- removing the accompanying circle at step 600\n")
    print("| run | seed | all circles | main circles only | drop |")
    print("| --- | --- | --- | --- | --- |")
    for r in sorted(pizza_runs, key=lambda r: r["seed"]):
        try:
            model, _ = load_checkpoint(r["run_id"], 600)
        except FileNotFoundError:
            print(f"| `{r['run_id']}` | {r['seed']} | *checkpoint not on this host* | | |")
            continue
        pca = C.embedding_pca(model)
        main, acc = C.classify_circles(C.find_circles(model, pca=pca), model, data, pca=pca)
        if not main:
            print(f"| `{r['run_id']}` | {r['seed']} | *no circles at step 600* | | |")
            continue

        def accuracy_of(keep):
            with torch.no_grad():
                return M.accuracy(C.isolated_model(model, keep, pca=pca)
                                  .final_logits(data.inputs), data.labels) * 100

        with_all = accuracy_of(C.circle_pcs(main + acc))
        without = accuracy_of(C.circle_pcs(main))
        print(f"| `{r['run_id']}` | {r['seed']} | {with_all:.2f}% | {without:.2f}% "
              f"| {with_all - without:+.2f} |")
    print(f"\nApp. K reports {APP_K_WITH}% falling to {APP_K_WITHOUT}% "
          f"(a drop of {APP_K_WITH - APP_K_WITHOUT:.1f} points).")

    # --------------------------------------------------- Fig. 22-23: isolation over time
    if not args.no_figures:
        clock_runs = [r for r in runs if r["config"]["attn_coeff"] == 1.0]
        for tag, group, steps in (("clock", clock_runs, (90, 210, 300, 510, 840)),
                                  ("pizza", pizza_runs, (600,))):
            if not group:
                continue
            run = sorted(group, key=lambda r: r["seed"])[0]
            series = []
            for step in steps:
                try:
                    model, _ = load_checkpoint(run["run_id"], step)
                except FileNotFoundError:
                    continue
                series.append((step, model))
            if series:
                made[f"fig22_{tag}"] = plots.fig22_isolation_over_training(
                    series, outdir=None, tag=tag)

        histories = {}
        for r in sorted(runs, key=lambda r: (r["config"]["attn_coeff"], r["seed"]))[:4]:
            try:
                _, payload = load_run(r["run_id"])
            except FileNotFoundError:
                continue
            if payload.get("history", {}).get("step"):
                histories[f"alpha={r['config']['attn_coeff']} seed={r['seed']}"] = payload["history"]
        if histories:
            made["fig23"] = plots.fig23_training_curves(histories)

    print_figures(made)
    print(f"\nCheckpoints live in `{CHECKPOINT_DIR}/`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
