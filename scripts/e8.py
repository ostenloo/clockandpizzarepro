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

    # ------------------------------------------- App. K: removing the accompanying circle
    # An earlier E8 pass used a sparser checkpoint grid; those runs are still in the
    # registry under different ids. Keep, per (seed, alpha), whichever has more steps.
    best: dict[tuple, dict] = {}
    for r in runs:
        key = (r["seed"], r["config"]["attn_coeff"])
        if len(r["config"].get("checkpoint_steps", ())) > \
                len(best.get(key, {}).get("config", {}).get("checkpoint_steps", ())):
            best[key] = r
    runs = sorted(best.values(), key=lambda r: (r["config"]["attn_coeff"], r["seed"]))
    pizza_runs = [r for r in runs if r["config"]["attn_coeff"] == 0.0]
    print("## App. K -- removing the accompanying circle\n")
    print("A drop needs an accompanying circle to remove. The paper looks at step 600; "
          "every checkpointed step is scanned here, because a run that has not yet grown "
          "its accompanying circles would otherwise report a drop of zero and look like "
          "agreement.\n")
    print("| run | seed | step | circles (main + accompanying) | all circles "
          "| main circles only | drop |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    drops: list[tuple[int, float]] = []
    for r in sorted(pizza_runs, key=lambda r: r["seed"]):
        for step in r["config"].get("checkpoint_steps", ()):
            try:
                model, _ = load_checkpoint(r["run_id"], step)
            except FileNotFoundError:
                continue
            pca = C.embedding_pca(model)
            main, acc = C.classify_circles(C.find_circles(model, pca=pca), model, data,
                                           pca=pca)
            if not main:
                print(f"| `{r['run_id']}` | {r['seed']} | {step} | none | - | - | - |")
                continue

            def accuracy_of(keep, _model=model, _pca=pca):
                with torch.no_grad():
                    return M.accuracy(C.isolated_model(_model, keep, pca=_pca)
                                      .final_logits(data.inputs), data.labels) * 100

            with_all = accuracy_of(C.circle_pcs(main + acc))
            without = accuracy_of(C.circle_pcs(main))
            if acc:
                drops.append((step, with_all - without))
            drop = "-" if not acc else f"{with_all - without:+.2f}"
            print(f"| `{r['run_id']}` | {r['seed']} | {step} | {len(main)} + {len(acc)} "
                  f"| {with_all:.2f}% | {without:.2f}% | {drop} |")
    print(f"\nApp. K reports {APP_K_WITH}% falling to {APP_K_WITHOUT}% "
          f"(a drop of {APP_K_WITH - APP_K_WITHOUT:.1f} points).")
    if drops:
        import statistics as st

        at_600 = [d for step, d in drops if step == 600]
        print(f"\nAcross {len(drops)} checkpoints that had an accompanying circle to "
              f"remove, the drop was positive in {sum(d > 0 for _, d in drops)} of them, "
              f"median {st.median(d for _, d in drops):+.2f} points "
              f"(range {min(d for _, d in drops):+.2f} to {max(d for _, d in drops):+.2f}).")
        print(f"At step 600 specifically -- the step App. K uses -- "
              f"{'no run had' if not at_600 else f'{len(at_600)} runs had'} an "
              f"accompanying circle yet, so the paper's comparison cannot be made there "
              f"on these runs. The structure appears between steps "
              f"{min(step for step, _ in drops)} and {max(step for step, _ in drops)}.")

    # --------------------------------------------------- Fig. 22-23: isolation over time
    if not args.no_figures:
        clock_runs = [r for r in runs if r["config"]["attn_coeff"] == 1.0]
        for tag, group in (("clock", clock_runs), ("pizza", pizza_runs)):
            if not group:
                continue
            run = sorted(group, key=lambda r: r["seed"])[0]
            series = []
            for step in run["config"].get("checkpoint_steps", ()):
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
