#!/usr/bin/env python
"""E9 (spec §10, App. L): weight-level analysis of one linear beta run.

Evaluates T17 -- that dropping the model's second ReLU leaves accuracy at 100% and does
not raise the loss (the paper reports 6.20e-7 -> 5.89e-7 over all 3,481 pairs) -- and
draws Fig. 25 and Fig. 26. Picks a circular, Pizza-like beta run from E5 unless a run
id is given. Resumable.

Usage: python scripts/e9.py [--run-id ...] [--skip-training]
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from clockpizza import metrics as M  # noqa: E402
from clockpizza import plots  # noqa: E402
from clockpizza.data import make_dataset  # noqa: E402
from clockpizza.summary import Targets, by_experiment, circular, perfect, print_figures  # noqa: E402
from clockpizza.sweeps import e9_config  # noqa: E402
from clockpizza.train import (  # noqa: E402
    append_record, cross_entropy_f64, load_registry, load_run, registry_ids, train_solo,
)

EXPERIMENT = "E9"
PAPER_LOSS_WITH, PAPER_LOSS_WITHOUT = 6.20e-7, 5.89e-7


def pick_run(registry) -> dict | None:
    """A circular, Pizza-like linear beta run -- from E9 if present, else from E5."""
    for experiment in (EXPERIMENT, "E5"):
        cands = [r for r in by_experiment(registry, experiment)
                 if r["config"].get("model_type") == "beta"]
        cands = circular(perfect(cands))
        if cands:
            pizza = [r for r in cands if r["di"] < 0.4] or cands
            return max(pizza, key=lambda r: r["circularity"])
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--device", default=None)
    ap.add_argument("--skip-training", action="store_true")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    registry = load_registry()
    chosen = None
    if args.run_id:
        chosen = next((r for r in registry if r["run_id"] == args.run_id), None)
    else:
        chosen = pick_run(registry)

    if chosen is None and not args.skip_training:
        # no suitable beta run anywhere yet: train seeds until one is circular
        for seed in range(16):
            cfg = e9_config(seed=seed, steps=args.steps)
            if cfg.run_id in registry_ids():
                continue
            print(f"training linear beta seed {seed}", flush=True)
            rec = train_solo(cfg, device=args.device, progress=True)
            append_record(rec)
            print(f"  val {rec['val_accuracy']:.4f}  DI {rec['di']:.4f}  "
                  f"circ {rec['circularity']:.4f}", flush=True)
            if rec["circular"] and rec["val_accuracy"] == 1.0:
                chosen = rec
                break
        registry = load_registry()
        chosen = chosen or pick_run(registry)

    print("# E9 -- App. L, the linear beta model\n")
    if chosen is None:
        print("*No circular linear beta run available; run `scripts/e5.py` first.*")
        return 0

    try:
        model, _ = load_run(chosen["run_id"])
    except FileNotFoundError:
        print(f"*Run `{chosen['run_id']}` is in the registry but its weights are not on "
              f"this host.*")
        return 0

    print(f"Run `{chosen['run_id']}` (seed {chosen['seed']}): validation "
          f"{chosen['val_accuracy'] * 100:.2f}%, DI {chosen['di']:.4f}, "
          f"GS {chosen['gs']:.4f}, circularity {chosen['circularity']:.4f}.\n")

    # ------------------------------------------------------ T17: drop the second ReLU
    data = make_dataset()
    targets = Targets()
    print("## Dropping the second ReLU (all 3,481 pairs)\n")
    print("| second ReLU | accuracy | loss |")
    print("| --- | --- | --- |")
    results = {}
    for keep in (True, False):
        with torch.no_grad():
            logits = model(data.inputs, second_relu=keep)
            loss = cross_entropy_f64(logits, data.labels).item()
        acc = M.accuracy(logits, data.labels) * 100
        results[keep] = (acc, loss)
        print(f"| {'kept' if keep else '**removed**'} | {acc:.2f}% | {loss:.3e} |")
    print(f"\nApp. L reports {PAPER_LOSS_WITH:.2e} -> {PAPER_LOSS_WITHOUT:.2e}.")

    acc_off, loss_off = results[False]
    _, loss_on = results[True]
    targets.record("T17", "linear beta without its second ReLU",
                   f"100%, {PAPER_LOSS_WITH:.2e} -> {PAPER_LOSS_WITHOUT:.2e}",
                   f"{acc_off:.2f}%, {loss_on:.3e} -> {loss_off:.3e}",
                   "accuracy 100% and loss not higher",
                   acc_off >= 99.995 and loss_off <= loss_on)

    made = {}
    if not args.no_figures:
        made["fig24"] = plots.fig24_second_relu(model)
        made["fig25"] = plots.fig25_aligned_weights(model)
        made["fig26"] = plots.fig26_second_harmonic_fit(model)
    print_figures(made)
    return targets.print("E9")


if __name__ == "__main__":
    raise SystemExit(main())
