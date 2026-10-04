"""Samplers and the resumable sweep runner (spec §10-11).

A sweep is a list of :class:`RunConfig`. The runner groups them into ensembles that
share width, depth and variant -- the §11 requirement -- then trains each group with
:func:`clockpizza.train.train_ensemble`, appending every finished run to the registry.
Run ids already in the registry are skipped, so a sweep resumes after a crash.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Callable, Iterable, Optional, Sequence

from .train import RunConfig, append_record, registry_ids, train_ensemble

# §10: the authors drew d = int(2 ** random.uniform(5, 9)). Ensembles need shared
# widths, so E3 uses a 24-point log grid over the same range instead (logged in
# DECISIONS.md as a deviation).
E3_WIDTH_POINTS = 24
E3_WIDTH_MIN, E3_WIDTH_MAX = 32, 511


def log_spaced_widths(n: int = E3_WIDTH_POINTS, lo: int = E3_WIDTH_MIN,
                      hi: int = E3_WIDTH_MAX) -> list[int]:
    """``n`` integer widths spaced evenly in log2 over [lo, hi], deduplicated."""
    out = []
    for i in range(n):
        w = int(round(lo * (hi / lo) ** (i / (n - 1))))
        if w not in out:
            out.append(w)
    return out


def e2_configs(n: int = 2400, seed: int = 0, steps: int = 20_000,
               experiment: str = "E2") -> list[RunConfig]:
    """Fig. 6, 7 top, 10 top: 1 layer, d = 128, alpha ~ U[0, 1]."""
    rng = random.Random(seed)
    return [RunConfig(seed=i, attn_coeff=rng.uniform(0, 1), d_model=128, n_layers=1,
                      steps=steps, experiment=experiment) for i in range(n)]


def e3_configs(n: int = 1200, seed: int = 1, steps: int = 20_000,
               experiment: str = "E3") -> list[RunConfig]:
    """Fig. 7 bottom, 10 bottom: alpha ~ U[0, 1], width on the log grid.

    Runs are spread evenly over the grid so each width gets about n / 24 runs.
    """
    rng = random.Random(seed)
    widths = log_spaced_widths()
    return [RunConfig(seed=i, attn_coeff=rng.uniform(0, 1), d_model=widths[i % len(widths)],
                      n_layers=1, steps=steps, experiment=experiment) for i in range(n)]


def e4_configs(n_per_depth: int = 800, depths: Sequence[int] = (2, 3, 4), seed: int = 2,
               steps: int = 20_000, experiment: str = "E4") -> list[RunConfig]:
    """Fig. 11: 2, 3 and 4 layers at d = 128, alpha ~ U[0, 1]."""
    rng = random.Random(seed)
    out = []
    for depth in depths:
        for i in range(n_per_depth):
            out.append(RunConfig(seed=i, attn_coeff=rng.uniform(0, 1), d_model=128,
                                 n_layers=depth, steps=steps, experiment=experiment))
    return out


def e7_configs(n: int = 500, seed: int = 3, steps: int = 20_000,
               experiment: str = "E7") -> list[RunConfig]:
    """App. I: GeLU, diff_vocab and eqn_sign variants, 1 layer, d = 128."""
    rng = random.Random(seed)
    out = []
    for kwargs in ({"act_fn": "GeLU"}, {"diff_vocab": True}, {"eqn_sign": True}):
        for i in range(n):
            out.append(RunConfig(seed=i, attn_coeff=rng.uniform(0, 1), d_model=128,
                                 n_layers=1, steps=steps, experiment=experiment, **kwargs))
    return out


def group_key(cfg: RunConfig) -> tuple:
    """Members of one ensemble must share everything but seed, alpha and split."""
    return (cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.act_fn, cfg.diff_vocab,
            cfg.eqn_sign, cfg.p, cfg.steps, cfg.lr, cfg.weight_decay, cfg.betas,
            cfg.eps, cfg.frac, cfg.warmup)


def plan(configs: Sequence[RunConfig], chunk: int = 64,
         skip_done: bool = True) -> list[list[RunConfig]]:
    """Group configs into trainable ensembles of at most ``chunk`` members."""
    done = registry_ids() if skip_done else set()
    by_key: dict[tuple, list[RunConfig]] = defaultdict(list)
    for cfg in configs:
        if cfg.run_id not in done:
            by_key[group_key(cfg)].append(cfg)
    batches = []
    for group in by_key.values():
        for i in range(0, len(group), chunk):
            batches.append(group[i:i + chunk])
    return batches


def chunk_for_width(d_model: int, budget_gb: float = 8.0) -> int:
    """Members that fit in ``budget_gb``. Memory is about linear in E and in d^2;
    measured at ~70 MiB per member at d = 128 (spec §11 planning numbers)."""
    per_member_gb = 0.07 * (d_model / 128) ** 2
    return max(1, min(128, int(budget_gb / max(per_member_gb, 1e-6))))


def run(configs: Sequence[RunConfig], device: Optional[str] = None, chunk: Optional[int] = None,
        budget_gb: float = 8.0, progress: bool = True,
        on_batch: Optional[Callable[[int, int, list[dict[str, Any]]], None]] = None
        ) -> int:
    """Train every config not already in the registry. Returns the number of new runs."""
    if chunk is not None:
        batches = plan(configs, chunk=chunk)
    else:
        # size each ensemble for its width, so wide runs use fewer members
        batches = []
        for group in plan(configs, chunk=10 ** 6):
            c = chunk_for_width(group[0].d_model, budget_gb)
            batches.extend(group[i:i + c] for i in range(0, len(group), c))

    total = sum(len(b) for b in batches)
    if progress:
        print(f"{total} runs to train in {len(batches)} ensembles", flush=True)
    trained = 0
    for i, batch in enumerate(batches, start=1):
        cfg = batch[0]
        if progress:
            print(f"[{i}/{len(batches)}] E = {len(batch)}  d = {cfg.d_model}  "
                  f"layers = {cfg.n_layers}  act = {cfg.act_fn}", flush=True)
        records = train_ensemble(batch, device=device, progress=False)
        for rec in records:
            append_record(rec)
        trained += len(records)
        if on_batch:
            on_batch(i, len(batches), records)
    return trained
