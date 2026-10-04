"""Test 7 (spec §12): the CI-sized smoke run.

3 seeds at alpha = 0, d = 128, 20,000 steps; at least 2 must reach 100% validation
accuracy. Marked ``slow`` -- 20,000 steps three times is minutes on a GPU and much
longer on CPU -- so it is opt-in: ``pytest -m slow tests/test_smoke.py``.
"""

import pytest
import torch

from clockpizza.train import RunConfig, train_solo


@pytest.mark.slow
@pytest.mark.smoke
def test_three_seeds_mostly_reach_full_validation_accuracy():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    perfect = 0
    for seed in range(3):
        cfg = RunConfig(seed=seed, attn_coeff=0.0, steps=20_000, log_every=10 ** 9,
                        experiment="smoke")
        rec = train_solo(cfg, device=device, save_weights=False)
        perfect += rec["val_accuracy"] == 1.0
    assert perfect >= 2, f"only {perfect} of 3 seeds reached 100% validation accuracy"
