"""The App. E linear models and E8's mid-training checkpoints."""

import pathlib
import shutil

import pytest
import torch

from clockpizza import circles as C
from clockpizza import metrics as M
from clockpizza.data import make_dataset
from clockpizza.metrics import P
from clockpizza.model import LINEAR_MODELS
from clockpizza.train import RunConfig, load_checkpoint, train_solo

KINDS = ["alpha", "alpha_prime", "beta", "gamma", "delta"]


@pytest.mark.parametrize("kind", KINDS)
def test_forward_from_embeddings_agrees_with_forward(kind):
    """Gradient symmetricity goes through the embedding path, so it must match."""
    data = make_dataset()
    model = LINEAR_MODELS[kind](seed=0)
    with torch.no_grad():
        direct = model.final_logits(data.inputs)
        viaemb = model.forward_from_embeddings(model.embed_tokens(data.inputs))[:, -1, :]
    assert torch.allclose(direct, viaemb, atol=1e-6)
    assert direct.shape == (len(data), P)


def test_gradient_symmetricity_separates_the_linear_models():
    """alpha, alpha', beta and gamma are functions of e_a + e_b, so GS is exactly 1;
    delta concatenates instead, so its two gradients are unrelated."""
    for kind in ("alpha", "alpha_prime", "beta", "gamma"):
        assert M.gradient_symmetricity(LINEAR_MODELS[kind](seed=0)) == pytest.approx(1.0, abs=1e-5)
    assert abs(M.gradient_symmetricity(LINEAR_MODELS["delta"](seed=0))) < 0.2


@pytest.mark.parametrize("kind", KINDS)
def test_linear_models_train(kind):
    cfg = RunConfig(seed=0, attn_coeff=0.0, model_type=kind, betas=(0.9, 0.999),
                    warmup=0, steps=60, experiment="test")
    rec = train_solo(cfg, device="cpu", save_weights=False)
    assert rec["finished"] and 0.0 <= rec["val_accuracy"] <= 1.0
    assert rec["config"]["model_type"] == kind
    for key in ("gs", "di", "circularity", "train_loss"):
        assert rec[key] == rec[key]  # not NaN


def test_constant_learning_rate_when_warmup_is_zero():
    from clockpizza.train import make_scheduler

    cfg = RunConfig(seed=0, attn_coeff=0.0, warmup=0, steps=10)
    opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
    sched = make_scheduler(opt, cfg)
    assert opt.param_groups[0]["lr"] == pytest.approx(1e-3)  # no step-0 zero
    sched.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(1e-3)

    warm = make_scheduler(torch.optim.AdamW(
        [torch.nn.Parameter(torch.zeros(1))], lr=1e-3), RunConfig(seed=0, attn_coeff=0.0))
    assert warm.optimizer.param_groups[0]["lr"] == pytest.approx(0.0)  # warmup starts at 0


def test_isolation_works_on_linear_models():
    """Circle isolation must handle the linear models' (p, d) table as well as the
    transformer's transposed W_E."""
    data = make_dataset()
    model = LINEAR_MODELS["beta"](seed=0)
    pca = C.embedding_pca(model)
    assert pca.proj.shape[0] == P

    full = C.isolated_model(model, list(range(P)), pca=pca)
    with torch.no_grad():
        assert torch.allclose(model.final_logits(data.inputs),
                              full.final_logits(data.inputs), atol=1e-4)
    rank2 = C.isolated_model(model, [0, 1], pca=pca)
    with torch.no_grad():
        assert not torch.allclose(model.final_logits(data.inputs),
                                  rank2.final_logits(data.inputs), atol=1e-3)
    # the original model is untouched
    assert torch.equal(model.embed_table.weight, LINEAR_MODELS["beta"](seed=0).embed_table.weight)


def test_beta_second_relu_switch_changes_the_output():
    data = make_dataset()
    model = LINEAR_MODELS["beta"](seed=0)
    with torch.no_grad():
        on = model(data.inputs, second_relu=True)
        off = model(data.inputs, second_relu=False)
    assert not torch.allclose(on, off)


def test_mid_training_checkpoints(tmp_path, monkeypatch):
    import clockpizza.train as T

    monkeypatch.setattr(T, "WEIGHTS_DIR", tmp_path / "runs")
    monkeypatch.setattr(T, "CHECKPOINT_DIR", tmp_path / "runs/checkpoints")
    cfg = RunConfig(seed=0, attn_coeff=1.0, steps=40, checkpoint_steps=(10, 25),
                    experiment="test")
    T.train_solo(cfg, device="cpu")

    files = sorted(p.name for p in (tmp_path / "runs/checkpoints").glob("*.pt"))
    assert files == [f"{cfg.run_id}_step000010.pt", f"{cfg.run_id}_step000025.pt"]

    model, payload = T.load_checkpoint(cfg.run_id, 10, dir=tmp_path / "runs/checkpoints")
    assert payload["step"] == 10
    later, _ = T.load_checkpoint(cfg.run_id, 25, dir=tmp_path / "runs/checkpoints")
    # the model kept training between the two checkpoints
    assert not torch.equal(model.embed.W_E, later.embed.W_E)


def test_checkpoint_steps_are_part_of_the_run_id():
    base = RunConfig(seed=0, attn_coeff=1.0, steps=40)
    with_ckpt = RunConfig(seed=0, attn_coeff=1.0, steps=40, checkpoint_steps=(10,))
    assert base.run_id != with_ckpt.run_id
