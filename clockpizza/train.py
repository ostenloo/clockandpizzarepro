"""Solo training loop (spec §7) and the run registry (spec §11).

Every run is 20,000 full-batch AdamW steps with weight decay 2.0 and a float64
cross-entropy. No early stopping: a run is finished after 20,000 steps, and analyses
keep only finished runs whose final validation accuracy is 100%.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import pathlib
import platform
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import torch
import torch.nn.functional as F

from . import circles as C
from . import metrics as M
from .data import make_dataset, split_masks
from .model import Transformer, from_config

REGISTRY = pathlib.Path("results/runs.jsonl")
WEIGHTS_DIR = pathlib.Path("runs")


# ----------------------------------------------------------------------------- config

@dataclass(frozen=True)
class RunConfig:
    """One run. ``seed`` drives both initialization and the train/validation split."""

    seed: int
    attn_coeff: float
    d_model: int = 128
    n_layers: int = 1
    n_heads: int = 4
    act_fn: str = "ReLU"
    diff_vocab: bool = False
    eqn_sign: bool = False
    p: int = 59
    steps: int = 20_000
    lr: float = 1e-3
    weight_decay: float = 2.0
    betas: tuple[float, float] = (0.9, 0.98)
    eps: float = 1e-8
    frac: float = 0.8
    warmup: int = 10
    log_every: int = 100
    experiment: str = ""

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @property
    def run_id(self) -> str:
        """Deterministic id from the config, so sweeps can skip finished runs."""
        payload = {k: v for k, v in sorted(self.as_dict().items()) if k != "log_every"}
        digest = hashlib.blake2s(json.dumps(payload, sort_keys=True).encode(), digest_size=5)
        return digest.hexdigest()

    def build(self) -> Transformer:
        vocab = self.p * (2 if self.diff_vocab else 1) + (1 if self.eqn_sign else 0)
        return Transformer(
            n_layers=self.n_layers,
            d_vocab=vocab,
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_head=self.d_head,
            n_ctx=2 + (1 if self.eqn_sign else 0),
            act_type=self.act_fn,
            attn_coeff=self.attn_coeff,
            seed=self.seed,
        )


# ------------------------------------------------------------------------------- loss

def cross_entropy_f64(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Mean cross-entropy with a float64 log_softmax.

    float32 log_softmax underflows on confident data and quantizes to multiples of
    1.2e-7, which shows up as loss spikes near 1e-7.
    """
    logprobs = F.log_softmax(logits.to(torch.float64), dim=-1)
    return -logprobs.gather(-1, labels[:, None]).mean()


def set_precision() -> None:
    """float32 parameters and matmuls at highest precision: no TF32, AMP or bf16."""
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


# ------------------------------------------------------------------------------ train

def train_solo(
    cfg: RunConfig,
    device: Optional[str] = None,
    save_weights: bool = True,
    progress: bool = False,
    return_model: bool = False,
    dtype: torch.dtype = torch.float32,
) -> dict[str, Any] | tuple[dict[str, Any], Transformer]:
    """Train one model for ``cfg.steps`` full-batch steps and return its record.

    With ``return_model`` also returns the trained model, moved to CPU. ``dtype`` is for
    tests only: §7 mandates float32 parameters and matmuls.
    """
    set_precision()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    data = make_dataset(p=cfg.p, diff_vocab=cfg.diff_vocab, eqn_sign=cfg.eqn_sign)
    inputs = data.inputs.to(device)
    labels = data.labels.to(device)
    train_mask, val_mask = (m.to(device) for m in split_masks(cfg.seed, n=len(data), frac=cfg.frac))

    model = cfg.build().to(device=device, dtype=dtype)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=cfg.betas, eps=cfg.eps,
                            weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(s / cfg.warmup, 1.0))

    history: dict[str, list[float]] = {k: [] for k in
                                       ("step", "train_loss", "val_loss", "train_acc", "val_acc", "norm")}
    start = time.perf_counter()
    for step in range(cfg.steps):
        logits = model(inputs)[:, -1, :]
        loss = cross_entropy_f64(logits[train_mask], labels[train_mask])
        loss.backward()
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)

        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            with torch.no_grad():
                val_loss = cross_entropy_f64(logits[val_mask], labels[val_mask]).item()
                pred = logits[:, : cfg.p].argmax(dim=-1)
                correct = (pred == labels).double()
                history["step"].append(step)
                history["train_loss"].append(loss.item())
                history["val_loss"].append(val_loss)
                history["train_acc"].append(correct[train_mask].mean().item())
                history["val_acc"].append(correct[val_mask].mean().item())
                history["norm"].append(model.parameters_norm())
            if progress and step % (cfg.log_every * 20) == 0:
                print(f"  step {step:6d}  train {loss.item():.3e}  val {val_loss:.3e}  "
                      f"acc {history['train_acc'][-1]:.3f}/{history['val_acc'][-1]:.3f}", flush=True)

    wall = time.perf_counter() - start

    record = finalize(model.eval(), cfg, labels.cpu(), inputs, train_mask.cpu(), val_mask.cpu(),
                      wall=wall, device=device, history=history, save_weights=save_weights)
    return (record, model) if return_model else record


def finalize(
    model: Transformer,
    cfg: RunConfig,
    labels: torch.Tensor,
    inputs: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    wall: float,
    device: str,
    history: Optional[dict[str, list[float]]] = None,
    save_weights: bool = True,
) -> dict[str, Any]:
    """Final metrics (§8) for one finished model, plus the saved weights payload."""
    model = model.to("cpu").eval()
    inputs = inputs.cpu()
    with torch.no_grad():
        logits = model(inputs)[:, -1, :]
        train_loss = cross_entropy_f64(logits[train_mask], labels[train_mask]).item()
        val_loss = cross_entropy_f64(logits[val_mask], labels[val_mask]).item()

    correct = (logits[:, : cfg.p].argmax(dim=-1) == labels).double()
    circularity = M.model_circularity(model, p=cfg.p)
    record: dict[str, Any] = dict(
        run_id=cfg.run_id,
        config=cfg.as_dict(),
        seed=cfg.seed,
        train_accuracy=correct[train_mask].mean().item(),
        val_accuracy=correct[val_mask].mean().item(),
        accuracy=correct.mean().item(),
        train_loss=train_loss,
        val_loss=val_loss,
        parameter_norm=model.parameters_norm(),
        di=M.distance_irrelevance_from_logits(logits, p=cfg.p),
        di_top_wrong=M.distance_irrelevance_from_logits(logits, p=cfg.p, variant="top_wrong"),
        gs=M.gradient_symmetricity(model, p=cfg.p, diff_vocab=cfg.diff_vocab, eqn_sign=cfg.eqn_sign),
        circularity=circularity,
        circular=M.is_circular(circularity),
        finished=True,
        wall_seconds=wall,
        device=torch.cuda.get_device_name(0) if str(device).startswith("cuda") else "cpu",
        versions=versions(),
    )
    record["label"] = M.label(record["gs"], record["di"])

    if save_weights:
        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "config": cfg.as_dict(),
                    "metrics": {k: v for k, v in record.items() if k != "config"},
                    "history": history or {}},
                   WEIGHTS_DIR / f"{cfg.run_id}.pt")
    return record


SHARED_KEYS = ("d_model", "n_layers", "n_heads", "act_fn", "diff_vocab", "eqn_sign", "p",
               "steps", "lr", "weight_decay", "betas", "eps", "frac", "warmup")


def train_ensemble(
    cfgs: Sequence[RunConfig],
    device: Optional[str] = None,
    save_weights: bool = True,
    progress: bool = False,
    return_models: bool = False,
    dtype: torch.dtype = torch.float32,
) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], list[Transformer]]:
    """Train every config in one process on stacked parameters (spec §11).

    Members must share width, depth, variant and schedule; they differ in seed,
    attention rate and split. Returns one record per member, each identical in form to
    :func:`train_solo`'s; with ``return_models`` also the exported single models.
    """
    from .ensemble import EnsembleTransformer, ensemble_loss

    assert cfgs, "no configs"
    base = cfgs[0].as_dict()
    for cfg in cfgs[1:]:
        d = cfg.as_dict()
        mismatch = [k for k in SHARED_KEYS if d[k] != base[k]]
        assert not mismatch, f"ensemble members differ in {mismatch}"

    set_precision()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg0 = cfgs[0]

    data = make_dataset(p=cfg0.p, diff_vocab=cfg0.diff_vocab, eqn_sign=cfg0.eqn_sign)
    inputs = data.inputs.to(device)
    labels = data.labels.to(device)
    masks = [split_masks(c.seed, n=len(data), frac=c.frac) for c in cfgs]
    train_mask = torch.stack([m[0] for m in masks]).to(device)
    val_mask = torch.stack([m[1] for m in masks]).to(device)

    vocab = cfg0.p * (2 if cfg0.diff_vocab else 1) + (1 if cfg0.eqn_sign else 0)
    model = EnsembleTransformer.from_seeds(
        [c.seed for c in cfgs], [c.attn_coeff for c in cfgs],
        n_layers=cfg0.n_layers, d_vocab=vocab, d_model=cfg0.d_model, n_heads=cfg0.n_heads,
        d_head=cfg0.d_head, n_ctx=2 + (1 if cfg0.eqn_sign else 0), act_type=cfg0.act_fn,
    ).to(device=device, dtype=dtype)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg0.lr, betas=cfg0.betas, eps=cfg0.eps,
                            weight_decay=cfg0.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(s / cfg0.warmup, 1.0))

    histories: list[dict[str, list[float]]] = [
        {k: [] for k in ("step", "train_loss", "val_loss", "train_acc", "val_acc")} for _ in cfgs]
    start = time.perf_counter()
    for step in range(cfg0.steps):
        logits = model(inputs)
        loss, per_member = ensemble_loss(logits, labels, train_mask)
        loss.backward()
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)

        if step % cfg0.log_every == 0 or step == cfg0.steps - 1:
            with torch.no_grad():
                _, val_per = ensemble_loss(logits.detach(), labels, val_mask)
                correct = (logits[:, :, : cfg0.p].argmax(dim=-1) == labels).double()
                tr_acc = (correct * train_mask).sum(1) / train_mask.sum(1)
                va_acc = (correct * val_mask).sum(1) / val_mask.sum(1)
            for e, hist in enumerate(histories):
                hist["step"].append(step)
                hist["train_loss"].append(per_member[e].item())
                hist["val_loss"].append(val_per[e].item())
                hist["train_acc"].append(tr_acc[e].item())
                hist["val_acc"].append(va_acc[e].item())
            if progress and step % (cfg0.log_every * 20) == 0:
                print(f"  step {step:6d}  mean train {per_member.mean().item():.3e}  "
                      f"mean val acc {va_acc.mean().item():.3f}", flush=True)

    wall = time.perf_counter() - start
    per_run_wall = wall / len(cfgs)

    records: list[dict[str, Any]] = []
    members: list[Transformer] = []
    for e, cfg in enumerate(cfgs):
        member = model.export_member(e).to(torch.float32 if dtype == torch.float32 else dtype)
        rec = finalize(member, cfg, labels.cpu(), inputs, train_mask[e].cpu(), val_mask[e].cpu(),
                       wall=per_run_wall, device=device, history=histories[e],
                       save_weights=save_weights)
        rec["ensemble_size"] = len(cfgs)
        rec["ensemble_wall_seconds"] = wall
        records.append(rec)
        members.append(member)
    return (records, members) if return_models else records


def versions() -> dict[str, str]:
    import numpy

    # str() matters: torch.__version__ is a TorchVersion, which torch.load refuses
    # under weights_only=True when it is pickled into a checkpoint payload.
    return dict(
        python=sys.version.split()[0],
        torch=str(torch.__version__),
        cuda=str(torch.version.cuda),
        numpy=numpy.__version__,
        platform=platform.platform(),
        driver=_driver_version(),
    )


def _driver_version() -> str:
    try:
        import subprocess

        out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip().splitlines()[0] if out.returncode == 0 else ""
    except Exception:  # noqa: BLE001
        return ""


# --------------------------------------------------------------------------- registry

def registry_ids(path: pathlib.Path = REGISTRY) -> set[str]:
    """Run ids already in the registry, so a sweep can resume after a crash."""
    if not path.exists():
        return set()
    ids = set()
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                ids.add(json.loads(line)["run_id"])
    return ids


def append_record(record: dict[str, Any], path: pathlib.Path = REGISTRY) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(record) + "\n")


def load_registry(path: pathlib.Path = REGISTRY) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_run(run_id: str, dir: pathlib.Path = WEIGHTS_DIR) -> tuple[Transformer, dict[str, Any]]:
    """Rebuild a saved run's model from ``runs/<run_id>.pt``."""
    # Runs saved before versions() cast the torch version to str carry a TorchVersion
    # in their metrics; allow it so those checkpoints still load under weights_only.
    torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])
    payload = torch.load(dir / f"{run_id}.pt", map_location="cpu")
    cfg = RunConfig(**payload["config"])
    model = cfg.build()
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model, payload
