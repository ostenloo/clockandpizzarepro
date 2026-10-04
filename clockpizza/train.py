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
from typing import Any, Optional

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
) -> dict[str, Any]:
    """Train one model for ``cfg.steps`` full-batch steps and return its record."""
    set_precision()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    data = make_dataset(p=cfg.p, diff_vocab=cfg.diff_vocab, eqn_sign=cfg.eqn_sign)
    inputs = data.inputs.to(device)
    labels = data.labels.to(device)
    train_mask, val_mask = (m.to(device) for m in split_masks(cfg.seed, n=len(data), frac=cfg.frac))

    model = cfg.build().to(device)
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

    model.eval()
    with torch.no_grad():
        logits = model(inputs)[:, -1, :]
        final_train_loss = cross_entropy_f64(logits[train_mask], labels[train_mask]).item()
        final_val_loss = cross_entropy_f64(logits[val_mask], labels[val_mask]).item()
    cpu_logits = logits.detach().float().cpu()
    cpu_labels = labels.cpu()
    cpu_model = model.to("cpu")

    pred = cpu_logits[:, : cfg.p].argmax(dim=-1)
    correct = (pred == cpu_labels).double()
    circularity = M.model_circularity(cpu_model, p=cfg.p)
    record: dict[str, Any] = dict(
        run_id=cfg.run_id,
        config=cfg.as_dict(),
        seed=cfg.seed,
        train_accuracy=correct[train_mask.cpu()].mean().item(),
        val_accuracy=correct[val_mask.cpu()].mean().item(),
        accuracy=correct.mean().item(),
        train_loss=final_train_loss,
        val_loss=final_val_loss,
        parameter_norm=cpu_model.parameters_norm(),
        di=M.distance_irrelevance_from_logits(cpu_logits, p=cfg.p),
        di_top_wrong=M.distance_irrelevance_from_logits(cpu_logits, p=cfg.p, variant="top_wrong"),
        gs=M.gradient_symmetricity(cpu_model, p=cfg.p, diff_vocab=cfg.diff_vocab, eqn_sign=cfg.eqn_sign),
        circularity=circularity,
        circular=M.is_circular(circularity),
        finished=True,
        wall_seconds=wall,
        device=torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        versions=versions(),
    )
    record["label"] = M.label(record["gs"], record["di"])

    if save_weights:
        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": cpu_model.state_dict(), "config": cfg.as_dict(),
                    "metrics": {k: v for k, v in record.items() if k != "config"},
                    "history": history},
                   WEIGHTS_DIR / f"{cfg.run_id}.pt")
    return record


def versions() -> dict[str, str]:
    import numpy

    return dict(
        python=sys.version.split()[0],
        torch=torch.__version__,
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
    payload = torch.load(dir / f"{run_id}.pt", map_location="cpu")
    cfg = RunConfig(**payload["config"])
    model = cfg.build()
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model, payload
