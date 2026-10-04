"""One function per paper figure (spec §10). Files land in ``figures/figNN_<panel>.png``."""

from __future__ import annotations

import pathlib
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from . import circles as C  # noqa: E402
from . import metrics as M  # noqa: E402
from .data import make_dataset  # noqa: E402
from .metrics import P  # noqa: E402

FIGURES = pathlib.Path("figures")
FIG2_TRIPLES = ((7, 49, 33), (37, 28, 16))


def _save(fig, name: str, outdir: Optional[pathlib.Path] = None) -> pathlib.Path:
    outdir = FIGURES if outdir is None else outdir
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _logits(model, data=None) -> torch.Tensor:
    data = make_dataset() if data is None else data
    with torch.no_grad():
        return model.final_logits(data.inputs)


# ------------------------------------------------------------------- Fig. 2: gradients

def fig02_gradient_symmetry(model, triples: Sequence[tuple[int, int, int]] = FIG2_TRIPLES,
                            n_pcs: int = 6, outdir=None, tag: str = "") -> pathlib.Path:
    """Both input gradients projected onto the top ``n_pcs`` PCs of W_E, first-token
    component against second-token component, around y = x.

    On the diagonal means the two gradients agree -- the Pizza signature (high GS).
    """
    pca = C.embedding_pca(model.embed.W_E)
    basis = pca.Vt[:n_pcs].to(torch.float32)  # (n_pcs, d)

    tokens = torch.tensor([[a, b] for a, b, _ in triples], dtype=torch.long)
    cs = torch.tensor([c for _, _, c in triples], dtype=torch.long)
    e = model.embed(tokens).detach().clone().requires_grad_(True)
    logits = model.forward_from_embeddings(e)[:, -1, :]
    (grad,) = torch.autograd.grad(logits[torch.arange(len(triples)), cs].sum(), e)

    fig, axes = plt.subplots(1, len(triples), figsize=(5 * len(triples), 4.6), squeeze=False)
    for i, (triple, ax) in enumerate(zip(triples, axes[0])):
        first = (grad[i, 0] @ basis.T).numpy()
        second = (grad[i, 1] @ basis.T).numpy()
        lim = max(np.abs(first).max(), np.abs(second).max()) * 1.25 + 1e-9
        ax.plot([-lim, lim], [-lim, lim], color="0.6", lw=1, ls="--", label="y = x", zorder=1)
        ax.axhline(0, color="0.85", lw=0.8, zorder=0)
        ax.axvline(0, color="0.85", lw=0.8, zorder=0)
        ax.scatter(first, second, c=range(n_pcs), cmap="viridis", s=90, zorder=2,
                   edgecolor="white", linewidth=0.7)
        for pc in range(n_pcs):
            ax.annotate(f"PC{pc + 1}", (first[pc], second[pc]), fontsize=8,
                        xytext=(4, 4), textcoords="offset points")
        cos = float(torch.nn.functional.cosine_similarity(
            grad[i, 0].double(), grad[i, 1].double(), dim=0))
        a, b, c = triple
        ax.set_title(f"(a, b, c) = ({a}, {b}, {c})    cosine {cos:.3f}")
        ax.set_xlabel("gradient w.r.t. first token, PC component")
        ax.set_ylabel("gradient w.r.t. second token, PC component")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        if i == 0:
            ax.legend(loc="upper left", fontsize=8)
    fig.suptitle(f"Fig. 2 analog -- input-gradient symmetry{(' ' + tag) if tag else ''}", y=1.01)
    return _save(fig, f"fig02_gradient_symmetry{('_' + tag) if tag else ''}.png", outdir)


# ------------------------------------------------------- Fig. 3: correct-logit heatmap

def fig03_correct_logit_heatmap(model, data=None, outdir=None, tag: str = "") -> pathlib.Path:
    """Correct-logit heatmap with rows a - b and columns a + b (the full model)."""
    data = make_dataset() if data is None else data
    # correct_logit_matrix is indexed [(a+b), (a-b)]; Fig. 3 wants the transpose
    M_mat = M.correct_logit_matrix(_logits(model, data), p=data.p).T.double().numpy()
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(M_mat, cmap="Reds_r", aspect="auto", origin="upper")
    ax.set_xlabel("(a + b) mod p")
    ax.set_ylabel("(a - b) mod p")
    di = M.distance_irrelevance(torch.tensor(M_mat.T))
    ax.set_title(f"Fig. 3 analog -- correct logits{(' ' + tag) if tag else ''}\n"
                 f"distance irrelevance {di:.3f}")
    fig.colorbar(im, ax=ax, label="logit of the correct class")
    return _save(fig, f"fig03_correct_logits{('_' + tag) if tag else ''}.png", outdir)


# ------------------------------------------------- Fig. 4 / 13: circles under isolation

def _circle_panels(model, circles: Sequence[C.Circle], data, pca, title: str, fname: str,
                   accompanying: bool, outdir) -> pathlib.Path:
    n = len(circles)
    fig, axes = plt.subplots(2, n, figsize=(4.6 * n, 8.4), squeeze=False)
    for j, circle in enumerate(circles):
        logits = _logits(C.isolated_model(model, list(circle.pcs), pca=pca), data)
        heat = C.logit_heatmap(logits, circle.k, p=data.p)
        acc = M.accuracy(logits, data.labels) * 100

        ax = axes[0][j]
        im = ax.imshow(heat, cmap="Reds_r", aspect="auto")
        ax.set_xlabel("(a + b) mod p")
        ax.set_ylabel(f"(a - b) / {circle.delta} mod p")
        kind = f"accompanies k = {circle.accompanies}" if accompanying else "main"
        ax.set_title(f"circle k = {circle.k}, $\\delta$ = {circle.delta} ({kind})\n"
                     f"accuracy {acc:.1f}%")
        fig.colorbar(im, ax=ax, label="correct logit")

        ax = axes[1][j]
        x, y = pca.proj[:, circle.pcs[0]].numpy(), pca.proj[:, circle.pcs[1]].numpy()
        ax.scatter(x, y, c="crimson", s=22)
        for t in range(data.p):
            ax.annotate(str(t), (x[t], y[t]), fontsize=7, xytext=(3, 3),
                        textcoords="offset points")
        ax.set_xlabel(f"PC {circle.pcs[0] + 1}")
        ax.set_ylabel(f"PC {circle.pcs[1] + 1}")
        ax.set_title(f"neighbours step by {circle.delta}: 0, {circle.delta}, "
                     f"{2 * circle.delta % data.p}, ...")
        ax.set_aspect("equal", adjustable="datalim")
    fig.suptitle(title, y=1.0)
    return _save(fig, fname, outdir)


def fig04_main_circles(model, main: Sequence[C.Circle], data=None, pca=None,
                       outdir=None) -> pathlib.Path:
    """Each main circle isolated: its correct-logit heatmap and its PC-pair scatter."""
    data = make_dataset() if data is None else data
    pca = C.embedding_pca(model.embed.W_E) if pca is None else pca
    return _circle_panels(model, main, data, pca,
                          "Fig. 4 analog -- main circles under isolation",
                          "fig04_main_circles.png", accompanying=False, outdir=outdir)


def fig13_accompanying_circles(model, acc: Sequence[C.Circle], data=None, pca=None,
                               outdir=None) -> pathlib.Path:
    """The accompanying circles, at twice the frequency and half the spacing."""
    data = make_dataset() if data is None else data
    pca = C.embedding_pca(model.embed.W_E) if pca is None else pca
    return _circle_panels(model, acc, data, pca,
                          "Fig. 13 analog -- accompanying circles",
                          "fig13_accompanying_circles.png", accompanying=True, outdir=outdir)


# --------------------------------------------- Fig. 5: accuracy per difference a - b

def fig05_accuracy_per_difference(model, main: Sequence[C.Circle], data=None, pca=None,
                                  outdir=None) -> pathlib.Path:
    """Accuracy against a - b for each isolated circle, plus all circles together.

    A Pizza circle collapses to 0% at specific differences (the near-antipodal inputs);
    keeping every circle restores full accuracy.
    """
    data = make_dataset() if data is None else data
    pca = C.embedding_pca(model.embed.W_E) if pca is None else pca

    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    for i, circle in enumerate(main, start=1):
        logits = _logits(C.isolated_model(model, list(circle.pcs), pca=pca), data)
        per_diff = C.accuracy_per_diff(logits, data.labels, p=data.p) * 100
        ax.plot(range(data.p), per_diff, marker="o", ms=3, lw=1.2,
                label=f"circle #{i} alone (k = {circle.k}): {per_diff.mean():.1f}%")
    if len(main) > 1:
        logits = _logits(C.isolated_model(model, C.circle_pcs(main), pca=pca), data)
        per_diff = C.accuracy_per_diff(logits, data.labels, p=data.p) * 100
        ax.plot(range(data.p), per_diff, color="black", lw=2,
                label=f"all main circles: {per_diff.mean():.1f}%")
    ax.set_xlabel("(a - b) mod p")
    ax.set_ylabel("accuracy (%)")
    ax.set_ylim(-3, 103)
    ax.set_title("Fig. 5 analog -- accuracy by difference under circle isolation")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    return _save(fig, "fig05_accuracy_per_difference.png", outdir)


# ---------------------------------------------------------------- Fig. 9: non-circular

def fig09_pc_plots(models: Sequence, labels: Sequence[str], n_pcs: int = 4,
                   outdir=None) -> pathlib.Path:
    """PC plots of runs that did not learn circular embeddings (App. B)."""
    fig, axes = plt.subplots(len(models), n_pcs, figsize=(3.1 * n_pcs, 3.0 * len(models)),
                             squeeze=False)
    for row, (model, label) in enumerate(zip(models, labels)):
        pca = C.embedding_pca(model.embed.W_E)
        scores = pca.scores(n_pcs)
        for col in range(n_pcs):
            ax = axes[row][col]
            ax.plot(pca.proj[:, col].numpy(), lw=1.2)
            ax.set_title(f"{label}  PC {col + 1}  (c = {scores[col]:.2f})", fontsize=9)
            ax.set_xlabel("token")
    fig.suptitle("Fig. 9 analog -- principal components of non-circular runs", y=1.0)
    return _save(fig, "fig09_non_circular_pcs.png", outdir)


def all_e1_figures(model_pizza, model_clock, outdir=None) -> dict[str, pathlib.Path]:
    """Every E1 figure: Fig. 2-5 for the Pizza run, Fig. 13 for its accompanying
    circles, and Fig. 3 for the Clock run."""
    data = make_dataset()
    out: dict[str, pathlib.Path] = {}

    pca = C.embedding_pca(model_pizza.embed.W_E)
    found = C.find_circles(model_pizza.embed.W_E, pca=pca)
    main, acc = C.classify_circles(found, model_pizza, data, pca=pca)

    out["fig02_pizza"] = fig02_gradient_symmetry(model_pizza, outdir=outdir, tag="pizza")
    out["fig03_pizza"] = fig03_correct_logit_heatmap(model_pizza, data, outdir, tag="pizza")
    out["fig04"] = fig04_main_circles(model_pizza, main, data, pca, outdir)
    out["fig05"] = fig05_accuracy_per_difference(model_pizza, main, data, pca, outdir)
    if acc:
        out["fig13"] = fig13_accompanying_circles(model_pizza, acc, data, pca, outdir)

    out["fig03_clock"] = fig03_correct_logit_heatmap(model_clock, data, outdir, tag="clock")
    out["fig02_clock"] = fig02_gradient_symmetry(model_clock, outdir=outdir, tag="clock")
    return out


# ------------------------------------------------- Fig. 6 / 7 / 10: sweeps over alpha

def fig06_di_vs_gs(runs: Sequence[dict], outdir=None, di_key: str = "di") -> pathlib.Path:
    """Distance irrelevance against gradient symmetricity for every sweep run.

    The two algorithms separate into opposite corners: Pizza high-GS / low-DI,
    Clock low-GS / high-DI.
    """
    gs = np.array([r["gs"] for r in runs])
    di = np.array([r[di_key] for r in runs])
    alpha = np.array([r["config"]["attn_coeff"] for r in runs])

    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    sc = ax.scatter(gs, di, c=alpha, cmap="coolwarm", s=12, alpha=0.75, edgecolor="none")
    ax.axvline(0.98, color="0.4", lw=1, ls="--")
    ax.axhline(0.6, color="0.4", lw=1, ls=":")
    ax.set_xlabel("gradient symmetricity")
    ax.set_ylabel(f"distance irrelevance ({'correct' if di_key == 'di' else 'top-wrong'})")
    ax.set_title(f"Fig. 6 analog -- {len(runs)} runs")
    fig.colorbar(sc, ax=ax, label="attention rate $\\alpha$")
    ax.grid(alpha=0.25)
    return _save(fig, f"fig06_di_vs_gs{'_topwrong' if di_key != 'di' else ''}.png", outdir)


def logistic_boundary(alpha: np.ndarray, label: np.ndarray) -> Optional[dict]:
    """Unregularized logistic regression of a binary label on alpha.

    Returns the fit and ``alpha_star``, where the predicted probability is 0.5.
    """
    from sklearn.linear_model import LogisticRegression

    label = np.asarray(label, dtype=int)
    if label.min() == label.max() or len(label) < 3:
        return None
    # C = inf is the unregularized fit the spec asks for; penalty=None is deprecated
    fit = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=1000)
    fit.fit(np.asarray(alpha).reshape(-1, 1), label)
    coef = float(fit.coef_[0, 0])
    if abs(coef) < 1e-9:
        return None
    return {"coef": coef, "intercept": float(fit.intercept_[0]),
            "alpha_star": -float(fit.intercept_[0]) / coef, "fit": fit, "n": len(label)}


def fig07_phase_boundary(runs: Sequence[dict], outdir=None, tag: str = "top") -> pathlib.Path:
    """GS and DI against alpha for circular runs, with the fitted phase boundary."""
    alpha = np.array([r["config"]["attn_coeff"] for r in runs])
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))
    for ax, key, label, thresh, above in (
        (axes[0], "gs", "gradient symmetricity", 0.98, True),
        (axes[1], "di", "distance irrelevance (correct)", 0.6, False),
    ):
        y = np.array([r[key] for r in runs])
        ax.scatter(alpha, y, s=10, alpha=0.6, color="tab:blue", edgecolor="none")
        ax.axhline(thresh, color="0.4", lw=1, ls="--", label=f"{label} = {thresh}")
        bound = logistic_boundary(alpha, (y > thresh) if above else (y < thresh))
        if bound is not None:
            ax.axvline(bound["alpha_star"], color="crimson", lw=1.8,
                       label=f"$\\alpha^*$ = {bound['alpha_star']:.2f}")
        ax.set_xlabel("attention rate $\\alpha$")
        ax.set_ylabel(label)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    fig.suptitle(f"Fig. 7 ({tag}) analog -- phase boundary, {len(runs)} circular runs")
    return _save(fig, f"fig07_{tag}_phase_boundary.png", outdir)


def fig10_circular_share(runs: Sequence[dict], outdir=None, bins: int = 10,
                         tag: str = "top") -> pathlib.Path:
    """Share of finished runs that are circular, by attention rate."""
    alpha = np.array([r["config"]["attn_coeff"] for r in runs])
    circ = np.array([bool(r["circular"]) for r in runs])
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(alpha, edges) - 1, 0, bins - 1)

    centres, shares, counts = [], [], []
    for b in range(bins):
        sel = idx == b
        if sel.sum():
            centres.append((edges[b] + edges[b + 1]) / 2)
            shares.append(circ[sel].mean() * 100)
            counts.append(int(sel.sum()))

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.bar(centres, shares, width=(edges[1] - edges[0]) * 0.85, color="tab:purple", alpha=0.8)
    for c, s, n in zip(centres, shares, counts):
        ax.annotate(f"n={n}", (c, s), ha="center", va="bottom", fontsize=7)
    ax.axhline(circ.mean() * 100, color="0.3", ls="--", lw=1,
               label=f"overall {circ.mean() * 100:.1f}%")
    ax.set_xlabel("attention rate $\\alpha$")
    ax.set_ylabel("circular runs (%)")
    ax.set_title(f"Fig. 10 ({tag}) analog -- circular share by $\\alpha$, {len(runs)} runs")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")
    return _save(fig, f"fig10_{tag}_circular_share.png", outdir)
