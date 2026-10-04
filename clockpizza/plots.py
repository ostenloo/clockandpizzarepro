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
    # too few points, or all one class: there is no boundary to fit
    if len(label) < 3 or label.min() == label.max():
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


# ------------------------------------------------- Fig. 7 / 10 bottom, Fig. 11: width and depth

def boundary_by_group(runs: Sequence[dict], key: str, thresh: float, above: bool,
                      group: str = "d_model") -> dict[Any, float]:
    """Per-group logistic boundary alpha*, e.g. one per width or per depth."""
    out: dict[Any, float] = {}
    groups: dict[Any, list[dict]] = {}
    for r in runs:
        groups.setdefault(r["config"][group], []).append(r)
    for g, rs in sorted(groups.items()):
        y = np.array([r[key] for r in rs])
        a = np.array([r["config"]["attn_coeff"] for r in rs])
        b = logistic_boundary(a, (y > thresh) if above else (y < thresh))
        if b is not None:
            out[g] = b["alpha_star"]
    return out


def fig07_width_phase(runs: Sequence[dict], outdir=None) -> pathlib.Path:
    """Fig. 7 bottom: (alpha, log2 d) scatter coloured by GS and by DI, with the
    per-width phase boundary."""
    alpha = np.array([r["config"]["attn_coeff"] for r in runs])
    width = np.array([r["config"]["d_model"] for r in runs], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.9))
    for ax, key, label, thresh, above in (
        (axes[0], "gs", "gradient symmetricity", 0.98, True),
        (axes[1], "di", "distance irrelevance (correct)", 0.6, False),
    ):
        y = np.array([r[key] for r in runs])
        sc = ax.scatter(alpha, np.log2(width), c=y, cmap="coolwarm", s=14, alpha=0.85,
                        edgecolor="none")
        bounds = boundary_by_group(runs, key, thresh, above)
        if bounds:
            ws = sorted(bounds)
            ax.plot([bounds[w] for w in ws], np.log2(ws), color="black", marker="o",
                    ms=4, lw=1.8, label="$\\alpha^*$ per width")
            ax.legend(fontsize=8)
        ax.set_xlabel("attention rate $\\alpha$")
        ax.set_ylabel("$\\log_2 d$")
        ax.set_title(label)
        fig.colorbar(sc, ax=ax, label=label)
    fig.suptitle(f"Fig. 7 (bottom) analog -- width sweep, {len(runs)} circular runs")
    return _save(fig, "fig07_bottom_width_phase.png", outdir)


def fig10_width_circular_share(runs: Sequence[dict], outdir=None) -> pathlib.Path:
    """Fig. 10 bottom: circular share against width."""
    widths = sorted({r["config"]["d_model"] for r in runs})
    shares, counts = [], []
    for w in widths:
        g = [r for r in runs if r["config"]["d_model"] == w]
        shares.append(100 * sum(bool(r["circular"]) for r in g) / len(g))
        counts.append(len(g))

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.plot(widths, shares, marker="o", lw=1.6, color="tab:purple")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("width $d$")
    ax.set_ylabel("circular runs (%)")
    ax.set_title(f"Fig. 10 (bottom) analog -- circular share by width, {len(runs)} runs")
    ax.grid(alpha=0.3)
    return _save(fig, "fig10_bottom_width_circular_share.png", outdir)


def fig11_depth(runs: Sequence[dict], outdir=None) -> pathlib.Path:
    """Fig. 11: GS and DI against alpha at 2, 3 and 4 layers, plus circular share."""
    depths = sorted({r["config"]["n_layers"] for r in runs})
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.6))
    for ax, key, label, thresh, above in (
        (axes[0], "gs", "gradient symmetricity", 0.98, True),
        (axes[1], "di", "distance irrelevance (correct)", 0.6, False),
    ):
        for depth in depths:
            g = [r for r in runs if r["config"]["n_layers"] == depth]
            ax.scatter([r["config"]["attn_coeff"] for r in g], [r[key] for r in g],
                       s=9, alpha=0.5, label=f"{depth} layers")
        ax.axhline(thresh, color="0.4", lw=1, ls="--")
        ax.set_xlabel("attention rate $\\alpha$")
        ax.set_ylabel(label)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)

    ax = axes[2]
    shares = []
    for depth in depths:
        g = [r for r in runs if r["config"]["n_layers"] == depth]
        perfect = [r for r in g if r["val_accuracy"] == 1.0]
        shares.append(100 * sum(bool(r["circular"]) for r in perfect) / len(perfect)
                      if perfect else 0.0)
    ax.bar([str(d) for d in depths], shares, color="tab:purple", alpha=0.85)
    for i, s in enumerate(shares):
        ax.annotate(f"{s:.1f}%", (i, s), ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("layers")
    ax.set_ylabel("circular share of 100%-validation runs (%)")
    ax.grid(alpha=0.25, axis="y")
    fig.suptitle(f"Fig. 11 analog -- depth sweep, {len(runs)} runs")
    return _save(fig, "fig11_depth.png", outdir)


# ------------------------------------------------- Fig. 14-16: the App. E linear models

LINEAR_LABELS = {"alpha": "$\\alpha$", "alpha_prime": "$\\alpha'$", "beta": "$\\beta$",
                 "gamma": "$\\gamma$", "delta": "$\\delta$"}


def fig14_linear_metrics(runs: Sequence[dict], outdir=None) -> pathlib.Path:
    """DI against GS, and circularity, for each linear model."""
    kinds = [k for k in LINEAR_LABELS if any(r["config"]["model_type"] == k for r in runs)]
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8))
    for kind in kinds:
        g = [r for r in runs if r["config"]["model_type"] == kind]
        axes[0].scatter([r["gs"] for r in g], [r["di"] for r in g], s=14, alpha=0.6,
                        label=LINEAR_LABELS[kind])
        axes[1].scatter([r["circularity"] for r in g], [r["di"] for r in g], s=14, alpha=0.6,
                        label=LINEAR_LABELS[kind])
    axes[0].set_xlabel("gradient symmetricity")
    axes[0].set_ylabel("distance irrelevance")
    axes[1].set_xlabel("circularity")
    axes[1].set_ylabel("distance irrelevance")
    axes[1].axvline(0.995, color="0.4", lw=1, ls="--", label="circular threshold")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    fig.suptitle(f"Fig. 14 analog -- linear models, {len(runs)} runs")
    return _save(fig, "fig14_linear_metrics.png", outdir)


def fig15_linear_di_hist(runs: Sequence[dict], outdir=None) -> pathlib.Path:
    """Distance-irrelevance histogram per linear model (the T18 ordering, visually)."""
    kinds = [k for k in LINEAR_LABELS if any(r["config"]["model_type"] == k for r in runs)]
    fig, axes = plt.subplots(1, len(kinds), figsize=(3.3 * len(kinds), 3.5), squeeze=False)
    bins = np.linspace(0, 1, 26)
    for ax, kind in zip(axes[0], kinds):
        di = [r["di"] for r in runs if r["config"]["model_type"] == kind]
        ax.hist(di, bins=bins, color="tab:blue", alpha=0.85)
        ax.axvline(float(np.median(di)), color="crimson", lw=1.6,
                   label=f"median {np.median(di):.2f}")
        ax.set_title(f"{LINEAR_LABELS[kind]}  (n = {len(di)})")
        ax.set_xlabel("distance irrelevance")
        ax.legend(fontsize=8)
    fig.suptitle("Fig. 15 analog -- distance irrelevance by linear model", y=1.02)
    return _save(fig, "fig15_linear_di_hist.png", outdir)


def fig16_linear_isolation(models: dict, outdir=None) -> pathlib.Path:
    """Circle isolation applied to one circular linear model of each named kind."""
    from . import circles as C

    data = make_dataset()
    items = [(name, m) for name, m in models.items() if m is not None]
    fig, axes = plt.subplots(2, len(items), figsize=(4.8 * len(items), 8.2), squeeze=False)
    for j, (name, model) in enumerate(items):
        pca = C.embedding_pca(model)
        found = C.find_circles(model, pca=pca)
        circle = found[0] if found else None
        ax = axes[0][j]
        if circle is None:
            ax.text(0.5, 0.5, "no circle found", ha="center", va="center")
            ax.set_axis_off()
        else:
            iso = C.isolated_model(model, list(circle.pcs), pca=pca)
            with torch.no_grad():
                logits = iso.final_logits(data.inputs)
            im = ax.imshow(C.logit_heatmap(logits, circle.k), cmap="Reds_r", aspect="auto")
            acc = M.accuracy(logits, data.labels) * 100
            ax.set_title(f"{name}: k = {circle.k}, $\\delta$ = {circle.delta}, {acc:.1f}%")
            ax.set_xlabel("(a + b) mod p")
            ax.set_ylabel(f"(a - b) / {circle.delta} mod p")
            fig.colorbar(im, ax=ax, label="correct logit")
        ax2 = axes[1][j]
        pcs = circle.pcs if circle else (0, 1)
        x, y = pca.proj[:, pcs[0]].numpy(), pca.proj[:, pcs[1]].numpy()
        ax2.scatter(x, y, c="crimson", s=20)
        for t in range(data.p):
            ax2.annotate(str(t), (x[t], y[t]), fontsize=6, xytext=(3, 3),
                         textcoords="offset points")
        ax2.set_xlabel(f"PC {pcs[0] + 1}")
        ax2.set_ylabel(f"PC {pcs[1] + 1}")
        ax2.set_aspect("equal", adjustable="datalim")
    fig.suptitle("Fig. 16 analog -- circle isolation on linear models", y=1.0)
    return _save(fig, "fig16_linear_isolation.png", outdir)


# ------------------------------------------------------ Fig. 24-26: App. L, the linear beta

def fig25_aligned_weights(model, outdir=None) -> pathlib.Path:
    """W1 and W2 of a linear beta run, rows ordered by the phase they respond to."""
    W1 = model.l1.weight.detach().cpu().numpy()
    W2 = model.l2.weight.detach().cpu().numpy()
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    for ax, W, name in ((axes[0], W1, "$W_1$"), (axes[1], W2, "$W_2$")):
        order = np.argsort(np.arctan2(W[:, 1], W[:, 0]))
        im = ax.imshow(W[order], aspect="auto", cmap="RdBu_r",
                       vmin=-np.abs(W).max(), vmax=np.abs(W).max())
        ax.set_title(f"{name}, rows sorted by phase")
        ax.set_xlabel("input unit")
        ax.set_ylabel("hidden unit (sorted)")
        fig.colorbar(im, ax=ax)
    fig.suptitle("Fig. 25 analog -- aligned weights of the linear $\\beta$ model")
    return _save(fig, "fig25_aligned_weights.png", outdir)


def fig26_second_harmonic_fit(model, outdir=None, n: int = 721) -> pathlib.Path:
    """Fit f(cos t, sin t) to A cos(2t + phi) for the beta model's hidden layer (App. L).

    The second harmonic is the signature App. L reports: the layer responds to twice
    the input angle.
    """
    pca_dirs = model.l1.weight.detach().cpu().numpy()[:, :2]
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    unit = np.stack([np.cos(t), np.sin(t)], axis=1)          # (n, 2)
    h = np.maximum(unit @ pca_dirs.T, 0.0)                    # ReLU hidden response
    f = h.mean(axis=1)                                        # average over hidden units

    # least squares against a constant plus the second harmonic
    design = np.stack([np.ones_like(t), np.cos(2 * t), np.sin(2 * t)], axis=1)
    coef, *_ = np.linalg.lstsq(design, f, rcond=None)
    fit = design @ coef
    amp = float(np.hypot(coef[1], coef[2]))
    phase = float(np.arctan2(-coef[2], coef[1]))
    ss = 1 - np.var(f - fit) / np.var(f) if np.var(f) > 0 else float("nan")

    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    ax.plot(t, f, lw=1.8, label="$f(\\cos t, \\sin t)$")
    ax.plot(t, fit, lw=1.4, ls="--",
            label=f"${amp:.3f}\\cos(2t {phase:+.2f})$ + const, FVE {ss * 100:.1f}%")
    ax.set_xlabel("$t$")
    ax.set_ylabel("mean hidden response")
    ax.set_title("Fig. 26 analog -- second-harmonic fit, linear $\\beta$")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    return _save(fig, "fig26_second_harmonic_fit.png", outdir)


# ------------------------------------------- Fig. 18-21: App. I setup variants

def fig18_variant_metrics(runs: Sequence[dict], outdir=None) -> pathlib.Path:
    """DI against GS, and each against alpha, for the GeLU / diff_vocab / eqn_sign runs."""
    def variant(r):
        c = r["config"]
        if c.get("diff_vocab"):
            return "diff_vocab"
        if c.get("eqn_sign"):
            return "eqn_sign"
        return c.get("act_fn", "ReLU")

    kinds = sorted({variant(r) for r in runs})
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.6))
    for kind in kinds:
        g = [r for r in runs if variant(r) == kind]
        a = [r["config"]["attn_coeff"] for r in g]
        axes[0].scatter([r["gs"] for r in g], [r["di"] for r in g], s=10, alpha=0.5, label=kind)
        axes[1].scatter(a, [r["gs"] for r in g], s=10, alpha=0.5, label=kind)
        axes[2].scatter(a, [r["di"] for r in g], s=10, alpha=0.5, label=kind)
    axes[0].set_xlabel("gradient symmetricity")
    axes[0].set_ylabel("distance irrelevance")
    axes[1].set_xlabel("attention rate $\\alpha$")
    axes[1].set_ylabel("gradient symmetricity")
    axes[1].axhline(0.98, color="0.4", lw=1, ls="--")
    axes[2].set_xlabel("attention rate $\\alpha$")
    axes[2].set_ylabel("distance irrelevance")
    axes[2].axhline(0.6, color="0.4", lw=1, ls="--")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    fig.suptitle(f"Fig. 18-19 analog -- setup variants, {len(runs)} runs")
    return _save(fig, "fig18_variant_metrics.png", outdir)


def fig20_aligned_embeddings(model, p: int = P, outdir=None) -> pathlib.Path:
    """Fig. 20: with ``diff_vocab`` the two token tables are separate; plotted on the
    same PCs they should still trace the same circle."""
    from . import circles as C

    W = model.embed.W_E.T.detach().cpu().double()
    first, second = W[:p], W[p:2 * p]
    mu = first.mean(dim=0)
    _, _, Vt = torch.linalg.svd(first - mu, full_matrices=False)
    pa = ((first - mu) @ Vt.T).numpy()
    pb = ((second - mu) @ Vt.T).numpy()

    fig, ax = plt.subplots(figsize=(6.2, 5.8))
    ax.scatter(pa[:, 0], pa[:, 1], c="crimson", s=26, label="first-token table")
    ax.scatter(pb[:, 0], pb[:, 1], c="tab:blue", s=26, marker="x", label="second-token table")
    for t in range(p):
        ax.annotate(str(t), (pa[t, 0], pa[t, 1]), fontsize=6, xytext=(3, 3),
                    textcoords="offset points")
    ax.set_xlabel("PC 1 (of the first-token table)")
    ax.set_ylabel("PC 2")
    ax.set_title("Fig. 20 analog -- aligned embeddings, `diff_vocab`")
    ax.legend(fontsize=8)
    ax.set_aspect("equal", adjustable="datalim")
    return _save(fig, "fig20_aligned_embeddings.png", outdir)


# ------------------------------------------------ Fig. 22-23: App. J-K training dynamics

def fig22_isolation_over_training(checkpoints: Sequence[tuple[int, Any]], pcs=(0, 1),
                                  outdir=None, tag: str = "clock") -> pathlib.Path:
    """Correct-logit heatmap of one isolated PC pair at a sequence of training steps."""
    from . import circles as C

    data = make_dataset()
    fig, axes = plt.subplots(1, len(checkpoints), figsize=(4.3 * len(checkpoints), 4.4),
                             squeeze=False)
    for ax, (step, model) in zip(axes[0], checkpoints):
        pca = C.embedding_pca(model)
        iso = C.isolated_model(model, list(pcs), pca=pca)
        with torch.no_grad():
            logits = iso.final_logits(data.inputs)
        k = C.dominant_freq(pca.proj[:, pcs[0]])
        im = ax.imshow(C.logit_heatmap(logits, k), cmap="Reds_r", aspect="auto")
        acc = M.accuracy(logits, data.labels) * 100
        ax.set_title(f"step {step}  (k = {k}, {acc:.1f}%)")
        ax.set_xlabel("(a + b) mod p")
        ax.set_ylabel(f"(a - b) / {C.delta_from_k(k)} mod p")
        fig.colorbar(im, ax=ax)
    fig.suptitle(f"Fig. 22-23 analog -- PCs {pcs[0] + 1},{pcs[1] + 1} isolated over training "
                 f"({tag})")
    return _save(fig, f"fig22_isolation_over_training_{tag}.png", outdir)


def fig23_training_curves(histories: dict[str, dict], outdir=None) -> pathlib.Path:
    """Train/validation loss and accuracy against step, from the saved histories."""
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))
    for label, h in histories.items():
        axes[0].plot(h["step"], h["train_loss"], lw=1.2, label=f"{label} train")
        axes[0].plot(h["step"], h["val_loss"], lw=1.2, ls="--", label=f"{label} val")
        axes[1].plot(h["step"], h["train_acc"], lw=1.2, label=f"{label} train")
        axes[1].plot(h["step"], h["val_acc"], lw=1.2, ls="--", label=f"{label} val")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("cross-entropy")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("accuracy")
    for ax in axes:
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25)
    fig.suptitle("Fig. 23 analog -- training dynamics")
    return _save(fig, "fig23_training_curves.png", outdir)
