"""The paper's result figures. Every function takes score / point tables and returns a matplotlib Figure."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from cmsteer.analysis.metrics import DIRECTION_LABEL, RANGES, TRANSFER, concept_rate  # noqa: E402
from cmsteer.analysis.tables import MODEL_NAMES, recovery_by_range  # noqa: E402
from cmsteer.concepts import CONCEPTS  # noqa: E402
from cmsteer.config import CONFIG_ALIASES  # noqa: E402

GREEN, RED, GREY = "#009E73", "#D62728", "#444444"
DEPTH = {"early": ("#56B4E9", "#0072B2"), "mid": ("#E69F00", "#D55E00"), "late": ("#5FD0B4", "#009E73")}
CONCEPT_COLORS = {"emotion": "#2a78d6", "age": "#eb6834", "cleanness": "#1baf7a", "chaos": "#eda100",
                  "size": "#e87ba4", "near_far": "#008300", "spatial_lr": "#4a3aa7", "color": "#c0392b"}
plt.rcParams.update({"pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})


def concept_name(c: str) -> str:
    return CONCEPTS[c].name if c in CONCEPTS else c


def _curve(cr: pd.DataFrame, concept: str, quadrant: str, config: str, xnorm: bool = True):
    """(x, rate) of one concept-rate curve with the (0, 0) anchor; x divided by the grid edge when xnorm."""
    g = cr[(cr.concept == concept) & (cr.quadrant == quadrant) & (cr.config == config)].sort_values("alpha")
    x = np.r_[g["alpha"].to_numpy(), 0.0]
    y = np.r_[g["rate"].to_numpy(), 0.0]
    o = np.argsort(x)
    x, y = x[o], y[o]
    if xnorm and len(x) > 1:
        x = np.round(x / np.abs(x).max(), 6)
    return x, y


def transfer_curves(scores: pd.DataFrame, config: str = "mid", concepts=None) -> plt.Figure:
    """Figures 2 and 10-12: intra-modality (black) vs transferred vector (green where above, red where below), one
    model, one layer configuration. Top row: image generation, bottom row: text generation."""
    concepts = concepts or [c for c in CONCEPTS if c in set(scores["concept"])]
    cr = concept_rate(scores)
    rows = [("txt2img", "img steering"), ("img2txt", "txt steering")]
    fig, axes = plt.subplots(2, len(concepts), figsize=(7.6, 2.9), squeeze=False)
    for i, c in enumerate(concepts):
        for r, (tq, label) in enumerate(rows):
            ax = axes[r][i]
            xt, yt = _curve(cr, c, tq, config)
            xi, yi = _curve(cr, c, TRANSFER[tq], config)
            xs = np.union1d(xt, xi)
            mt, mi = np.interp(xs, xt, yt), np.interp(xs, xi, yi)
            ax.fill_between(xs, mt, mi, where=mt >= mi, color=GREEN, alpha=0.45, lw=0, interpolate=True)
            ax.fill_between(xs, mt, mi, where=mt < mi, color=RED, alpha=0.45, lw=0, interpolate=True)
            ax.plot(xs, mi, color="black", lw=1.0, zorder=4)
            seg = np.stack([np.c_[xs[:-1], mt[:-1]], np.c_[xs[1:], mt[1:]]], axis=1)
            gap = (mt[:-1] + mt[1:]) / 2 - (mi[:-1] + mi[1:]) / 2
            colors = np.where(gap > 0.01, GREEN, np.where(gap < -0.01, RED, GREY))
            ax.add_collection(LineCollection(seg, colors=colors, linewidths=1.2, zorder=5))
            ax.set_xlim(-1, 1)
            ax.set_ylim(-0.02, 1.02)
            ax.set_xticks([-1, 0, 1])
            ax.set_yticks([0, 0.5, 1])
            ax.set_box_aspect(1)
            ax.tick_params(labelsize=6)
            ax.grid(True, alpha=0.18, lw=0.4)
            if r == 0:
                ax.set_title(concept_name(c), fontsize=8)
                ax.set_xticklabels([])
            if i == 0:
                ax.set_ylabel(label, fontsize=8)
    fig.supxlabel("steering strength (normalized)", fontsize=8, y=0.02)
    fig.supylabel("concept rate", fontsize=8, x=0.005)
    handles = [Line2D([0], [0], color="black", lw=1.0), Line2D([0], [0], color=GREEN, lw=2.2),
               Line2D([0], [0], color=RED, lw=2.2)]
    fig.legend(handles, ["intra-modality", "transfer above intra-modality", "transfer below intra-modality"],
               loc="upper center", ncol=3, fontsize=7, frameon=False)
    fig.subplots_adjust(left=0.07, right=0.995, top=0.86, bottom=0.12, wspace=0.35, hspace=0.12)
    return fig


def layer_grid(scores: pd.DataFrame, layer_configs, concepts=None) -> plt.Figure:
    """Figures 13-15: concept rate vs normalized strength, every concept (rows) and setting (columns), one line per
    layer configuration. `layer_configs` is the model's full ordered list of config names (single layers, ranges,
    "all"): the single layers are early, middle, late in that order. Configs the scores lack are not drawn."""
    concepts = concepts or [c for c in CONCEPTS if c in set(scores["concept"])]
    cr = concept_rate(scores)
    quads = ["img2img", "txt2img", "img2txt", "txt2txt"]
    singles = [c for c in layer_configs if c not in RANGES and c != "all"]
    style = {"all": ("black", 1.9, "-", "all layers")}
    for d, s in zip(RANGES, singles, strict=False):
        style[s] = (DEPTH[d][0], 0.8, (0, (3.6, 1.6)), f"{dict(early='early', mid='middle', late='late')[d]} layer")
    for d in RANGES:
        style[d] = (DEPTH[d][1], 1.15, "-", f"{dict(early='early', mid='middle', late='late')[d]} range")
    fig, axes = plt.subplots(len(concepts), len(quads), figsize=(8.6, 1.5 * len(concepts) + 0.4), squeeze=False)
    for i, c in enumerate(concepts):
        for j, q in enumerate(quads):
            ax = axes[i][j]
            for cfg in layer_configs:
                x, y = _curve(cr, c, q, cfg)
                if len(x) < 2:
                    continue
                col, lw, ls, lab = style.get(cfg, ("grey", 0.8, "-", cfg))
                ax.plot(x, y, color=col, lw=lw, ls=ls, label=lab, zorder=3 if cfg == "all" else 2)
            ax.set_xlim(-1, 1)
            ax.set_ylim(-0.02, 1.0)
            ax.set_xticks([-1, -0.5, 0, 0.5, 1])
            ax.set_yticks([0, 0.5, 1])
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.2, lw=0.4)
            if i == 0:
                ax.set_title(DIRECTION_LABEL[q], fontsize=10)
            if j == 0:
                ax.set_ylabel(concept_name(c), fontsize=9)
            if i == len(concepts) - 1:
                ax.set_xlabel(r"$\alpha/\alpha_{\max}$", fontsize=9)
    seen = {}                                                # a panel may be empty when the run lacks its setting
    for ax in axes.flat:
        seen.update({lab: h for h, lab in zip(*ax.get_legend_handles_labels(), strict=True) if lab not in seen})
    if seen:
        fig.legend(list(seen.values()), list(seen), loc="upper center", ncol=len(seen), fontsize=8, frameon=False)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.95, bottom=0.045, wspace=0.26, hspace=0.42)
    return fig


def recovery_bars(scores: pd.DataFrame) -> plt.Figure:
    """Figure 3: mean cross-modal recovery over concepts per model and layer range, both directions."""
    t = recovery_by_range(scores)
    models = list(dict.fromkeys(t.index.get_level_values("model")))
    fig, ax = plt.subplots(figsize=(5.2, 2.2))
    w, x0 = 0.38, 0
    ticks, labels = [], []
    for m in models:
        for k, rng in enumerate(t.columns):
            x = x0 + k
            for off, d, col in ((-w / 2, "txt2img", "#A61B1B"), (w / 2, "img2txt", "#1F5FA6")):
                if (m, d) not in t.index or np.isnan(t.loc[(m, d), rng]):
                    continue
                v = t.loc[(m, d), rng]
                ax.bar(x + off, v, w, color=col, zorder=2)
                ax.text(x + off, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=5.5)
            ticks.append(x)
            labels.append(rng)
        ax.text(x0 + (len(t.columns) - 1) / 2, -0.32, MODEL_NAMES.get(m, m), ha="center", fontsize=8,
                transform=ax.get_xaxis_transform())
        x0 += len(t.columns) + 0.8
    ax.axhline(1.0, color="black", ls="--", lw=0.8)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("mean cross-modal recovery", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_ylim(0, max(1.1, float(np.nan_to_num(t.values).max()) + 0.15))
    fig.legend([Patch(color="#A61B1B"), Patch(color="#1F5FA6")], ["txt→img", "img→txt"], fontsize=7, frameon=False,
               ncol=2, loc="upper center")
    fig.subplots_adjust(bottom=0.25, top=0.88)
    return fig


def cosine_scatter(points: pd.DataFrame, corr: pd.DataFrame) -> plt.Figure:
    """Figure 5: cos(v_txt, v_img) over each layer range vs recovery, per model (columns) and direction (rows)."""
    models = list(dict.fromkeys(points["model"]))
    markers = {"early": "^", "mid": "o", "late": "s"}
    directions = [d for d in ("txt2img", "img2txt") if d in set(points["direction"])]
    fig, axes = plt.subplots(len(directions), len(models), figsize=(7.4, 2.2 * len(directions)), squeeze=False)
    for r, d in enumerate(directions):
        for j, m in enumerate(models):
            ax = axes[r][j]
            g = points[(points.model == m) & (points.direction == d)]
            for _, p in g.iterrows():
                ax.scatter(p["cos"], p["recovery"], color=CONCEPT_COLORS.get(p["concept"], "grey"),
                           marker=markers[p["config"]], s=26, edgecolor="white", linewidth=0.5, zorder=3)
            c = corr[(corr.model == m) & (corr.direction == d)].iloc[0]
            star = lambda p: "*" if p < 0.05 else ""
            ax.set_title(f"{MODEL_NAMES.get(m, m)}  ρ={c.spearman:.2f}{star(c.p_spearman)}, r={c.pearson:.2f}",
                         fontsize=7.5, loc="left")
            ax.axhline(1.0, color="#cccccc", lw=0.8, zorder=1)
            ax.tick_params(labelsize=6.5)
            ax.grid(True, alpha=0.2, lw=0.4)
            if j == 0:
                ax.set_ylabel(f"recovery ({DIRECTION_LABEL[d]})", fontsize=8)
    fig.supxlabel(r"cos($\hat v_{txt}$, $\hat v_{img}$), mean over the layer range", fontsize=8)
    h = [Line2D([], [], marker="o", ls="", color=CONCEPT_COLORS[c], label=concept_name(c))
         for c in dict.fromkeys(points["concept"])]
    h += [Line2D([], [], marker=mk, ls="", color=GREY, label=w) for w, mk in markers.items()]
    fig.legend(handles=h, ncol=len(h), fontsize=6.5, frameon=False, loc="upper center", handletextpad=0.2,
               columnspacing=0.8)
    fig.subplots_adjust(top=0.88, hspace=0.45, wspace=0.25, bottom=0.11)
    return fig


def cosine_by_layer(cosines: dict, dim: int) -> plt.Figure:
    """Figure 16a: per-layer cos(v_txt, v_img) per concept, with the +-2 sigma band of random directions."""
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    n = len(next(iter(cosines.values())))
    band = 2 / np.sqrt(dim)
    ax.axhspan(-band, band, color="#dddddd", zorder=0, label="random (±2σ)")
    for c, v in cosines.items():
        ax.plot(range(n), v, color=CONCEPT_COLORS.get(c, "grey"), lw=1.2, label=concept_name(c))
    ax.set_xlabel("layer", fontsize=8)
    ax.set_ylabel(r"cos($\hat v_{txt}$, $\hat v_{img}$)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, frameon=False, ncol=3)
    return fig


def image_strips(run, concept: str, prompt_idx: int, config: str = "mid", quadrants=("img2img", "txt2img"),
                 steps=(-3, -2, -1, 0, 1, 2, 3)) -> plt.Figure:
    """Figure 4 (a): one prompt's images across strengths, one row per setting. `steps` index the strength grid
    from the baseline (0), e.g. -1 = the smallest negative strength. Needs the run's generations/images."""
    from pathlib import Path

    from PIL import Image

    gen = Path(run) / "generations"
    rows = pd.read_csv(gen / "sweep_image.csv")
    rows["config"] = rows["config"].replace(CONFIG_ALIASES)
    rows = rows[(rows.concept == concept) & (rows.prompt_idx == prompt_idx) & (rows.config == config)]
    fig, axes = plt.subplots(len(quadrants), len(steps), figsize=(1.2 * len(steps), 1.3 * len(quadrants)),
                             squeeze=False)
    for i, q in enumerate(quadrants):
        g = rows[rows.quadrant == q].sort_values("alpha")
        if g.empty:
            raise FileNotFoundError(f"no {q} images for {concept}, prompt {prompt_idx}, {config} in {gen}")
        alphas = list(g["alpha"])
        zero = alphas.index(0)
        top = max(abs(a) for a in alphas)
        for j, s in enumerate(steps):
            ax = axes[i][j]
            ax.axis("off")
            if 0 <= zero + s < len(alphas):
                r = g.iloc[zero + s]
                ax.imshow(Image.open(gen / r["image_path"]))
                if i == 0:
                    ax.set_title("baseline" if s == 0 else f"{r['alpha'] / top:+.1f}", fontsize=7)
        axes[i][0].text(-0.08, 0.5, DIRECTION_LABEL[q], transform=axes[i][0].transAxes, rotation=90, ha="right",
                        va="center", fontsize=8)
    fig.subplots_adjust(wspace=0.04, hspace=0.12)
    return fig
