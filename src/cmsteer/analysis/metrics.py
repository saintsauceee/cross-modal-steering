"""Concept rate, AUC and cross-modal recovery (paper §3.1).

Input is a judge-score table with one row per steered output:
    concept, quadrant, config, alpha, prompt_idx, correct      (correct = 1 when the judge saw the steering)

- concept rate CR(c, alpha): mean of `correct` over the prompts of one (concept, quadrant, config, alpha).
- AUC: trapezoid of the concept-rate curve over the sampled non-zero strengths, divided by the strength range, so it
  lies in [0, 1] and does not depend on the units of alpha (each setting has its own grid).
- recovery: AUC of the transferred vector / AUC of the intra-modality vector on the same target,
      Rec_img = AUC(txt2img) / AUC(img2img),   Rec_txt = AUC(img2txt) / AUC(txt2txt).
  It is left undefined (NaN) where intra-modality steering barely works (intra AUC < MIN_INTRA_AUC).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cmsteer.config import CONFIG_ALIASES

QUADRANTS = ("txt2txt", "img2txt", "img2img", "txt2img")
# transfer setting -> the intra-modality setting with the same target modality
TRANSFER = {"txt2img": "img2img", "img2txt": "txt2txt"}
DIRECTION_LABEL = {"txt2img": "txt→img", "img2txt": "img→txt", "img2img": "img→img", "txt2txt": "txt→txt"}
RANGES = ("early", "mid", "late")
MIN_INTRA_AUC = 0.2


def load_scores(path: str | Path) -> pd.DataFrame:
    """Read a concept-rate score table (a CSV, or a release folder holding judge/concept-rate_scores.csv)."""
    path = Path(path)
    if path.is_dir():
        path = path / "judge" / "concept-rate_scores.csv"
    df = pd.read_csv(path)
    df["config"] = df["config"].replace(CONFIG_ALIASES)
    df["alpha"] = df["alpha"].astype(float)
    return df[df["alpha"] != 0]


def concept_rate(scores: pd.DataFrame) -> pd.DataFrame:
    """One row per (concept, quadrant, config, alpha): rate and number of judged prompts."""
    g = scores.dropna(subset=["correct"]).groupby(["concept", "quadrant", "config", "alpha"])["correct"]
    return g.agg(rate="mean", n="size").reset_index()


def normalized_auc(alpha, rate) -> float:
    """Trapezoid area under rate(alpha) over the sampled strengths, divided by the strength range."""
    x = np.asarray(alpha, dtype=float)
    y = np.asarray(rate, dtype=float)
    o = np.argsort(x)
    x, y = x[o], y[o]
    return float(np.trapezoid(y, x) / (x[-1] - x[0])) if len(x) > 1 else float("nan")


def auc_table(scores: pd.DataFrame) -> pd.DataFrame:
    """One row per (concept, quadrant, config) with its normalized AUC."""
    cr = concept_rate(scores)
    rows = [(*k, normalized_auc(g["alpha"], g["rate"])) for k, g in cr.groupby(["concept", "quadrant", "config"])]
    return pd.DataFrame(rows, columns=["concept", "quadrant", "config", "auc"])


def recovery_table(scores: pd.DataFrame, min_intra_auc: float | None = MIN_INTRA_AUC) -> pd.DataFrame:
    """One row per (concept, config, transfer quadrant): transfer AUC, intra AUC and recovery.

    Recovery is NaN when the intra AUC is below `min_intra_auc` (the paper's Table 1, Table 6 and Figure 3), or only
    when the intra AUC is 0 if `min_intra_auc` is None (Figure 5 and its correlations use every point)."""
    a = auc_table(scores).set_index(["concept", "quadrant", "config"])["auc"]
    floor = 0.0 if min_intra_auc is None else min_intra_auc
    rows = []
    for (concept, quadrant, config), transfer in a.items():
        if quadrant not in TRANSFER:
            continue
        intra = a.get((concept, TRANSFER[quadrant], config), np.nan)
        rec = transfer / intra if (intra > 0 and intra >= floor) else np.nan
        rows.append((concept, config, quadrant, transfer, intra, rec))
    return pd.DataFrame(rows, columns=["concept", "config", "direction", "auc_transfer", "auc_intra", "recovery"])


def load_release(models, root) -> pd.DataFrame:
    """Concatenate the score tables of several models (`root/<model>/judge/concept-rate_scores.csv`) with a `model`
    column."""
    frames = []
    for m in models:
        df = load_scores(Path(root) / m)
        df.insert(0, "model", m)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def recovery_by_model(scores: pd.DataFrame, **kw) -> pd.DataFrame:
    """recovery_table for every model in a multi-model score table."""
    return pd.concat([recovery_table(g, **kw).assign(model=m) for m, g in scores.groupby("model", sort=False)],
                     ignore_index=True)


def load_quality(run: str | Path, kind: str) -> pd.DataFrame | None:
    """Quality scores of a run, as concept, quadrant, config, alpha, prompt_idx, score (baselines: quadrant
    "baseline"). kind "fluency" reads scores/fluency_scores.csv, or the release's scores/axbench_scores.csv (its F
    column); kind "coherence" reads scores/coherence_scores.csv. None when the run has no such scores."""
    run = Path(run)
    p = run / "scores" / f"{kind}_scores.csv"
    if p.exists():
        df = pd.read_csv(p)
    elif kind == "fluency" and (run / "scores" / "axbench_scores.csv").exists():
        df = pd.read_csv(run / "scores" / "axbench_scores.csv").rename(columns={"F": "score"})
    else:
        return None
    df["config"] = df["config"].replace(CONFIG_ALIASES)
    return df[["concept", "quadrant", "config", "alpha", "prompt_idx", "score"]].dropna(subset=["score"])


def load_dreamsim(run: str | Path) -> pd.DataFrame | None:
    p = Path(run) / "scores" / "dreamsim_scores.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df["config"] = df["config"].replace(CONFIG_ALIASES)
    return df


def strength(df: pd.DataFrame) -> pd.Series:
    """|alpha| divided by the largest |alpha| of each setting, rounded to one decimal (the normalized strength)."""
    top = df.groupby("quadrant")["alpha"].transform(lambda a: a.abs().max())
    return (df["alpha"].abs() / top).round(1)


def quality_vs_strength(scores: pd.DataFrame, quality: pd.DataFrame, config: str = "mid") -> pd.DataFrame:
    """Per setting and normalized strength: concept rate and quality relative to the unsteered outputs, for one
    model at one layer configuration (paper Figure 8). The row at strength 0 is (0, 1) by definition."""
    base = quality[quality.quadrant == "baseline"]["score"].mean()
    q = quality[(quality.quadrant != "baseline") & (quality.config == config)].copy()
    s = scores[scores.config == config].copy()
    q["r"], s["r"] = strength(q), strength(s)
    rows = []
    for quad in sorted(set(q.quadrant) & set(s.quadrant)):
        cr = s[s.quadrant == quad].groupby("r")["correct"].mean()
        ql = q[q.quadrant == quad].groupby("r")["score"].mean() / base
        rows.append((quad, 0.0, 0.0, 1.0))
        rows += [(quad, r, cr[r], ql[r]) for r in cr.index if r in ql.index]
    return pd.DataFrame(rows, columns=["quadrant", "strength", "concept_rate", "quality"])
