"""The paper's result tables, computed from judge scores (and vectors, for the geometry table)."""

from __future__ import annotations

import pandas as pd

from cmsteer.analysis.geometry import correlate, cosine_table
from cmsteer.analysis.metrics import RANGES, recovery_by_model

MODEL_NAMES = {"uniar": "UniAR", "emu35": "Emu3.5", "liquid": "Liquid"}


def recovery_by_range(scores: pd.DataFrame) -> pd.DataFrame:
    """Figure 3: mean recovery over concepts, per model, layer range and direction (rows: model, direction)."""
    r = recovery_by_model(scores)
    r = r[r["config"].isin(RANGES)]
    t = r.groupby(["model", "direction", "config"], sort=False)["recovery"].mean().unstack("config")
    return t.reindex(columns=[r for r in RANGES if r in t.columns])


def recovery_per_concept(scores: pd.DataFrame, concepts=None) -> pd.DataFrame:
    """Table 1: recovery per concept, averaged over models and layer ranges. The `mean` row averages the
    (model, layer range) cells of Figure 3, so every model and range weighs the same."""
    r = recovery_by_model(scores)
    r = r[r["config"].isin(RANGES)]
    t = r.groupby(["concept", "direction"])["recovery"].mean().unstack("direction")
    t = t.reindex(columns=["txt2img", "img2txt"])                  # a run may hold only one transfer setting
    if concepts is not None:
        t = t.loc[[c for c in concepts if c in t.index]]
    cells = recovery_by_range(scores).stack().groupby(level="direction").mean()
    t.loc["mean"] = [cells.get("txt2img", float("nan")), cells.get("img2txt", float("nan"))]
    return t


def recovery_full(scores: pd.DataFrame, concepts=None) -> pd.DataFrame:
    """Table 6: recovery for every model, layer range, direction and concept (NaN where the intra-modality AUC is
    below 0.2), plus the mean over concepts."""
    r = recovery_by_model(scores)
    r = r[r["config"].isin(RANGES)]
    t = r.pivot_table(index=["model", "config", "direction"], columns="concept", values="recovery", dropna=False)
    if concepts is not None:
        t = t[[c for c in concepts if c in t.columns]]
    t["mean"] = t.mean(axis=1)
    order = [(m, w, d) for m in scores["model"].unique() for w in RANGES for d in ("txt2img", "img2txt")]
    return t.reindex([o for o in order if o in t.index])


def cosine_vs_recovery(scores: pd.DataFrame, vectors_dirs: dict, layer_ranges: dict, concepts) -> tuple:
    """Figure 5 / Table 7: per (model, concept, layer range) the text-image vector cosine next to the recovery of
    each direction (every point, no low-AUC exclusion). Returns (points, correlations)."""
    rec = recovery_by_model(scores, min_intra_auc=None)
    rec = rec[rec["config"].isin(RANGES) & rec["concept"].isin(concepts)]
    cos = pd.concat([cosine_table(vectors_dirs[m], concepts, layer_ranges[m]).assign(model=m) for m in vectors_dirs])
    pts = rec.merge(cos, on=["model", "concept", "config"])
    corr = []
    for (m, d), g in pts.groupby(["model", "direction"], sort=False):
        g = g.dropna(subset=["recovery"])
        corr.append({"model": m, "direction": d, **correlate(g["cos"], g["recovery"])})
    both = pts.pivot_table(index=["model", "concept", "config"], columns="direction", values="recovery")
    both = both.mean(axis=1).rename("recovery").reset_index().merge(cos, on=["model", "concept", "config"]).dropna()
    corr.append({"model": "all", "direction": "mean of both", **correlate(both["cos"], both["recovery"])})
    return pts, pd.DataFrame(corr)
