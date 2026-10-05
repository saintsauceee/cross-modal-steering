"""Geometry of steering vectors (paper §3.4): text-vs-image cosine per layer, and its relation to recovery."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from cmsteer.vectors import load_vector


def layer_cosines(v_txt, v_img) -> np.ndarray:
    """Per-layer cosine between two [n_layers, d] vectors."""
    a = v_txt / np.linalg.norm(v_txt, axis=-1, keepdims=True)
    b = v_img / np.linalg.norm(v_img, axis=-1, keepdims=True)
    return (a * b).sum(-1)


def cosine_table(vectors_dir: str | Path, concepts, layer_ranges: dict[str, list[int]]) -> pd.DataFrame:
    """Mean per-layer cos(v_txt, v_img) over each layer range, for vectors stored as <dir>/<concept>/{txt,img}.pt."""
    rows = []
    for c in concepts:
        cos = layer_cosines(load_vector(Path(vectors_dir) / c / "txt.pt").numpy(),
                            load_vector(Path(vectors_dir) / c / "img.pt").numpy())
        for name, layers in layer_ranges.items():
            rows.append((c, name, float(cos[list(layers)].mean())))
    return pd.DataFrame(rows, columns=["concept", "config", "cos"])


def random_cosine_sd(dim: int) -> float:
    """Standard deviation of the cosine between two random directions in `dim` dimensions."""
    return 1.0 / np.sqrt(dim)


def correlate(x, y) -> dict:
    nan = float("nan")
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:             # too few points for a correlation
        return {"spearman": nan, "p_spearman": nan, "pearson": nan, "p_pearson": nan, "n": len(x)}
    rho, p_rho = spearmanr(x, y)
    r, p_r = pearsonr(x, y)
    return {"spearman": float(rho), "p_spearman": float(p_rho), "pearson": float(r), "p_pearson": float(p_r),
            "n": len(x)}
