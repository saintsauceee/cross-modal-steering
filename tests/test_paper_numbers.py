"""The released judge scores and vectors reproduce the paper's numbers (Figure 3, Table 1, Figure 5)."""

import numpy as np
import pytest

from cmsteer.analysis import tables
from cmsteer.analysis.metrics import load_release, normalized_auc
from cmsteer.analysis.report import GEOMETRY_CONCEPTS
from cmsteer.config import load_config

MODELS = ["uniar", "emu35", "liquid"]


def test_normalized_auc():
    assert normalized_auc([-1, 1], [1, 1]) == pytest.approx(1.0)
    assert normalized_auc([-2, -1, 1, 2], [0, 1, 1, 0]) == pytest.approx(0.75)
    assert normalized_auc([10, -10], [0.5, 0.5]) == pytest.approx(0.5)      # order and units do not matter


@pytest.fixture(scope="module")
def scores(release_root):
    return load_release(MODELS, release_root)


def test_figure3(scores):
    t = tables.recovery_by_range(scores).round(2)
    paper = {("uniar", "txt2img"): [0.82, 0.84, 0.80], ("uniar", "img2txt"): [0.96, 0.74, 0.83],
             ("emu35", "txt2img"): [1.15, 1.23, 0.95], ("emu35", "img2txt"): [0.46, 0.59, 0.87],
             ("liquid", "txt2img"): [1.07, 1.29, 1.24], ("liquid", "img2txt"): [0.37, 0.39, 0.91]}
    for k, v in paper.items():
        assert t.loc[k].tolist() == v, k


def test_table1(scores):
    t = tables.recovery_per_concept(scores).round(2)
    paper = {"emotion": (0.95, 0.95), "age": (1.45, 0.60), "cleanness": (0.97, 0.69), "chaos": (1.06, 0.70),
             "size": (1.12, 0.32), "near_far": (0.93, 0.82), "spatial_lr": (0.08, 0.01), "color": (0.97, 0.88),
             "mean": (1.04, 0.68)}
    for c, (t2i, i2t) in paper.items():
        assert (t.loc[c, "txt2img"], t.loc[c, "img2txt"]) == (t2i, i2t), c


def test_figure5_correlations(scores, release_root):
    cfgs = {m: load_config(m) for m in MODELS}
    _, corr = tables.cosine_vs_recovery(scores, {m: release_root / m / "vectors" for m in MODELS},
                                        {m: cfgs[m].layer_ranges for m in MODELS}, GEOMETRY_CONCEPTS)
    c = corr.set_index(["model", "direction"]).round(2)
    paper = {("uniar", "txt2img"): (0.76, 0.58), ("emu35", "txt2img"): (-0.27, -0.19),
             ("liquid", "txt2img"): (-0.12, -0.13), ("uniar", "img2txt"): (0.17, 0.19),
             ("emu35", "img2txt"): (0.47, 0.62), ("liquid", "img2txt"): (0.69, 0.60),
             ("all", "mean of both"): (0.38, 0.22)}
    for k, (rho, r) in paper.items():
        assert (c.loc[k, "spearman"], c.loc[k, "pearson"]) == (rho, r), k
    assert int(c.loc[("all", "mean of both"), "n"]) == 63
    assert np.isclose(c.loc[("uniar", "txt2img"), "p_spearman"], 0, atol=0.05)


def test_results_on_a_partial_run(scores, release_root, tmp_path):
    """A run with one concept, one layer range and no vectors still builds its tables and figures."""
    from cmsteer.analysis.report import make_all
    part = scores[(scores.model == "uniar") & (scores.concept == "emotion") & (scores.config == "mid")]
    (tmp_path / "uniar" / "judge").mkdir(parents=True)
    part.drop(columns="model").to_csv(tmp_path / "uniar" / "judge" / "concept-rate_scores.csv", index=False)
    written = make_all(tmp_path, tmp_path / "out", ["uniar"])
    assert {"table1_recovery_per_concept", "fig_transfer_mid_uniar", "fig_layers_uniar"} <= set(written)
    text_only = part[part.quadrant.isin(["txt2txt", "img2txt"])]
    text_only.drop(columns="model").to_csv(tmp_path / "uniar" / "judge" / "concept-rate_scores.csv", index=False)
    assert "table1_recovery_per_concept" in make_all(tmp_path, tmp_path / "out2", ["uniar"])
    # text-only run that also has vectors: the cosine figure draws only the direction it has
    (tmp_path / "uniar" / "vectors").symlink_to(release_root / "uniar" / "vectors")
    written = make_all(tmp_path, tmp_path / "out3", ["uniar"])
    assert "fig5_cosine_vs_recovery" in written and "fig16a_cosine_by_layer_uniar" in written
