"""Build every main-paper table and figure from judge scores and vectors.

    cmsteer results --out results/                 # from the released scores and vectors (downloads ~120 MB)
    cmsteer results --root runs/ --out results/    # from your own runs: runs/<model>/judge/concept-rate_scores.csv
                                                   # and runs/<model>/vectors/<concept>/{txt,img}.pt
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from cmsteer.analysis import figures, tables
from cmsteer.analysis.geometry import layer_cosines
from cmsteer.analysis.metrics import load_dreamsim, load_quality, load_release, quality_vs_strength
from cmsteer.config import load_config
from cmsteer.vectors import load_vector

GEOMETRY_CONCEPTS = ["emotion", "age", "cleanness", "chaos", "size", "near_far", "color"]   # spatial excluded (§3.4)


def make_all(root: str | Path, out: str | Path, models=("uniar", "emu35", "liquid"), pairs_dir=None) -> dict:
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    scores = load_release(models, root)
    cfgs = {m: load_config(m) for m in models}
    concepts = list(dict.fromkeys(c for m in models for c in cfgs[m].concepts))
    written = {}

    def save_table(name, df):
        df.to_csv(out / f"{name}.csv")
        (out / f"{name}.md").write_text(df.to_markdown(floatfmt=".2f"))
        written[name] = out / f"{name}.csv"

    def save_fig(name, fig):
        fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
        fig.savefig(out / f"{name}.png", dpi=160, bbox_inches="tight")
        written[name] = out / f"{name}.pdf"
        figures.plt.close(fig)

    def config_names(m):                                   # the layer configs this run actually has
        have = set(scores[scores.model == m]["config"])
        return [c for c in cfgs[m].layer_configs if c in have]

    has_recovery = not tables.recovery_by_range(scores).empty      # needs a transfer and its intra-modality setting
    if has_recovery:
        save_table("table1_recovery_per_concept", tables.recovery_per_concept(scores, concepts))
        save_table("table6_recovery_full", tables.recovery_full(scores, concepts))
        save_table("fig3_recovery_by_range", tables.recovery_by_range(scores))
        save_fig("fig3_recovery_by_range", figures.recovery_bars(scores))

    for m in models:
        s = scores[scores.model == m]
        mine = [c for c in concepts if c in set(s["concept"])]
        if "mid" in set(s["config"]):
            save_fig(f"fig_transfer_mid_{m}", figures.transfer_curves(s, "mid", mine))       # Fig 2 / 10-12
        save_fig(f"fig_layers_{m}", figures.layer_grid(s, list(cfgs[m].layer_configs), mine))   # Figs 13-15
        try:                                     # Figure 4 (a), only when the images were downloaded
            save_fig(f"fig4_images_emotion_{m}", figures.image_strips(root / m, "emotion", 0))
        except FileNotFoundError:
            pass

    # Figure 4 (b): text samples; Figure 6: minimal pairs; Figure 7: DreamSim; Figure 8: quality vs concept rate
    for m in models:
        if (root / m / "generations" / "sweep_text.csv").exists():
            t = tables.text_samples(root / m, "emotion")
            if not t.empty:
                save_table(f"fig4b_text_samples_emotion_{m}", t.set_index(["setting", "alpha"]))
    if pairs_dir is not None and Path(pairs_dir).is_dir():
        try:
            save_fig("fig6_minimal_pairs", figures.minimal_pairs(pairs_dir, concepts))
        except FileNotFoundError:
            pass
    ds = {m: d for m in models if (d := load_dreamsim(root / m)) is not None}
    if ds:
        save_fig("fig7_dreamsim", figures.dreamsim_curves(ds))
    parts = []
    for m in models:
        s = scores[scores.model == m]
        for kind in ("fluency", "coherence"):
            q = load_quality(root / m, kind)
            if q is not None and "mid" in set(s["config"]):
                want = ("txt2txt", "img2txt") if kind == "fluency" else ("img2img", "txt2img")
                t = quality_vs_strength(s[s.quadrant.isin(want)], q, "mid")
                parts.append(t.assign(model=m))
    if parts:
        avg = pd.concat(parts).groupby(["quadrant", "strength"])[["concept_rate", "quality"]].mean().reset_index()
        save_table("fig8_quality_vs_concept_rate", avg.set_index(["quadrant", "strength"]))
        save_fig("fig8_quality_vs_concept_rate", figures.quality_curves(avg))

    def with_vectors(m, cs):                               # concepts whose two vectors are both there
        return [c for c in cs if all((root / m / "vectors" / c / f"{k}.pt").exists() for k in ("txt", "img"))]

    geo = {m: with_vectors(m, [c for c in GEOMETRY_CONCEPTS if c in concepts]) for m in models}
    common = [c for c in GEOMETRY_CONCEPTS if all(c in geo[m] for m in models)]
    if has_recovery and common:
        vec_dirs = {m: root / m / "vectors" for m in models}
        ranges = {m: {k: v for k, v in cfgs[m].layer_ranges.items() if k in config_names(m)} for m in models}
        points, corr = tables.cosine_vs_recovery(scores, vec_dirs, ranges, common)
        save_table("table7_cosine_correlation", corr.set_index(["model", "direction"]))
        points.to_csv(out / "fig5_points.csv", index=False)
        save_fig("fig5_cosine_vs_recovery", figures.cosine_scatter(points, corr))
    for m in models:
        have = with_vectors(m, concepts)
        if have:
            cos = {c: layer_cosines(load_vector(root / m / "vectors" / c / "txt.pt").numpy(),
                                    load_vector(root / m / "vectors" / c / "img.pt").numpy()) for c in have}
            dim = load_vector(root / m / "vectors" / have[0] / "txt.pt").shape[1]
            save_fig(f"fig16a_cosine_by_layer_{m}", figures.cosine_by_layer(cos, dim))
    (out / "index.json").write_text(json.dumps({k: str(v) for k, v in written.items()}, indent=1))
    return written

