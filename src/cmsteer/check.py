"""Check a model setup against the release: regenerate a sample of released rows and compare.

    cmsteer check --model uniar

- text: greedy outputs at a few strengths of the middle layer range. In bf16, greedy text depends on how a batch is
  padded (on UniAR even unsteered rows change when the other rows of their batch change), so outputs agree with the
  release for a long prefix and then may drift. The check reports character similarity next to exact matches;
  the comparison that matters is the judged concept rate;
- images: three seeded batches (two strengths of img2img, one of txt2img), made exactly as the paper's run made them
  and steered with the released vectors (pixel match for UniAR, near-identical for Liquid, on the A100 GPUs the
  paper's runs used; Emu3.5's released images came from a different engine);
- vectors: the text vector of one concept re-extracted from the minimal pairs (cosine with the released one).
"""

from __future__ import annotations

import csv
import difflib
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from cmsteer import hub
from cmsteer.config import CONFIG_ALIASES, load_config
from cmsteer.steering import Steer
from cmsteer.sweep import fmt_alpha, image_prompts, source, text_prompts
from cmsteer.vectors import load_vector, unit


def _rows(path):
    csv.field_size_limit(sys.maxsize)
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def check(model_name: str, model=None, concept: str = "emotion", n_text: int = 8, images: bool = True,
          vectors: bool = True, log=print) -> dict:
    from cmsteer.models import load_model

    cfg = load_config(model_name)
    root = hub.fetch(model_name, ("vectors", "generations"))
    vec = {m: load_vector(root / "vectors" / concept / f"{m}.pt") for m in ("txt", "img")}
    if model is None:
        model = load_model(cfg.model, **cfg.model_args)
    layers = cfg.layers("mid", model.n_layers)
    report = {}

    # ---- text ----
    alias = lambda c: CONFIG_ALIASES.get(c, c)                     # Emu3.5's release calls its ranges early8/mid8/late8
    rel = {(r["quadrant"], alias(r["config"]), fmt_alpha(float(r["alpha"])), int(r["prompt_idx"])): r["output"]
           for r in _rows(root / "generations" / "sweep_text.csv") if r["concept"] == concept}
    prompts = text_prompts(concept)
    jobs = []
    for q in ("txt2txt", "img2txt"):
        grid = cfg.alpha_grid(q)
        z = grid.index(0)
        for a in (0.0, grid[z + 1], grid[z - 1], grid[z + 2]):   # baseline, +1 step, -1 step, +2 steps
            jobs += [(q, a, pi) for pi in range(n_text)]
    outs = model.generate_text([prompts[pi][0] for _, _, pi in jobs], [prompts[pi][1] for _, _, pi in jobs],
                               [None if a == 0 else Steer(vec[source(q)], layers, a) for q, a, _ in jobs],
                               max_new_tokens=cfg.text.get("max_new_tokens", 96))
    same, sims = 0, []
    for (q, a, pi), o in zip(jobs, outs, strict=True):
        ref = rel[(q, "mid", fmt_alpha(a), pi)]
        same += o == ref
        sims.append(difflib.SequenceMatcher(None, o, ref).ratio())
    report["text_exact"] = (same, len(jobs))
    report["text_similarity"] = float(np.mean(sims))
    log(f"[text] exact {same}/{len(jobs)}, mean character similarity {np.mean(sims):.3f}")

    # ---- images: three seeded batches, each made exactly as the paper's run made it (rows of a (concept, setting,
    # mid) unit sorted by (alpha, prompt), image_batch rows per call): the first and a middle batch of img2img and a
    # middle batch of txt2img, so several strengths and both vector sources are covered ----
    if images and hasattr(model, "generate_images"):
        from huggingface_hub import hf_hub_download

        descs = image_prompts(concept)
        rel_rows = [r for r in _rows(root / "generations" / "sweep_image.csv")
                    if r["concept"] == concept and alias(r["config"]) == "mid"]
        diffs = []
        for q, which in (("img2img", "first"), ("img2img", "middle"), ("txt2img", "middle")):
            items = [(a, pi) for a in cfg.alpha_grid(q) if a != 0 for pi in range(len(descs))]
            batches = [items[k:k + cfg.image_batch] for k in range(0, len(items), cfg.image_batch)]
            batch = batches[0] if which == "first" else batches[len(batches) // 2]
            gen = model.generate_images([descs[pi] for _, pi in batch],
                                        [Steer(vec[source(q)], layers, a) for a, _ in batch], seed=cfg.seed)
            rel_img = {(fmt_alpha(float(r["alpha"])), int(r["prompt_idx"])): r["image_path"]
                       for r in rel_rows if r["quadrant"] == q}
            d = []
            for (a, pi), im in zip(batch, gen, strict=True):
                p = hf_hub_download(hub.repo_id(model_name), "generations/" + rel_img[(fmt_alpha(a), pi)],
                                    repo_type="dataset", local_dir=root)
                ref = np.asarray(Image.open(p).convert("RGB"), dtype=np.int16)
                out = np.asarray(im.convert("RGB"), dtype=np.int16)
                d.append(np.abs(out - ref).mean() if out.shape == ref.shape else np.inf)
            log(f"[image] {q}, {which} batch (alpha {batch[0][0]:g} to {batch[-1][0]:g}): pixel-identical "
                f"{sum(x == 0 for x in d)}/{len(d)}, mean |difference| {np.mean(d):.3f} / 255")
            diffs += d
        report["image_identical"] = (int(sum(d == 0 for d in diffs)), len(diffs))
        report["image_within_1_of_255"] = (int(sum(d < 1 for d in diffs)), len(diffs))
        report["image_mean_abs_diff"] = float(np.mean(diffs))
        log(f"[image] all batches: pixel-identical {report['image_identical'][0]}/{len(diffs)}, mean |difference| "
            f"below 1/255 {report['image_within_1_of_255'][0]}/{len(diffs)}")

    # ---- vectors ----
    if vectors:
        from cmsteer.extract import text_vector

        v, _, _ = text_vector(model, concept)
        cos = (unit(v) * unit(vec["txt"])).sum(-1)
        report["text_vector_cosine"] = (float(cos.min()), float(cos.mean()))
        log(f"[vector] {concept}/txt cosine with the released vector: min {cos.min():.4f}, mean {cos.mean():.4f}")
    return report


def save_report(report: dict, path: str | Path) -> None:
    import json
    Path(path).write_text(json.dumps(report, indent=1))
