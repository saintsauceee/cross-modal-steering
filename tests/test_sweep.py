"""The sweep engine with a fake model: rows, baselines, batching, resume, shards."""

import csv

import torch
from PIL import Image

from cmsteer.config import load_config
from cmsteer.sweep import merge_shards, plan, run_sweep


class FakeModel:
    name = "fake"
    n_layers = 36

    def __init__(self):
        self.text_calls, self.image_calls = [], []

    def generate_text(self, prompts, spans, steer, max_new_tokens=96):
        self.text_calls.append(len(prompts))
        return [f"{s}|{'-' if r is None else r.alpha}|{'' if r is None else r.layers}"
                for s, r in zip(spans, steer, strict=True)]

    def generate_images(self, prompts, steer, seed=42):
        self.image_calls.append([None if r is None else r.alpha for r in steer])
        return [Image.new("RGB", (4, 4)) for _ in prompts]


def small_config():
    cfg = load_config("uniar", concepts=["emotion", "age"], n_prompts=3)
    cfg.layer_configs = {"L18": [18], "mid": [16, 17, 18, 19]}
    return cfg


def vectors(cfg):
    return {(c, s): torch.randn(36, 8) for c in cfg.concepts for s in ("txt", "img")}


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_full_run(tmp_path):
    cfg = small_config()
    m = FakeModel()
    run_sweep(cfg, m, vectors(cfg), tmp_path, text_batch=16, image_batch=cfg.image_batch, log=lambda *_: None)
    text = read(tmp_path / "generations/sweep_text.csv")
    image = read(tmp_path / "generations/sweep_image.csv")
    assert len(text) == len(plan(cfg, "text")) == 2 * 2 * 2 * 11 * 3
    assert len(image) == len(plan(cfg, "image"))
    # alpha 0 rows: one unsteered generation per (concept, prompt), shared by every setting and layer config
    base = [r for r in text if float(r["alpha"]) == 0]
    assert all(r["output"].split("|")[1] == "-" for r in base)
    assert len({(r["concept"], r["prompt_idx"], r["output"]) for r in base}) == 2 * 3
    # steered rows carry their own strength and layers
    r = next(r for r in text if r["config"] == "mid" and float(r["alpha"]) == 40)
    assert r["output"].endswith("|40|[16, 17, 18, 19]")
    # image calls: baselines first, then each unit's rows sorted by alpha, image_batch per call
    assert m.image_calls[0] == [None] * 6
    assert m.image_calls[1][:3] == [-10.0] * 3 and len(m.image_calls[1]) == cfg.image_batch
    imgs = {r["image_path"] for r in image}
    assert all((tmp_path / "generations" / p).exists() for p in imgs)


def test_resume_does_nothing(tmp_path):
    cfg = small_config()
    run_sweep(cfg, FakeModel(), vectors(cfg), tmp_path, log=lambda *_: None)
    m = FakeModel()
    run_sweep(cfg, m, vectors(cfg), tmp_path, log=lambda *_: None)
    assert m.text_calls == [] and m.image_calls == []
    assert len(read(tmp_path / "generations/sweep_text.csv")) == len(plan(cfg, "text"))


def test_shards_merge_to_the_full_run(tmp_path):
    cfg = small_config()
    vec = vectors(cfg)
    for i in range(3):
        run_sweep(cfg, FakeModel(), vec, tmp_path, shard=i, nshard=3, log=lambda *_: None)
    merge_shards(tmp_path)
    for medium in ("text", "image"):
        rows = read(tmp_path / f"generations/sweep_{medium}.csv")
        keys = {(r["concept"], r["quadrant"], r["config"], r["alpha"], r["prompt_idx"]) for r in rows}
        assert len(rows) == len(keys) == len(plan(cfg, medium))
    assert not list((tmp_path / "generations").glob("*.part*"))


def test_grid_without_baseline_is_rejected():
    import pytest
    cfg = small_config()
    cfg.alphas["txt2txt"] = {"values": [-1, 1]}
    with pytest.raises(ValueError):
        cfg.alpha_grid("txt2txt")
