"""The steering sweep: every (concept, setting, layer configuration, strength, prompt) of a run config, generated
once, written to the same tables as the release:

    <out>/generations/sweep_text.csv     concept, quadrant, config, alpha, prompt_idx, prompt, output
    <out>/generations/sweep_image.csv    concept, quadrant, config, alpha, prompt_idx, prompt, image_path
    <out>/generations/images/<concept>/*.png

A setting ("quadrant") is <vector source>2<output>: txt2txt, img2txt (text output) and img2img, txt2img (image output).
alpha = 0 is the unsteered baseline; it is generated once per (concept, prompt) and shared by every setting and
layer configuration.

Batching follows the paper's runs, so seeded image generation reproduces their batches: steered images are made per
(concept, setting, layer config) unit, rows sorted by (alpha, prompt), `image_batch` rows per call; baselines are made
over the sorted (concept, prompt) list in calls of `image_batch`. Every call seeds the RNG with `seed`.

Runs resume: rows already in the CSV are skipped. `shard` / `nshard` split the work round-robin for several GPUs; each
shard writes sweep_<medium>.part<i>of<n>.csv and `merge_shards()` joins them.
"""

from __future__ import annotations

import csv
import glob
import time
from dataclasses import dataclass
from pathlib import Path

from cmsteer.concepts import get as get_concept
from cmsteer.concepts import text_instruction
from cmsteer.config import RunConfig
from cmsteer.steering import Steer

TEXT_COLUMNS = ["concept", "quadrant", "config", "alpha", "prompt_idx", "prompt", "output"]
IMAGE_COLUMNS = ["concept", "quadrant", "config", "alpha", "prompt_idx", "prompt", "image_path"]


def source(quadrant: str) -> str:
    return quadrant.split("2")[0]


def target(quadrant: str) -> str:
    return quadrant.split("2")[1]


def fmt_alpha(a: float) -> str:
    return f"{float(a):g}"


@dataclass(frozen=True)
class Row:
    concept: str
    quadrant: str
    config: str
    alpha: float
    prompt_idx: int

    @property
    def key(self):
        return (self.concept, self.quadrant, self.config, fmt_alpha(self.alpha), str(self.prompt_idx))


def text_prompts(concept: str, n: int | None = None) -> list[tuple[str, str]]:
    """(user message, subject span) per evaluation prompt."""
    tmpl = text_instruction()
    subj = get_concept(concept).text_subjects()[:n]
    return [(tmpl.format(subject=s), s) for s in subj]


def image_prompts(concept: str, n: int | None = None) -> list[str]:
    return get_concept(concept).image_descriptions()[:n]


def plan(cfg: RunConfig, medium: str) -> list[Row]:
    """Every row of the run for one medium ("text" or "image"), baselines included."""
    rows = []
    for c in cfg.concepts:
        n = len(text_prompts(c, cfg.n_prompts) if medium == "text" else image_prompts(c, cfg.n_prompts))
        for q in cfg.quadrants:
            if (target(q) == "txt") != (medium == "text"):
                continue
            for cn in cfg.layer_configs:
                for a in cfg.alpha_grid(q):
                    rows.extend(Row(c, q, cn, a, pi) for pi in range(n))
    return rows


def _done(paths) -> set:
    done = set()
    for p in paths:
        with open(p, newline="") as f:
            done |= {(r["concept"], r["quadrant"], r["config"], fmt_alpha(float(r["alpha"])), r["prompt_idx"])
                     for r in csv.DictReader(f)}
    return done


class _Writer:
    def __init__(self, path: Path, columns):
        new = not path.exists() or path.stat().st_size == 0
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "a", newline="")
        self.w = csv.writer(self.f)
        if new:
            self.w.writerow(columns)

    def write(self, rows):
        self.w.writerows(rows)
        self.f.flush()

    def close(self):
        self.f.close()


def _part(nshard, shard):
    return "" if nshard == 1 else f".part{shard}of{nshard}"


def run_sweep(cfg: RunConfig, model, vectors: dict, out: str | Path, *, media=("text", "image"), shard: int = 0,
              nshard: int = 1, text_batch: int = 64, image_batch: int = 20, log=print) -> None:
    """vectors: {(concept, "txt" | "img"): [n_layers, hidden] tensor}."""
    out = Path(out) / "generations"
    n_layers = model.n_layers
    layers = {cn: cfg.layers(cn, n_layers) for cn in cfg.layer_configs}

    def steer_for(r: Row):
        return None if r.alpha == 0 else Steer(vectors[(r.concept, source(r.quadrant))], layers[r.config], r.alpha)

    if "text" in media and any(target(q) == "txt" for q in cfg.quadrants):
        _text(cfg, model, steer_for, out, shard, nshard, text_batch, log)
    if "image" in media and any(target(q) == "img" for q in cfg.quadrants):
        _image(cfg, model, steer_for, out, shard, nshard, image_batch, log)


def _text(cfg, model, steer_for, out, shard, nshard, batch, log):
    prompts = {c: text_prompts(c, cfg.n_prompts) for c in cfg.concepts}
    path = out / f"sweep_text{_part(nshard, shard)}.csv"
    done = _done(glob.glob(str(out / "sweep_text*.csv")))
    # work is assigned from the full plan, so shards and resumed runs always agree on who makes which row; a
    # concept's baselines all go to one shard, so its unsteered outputs are generated once
    rows = plan(cfg, "text")
    owned = set(cfg.concepts[shard::nshard])
    mine = [r for r in rows if r.alpha == 0 and r.concept in owned] + [r for r in rows if r.alpha != 0][shard::nshard]
    todo = [r for r in mine if r.key not in done]
    if not todo:
        log(f"[text] nothing to do ({len(done)} rows done)")
        return
    w = _Writer(path, TEXT_COLUMNS)
    t0 = time.time()

    def gen(rows):
        msgs = [prompts[r.concept][r.prompt_idx] for r in rows]
        return model.generate_text([m for m, _ in msgs], [s for _, s in msgs], [steer_for(r) for r in rows],
                                   max_new_tokens=cfg.text.get("max_new_tokens", 96))

    base = sorted({(r.concept, r.prompt_idx) for r in todo if r.alpha == 0})
    base_out = {}
    for k in range(0, len(base), batch):
        ch = [Row(c, "txt2txt", "", 0.0, pi) for c, pi in base[k:k + batch]]
        base_out.update({(r.concept, r.prompt_idx): o for r, o in zip(ch, gen(ch), strict=True)})
    w.write([[r.concept, r.quadrant, r.config, fmt_alpha(r.alpha), r.prompt_idx, prompts[r.concept][r.prompt_idx][0],
              base_out[(r.concept, r.prompt_idx)]] for r in todo if r.alpha == 0])
    # rows with the same layers and similar strength share batches: fewer hooks, and rows that degenerate run to the
    # token limit together
    steered = sorted((r for r in todo if r.alpha != 0),
                     key=lambda r: (list(cfg.layer_configs).index(r.config), abs(r.alpha), r.concept, r.prompt_idx))
    for k in range(0, len(steered), batch):
        ch = steered[k:k + batch]
        w.write([[r.concept, r.quadrant, r.config, fmt_alpha(r.alpha), r.prompt_idx,
                  prompts[r.concept][r.prompt_idx][0], o] for r, o in zip(ch, gen(ch), strict=True)])
        log(f"[text] {min(k + batch, len(steered))}/{len(steered)} steered ({time.time() - t0:.0f}s)")
    w.close()


def _image(cfg, model, steer_for, out, shard, nshard, batch, log):
    prompts = {c: image_prompts(c, cfg.n_prompts) for c in cfg.concepts}
    path = out / f"sweep_image{_part(nshard, shard)}.csv"
    done = _done(glob.glob(str(out / "sweep_image*.csv")))
    full = plan(cfg, "image")
    rows = [r for r in full if r.key not in done]
    order = {"c": cfg.concepts, "q": cfg.quadrants, "l": list(cfg.layer_configs)}
    units = sorted({(r.concept, r.quadrant, r.config) for r in full if r.alpha != 0},     # from the full plan, as
                   key=lambda u: (order["c"].index(u[0]), order["q"].index(u[1]), order["l"].index(u[2])))
    units = units[shard::nshard]                                                          # in _text
    owned = cfg.concepts[shard::nshard]                       # baselines of a concept are made by one shard
    w = _Writer(path, IMAGE_COLUMNS)
    rel = lambda c, name: f"images/{c}/{name}.png"
    (out / "images").mkdir(parents=True, exist_ok=True)

    def save(img, c, name):
        (out / "images" / c).mkdir(exist_ok=True)
        img.save(out / rel(c, name))
        return rel(c, name)

    base_need = sorted({(c, pi) for c in owned for pi in range(len(prompts[c]))
                        if not (out / rel(c, f"{c}_baseline_p{pi:02d}")).exists()})
    for k in range(0, len(base_need), batch):
        ch = base_need[k:k + batch]
        for (c, pi), im in zip(ch, model.generate_images([prompts[c][pi] for c, pi in ch], [None] * len(ch),
                                                          seed=cfg.seed), strict=True):
            save(im, c, f"{c}_baseline_p{pi:02d}")
        log(f"[image] baselines {k + len(ch)}/{len(base_need)}")
    w.write([[r.concept, r.quadrant, r.config, 0, r.prompt_idx, prompts[r.concept][r.prompt_idx],
              rel(r.concept, f"{r.concept}_baseline_p{r.prompt_idx:02d}")]
             for r in rows if r.alpha == 0 and r.concept in owned])
    t0, n = time.time(), 0
    for c, q, cn in units:
        items = sorted((r for r in rows if (r.concept, r.quadrant, r.config) == (c, q, cn) and r.alpha != 0),
                       key=lambda r: (r.alpha, r.prompt_idx))
        for k in range(0, len(items), batch):
            ch = items[k:k + batch]
            imgs = model.generate_images([prompts[c][r.prompt_idx] for r in ch], [steer_for(r) for r in ch],
                                         seed=cfg.seed)
            w.write([[c, q, cn, fmt_alpha(r.alpha), r.prompt_idx, prompts[c][r.prompt_idx],
                      save(im, c, f"{c}_{q}_{cn}_a{fmt_alpha(r.alpha)}_p{r.prompt_idx:02d}")]
                     for r, im in zip(ch, imgs, strict=True)])
            n += len(ch)
        log(f"[image] {c}/{q}/{cn} done, {n} images ({time.time() - t0:.0f}s)")
    w.close()


def merge_shards(out: str | Path) -> None:
    """Join sweep_<medium>.part*.csv into sweep_<medium>.csv (keeping rows already there) and remove the parts."""
    out = Path(out) / "generations"
    for medium, cols in (("text", TEXT_COLUMNS), ("image", IMAGE_COLUMNS)):
        parts = sorted(out.glob(f"sweep_{medium}.part*.csv"))
        if not parts:
            continue
        main = out / f"sweep_{medium}.csv"
        seen = _done([main]) if main.exists() else set()
        w = _Writer(main, cols)
        for p in parts:
            with open(p, newline="") as f:
                for r in csv.DictReader(f):
                    k = (r["concept"], r["quadrant"], r["config"], fmt_alpha(float(r["alpha"])), r["prompt_idx"])
                    if k not in seen:
                        seen.add(k)
                        w.write([[r[c] for c in cols]])
        w.close()
        for p in parts:
            p.unlink()
