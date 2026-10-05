"""Extract steering vectors from minimal pairs (paper §2.2): v_l = mean_i (h+_{i,l} - h-_{i,l}).

Text pairs ship with the package (cmsteer/data/text_pairs). Image pairs are downloaded with
`cmsteer.hub.fetch_minimal_pairs("img")`, or come from your own folder laid out as <dir>/<concept>/<pair>/{pos,neg}.png.
Vectors are saved as <out>/<concept>/{txt,img}.pt, the layout `cmsteer sweep` and `cmsteer results` read.
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from cmsteer.concepts import get as get_concept
from cmsteer.vectors import diff_of_means, save_vector, split_half_cosine, unit


def text_vector(model, concept: str, batch: int = 50):
    pairs = get_concept(concept).text_pairs()
    neg = _encode(model.encode_text, [n for n, _ in pairs], batch)
    pos = _encode(model.encode_text, [p for _, p in pairs], batch)
    return diff_of_means(neg, pos), neg, pos


def image_vector(model, concept: str, pairs_dir: str | Path, batch: int = 25):
    dirs = sorted(d for d in (Path(pairs_dir) / concept).iterdir() if d.is_dir())
    if not dirs:
        raise FileNotFoundError(f"no image pairs under {Path(pairs_dir) / concept}")
    load = lambda p: Image.open(p).convert("RGB")
    neg = _encode(model.encode_images, [d / "neg.png" for d in dirs], batch, load)
    pos = _encode(model.encode_images, [d / "pos.png" for d in dirs], batch, load)
    return diff_of_means(neg, pos), neg, pos


def _encode(fn, items, batch, load=None):
    out = []
    for k in range(0, len(items), batch):
        chunk = items[k:k + batch]
        out.append(fn([load(x) for x in chunk] if load else chunk))
    return torch.cat(out)


def extract(model, concepts, out: str | Path, modalities=("txt", "img"), pairs_dir: str | Path | None = None,
            log=print) -> None:
    out = Path(out)
    for c in concepts:
        for mod in modalities:
            dst = out / c / f"{mod}.pt"
            if dst.exists():
                log(f"[skip] {dst} exists")
                continue
            if mod == "txt":
                v, neg, pos = text_vector(model, c)
            else:
                if pairs_dir is None:
                    raise ValueError("image vectors need pairs_dir (see cmsteer.hub.fetch_minimal_pairs)")
                v, neg, pos = image_vector(model, c, pairs_dir)
            stability = split_half_cosine(neg, pos)
            save_vector(dst, v, model=model.name, concept=c, modality=mod, n_pairs=int(neg.shape[0]),
                        pool="last_token" if mod == "txt" else "image_token_mean", estimator="diff_of_means",
                        split_half_cosine=stability.tolist())
            log(f"[vector] {c}/{mod}: {tuple(v.shape)}, split-half cosine (median over layers) "
                f"{stability.median():.3f}")


def compare_vectors(a_dir: str | Path, b_dir: str | Path, concepts, modalities=("txt", "img")) -> dict:
    """Per (concept, modality): min and mean over layers of cos(a, b), e.g. your extraction vs the released one."""
    from cmsteer.vectors import load_vector
    res = {}
    for c in concepts:
        for m in modalities:
            a, b = load_vector(Path(a_dir) / c / f"{m}.pt"), load_vector(Path(b_dir) / c / f"{m}.pt")
            cos = (unit(a) * unit(b)).sum(-1)
            res[(c, m)] = (float(cos.min()), float(cos.mean()))
    return res
