"""Download the released artifacts from the Hugging Face Hub.

Every model has one dataset repo holding the steering vectors, all generations, the judge verdicts and the scores;
one more repo holds the minimal pairs. `fetch()` downloads the parts you ask for and returns the local folder, laid
out as on the Hub:

    vectors/<concept>/{txt,img}.pt
    generations/sweep_{text,image}.csv, generations/images/<concept>/*.png
    judge/concept-rate_scores.csv, judge/verdicts_{text,image}.jsonl
    scores/{dreamsim,axbench}_scores.csv
    prompts/{text,image}_prompts.json

Set CMSTEER_HF_NAMESPACE to read from a mirror, and CMSTEER_DATA to change where downloads go
(default ~/.cache/cmsteer).
"""

from __future__ import annotations

import os
from pathlib import Path

NAMESPACE = os.environ.get("CMSTEER_HF_NAMESPACE", "saintsauce")
MODELS = ("uniar", "emu35", "liquid")

PARTS = {
    "vectors": ["vectors/**"],
    "scores": ["judge/concept-rate_scores.csv", "judge/summary.json", "scores/*"],
    "verdicts": ["judge/verdicts_*.jsonl"],
    "generations": ["generations/sweep_text.csv", "generations/sweep_image.csv"],
    "images": ["generations/images/**"],
    "prompts": ["prompts/*"],
}


def repo_id(model: str) -> str:
    if model == "minimal-pairs":
        return f"{NAMESPACE}/unified-vlm-steering-minimal-pairs"
    if model not in MODELS:
        raise ValueError(f"no released artifacts for {model!r}; released: {MODELS}")
    return f"{NAMESPACE}/unified-vlm-steering-{model}"


def data_root() -> Path:
    return Path(os.environ.get("CMSTEER_DATA", Path.home() / ".cache" / "cmsteer"))


def fetch(model: str, parts=("scores", "vectors"), local_dir: str | Path | None = None) -> Path:
    """Download `parts` (keys of PARTS) of a model's release; returns the local root. Already-downloaded files are
    not fetched again. "images" is large (6-8 GB per model)."""
    from huggingface_hub import snapshot_download

    unknown = set(parts) - set(PARTS)
    if unknown:
        raise ValueError(f"unknown parts {sorted(unknown)}; choose from {sorted(PARTS)}")
    local = Path(local_dir) if local_dir else data_root() / model
    patterns = ["README.md", "manifest.json"] + [p for part in parts for p in PARTS[part]]
    snapshot_download(repo_id(model), repo_type="dataset", local_dir=local, allow_patterns=patterns)
    return local


def fetch_examples(models=MODELS, concept: str = "emotion", prompt_idx: int = 0) -> None:
    """The few files the example figures need: one prompt's middle-range images (Figure 4a) and the first image
    minimal pair of every concept (Figure 6)."""
    from huggingface_hub import snapshot_download

    for m in models:
        patterns = [f"generations/images/{concept}/{concept}_*_mid*_p{prompt_idx:02d}.png",
                    f"generations/images/{concept}/{concept}_baseline*_p{prompt_idx:02d}.png"]
        snapshot_download(repo_id(m), repo_type="dataset", local_dir=data_root() / m, allow_patterns=patterns)
    snapshot_download(repo_id("minimal-pairs"), repo_type="dataset", local_dir=data_root() / "minimal-pairs",
                      allow_patterns=[f"img/{c}/000/*" for c in ("emotion", "age", "cleanness", "chaos", "size",
                                                                "near_far", "spatial_lr", "color")])


def fetch_minimal_pairs(modality: str = "txt", concepts=None, local_dir: str | Path | None = None) -> Path:
    """Download the minimal pairs: modality "txt" (small JSON files) or "img" (per pair a baseline and its two edits,
    1024 px PNGs)."""
    from huggingface_hub import snapshot_download

    local = Path(local_dir) if local_dir else data_root() / "minimal-pairs"
    sel = concepts or ["*"]
    patterns = ["README.md"] + [f"{modality}/{c}/**" for c in sel]
    snapshot_download(repo_id("minimal-pairs"), repo_type="dataset", local_dir=local, allow_patterns=patterns)
    return local
