"""The eight concepts of the paper, their poles, minimal text pairs and evaluation prompts.

A concept has two poles. A positive steering strength (alpha > 0) pushes toward the positive pole, a negative one
toward the negative pole, because every steering vector is mean(positive) - mean(negative).

Concept ids are the ones used in every published file (`near_far` is "depth" in the paper, `spatial_lr` is
"spatial"); `Concept.name` is the paper's name.

To add a concept, call `register()` (or add an entry to `CONCEPTS`) and provide text pairs and evaluation prompts:
see `examples/new_concept.py`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path


@dataclass(frozen=True)
class Concept:
    id: str
    name: str                      # name used in the paper and figures
    neg: str                       # negative pole, as it appears in the text pairs (alpha < 0)
    pos: str                       # positive pole (alpha > 0)
    judge_neg: str                 # how the judge prompt names each pole ("sadder", "happier", ...)
    judge_pos: str
    judge_neg_word: str            # the single word the judge question uses ("Which represents more of sad ...")
    judge_pos_word: str
    kind: str = "semantic"         # "semantic" or "visual"
    extra: dict = field(default_factory=dict, compare=False)

    def steered_pole(self, alpha: float) -> str:
        """Judge-side name of the pole a strength of this sign steers toward."""
        return self.judge_pos if alpha > 0 else self.judge_neg

    def opposite(self, judge_pole: str) -> str:
        return self.judge_neg if judge_pole == self.judge_pos else self.judge_pos

    def word(self, judge_pole: str) -> str:
        return self.judge_pos_word if judge_pole == self.judge_pos else self.judge_neg_word

    # ---- data ----
    # A concept you register yourself can carry its data in `extra` ("text_pairs": [(neg, pos), ...],
    # "subjects": [...], "descriptions": [...]) instead of the package data files.
    def text_pairs(self) -> list[tuple[str, str]]:
        """The text minimal pairs as (negative, positive) sentences (100 per paper concept)."""
        if "text_pairs" in self.extra:
            return [tuple(p) for p in self.extra["text_pairs"]]
        d = _load_json(f"text_pairs/{self.id}.json")
        return list(zip(d["neg"], d["pos"], strict=True))

    def text_subjects(self) -> list[str]:
        """Subject phrases of the 20 evaluation prompts (filled into the text instruction)."""
        if "subjects" in self.extra:
            return list(self.extra["subjects"])
        return _load_json("text_prompts.json")["subjects"][self.id]

    def image_descriptions(self) -> list[str]:
        """The 20 full scene descriptions used as image-generation prompts."""
        if "descriptions" in self.extra:
            return list(self.extra["descriptions"])
        return _load_json("image_prompts.json")["prompts"][self.id]


DATA = Path(__file__).parent / "data"

CONCEPTS: dict[str, Concept] = {c.id: c for c in [
    Concept("emotion", "emotion", "sad", "happy", "sadder", "happier", "sad", "happy"),
    Concept("age", "age", "young", "old", "younger", "older", "young", "old"),
    Concept("cleanness", "cleanness", "clean", "dirty", "cleaner", "dirtier", "clean", "dirty"),
    Concept("chaos", "chaos", "peaceful", "chaotic", "more peaceful", "more chaotic", "peaceful", "chaotic"),
    Concept("size", "size", "small", "large", "smaller", "bigger", "small", "big", kind="visual"),
    Concept("near_far", "depth", "far", "near", "farther", "closer", "far", "near", kind="visual"),
    Concept("spatial_lr", "spatial", "left", "right", "further left", "further right", "left", "right", kind="visual"),
    Concept("color", "color", "blue", "red", "bluer", "redder", "blue", "red", kind="visual"),
]}

PAPER_CONCEPTS = list(CONCEPTS)


def get(concept_id: str) -> Concept:
    try:
        return CONCEPTS[concept_id]
    except KeyError:
        raise KeyError(f"unknown concept {concept_id!r}; known: {sorted(CONCEPTS)}") from None


def register(concept: Concept) -> Concept:
    CONCEPTS[concept.id] = concept
    return concept


def text_instruction() -> str:
    """The text-generation instruction, with a `{subject}` slot (paper Appendix A.3)."""
    return _load_json("text_prompts.json")["template"]


@cache
def _load_json(rel: str) -> dict:
    return json.loads((DATA / rel).read_text())
