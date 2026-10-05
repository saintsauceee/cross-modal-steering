"""Run configs: one YAML file defines a model's full sweep (see configs/*.yaml)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO_CONFIGS = Path(__file__).resolve().parents[2] / "configs"
QUADRANTS = ("txt2txt", "img2txt", "img2img", "txt2img")
# Emu3.5's released files call its eight-layer ranges early8 / mid8 / late8.
CONFIG_ALIASES = {"early8": "early", "mid8": "mid", "late8": "late"}


@dataclass
class RunConfig:
    model: str
    concepts: list[str]
    layer_configs: dict[str, list[int] | str]
    alphas: dict[str, dict]
    model_args: dict = field(default_factory=dict)
    text: dict = field(default_factory=lambda: {"max_new_tokens": 96})
    seed: int = 42
    quadrants: list[str] = field(default_factory=lambda: list(QUADRANTS))
    vectors: str = "release"               # "release" (download the paper's vectors) or a directory
    n_prompts: int | None = None           # use only the first n evaluation prompts per concept
    text_batch: int = 64                   # rows per text generate() call (memory; padding can shift bf16 rounding)
    image_batch: int = 20                  # rows per image call; seeded images depend on it (keep the paper's value)

    def alpha_grid(self, quadrant: str) -> list[float]:
        """Five evenly spaced strengths per sign plus 0, e.g. {max: 10, step: 2} -> [-10, -8, ..., 0, ..., 10]."""
        a = self.alphas[quadrant]
        if "values" in a:
            grid = sorted(float(x) for x in a["values"])
            if 0.0 not in grid:
                raise ValueError(f"alphas.{quadrant}.values must include 0, the unsteered baseline every output is "
                                 "judged against")
            return grid
        n = round(a["max"] / a["step"])
        return [round(k * a["step"], 6) for k in range(-n, n + 1)]

    def layers(self, config_name: str, n_layers: int) -> list[int]:
        v = self.layer_configs[config_name]
        return list(range(n_layers)) if v == "all" else list(v)

    @property
    def layer_ranges(self) -> dict[str, list[int]]:
        """The early / mid / late contiguous ranges."""
        return {k: list(self.layer_configs[k]) for k in ("early", "mid", "late") if k in self.layer_configs}


def load_config(name_or_path: str | Path, **overrides) -> RunConfig:
    """Load configs/<name>.yaml (by model name) or a YAML path; keyword overrides replace top-level keys."""
    p = Path(name_or_path)
    if not p.exists():
        p = REPO_CONFIGS / f"{name_or_path}.yaml"
    d = yaml.safe_load(p.read_text())
    d.update({k: v for k, v in overrides.items() if v is not None})
    return RunConfig(**d)
