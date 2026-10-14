"""Quality judges, DreamSim pole score, and the tables and figures built on them (no API calls)."""

import csv

import numpy as np
import pandas as pd
import pytest

from cmsteer.analysis.metrics import quality_vs_strength
from cmsteer.judge.dreamsim import pole_score
from cmsteer.judge.quality import COHERENCE_SCORE, FLUENCY_PROMPT, items


def test_pole_score():
    pos, neg = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    assert pole_score(pos, pos, neg) == pytest.approx(100.0)
    assert pole_score(neg, pos, neg) == pytest.approx(0.0)
    mid = np.array([1.0, 1.0]) / np.sqrt(2)
    assert pole_score(mid, pos, neg) == pytest.approx(50.0)


def test_prompts():
    assert "{text}" not in FLUENCY_PROMPT.format(text="hello")
    assert FLUENCY_PROMPT.format(text="hello").endswith('{"score": 0 | 1 | 2, "reasoning": "<one concise sentence>"}\n')
    assert COHERENCE_SCORE == {"intact": 2, "partly_broken": 1, "broken": 0}


def test_items_dedupe_baselines_and_filter_configs(tmp_path):
    (tmp_path / "generations").mkdir()
    rows = [("emotion", q, cn, a, pi) for q in ("txt2txt", "img2txt") for cn in ("mid", "L18")
            for a in (-1, 0, 1) for pi in (0, 1)]
    with open(tmp_path / "generations" / "sweep_text.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["concept", "quadrant", "config", "alpha", "prompt_idx", "prompt", "output"])
        w.writerows([[c, q, cn, a, pi, "p", f"out {q} {cn} {a} {pi}"] for c, q, cn, a, pi in rows])
    every = items(tmp_path, "text")
    assert sum(r["quadrant"] == "baseline" for r in every) == 2               # one per (concept, prompt)
    assert sum(r["quadrant"] != "baseline" for r in every) == 2 * 2 * 2 * 2
    mid = items(tmp_path, "text", configs=["mid"])
    assert {r["config"] for r in mid} == {"mid", "none"} and len(mid) == 2 + 8
    assert len({r["id"] for r in every}) == len(every)


def test_quality_vs_strength():
    scores = pd.DataFrame([("emotion", "txt2txt", "mid", a, p, int(abs(a) >= 10))
                           for a in (-20, -10, 10, 20) for p in range(4)],
                          columns=["concept", "quadrant", "config", "alpha", "prompt_idx", "correct"])
    quality = pd.DataFrame([("emotion", "baseline", "none", 0, p, 2) for p in range(4)]
                           + [("emotion", "txt2txt", "mid", a, p, 2 if abs(a) == 10 else 1)
                              for a in (-20, -10, 10, 20) for p in range(4)],
                           columns=["concept", "quadrant", "config", "alpha", "prompt_idx", "score"])
    t = quality_vs_strength(scores, quality, "mid").set_index("strength")
    assert t.loc[0.0, "quality"] == 1.0 and t.loc[0.0, "concept_rate"] == 0.0
    assert t.loc[0.5, "quality"] == pytest.approx(1.0) and t.loc[1.0, "quality"] == pytest.approx(0.5)
    assert t.loc[1.0, "concept_rate"] == pytest.approx(1.0)
