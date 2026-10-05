"""Prompts, grids and judge strings match the paper (Appendix A)."""

from cmsteer.concepts import CONCEPTS, text_instruction
from cmsteer.config import load_config
from cmsteer.judge.concept_rate import INSTRUCTION, label, question, text_shots
from cmsteer.sweep import plan


def test_concepts_and_pairs():
    assert list(CONCEPTS) == ["emotion", "age", "cleanness", "chaos", "size", "near_far", "spatial_lr", "color"]
    for c in CONCEPTS.values():
        pairs = c.text_pairs()
        assert len(pairs) == 100
        neg, pos = pairs[0]
        assert neg != pos
        assert len(c.text_subjects()) == 20 and len(c.image_descriptions()) == 20
    assert CONCEPTS["emotion"].text_pairs()[0] == ("a person is sad", "a person is happy")


def test_text_instruction():
    assert text_instruction() == ("No image is provided; write from imagination. Answer in two or three plain "
                                  "sentences, no title, no list. Describe your day in the city. Your description "
                                  "must include {subject}.")


def test_paper_grids_and_row_counts():
    expect = {"uniar": {"txt2txt": 100, "img2txt": 100, "img2img": 10, "txt2img": 15},
              "emu35": {"txt2txt": 25, "img2txt": 25, "img2img": 1.5, "txt2img": 5},
              "liquid": {"txt2txt": 25, "img2txt": 25, "img2img": 0.5, "txt2img": 1.5}}
    for m, grids in expect.items():
        cfg = load_config(m)
        for q, top in grids.items():
            g = cfg.alpha_grid(q)
            assert len(g) == 11 and g[0] == -top and g[-1] == top and g[5] == 0
        # 8 concepts x 2 settings x 7 layer configs x 11 strengths x 20 prompts per medium
        assert len(plan(cfg, "text")) == len(plan(cfg, "image")) == 8 * 2 * 7 * 11 * 20
        assert len(cfg.layer_ranges) == 3


def test_judge_question_and_labels():
    assert question("happier", INSTRUCTION).startswith("Which represents more of happy (or less of sad)?\nChoose A")
    assert question("closer", "x", "image") == "Which image represents more of near (or less of far)?\nx\n" \
                                               "Answer with one word (A, B, or Tie)."
    assert len(text_shots()) == 10
    # steered shown as A, asked for the steered pole: picking A is a hit
    assert label("happier", "happier", True, "A") == "D"
    # asked for the opposite pole: the baseline (B) is the expected answer
    assert label("sadder", "happier", True, "B") == "D"
    assert label("sadder", "happier", True, "A") == "N"
    assert label("sadder", "happier", True, "Tie") == "T"
