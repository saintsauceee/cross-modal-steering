"""A concept of your own, end to end: define it, extract its text vector, steer text and images with it, judge.

The paper's language-only concepts (Section 3.3) were built this way. A real vector should use about 100 minimal
pairs; the eight below only show the format.

    python examples/new_concept.py --model uniar --out runs/formality
"""

import argparse

from cmsteer.concepts import Concept, register
from cmsteer.config import load_config
from cmsteer.extract import text_vector
from cmsteer.models import load_model
from cmsteer.sweep import run_sweep
from cmsteer.vectors import save_vector

PAIRS = [  # (negative pole, positive pole): one sentence in two registers, the difference at the end
    ("I gotta go", "I must leave"),
    ("thanks a bunch", "thank you very much"),
    ("what's up", "how do you do"),
    ("gonna be late", "I will be late"),
    ("that's super cool", "that is remarkable"),
    ("hang on a sec", "please wait a moment"),
    ("no worries", "it is no trouble at all"),
    ("see ya", "goodbye"),
]

formality = register(Concept(
    id="formality", name="formality", neg="colloquial", pos="formal",
    judge_neg="more colloquial", judge_pos="more formal", judge_neg_word="colloquial", judge_pos_word="formal",
    extra={
        "text_pairs": PAIRS,
        # text prompts: the subject is filled into the paper's instruction and steered on the prompt pass
        "subjects": ["a person at a desk", "a pair of shoes", "a menu", "a basket of laundry"],
        # image prompts: full descriptions
        "descriptions": ["A person sitting at a desk by a window, photographed in daylight.",
                         "A pair of shoes on a wooden floor, plain photograph.",
                         "A restaurant menu lying on a table, top-down photograph.",
                         "A basket of laundry next to a washing machine, plain photograph."],
    }))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="uniar")
    ap.add_argument("--out", default="runs/formality")
    ap.add_argument("--judge", action="store_true", help="also run the concept-rate judge (needs GEMINI_API_KEY)")
    a = ap.parse_args()

    cfg = load_config(a.model, concepts=["formality"], quadrants=["txt2txt", "txt2img"])
    cfg.layer_configs = {"mid": cfg.layer_configs["mid"]}           # one layer range to keep it small
    model = load_model(cfg.model, **cfg.model_args)

    v, _, _ = text_vector(model, "formality")
    save_vector(f"{a.out}/vectors/formality/txt.pt", v, model=cfg.model, concept="formality", n_pairs=len(PAIRS))
    run_sweep(cfg, model, {("formality", "txt"): v}, a.out, text_batch=cfg.text_batch, image_batch=cfg.image_batch)
    print(f"generations in {a.out}/generations/")

    if a.judge:                                                      # the concept must be registered in this process
        from cmsteer.analysis.metrics import auc_table, load_scores
        from cmsteer.judge.concept_rate import judge
        judge(a.out)
        print(auc_table(load_scores(a.out)))


if __name__ == "__main__":
    main()
