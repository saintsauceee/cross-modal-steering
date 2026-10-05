"""The concept-rate judge (paper §3.1 and Appendix A.5).

Every steered output is compared with its own unsteered (alpha = 0) output. Gemini sees the two in random order
(A/B) and is asked which shows more of one pole of the concept; the asked pole is the steered pole for half of the
pairs and the opposite pole for the other half, so recognizing steering artifacts does not score. Ten few-shot
examples come first: one grid image with an answer key for image pairs, ten worked text examples for text pairs.

Each verdict is recoded as D (the judge picked the output the steering predicts), N (the other one) or T (tie); the
concept rate counts D as 1, N and T as 0.

    judge(run_dir)   reads <run>/generations/sweep_{text,image}.csv
                     writes <run>/judge/verdicts_{text,image}.jsonl and <run>/judge/concept-rate_scores.csv

Needs GEMINI_API_KEY. Judging is resumable: an output already in verdicts_*.jsonl is never sent again.
"""

from __future__ import annotations

import csv
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

from cmsteer.concepts import DATA
from cmsteer.concepts import get as get_concept

MODEL = "gemini-3.7-flash"
SEED = 280
ROLE_PREAMBLE = "You are a careful annotator comparing two model outputs."
FEWSHOT_HEADER = "Here are examples of how this rater judges such pairs:"
FEWSHOT_FOOTER_IMAGE = "Now judge this new pair the same way:"
FEWSHOT_FOOTER_TEXT = "Now judge this pair the same way:"
EXAMPLE_INSTRUCTION = "Choose A or B, or Tie if it's not obvious."
INSTRUCTION = (
    "Choose A or B only if one clearly and obviously shows more of the attribute at a glance. "
    "If the difference is small, faint, or you have to look hard, answer Tie.\n"
    "Judge only the attribute - ignore style, grammar, and quality. A candidate that looks odd, broken, or "
    "repetitive can still clearly show the attribute (e.g. it repeats the trait), and is not automatically a "
    "Tie; decide only by whether the attribute itself is clearly stronger in one.")
SCHEMA = {"type": "object", "properties": {"verdict": {"type": "string", "enum": ["A", "B", "Tie"]}},
          "required": ["verdict"]}
N_IMAGE_EXEMPLARS = 10        # the text exemplars' A/B flips continue the random stream after the image exemplars'


def _data(name) -> Path:
    return DATA / "judge" / name


def _concept_of_pole(judge_pole: str):
    from cmsteer.concepts import CONCEPTS
    for c in CONCEPTS.values():
        if judge_pole in (c.judge_pos, c.judge_neg):
            return c
    raise KeyError(judge_pole)


def question(asked_pole: str, instruction: str, medium: str = "") -> str:
    c = _concept_of_pole(asked_pole)
    which = f"Which {medium}".strip()
    return (f"{which} represents more of {c.word(asked_pole)} (or less of {c.word(c.opposite(asked_pole))})?\n"
            f"{instruction}\nAnswer with one word (A, B, or Tie).")


def text_shots() -> list[str]:
    ex = json.loads(_data("text_exemplars.json").read_text())["exemplars"]
    rng = random.Random(SEED + 7)
    flips = [rng.random() < 0.5 for _ in range(N_IMAGE_EXEMPLARS + len(ex))][N_IMAGE_EXEMPLARS:]
    shots = []
    for e, steered_is_a in zip(ex, flips, strict=True):
        a, b = (e["steered"], e["original"]) if steered_is_a else (e["original"], e["steered"])
        ans = "Tie" if e["choice"] == "tie" else ("A" if (e["choice"] == "b") == steered_is_a else "B")
        shots.append(f"Example:\nCandidate A:\n{a}\n\nCandidate B:\n{b}\n\n"
                     f"{question(e['asked_pole'], EXAMPLE_INSTRUCTION, 'text')}\nAnswer: {ans}")
    return shots


def eid(r) -> str:
    return f"{r['quadrant']}:{r['concept']}:{r['prompt_idx']}:{r['config']}:{float(r['alpha']):g}"


def label(asked_pole: str, steer_pole: str, steered_is_a: bool, verdict: str | None) -> str | None:
    if verdict == "Tie":
        return "T"
    if verdict not in ("A", "B"):
        return None
    picked_steered = (verdict == "A") == steered_is_a
    return "D" if picked_steered == (asked_pole == steer_pole) else "N"


def pairs(run: str | Path, medium: str, replay: str | Path | None = None) -> list[dict]:
    """One record per steered output: the steered and baseline content, the steered pole, the asked pole and the
    A/B order. Order and asked pole are drawn per medium from Random(SEED) and Random(SEED + 1) over the sorted ids
    of all steered outputs in the run; with `replay` (a released verdicts_*.jsonl) they are copied from it instead,
    which re-asks exactly the paper's questions."""
    csv.field_size_limit(sys.maxsize)
    gen = Path(run) / "generations"
    col = "output" if medium == "text" else "image_path"
    with open(gen / f"sweep_{medium}.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    base = {}
    for r in rows:
        if float(r["alpha"]) == 0:
            base.setdefault((r["concept"], r["quadrant"], int(r["prompt_idx"])), r[col])
    steered = [r for r in rows if float(r["alpha"]) != 0]
    ids = sorted({eid(r) for r in steered})
    r1, r2 = random.Random(SEED), random.Random(SEED + 1)
    flip = {i: r1.random() < 0.5 for i in ids}
    same = {i: r2.random() < 0.5 for i in ids}
    if replay:
        with open(replay) as f:
            old = {o["eid"]: o for o in map(json.loads, f)}
    out = []
    for r in steered:
        i = eid(r)
        c = get_concept(r["concept"])
        steer = c.steered_pole(float(r["alpha"]))
        rec = {"eid": i, "medium": medium, "concept": r["concept"], "quadrant": r["quadrant"], "config": r["config"],
               "alpha": f"{float(r['alpha']):g}", "prompt_idx": int(r["prompt_idx"]), "steered": r[col],
               "baseline": base[(r["concept"], r["quadrant"], int(r["prompt_idx"]))], "steer_pole": steer,
               "asked_pole": steer if same[i] else c.opposite(steer), "steered_is_a": flip[i]}
        if replay and i in old:
            rec["asked_pole"], rec["steered_is_a"] = old[i]["asked_pole"], old[i]["steered_is_a"]
        out.append(rec)
    return out


def request_parts(rec: dict, run: str | Path) -> list:
    """The judgment as an ordered list of parts: strings and ("image", path) tuples."""
    a, b = (rec["steered"], rec["baseline"]) if rec["steered_is_a"] else (rec["baseline"], rec["steered"])
    q = question(rec["asked_pole"], INSTRUCTION)
    if rec["medium"] == "image":
        grid = ("image", str(_data("image_exemplar_grid.png")))
        key = _data("image_exemplar_key.txt").read_text()
        gen = Path(run) / "generations"
        return [ROLE_PREAMBLE, key, grid, FEWSHOT_FOOTER_IMAGE, "A:", ("image", str(gen / a)), "B:",
                ("image", str(gen / b)), q]
    return ([ROLE_PREAMBLE, FEWSHOT_HEADER] + text_shots() +
            [FEWSHOT_FOOTER_TEXT, f"Candidate A:\n{a}\n\nCandidate B:\n{b}\n\n{q}"])


def _call(client, parts):
    from google.genai import types
    contents = [p if isinstance(p, str) else types.Part.from_bytes(data=Path(p[1]).read_bytes(), mime_type="image/png")
                for p in parts]
    cfg = types.GenerateContentConfig(
        response_mime_type="application/json", response_schema=SCHEMA,
        thinking_config=types.ThinkingConfig(thinking_level="low", include_thoughts=True),
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))
    resp = client.models.generate_content(model=MODEL, contents=contents, config=cfg)
    cand = resp.candidates[0].content.parts
    thoughts = "\n".join(p.text for p in cand if getattr(p, "thought", False) and p.text)
    answer = "".join(p.text for p in cand if not getattr(p, "thought", False) and p.text)
    return json.loads(answer)["verdict"], thoughts


def _locked_writer(f):
    lock = threading.Lock()

    def write(rec):
        with lock:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
    return write


def _judge_one(client, run, rec, *, write, log):
    for attempt in range(4):
        try:
            verdict, why = _call(client, request_parts(rec, run))
            break
        except Exception as e:                                   # rate limits, transient API errors
            if attempt == 3:
                log(f"  failed {rec['eid']}: {e}")
                return False
            time.sleep(2 ** attempt)
    write(dict(rec, verdict=verdict, reasoning=why,
               label=label(rec["asked_pole"], rec["steer_pole"], rec["steered_is_a"], verdict)))
    return True


def judge(run: str | Path, media=("text", "image"), workers: int = 32, limit: int | None = None,
          replay_dir: str | Path | None = None, log=print) -> Path:
    from google import genai

    run = Path(run)
    (run / "judge").mkdir(parents=True, exist_ok=True)
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    failed = 0
    for medium in media:
        if not (run / "generations" / f"sweep_{medium}.csv").exists():
            log(f"[judge {medium}] no generations/sweep_{medium}.csv in {run} (shards not merged?), skipped")
            continue
        replay = Path(replay_dir) / f"verdicts_{medium}.jsonl" if replay_dir else None
        path = run / "judge" / f"verdicts_{medium}.jsonl"
        done = {json.loads(line)["eid"] for line in open(path)} if path.exists() else set()
        todo = [r for r in pairs(run, medium, replay) if r["eid"] not in done][:limit]
        log(f"[judge {medium}] {len(todo)} to judge ({len(done)} done)")
        with open(path, "a") as f:
            write = _locked_writer(f)
            with ThreadPoolExecutor(workers) as ex:
                failed += sum(not ok for ok in ex.map(partial(_judge_one, client, run, write=write, log=log), todo))
    scores = write_scores(run)
    if failed:                                                # a missing pair would silently shrink a concept rate
        raise RuntimeError(f"{failed} pairs could not be judged and are missing from {scores}; run again to retry them")
    return scores


def write_scores(run: str | Path) -> Path:
    """judge/concept-rate_scores.csv from the verdict files (correct = 1 when the label is D)."""
    run = Path(run)
    out = run / "judge" / "concept-rate_scores.csv"
    cols = ["concept", "quadrant", "config", "alpha", "prompt_idx", "verdict", "correct"]
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for medium in ("image", "text"):
            p = run / "judge" / f"verdicts_{medium}.jsonl"
            if not p.exists():
                continue
            for r in sorted(map(json.loads, open(p)), key=lambda r: r["eid"]):
                if r.get("label") is None:
                    continue
                w.writerow([r["concept"], r["quadrant"], r["config"], r["alpha"], r["prompt_idx"], r["verdict"],
                            int(r["label"] == "D")])
    return out
