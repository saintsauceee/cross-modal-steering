"""cmsteer: reproduce the paper's results and run your own cross-modal steering experiments.

  cmsteer results  [--root DIR] --out DIR          tables and figures from judge scores and vectors
  cmsteer fetch    --model M --parts scores,vectors download released artifacts
  cmsteer extract  --model M --out DIR              steering vectors from minimal pairs
  cmsteer sweep    --config M --out DIR             steered generations for a run config
  cmsteer merge    --out DIR                        join the shard files of a multi-GPU sweep
  cmsteer judge    --run DIR                        concept-rate judge (Gemini) over a sweep
  cmsteer steer    --model M --concept C ...        one steered generation, to try things out
  cmsteer check    --model M                        regenerate released rows and compare (setup check)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

MODELS = ("uniar", "emu35", "liquid")


def _list(s):
    return [x for x in s.split(",") if x] if s else None


def cmd_results(a):
    from cmsteer import hub
    from cmsteer.analysis.report import make_all

    models = _list(a.models) or list(MODELS)
    root = Path(a.root) if a.root else None
    if root is None:
        for m in models:
            hub.fetch(m, ("scores", "vectors"))
        root = hub.data_root()
    written = make_all(root, a.out, models)
    for name, path in written.items():
        print(f"{name:34s} {path}")


def cmd_fetch(a):
    from cmsteer import hub
    if a.model == "minimal-pairs":
        print(hub.fetch_minimal_pairs(a.modality, _list(a.concepts)))
    else:
        print(hub.fetch(a.model, tuple(_list(a.parts))))


def cmd_extract(a):
    from cmsteer import hub
    from cmsteer.config import load_config
    from cmsteer.extract import extract
    from cmsteer.models import load_model

    if not (a.config or a.model):
        raise SystemExit("cmsteer extract: give --model or --config")
    cfg = load_config(a.config or a.model)
    concepts = _list(a.concepts) or cfg.concepts
    mods = _list(a.modalities)
    pairs = a.pairs
    if "img" in mods and pairs is None:
        pairs = hub.fetch_minimal_pairs("img", concepts) / "img"
    kw = dict(cfg.model_args)
    if cfg.model == "uniar":
        kw["load_decoder"] = False                       # reading text and images needs no image decoder
    extract(load_model(cfg.model, **kw), concepts, a.out, mods, pairs)


def _vectors(cfg, spec, concepts):
    from cmsteer import hub
    from cmsteer.vectors import load_vector

    root = hub.fetch(cfg.model, ("vectors",)) / "vectors" if spec == "release" else Path(spec)
    return {(c, m): load_vector(root / c / f"{m}.pt") for c in concepts for m in ("txt", "img")
            if (root / c / f"{m}.pt").exists()}


def cmd_sweep(a):
    from cmsteer.config import load_config
    from cmsteer.sweep import plan, run_sweep, source, target

    cfg = load_config(a.config, concepts=_list(a.concepts), quadrants=_list(a.quadrants), n_prompts=a.n_prompts)
    if a.layer_configs:
        cfg.layer_configs = {k: cfg.layer_configs[k] for k in _list(a.layer_configs)}
    media = _list(a.media)
    if a.dry_run:
        for m in media:
            rows = plan(cfg, m)
            print(f"{m}: {len(rows)} rows ({sum(r.alpha != 0 for r in rows)} steered)")
        return
    from cmsteer.models import load_model

    vec = _vectors(cfg, a.vectors or cfg.vectors, cfg.concepts)
    wanted = {"text": "txt", "image": "img"}
    targets = {wanted[m] for m in media}
    need = {(c, source(q)) for c in cfg.concepts for q in cfg.quadrants if target(q) in targets}
    missing = sorted(need - set(vec))
    if missing:
        raise SystemExit(f"missing steering vectors (concept, source): {missing}; "
                         "run `cmsteer extract` or pass --vectors")
    kw = dict(cfg.model_args)
    if cfg.model == "uniar" and "image" not in media:
        kw["load_decoder"] = False
    model = load_model(cfg.model, **kw)
    run_sweep(cfg, model, vec, a.out, media=media, shard=a.shard, nshard=a.nshard,
              text_batch=a.text_batch or cfg.text_batch, image_batch=cfg.image_batch)


def cmd_merge(a):
    from cmsteer.sweep import merge_shards
    merge_shards(a.out)


def cmd_judge(a):
    from cmsteer import hub
    from cmsteer.judge.concept_rate import judge

    replay = None
    if a.replay:
        replay = (hub.fetch(a.replay, ("verdicts",)) if a.replay in MODELS else Path(a.replay)) / "judge"
    print(judge(a.run, media=_list(a.media), workers=a.workers, limit=a.limit, replay_dir=replay))


def cmd_steer(a):
    from cmsteer.concepts import get, text_instruction
    from cmsteer.config import load_config
    from cmsteer.models import load_model
    from cmsteer.steering import Steer

    cfg = load_config(a.model)
    vec = _vectors(cfg, a.vectors or "release", [a.concept])[(a.concept, a.source)]
    kw = dict(cfg.model_args)
    if cfg.model == "uniar" and not a.image:
        kw["load_decoder"] = False
    model = load_model(cfg.model, **kw)
    if a.layers in cfg.layer_configs:
        layers = cfg.layers(a.layers, model.n_layers)
    else:
        layers = [int(x) for x in a.layers.split(",")]
    steer = Steer(vec, layers, a.alpha)
    if a.image:
        prompt = a.prompt or get(a.concept).image_descriptions()[0]
        imgs = model.generate_images([prompt, prompt], [None, steer], seed=cfg.seed)
        for name, im in zip(("unsteered", f"alpha{a.alpha:g}"), imgs, strict=True):
            path = Path(a.out) / f"{a.concept}_{a.source}2img_{a.layers}_{name}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            im.save(path)
            print(path)
    else:
        subject = a.prompt or get(a.concept).text_subjects()[0]
        msg = text_instruction().format(subject=subject)
        for name, out in zip(("unsteered", f"alpha {a.alpha:g}"),
                             model.generate_text([msg, msg], [subject, subject], [None, steer]), strict=True):
            print(f"--- {name}\n{out}")


def cmd_check(a):
    from cmsteer.check import check, save_report
    report = check(a.model, concept=a.concept, n_text=a.n_text, images=not a.no_images, vectors=not a.no_vectors)
    if a.report:
        save_report(report, a.report)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="cmsteer", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("results", help="tables and figures from judge scores and vectors")
    p.add_argument("--root", help="folder with <model>/judge/concept-rate_scores.csv (default: download the release)")
    p.add_argument("--out", default="results")
    p.add_argument("--models", default=",".join(MODELS))
    p.set_defaults(fn=cmd_results)

    p = sub.add_parser("fetch", help="download released artifacts")
    p.add_argument("--model", required=True, choices=list(MODELS) + ["minimal-pairs"])
    p.add_argument("--parts", default="scores,vectors", help="scores, vectors, verdicts, generations, images, prompts")
    p.add_argument("--modality", default="txt", choices=["txt", "img"], help="minimal-pairs only")
    p.add_argument("--concepts", help="minimal-pairs only")
    p.set_defaults(fn=cmd_fetch)

    p = sub.add_parser("extract", help="steering vectors from minimal pairs")
    p.add_argument("--model", choices=MODELS)
    p.add_argument("--config", help="run config (default: configs/<model>.yaml)")
    p.add_argument("--out", required=True, help="writes <out>/<concept>/{txt,img}.pt")
    p.add_argument("--concepts")
    p.add_argument("--modalities", default="txt,img")
    p.add_argument("--pairs", help="image pairs folder <pairs>/<concept>/<pair>/{pos,neg}.png (default: download)")
    p.set_defaults(fn=cmd_extract)

    p = sub.add_parser("sweep", help="steered generations for a run config")
    p.add_argument("--config", required=True, help="model name (configs/<name>.yaml) or a YAML path")
    p.add_argument("--out", required=True)
    p.add_argument("--vectors", help='"release" (default) or a folder <dir>/<concept>/{txt,img}.pt')
    p.add_argument("--media", default="text,image")
    p.add_argument("--concepts")
    p.add_argument("--quadrants", help="subset of txt2txt,img2txt,img2img,txt2img")
    p.add_argument("--layer-configs", help="subset of the config's layer configurations, e.g. mid")
    p.add_argument("--n-prompts", type=int)
    p.add_argument("--text-batch", type=int)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshard", type=int, default=1)
    p.add_argument("--dry-run", action="store_true", help="count the rows, load nothing")
    p.set_defaults(fn=cmd_sweep)

    p = sub.add_parser("merge", help="join the shard files of a sweep")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_merge)

    p = sub.add_parser("judge", help="concept-rate judge over a sweep (needs GEMINI_API_KEY)")
    p.add_argument("--run", required=True)
    p.add_argument("--media", default="text,image")
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--limit", type=int)
    p.add_argument("--replay", help="ask the paper's exact questions: a model name (downloads its verdicts) or a "
                                    "folder holding verdicts_{text,image}.jsonl")
    p.set_defaults(fn=cmd_judge)

    p = sub.add_parser("steer", help="one unsteered and one steered generation")
    p.add_argument("--model", required=True, choices=MODELS)
    p.add_argument("--concept", required=True)
    p.add_argument("--source", default="txt", choices=["txt", "img"], help="vector from text or image pairs")
    p.add_argument("--layers", default="mid", help="a layer config name or comma-separated layer indices")
    p.add_argument("--alpha", type=float, required=True)
    p.add_argument("--image", action="store_true", help="generate an image (default: text)")
    p.add_argument("--prompt", help="text: the subject; image: the description (default: the concept's first)")
    p.add_argument("--vectors", help='"release" (default) or a folder')
    p.add_argument("--out", default="steer_out")
    p.set_defaults(fn=cmd_steer)

    p = sub.add_parser("check", help="regenerate a sample of released rows and compare")
    p.add_argument("--model", required=True, choices=MODELS)
    p.add_argument("--concept", default="emotion")
    p.add_argument("--n-text", type=int, default=8)
    p.add_argument("--no-images", action="store_true")
    p.add_argument("--no-vectors", action="store_true")
    p.add_argument("--report", help="write the numbers to this JSON file")
    p.set_defaults(fn=cmd_check)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
