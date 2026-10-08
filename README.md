# cross-modal-steering

Code for **"Cross-Modal Steering Vectors Exist Across Both Image and Text Generation"**: extract steering vectors from
text or image minimal pairs and apply them to text and image generation in UniAR, Emu3.5 and Liquid.

## Install

```bash
pip install -e .                 # analysis and figures (CPU)
pip install -e ".[judge]"        # + the Gemini judge
pip install -e ".[uniar]"        # + one model: uniar, liquid or emu35 (one environment per model)
```

Copy `.env.example` to `.env`, fill in what you need, and load it:

```bash
set -a; source .env; set +a
```

| variable | needed for |
|---|---|
| `GEMINI_API_KEY` | `cmsteer judge` |
| `UNIAR_CKPT` | UniAR (optional, defaults to the Hub id) |
| `LIQUID_CKPT`, `LIQUID_VQGAN_DIR` | Liquid (also needs the `chameleon` package) |
| `EMU35_REPO`, `EMU35_CKPTS`, `EMU35_VAE` | Emu3.5 (a checkout of [baaivision/Emu3.5](https://github.com/baaivision/Emu3.5)) |

Each model runs on one 80 GB GPU.

## Quick start

Rebuild the paper's tables and figures from the released scores (CPU, about a minute):

```bash
cmsteer results --out results/
```

Try one steered generation:

```bash
cmsteer steer --model uniar --concept emotion --alpha 40            # text
cmsteer steer --model uniar --concept emotion --alpha 9 --image     # image
```

## Full pipeline

```bash
cmsteer extract --model uniar --out runs/uniar/vectors                        # 1. vectors from minimal pairs
cmsteer sweep   --config uniar --out runs/uniar --vectors runs/uniar/vectors  # 2. steered generations
cmsteer judge   --run runs/uniar                                              # 3. score with Gemini
cmsteer results --root runs --models uniar --out results/uniar                # 4. tables and figures
```

Skip step 1 with `--vectors release` to use the released vectors. To run a small slice, add for example
`--concepts emotion --layer-configs mid --n-prompts 2` to `sweep`.

## Commands

| command | what it does | main options |
|---|---|---|
| `cmsteer results` | tables and figures from judge scores | `--out`, `--root`, `--models` |
| `cmsteer extract` | steering vectors from minimal pairs | `--model`, `--out`, `--concepts`, `--modalities`, `--pairs` |
| `cmsteer sweep` | steered generations for a config | `--config`, `--out`, `--vectors`, `--media`, `--concepts`, `--quadrants`, `--layer-configs`, `--n-prompts`, `--shard`, `--nshard`, `--dry-run` |
| `cmsteer merge` | join the files of a multi-GPU sweep | `--out` |
| `cmsteer judge` | concept-rate judge over a sweep | `--run`, `--media`, `--workers`, `--limit`, `--replay` |
| `cmsteer steer` | one unsteered and one steered output | `--model`, `--concept`, `--alpha`, `--source`, `--layers`, `--image`, `--prompt` |
| `cmsteer check` | regenerate released outputs and compare | `--model`, `--concept`, `--no-images`, `--no-vectors` |
| `cmsteer fetch` | download released data | `--model`, `--parts` |

`cmsteer <command> --help` lists every option. Sweeps resume where they stopped. For N GPUs, run
`cmsteer sweep ... --shard i --nshard N` once per GPU, then `cmsteer merge`.

## Python API

```python
from cmsteer import hub
from cmsteer.models import load_model
from cmsteer.steering import Steer
from cmsteer.vectors import load_vector

model = load_model("uniar")                                   # "uniar", "emu35" or "liquid"
v = load_vector(hub.fetch("uniar", ["vectors"]) / "vectors/emotion/txt.pt")   # [n_layers, hidden]
steer = Steer(v, layers=[16, 17, 18, 19], alpha=9.0)          # adds alpha * unit(v) at these layers

# images: one Steer (or None) per prompt
images = model.generate_images(["A portrait of a person in a park."] * 2, [None, steer])

# text: `spans` is the phrase of each prompt that is steered, along with every generated token
msg = "Describe your day in the city. Your description must include a dog."
texts = model.generate_text([msg], spans=["a dog"], steer=[Steer(v, [16, 17, 18, 19], alpha=-40.0)])
```

Other entry points:

| | |
|---|---|
| `cmsteer.extract.text_vector(model, concept)`, `image_vector(model, concept, pairs_dir)` | extract a vector |
| `cmsteer.sweep.run_sweep(cfg, model, vectors, out)` | run a sweep from Python |
| `cmsteer.judge.concept_rate.judge(run)` | judge a sweep |
| `cmsteer.analysis.metrics` | `load_scores`, `concept_rate`, `auc_table`, `recovery_table` |
| `cmsteer.analysis.report.make_all(root, out, models)` | every table and figure |
| `cmsteer.concepts.register(Concept(...))` | add a concept (`examples/new_concept.py`) |
| `cmsteer.models.register(name, "module:Class")` | add a model (`examples/new_model.md`) |

## Configs

`configs/{uniar,emu35,liquid}.yaml` define a model's sweep: concepts, layer configurations, the strength grid of each
setting, decoding arguments and batch sizes. Pass a model name or your own YAML to `--config`.

Settings are named `<vector source>2<output>`: `txt2txt`, `img2txt`, `img2img`, `txt2img`.

## Outputs

```
runs/<model>/
  vectors/<concept>/{txt,img}.pt
  generations/sweep_{text,image}.csv        concept, quadrant, config, alpha, prompt_idx, prompt, output | image_path
  generations/images/<concept>/*.png
  judge/verdicts_{text,image}.jsonl
  judge/concept-rate_scores.csv             concept, quadrant, config, alpha, prompt_idx, verdict, correct
```

The released data has the same layout: one Hugging Face dataset per model, plus one for the minimal pairs.

## Notes

- With the released vectors, UniAR images regenerate pixel for pixel on an A100. With vectors you extract yourself they
  are a new sample of the same distribution.
- Text is greedy but depends on batch composition in bf16, so regenerated text matches the release in distribution.
- Released Emu3.5 images came from a vLLM engine; this code generates them with `transformers`.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT. The vendored UniAR code under `src/cmsteer/models/_vendor/uniar/` is Apache-2.0.

## Citation

```bibtex
@inproceedings{crossmodalsteering2027,
  title     = {Cross-Modal Steering Vectors Exist Across Both Image and Text Generation},
  author    = {Anonymous},
  booktitle = {Under review},
  year      = {2026}
}
```
