# cross-modal-steering

Code for **"Cross-Modal Steering Vectors Exist Across Both Image and Text Generation"**: extract steering vectors from
text or image minimal pairs and apply them to text and image generation in UniAR, Emu3.5 and Liquid.

## Install

```bash
pip install -e .                 # analysis and figures (CPU)
pip install -e ".[judge]"        # + the Gemini judge
pip install -e ".[uniar]"        # + UniAR, the model the examples below use
```

Copy `.env.example` to `.env`, fill in what you need, and load it:

```bash
set -a
source .env
set +a
```

| variable | needed for |
|---|---|
| `GEMINI_API_KEY` | `cmsteer judge` |
| `UNIAR_CKPT` | UniAR (optional, defaults to the Hub id) |

Liquid and Emu3.5 work the same way (`--model liquid`, `--model emu35`), each in its own environment
(`.[liquid]`, `.[emu35]`); `.env.example` lists what they need.

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
| `cmsteer sweep` | steered generations for a config | `--config`, `--out`, `--vectors`, `--media`, `--concepts`, `--quadrants`, `--layer-configs`, `--n-prompts`, `--dry-run` |
| `cmsteer judge` | concept-rate judge over a sweep | `--run`, `--media`, `--workers`, `--limit` |
| `cmsteer steer` | one unsteered and one steered output | `--model`, `--concept`, `--alpha`, `--source`, `--layers`, `--image`, `--prompt` |

`cmsteer <command> --help` lists every option. Sweeps resume where they stopped.

## Python API

```python
from cmsteer import hub
from cmsteer.models import load_model
from cmsteer.steering import Steer
from cmsteer.vectors import load_vector

model = load_model("uniar")
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

## License

MIT. The vendored UniAR code under `src/cmsteer/models/_vendor/uniar/` is Apache-2.0.
