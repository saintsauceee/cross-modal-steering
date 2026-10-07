# Cross-modal steering

Code for **"Cross-Modal Steering Vectors Exist Across Both Image and Text Generation"**.

A steering vector found in one modality of a unified multimodal model, from text minimal pairs (*a teacher is sad /
a teacher is happy*) or from image minimal pairs (one photo edited two ways), also steers generation in the other
modality, with no adaptation. This repository extracts such vectors, applies them to text and image generation in
UniAR, Emu3.5 and Liquid, judges the outputs, and rebuilds the paper's tables and figures.

```
txt→txt   text vector,  text output        (intra-modality reference)
img→img   image vector, image output       (intra-modality reference)
txt→img   text vector,  image output       (transfer)
img→txt   image vector, text output        (transfer)
```

## Install

```bash
pip install -e .                 # analysis, figures, vectors (CPU is enough)
pip install -e ".[judge]"        # + the Gemini concept-rate judge
```

Generation needs one environment per model, because the three models pin different `transformers` versions:

| model | extra | also needs | environment variables |
|---|---|---|---|
| UniAR (8B) | `.[uniar]` | nothing else (inference code is vendored) | `UNIAR_CKPT` (default: the Hub id) |
| Liquid (7B) | `.[liquid]` | the `chameleon` package and Liquid's VQGAN files (`vqgan.yaml`, `vqgan.ckpt`) | `LIQUID_CKPT`, `LIQUID_VQGAN_DIR` |
| Emu3.5 (34B) | `.[emu35]` | a checkout of [baaivision/Emu3.5](https://github.com/baaivision/Emu3.5), flash-attention | `EMU35_REPO`, `EMU35_CKPTS`, `EMU35_VAE` |

One 80 GB GPU runs each model.

`.env.example` lists every environment variable the package reads (the judge's API key, model paths). Copy it to
`.env`, fill it in, and load it with `set -a; source .env; set +a`.

## Reproduce the paper

Each step starts from a released artifact, so you can enter at whichever step your budget allows.

| step | command | starts from | needs |
|---|---|---|---|
| 1. tables and figures | `cmsteer results --out results/` | released judge scores and vectors | CPU, a minute |
| 2. judge | `cmsteer judge --run runs/uniar` | released or your own generations | `GEMINI_API_KEY` |
| 3. generate | `cmsteer sweep --config uniar --out runs/uniar` | released or your own vectors | GPU |
| 4. extract vectors | `cmsteer extract --model uniar --out runs/uniar/vectors` | released minimal pairs | GPU |

**Step 1** writes Table 1, Table 6, Figure 3, the concept-rate curves at the middle layer range (Figures 2 and 10-12),
all layer configurations (Figures 13-15), Figure 5 with its correlations (Table 7), and the per-layer cosines
(Figure 16a). `tests/test_paper_numbers.py` checks that these numbers equal the paper's.

**Check your setup first.** `cmsteer check --model uniar` regenerates a sample of released rows (greedy text, three
seeded image batches) and re-extracts one text vector, then reports how closely they match the release. These are the
numbers we get on one A100 80 GB:

| model | text, exact match (similarity) | images vs the release | text vector (cosine with the release) |
|---|---|---|---|
| UniAR | 15/64 (0.66) | pixel-identical | 1.0000 |
| Liquid | 47/64 (0.90) | near-identical (mean difference 0.01/255) | 1.0000 |
| Emu3.5 | 58/64 (0.95) | not comparable (see below) | 1.0000 |

Image vectors re-extracted with `cmsteer extract` have a cosine of 1.0000 (UniAR), 0.9995 (Liquid) and 0.9997
(Emu3.5) with the released ones.

Text differences come from batch composition, not from the steering. UniAR is the most sensitive: even its unsteered
outputs change when the other rows of their batch change (see the notes below).

**A full run** for one model (44,800 generations, both media):

```bash
cmsteer sweep --config uniar --out runs/uniar --vectors runs/uniar/vectors   # or --vectors release
cmsteer judge --run runs/uniar
cmsteer results --root runs --models uniar --out results/uniar
```

To spread a sweep over N GPUs, launch `cmsteer sweep ... --shard i --nshard N` once per GPU, then run
`cmsteer merge --out runs/uniar`. A sweep resumes where it stopped.

## Use it for your own experiments

```python
from cmsteer import hub
from cmsteer.models import load_model
from cmsteer.steering import Steer
from cmsteer.vectors import load_vector

model = load_model("uniar")                                    # or "emu35", "liquid"
v = load_vector(hub.fetch("uniar", ["vectors"]) / "vectors/emotion/txt.pt")   # text-derived happy - sad

# Steer an image with the text vector (txt→img) on the middle layers. Rows of a batch are steered independently.
imgs = model.generate_images(["A portrait of a person in a park."] * 2,
                             [None, Steer(v, layers=[16, 17, 18, 19], alpha=9.0)])

# Steer text: the subject phrase and every generated token are steered.
msg = "Describe your day in the city. Your description must include a dog."
print(model.generate_text([msg], spans=["a dog"], steer=[Steer(v, [16, 17, 18, 19], alpha=-40.0)]))
```

- **A new concept**: `examples/new_concept.py` defines the poles and minimal pairs, extracts a vector and runs a
  small sweep.
- **A new model**: subclass `cmsteer.models.SteerableModel` (four methods) and register it. See `examples/new_model.md`.
- **One-off tries**: `cmsteer steer --model uniar --concept emotion --source img --alpha 40` prints an unsteered and a
  steered completion. Add `--image` to generate an image instead.

## How it works

| | |
|---|---|
| vector | per layer, the mean over 100 minimal pairs of h(positive) - h(negative). Text uses the last token of the sentence. Images use the mean over the image's tokens (`cmsteer/extract.py`) |
| injection | h_l ← h_l + α · v_l / ‖v_l‖ at the output of every steered layer l (`cmsteer/steering.py`) |
| where | text: the subject phrase of the prompt plus every generated token. Images: generated image tokens only, never the prompt or the unconditional guidance stream |
| layers | three single layers, early / middle / late ranges of four layers (eight for Emu3.5), and all layers (`configs/*.yaml`) |
| strengths | five per sign plus 0, set per model and setting (paper Table 4, `configs/*.yaml`) |
| judge | Gemini 3.7 Flash compares each steered output with its unsteered one, blind to order and α (`cmsteer/judge/`) |
| metrics | concept rate per strength, its normalized AUC, and recovery = AUC(transfer) / AUC(intra-modality) (`cmsteer/analysis/metrics.py`) |

## Repository layout

```
configs/                 one YAML per model: layer configurations, strength grids, decoding
src/cmsteer/
  concepts.py            the eight concepts, poles, minimal text pairs, evaluation prompts
  steering.py            the steering hook (per-row vectors, layers and strengths)
  vectors.py, extract.py vector files; extraction from minimal pairs
  models/                UniAR, Emu3.5, Liquid behind one interface
  sweep.py               the generation sweep (resumable, shardable)
  judge/                 the concept-rate judge
  analysis/              metrics, tables and figures
  hub.py                 released artifacts on the Hugging Face Hub
  check.py               setup check against the release
  data/                  minimal text pairs, evaluation prompts, judge few-shot examples
examples/
tests/
```

## Released data

On the Hugging Face Hub, one dataset per model (`unified-vlm-steering-{uniar,emu35,liquid}`): vectors, every
generation, every judge verdict with its reasoning, DreamSim and AxBench scores. Another dataset holds the
minimal pairs (`unified-vlm-steering-minimal-pairs`). `cmsteer fetch` downloads them.

## Reproducibility notes

- Images use a fixed seed per generation call, and the sweep batches them as the paper's runs did. With the
  released vectors, UniAR images regenerate pixel for pixel: 840 of 840 in our end-to-end test (two concepts, middle
  range, every strength), on the same GPU type as the paper's runs (A100). Other hardware is untested.
- With vectors you extract yourself, images are a new sample of the same distribution. Re-extracted vectors equal the
  released ones to a cosine of 0.9999998 but not bit for bit, a handful of the 16,384 added bf16 values then round
  differently, and sampling amplifies that. In the same test only 124 of 840 images came out identical, while every
  concept-rate AUC stayed within 0.03 of the release.
- Text uses greedy decoding in bf16, which depends on batch composition: the padding of the other rows changes
  rounding, and that can flip a token late in an output, even for unsteered rows. Regenerated text therefore matches
  the release in distribution and usually for a long prefix, not always exactly.
- Emu3.5's released images were generated with a patched vLLM engine, which is about 13x faster. This repository
  generates them with plain `transformers`, which matches the release in distribution but not pixel for pixel.
- The paper drew the judge's A/B order and asked pole separately for the seven main concepts and for `color`, which
  was added later. `cmsteer judge --replay <model>` asks exactly the released questions. Without `--replay` the
  draws are made over the whole run.

## Citation

```bibtex
@inproceedings{crossmodalsteering2027,
  title     = {Cross-Modal Steering Vectors Exist Across Both Image and Text Generation},
  author    = {Anonymous},
  booktitle = {Under review},
  year      = {2026}
}
```
