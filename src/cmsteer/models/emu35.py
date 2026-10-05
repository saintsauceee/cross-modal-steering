"""Emu3.5 (BAAI/Emu3.5): a 34B dense decoder (64 layers) over a joint text + image-token vocabulary.

Needs a checkout of the official repository (https://github.com/baaivision/Emu3.5) for its modeling code, tokenizer
and vision tokenizer: set EMU35_REPO to it, EMU35_CKPTS to the weights and EMU35_VAE to the vision tokenizer.

Image generation here runs on Hugging Face transformers with the conditional and unconditional guidance rows in one
batch (`_batched_cfg.py`). It is distributionally identical to the reference processor but about 13x slower than
the patched vLLM engine the paper's image sweep used, so regenerated Emu3.5 images match the released ones in
distribution, not pixel for pixel.
"""

from __future__ import annotations

import os
import re
import sys
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from cmsteer.models.base import SteerableModel
from cmsteer.steering import left_pad, span_mask, steering

SYSTEM = "You are a helpful assistant."
SPECIAL = dict(BOS="<|extra_203|>", EOS="<|extra_204|>", PAD="<|endoftext|>", EOL="<|extra_200|>", EOF="<|extra_201|>",
               TMS="<|extra_202|>", IMG="<|image token|>", BOI="<|image start|>", EOI="<|image end|>",
               BSS="<|extra_100|>", ESS="<|extra_101|>", BOG="<|extra_60|>", EOG="<|extra_61|>", BOC="<|extra_50|>",
               EOC="<|extra_51|>")
TOKEN_ATTRS = {"bos_token": "BOS", "eos_token": "EOS", "pad_token": "PAD", "eol_token": "EOL", "eof_token": "EOF",
               "tms_token": "TMS", "img_token": "IMG", "boi_token": "BOI", "eoi_token": "EOI", "bss_token": "BSS",
               "ess_token": "ESS", "bog_token": "BOG", "eog_token": "EOG", "boc_token": "BOC", "eoc_token": "EOC"}
TEXT_PREFIX = "<|extra_203|>{system} USER: "
TEXT_SUFFIX = " ASSISTANT: <|extra_100|>"
IMAGE_TEMPLATE = "<|extra_203|>You are a helpful assistant for t2i task. USER: {q} ASSISTANT: <|extra_100|>"
IMAGE_UNCOND = "<|extra_203|>You are a helpful assistant. USER:  ASSISTANT: <|extra_100|>"


class Emu35(SteerableModel):
    name = "emu35"

    def __init__(self, model_path: str | None = None, device: str = "cuda", *, dtype=torch.bfloat16,
                 repo_path: str | None = None, vae_path: str | None = None, image_grid: int = 32,
                 guidance: float = 2.0, top_k: int = 10240, top_p: float = 1.0, temperature: float = 1.0,
                 text_guidance: float = 3.0, attn_implementation: str = "flash_attention_2"):
        repo = repo_path or os.environ["EMU35_REPO"]
        if repo not in sys.path:
            sys.path.insert(0, repo)
        from src.emu3p5 import Emu3Config, Emu3ForCausalLM
        from src.utils.generation_utils import build_logits_processor, multimodal_decode
        from src.utils.input_utils import build_image
        from src.vision_tokenizer import build_vision_tokenizer
        from transformers import AutoTokenizer

        self._build_logits_processor, self._multimodal_decode, self._build_image = (
            build_logits_processor, multimodal_decode, build_image)
        ckpt = model_path or os.environ["EMU35_CKPTS"]
        vae = vae_path or os.environ["EMU35_VAE"]
        tok_dir = os.path.join(repo, "src/tokenizer_emu3_ibq")
        self.model = Emu3ForCausalLM.from_pretrained(
            ckpt, config=Emu3Config.from_pretrained(ckpt, trust_remote_code=True), torch_dtype=dtype,
            device_map=device, attn_implementation=attn_implementation).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            tok_dir, special_tokens_file=os.path.join(tok_dir, "emu3_vision_tokens.txt"), trust_remote_code=True)
        for attr, key in TOKEN_ATTRS.items():
            setattr(self.tokenizer, attr, SPECIAL[key])
        self.vision_tokenizer = build_vision_tokenizer("ibq", vae, device=device)
        self.sid = {k: self.tokenizer.encode(v)[0] for k, v in SPECIAL.items()}
        self.G, self.guidance, self.text_guidance = image_grid, guidance, text_guidance
        self.top_k, self.top_p, self.temperature = top_k, top_p, temperature
        # only the last position's logits are needed while generating; skipping the rest saves memory on long batches
        self.model.lm_head.register_forward_pre_hook(
            lambda mod, args: (args[0][:, -1:, :],) if args[0].dim() == 3 and args[0].shape[1] > 1 else None)

    @property
    def layers(self):
        return list(self.model.model.layers)

    def _sampling(self, greedy: bool, max_new: int) -> dict:
        return dict(use_cache=True, text_top_k=1024, text_top_p=0.9, text_temperature=1.0, image_top_k=self.top_k,
                    image_top_p=self.top_p, image_temperature=self.temperature, top_k=131072, top_p=1.0,
                    temperature=1.0, num_beams_per_group=1, num_beam_groups=1, diversity_penalty=0.0,
                    max_new_tokens=max_new, guidance_scale=1.0, use_differential_sampling=True, do_sample=not greedy,
                    num_beams=1)

    def _generation_config(self, sp):
        from transformers import GenerationConfig
        return GenerationConfig(**sp, pad_token_id=self.sid["PAD"], eos_token_id=self.sid["EOS"])

    # ---- representations ----
    @torch.no_grad()
    def encode_text(self, sentences):
        out = []
        for s in sentences:
            ids = self.tokenizer(s, return_tensors="pt").to(self.device)
            hs = self.model(input_ids=ids["input_ids"], attention_mask=ids.get("attention_mask"),
                            output_hidden_states=True, use_cache=False).hidden_states
            out.append(torch.stack([h[0, -1].float() for h in hs[1:]]).cpu())
        return torch.stack(out)

    @torch.no_grad()
    def encode_images(self, images):
        """BOS + the image as Emu3.5 reads it (its own image string at 512 x 512); mean over the visual tokens,
        line-break tokens excluded."""
        cfg = SimpleNamespace(image_area=(self.G * 16) ** 2)
        out = []
        for im in images:
            s = self._build_image(im.convert("RGB"), cfg, self.tokenizer, self.vision_tokenizer)
            ids = self.tokenizer.encode(s, return_tensors="pt", add_special_tokens=False).to(self.device)
            ids = torch.cat([torch.tensor([[self.sid["BOS"]]], device=self.device, dtype=ids.dtype), ids], 1)
            hs = self.model.model(input_ids=ids, output_hidden_states=True, use_cache=False).hidden_states
            row = ids[0]
            ip = (row == self.sid["IMG"]).nonzero(as_tuple=True)[0]
            ep = (row == self.sid["EOI"]).nonzero(as_tuple=True)[0]
            lo = int(ip[0]) + 1 if len(ip) else 1
            hi = int(ep[0]) if len(ep) else row.shape[0]
            idx = torch.tensor([p for p in range(lo, hi) if int(row[p]) != self.sid["EOL"]], device=self.device)
            out.append(torch.stack([h[0, idx].float().mean(0) for h in hs[1:]]).cpu())
        return torch.stack(out)

    # ---- generation ----
    @torch.no_grad()
    def generate_text(self, prompts, spans, steer, max_new_tokens=96, seed=42):
        from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList

        prefix = TEXT_PREFIX.format(system=SYSTEM)
        seqs, masks = [], []
        for p, span in zip(prompts, spans, strict=True):
            ids, m = span_mask(self.tokenizer, prefix + p + TEXT_SUFFIX, span)
            seqs.append(ids)
            masks.append(m)
        # no attention mask is passed: generate() infers it from the pad token, as in the paper's runs
        ids, _attn, mask = left_pad(seqs, masks, self.sid["PAD"])
        sp = self._sampling(greedy=True, max_new=max_new_tokens)
        gcfg = SimpleNamespace(classifier_free_guidance=self.text_guidance, unconditional_type="no_text",
                               target_height=self.G, target_width=self.G, image_cfg_scale=1.0, sampling_params=sp,
                               special_token_ids=self.sid, streaming=False)
        unc = self.tokenizer.encode(prefix + TEXT_SUFFIX, return_tensors="pt", add_special_tokens=False).to(self.device)
        lp = self._build_logits_processor(gcfg, unc.repeat(len(seqs), 1), self.model, self.tokenizer, None,
                                          force_same_image_size=True)
        boi = self.sid["BOI"]

        class NoImage(LogitsProcessor):
            def __call__(self, input_ids, scores):
                scores[:, boi] = -1e30
                return scores

        torch.manual_seed(seed)
        with steering(self.layers, steer, prompt_mask=mask) as handle:
            _keep_unconditional_clean(lp, handle)
            out = self.model.generate(ids.to(self.device),
                                      self._generation_config(sp),
                                      logits_processor=LogitsProcessorList([lp, NoImage()]))
        res = []
        for row in out[:, ids.shape[1]:]:
            raw = self.tokenizer.decode(row, skip_special_tokens=False)
            t = re.split(r"<\|(image start|extra_101|endoftext|extra_204|extra_201)\|>", raw)[0]
            res.append(re.sub(r"<\|[^|]*\|>", "", t).strip())
        return res

    @torch.no_grad()
    def generate_images(self, prompts, steer, seed=42):
        from transformers.generation.logits_process import LogitsProcessorList

        from cmsteer.models._emu35_cfg import BatchedCFGProcessor

        b = len(prompts)
        sp = self._sampling(greedy=False, max_new=(self.G + 1) * self.G + 64)
        unc = self.tokenizer.encode(IMAGE_UNCOND, return_tensors="pt", add_special_tokens=False).to(self.device)
        seqs = []
        for p in prompts:
            s = self.tokenizer.encode(IMAGE_TEMPLATE.format(q=p), add_special_tokens=False)
            seqs.append(s if s[0] == self.sid["BOS"] else [self.sid["BOS"]] + s)
        seqs += [unc[0].tolist()] * b                                           # [cond rows; uncond rows]
        ids, _attn, _ = left_pad(seqs, None, self.sid["PAD"])
        lp = BatchedCFGProcessor(
            guidance_scale=self.guidance, model=self.model, tokenizer=self.tokenizer, unconditional_ids=unc,
            unconditional_type="no_text", target_height=self.G, target_width=self.G, image_cfg_scale=1.0,
            use_differential_sampling=True, text_top_k=sp["text_top_k"], text_top_p=sp["text_top_p"],
            text_temperature=sp["text_temperature"], image_top_k=sp["image_top_k"], image_top_p=sp["image_top_p"],
            image_temperature=sp["image_temperature"], force_same_image_size=True, do_sample=True)
        torch.manual_seed(seed)
        with steering(self.layers, steer, prompt_mask=None):
            out = self.model.generate(ids.to(self.device),
                                      self._generation_config(sp),
                                      logits_processor=LogitsProcessorList([lp]))
        images = []
        for row in out[:b, ids.shape[1]:]:
            im = None
            for kind, val in self._multimodal_decode(self.tokenizer.decode(row, skip_special_tokens=False),
                                                     self.tokenizer, self.vision_tokenizer):
                if kind == "image":
                    im = val if isinstance(val, Image.Image) else Image.fromarray(np.asarray(val))
                    break
            if im is None:
                raise RuntimeError(f"Emu3.5 produced no decodable image for prompt {len(images)} of this batch "
                                   f"({prompts[len(images)][:60]!r}); its steering strength may be past the point "
                                   "where the model still closes the image")
            images.append(im)
        return images


def _keep_unconditional_clean(processor, handle):
    """The reference CFG processor runs the unconditional prompt through the model in a separate forward; pause
    steering for it so only the conditional stream is steered."""
    original = processor.get_unconditional_logits

    def wrapped(x):
        with handle.paused():
            return original(x)

    processor.get_unconditional_logits = wrapped
