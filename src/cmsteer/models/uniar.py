"""UniAR (ShareLab-SII/UniAR-RL): Qwen3-VL backbone (36 layers) + BSQ image tokens rendered by an SD3 decoder.

Steering hooks sit on the Qwen3-VL text decoder layers, which run for text and for the image-token rollout alike.
Image generation batches classifier-free guidance as [cond rows; uncond rows]; only the conditional rows are steered.

Checkpoint: UNIAR_CKPT or model_path (a local copy of ShareLab-SII/UniAR-RL, or the Hub id).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from PIL import Image

from cmsteer.models.base import SteerableModel
from cmsteer.steering import Steer, left_pad, span_mask, steering

HF_ID = "ShareLab-SII/UniAR-RL"
_VENDOR = str(Path(__file__).parent / "_vendor" / "uniar")


class UniAR(SteerableModel):
    name = "uniar"

    def __init__(self, model_path: str | None = None, device: str = "cuda", *, dtype=torch.bfloat16,
                 load_decoder: bool = True, image_size: int = 512, upsampling_ratio: float = 1.067,
                 guidance: float = 1.5, temperature: float = 1.0, decoder_steps: int = 16,
                 decoder_guidance: float = 1.5, attn_implementation: str = "sdpa"):
        if _VENDOR not in sys.path:
            sys.path.insert(0, _VENDOR)
        import uniar  # noqa: F401  (registers the UniAR classes with the HF auto classes)
        from transformers import AutoProcessor
        from uniar import UniARForConditionalGeneration, UniARVisualDecoder

        ckpt = model_path or os.environ.get("UNIAR_CKPT", HF_ID)
        self.model = UniARForConditionalGeneration.from_pretrained(
            ckpt, torch_dtype=dtype, attn_implementation=attn_implementation).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(ckpt, padding_side="left")
        self.tokenizer = self.processor.tokenizer
        # prepare_visual_inputs replaces processor.chat_template with the generation template; keep the
        # understanding template for reading images
        self._read_template = self.processor.chat_template
        self.decoder = None
        if load_decoder:
            self.decoder = UniARVisualDecoder.from_pretrained(ckpt, device=device, torch_dtype=dtype)
        self.image_size, self.upsampling_ratio = image_size, upsampling_ratio
        self.guidance, self.temperature = guidance, temperature
        self.decoder_steps, self.decoder_guidance = decoder_steps, decoder_guidance
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    @property
    def layers(self):
        return list(self.model.model.language_model.layers)

    # ---- representations ----
    @torch.no_grad()
    def encode_text(self, sentences):
        self.model.set_generation_mode("text")
        out = []
        for s in sentences:
            ids = self.tokenizer(s, return_tensors="pt").to(self.device)
            hs = self.model(input_ids=ids["input_ids"], attention_mask=ids["attention_mask"],
                            output_hidden_states=True, use_cache=False).hidden_states
            out.append(torch.stack([h[0, -1].float() for h in hs[1:]]).cpu())
        return torch.stack(out)

    @torch.no_grad()
    def encode_images(self, images):
        self.model.set_generation_mode("text")
        out = []
        for im in images:
            msgs = [{"role": "user", "content": [{"type": "image", "image": im.convert("RGB")}]}]
            inputs = self.processor.apply_chat_template(
                msgs, chat_template=self._read_template, tokenize=True, add_generation_prompt=False,
                return_dict=True, return_tensors="pt").to(self.device)
            inputs.pop("mm_token_type_ids", None)
            hs = self.model(**inputs, output_hidden_states=True, use_cache=False).hidden_states
            pos = (inputs["input_ids"][0] == self.model.config.image_token_id).nonzero(as_tuple=True)[0]
            out.append(torch.stack([h[0, pos].float().mean(0) for h in hs[1:]]).cpu())
        return torch.stack(out)

    # ---- generation ----
    @torch.no_grad()
    def generate_text(self, prompts, spans, steer, max_new_tokens=96):
        self.model.set_generation_mode("text")
        seqs, masks = [], []
        for p, span in zip(prompts, spans, strict=True):
            text = self.tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                                      add_generation_prompt=True)
            ids, m = span_mask(self.tokenizer, text, span)
            seqs.append(ids)
            masks.append(m)
        pad = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        ids, attn, mask = left_pad(seqs, masks, pad)
        with steering(self.layers, steer, prompt_mask=mask):
            g = self.model.generate(input_ids=ids.to(self.device), attention_mask=attn.to(self.device),
                                    max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=pad)
        return [self.tokenizer.decode(row[ids.shape[1]:], skip_special_tokens=True).strip() for row in g]

    @torch.no_grad()
    def generate_images(self, prompts, steer, seed=42):
        from inference.visual_inputs import prepare_visual_inputs

        if self.decoder is None:
            raise RuntimeError("constructed with load_decoder=False")
        self.model.set_generation_mode("image")
        v = prepare_visual_inputs(list(prompts), self.model, self.processor, self.image_size, self.image_size)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        with steering(self.layers, steer, prompt_mask=None, eps=1e-6, scale_in_fp32=True):
            visual_ids = self.model.generate_visual(
                prefix_input_ids=v["prefix_input_ids"], attention_mask=v["attention_mask"],
                pos_ids_image=v["pos_ids_image"], pos_ids_all=v["pos_ids_all"],
                image_token_num=v["image_token_num"], temperature=self.temperature, cfg=self.guidance)
        gen = torch.Generator(device=self.device).manual_seed(seed)
        images = self.decoder.decode(visual_ids, ar_height=self.image_size, ar_width=self.image_size,
                                     upsampling_ratio=self.upsampling_ratio, num_inference_steps=self.decoder_steps,
                                     guidance_scale=self.decoder_guidance, generator=gen)
        return [im if isinstance(im, Image.Image) else Image.fromarray(im) for im in images]


__all__ = ["UniAR", "Steer"]
