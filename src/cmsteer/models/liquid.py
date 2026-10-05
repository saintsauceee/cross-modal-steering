"""Liquid (Junfeng5/Liquid_V1_7B): a stock Gemma-7B (28 layers) whose vocabulary is extended with 8,192 VQGAN image
codes. Text and image tokens share one embedding table and one transformer.

Image generation follows the Liquid repository's inference_t2i.py: classifier-free guidance with the conditional and
unconditional rows in one batch ([cond; uncond]), top-k / top-p sampling restricted to the image codes, 1,024 codes
decoded by Chameleon's VQGAN. Needs the `chameleon` package from the Liquid repository and its VQGAN files.

Checkpoint: LIQUID_CKPT or model_path; VQGAN folder (vqgan.yaml, vqgan.ckpt): LIQUID_VQGAN_DIR or vqgan_dir.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from PIL import Image

from cmsteer.models.base import SteerableModel
from cmsteer.steering import left_pad, span_mask, steering

HF_ID = "Junfeng5/Liquid_V1_7B"
N_CODES, BOI, EOI = 1024, 7, 8


def _filter(logits, top_k, top_p, mask_below):
    """Top-k / top-p filtering with every non-image token masked out (as in the reference implementation)."""
    logits = logits.clone()
    logits[:, :mask_below] = -float("inf")
    if top_k > 0:
        kth = torch.topk(logits, min(top_k, logits.size(-1)))[0][..., -1, None]
        logits = logits.masked_fill(logits < kth, -float("inf"))
    if top_p < 1.0:
        sorted_logits, idx = torch.sort(logits, descending=True)
        drop = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1) > top_p
        drop[..., 1:] = drop[..., :-1].clone()
        drop[..., 0] = False
        logits = logits.masked_fill(drop.scatter(1, idx, drop), -float("inf"))
    return logits


class Liquid(SteerableModel):
    name = "liquid"

    def __init__(self, model_path: str | None = None, device: str = "cuda", *, dtype=torch.bfloat16,
                 vqgan_dir: str | None = None, image_grid: int = 32, guidance: float = 7.0, top_k: int = 4096,
                 top_p: float = 0.96, temperature: float = 0.99):
        from chameleon.inference.image_tokenizer import ImageTokenizer
        from transformers import AutoModelForCausalLM, AutoTokenizer

        ckpt = model_path or os.environ.get("LIQUID_CKPT", HF_ID)
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt, padding_side="left")
        self.code_offset = len(self.tokenizer)
        self.model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=dtype).to(device).eval()
        vdir = vqgan_dir or os.environ["LIQUID_VQGAN_DIR"]
        self.vqgan = ImageTokenizer(cfg_path=f"{vdir}/vqgan.yaml", ckpt_path=f"{vdir}/vqgan.ckpt", device=device)
        if image_grid * image_grid != N_CODES:
            raise ValueError("Liquid generates 32 x 32 image codes")
        self.guidance, self.top_k, self.top_p, self.temperature = guidance, top_k, top_p, temperature

    @property
    def layers(self):
        return list(self.model.model.layers)

    @staticmethod
    def chat(prompt: str) -> str:
        return f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"

    # ---- representations ----
    @torch.no_grad()
    def encode_text(self, sentences):
        out = []
        for s in sentences:
            ids = self.tokenizer(s, return_tensors="pt").to(self.device)
            hs = self.model(**ids, output_hidden_states=True, use_cache=False).hidden_states
            out.append(torch.stack([h[0, -1].float() for h in hs[1:]]).cpu())
        return torch.stack(out)

    @torch.no_grad()
    def encode_images(self, images, batch_size: int = 25):
        """Liquid's understanding input: <bos><start_of_turn>user\\n<boi> codes <eoi>, image resized to 512 px."""
        pre = self.tokenizer("<start_of_turn>user\n")["input_ids"]
        p0 = len(pre) + 1
        out = []
        for k in range(0, len(images), batch_size):
            codes = torch.stack([
                torch.as_tensor(self.vqgan.img_tokens_from_pil(im.convert("RGB").resize((512, 512), Image.LANCZOS)))
                .view(-1).long() for im in images[k:k + batch_size]])
            b = codes.shape[0]
            ids = torch.cat([torch.tensor(pre + [BOI]).expand(b, -1), codes.cpu() + self.code_offset,
                             torch.full((b, 1), EOI)], 1).to(self.device)
            hs = self.model.model(input_ids=ids, output_hidden_states=True, use_cache=False).hidden_states
            out.append(torch.stack([h[:, p0:p0 + N_CODES].float().mean(1) for h in hs[1:]], 1).cpu())
        return torch.cat(out)

    # ---- generation ----
    @torch.no_grad()
    def generate_text(self, prompts, spans, steer, max_new_tokens=96):
        seqs, masks = [], []
        for p, span in zip(prompts, spans, strict=True):
            ids, m = span_mask(self.tokenizer, self.chat(p), span, add_special_tokens=True)
            seqs.append(ids)
            masks.append(m)
        pad = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        ids, attn, mask = left_pad(seqs, masks, pad)
        with steering(self.layers, steer, prompt_mask=mask, eps=1e-6):
            g = self.model.generate(input_ids=ids.to(self.device), attention_mask=attn.to(self.device),
                                    do_sample=False, max_new_tokens=max_new_tokens, pad_token_id=pad)
        return [self.tokenizer.decode(row[ids.shape[1]:], skip_special_tokens=True).strip() for row in g]

    @torch.no_grad()
    def generate_images(self, prompts, steer, seed=42):
        from transformers import StaticCache

        n = len(prompts)
        cond = [f"{p} Generate an image based on this description.<boi>" for p in prompts]
        texts = cond + ["<unconditional><boi>"] * n                            # [cond rows; uncond rows]
        enc = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.device)
        ids = enc["input_ids"]
        dtype = next(self.model.parameters()).dtype
        size = {"max_cache_len": ids.shape[1] + N_CODES + 8, "device": self.device, "dtype": dtype}
        try:
            cache = StaticCache(config=self.model.config, max_batch_size=ids.shape[0], **size)
        except TypeError:                                                      # older transformers
            cache = StaticCache(config=self.model.config, batch_size=ids.shape[0], **size)
        kw = {"attention_mask": enc["attention_mask"], "use_cache": True, "past_key_values": cache,
              "cache_position": torch.arange(ids.shape[1], device=self.device), "num_logits_to_keep": 1}
        torch.manual_seed(seed)
        codes = []
        with steering(self.layers, steer, prompt_mask=None, eps=1e-6):
            for _ in range(N_CODES):
                inputs = self.model.prepare_inputs_for_generation(ids, **kw)
                out = self.model(**inputs, return_dict=True)
                logits = out.logits[:, -1, :]
                c, u = logits[:n], logits[n:]
                scaled = (u + (c - u) * self.guidance) / max(self.temperature, 1e-5)
                probs = F.softmax(_filter(scaled, self.top_k, self.top_p, self.code_offset), dim=-1)
                nxt = torch.multinomial(probs, num_samples=1)                  # [n, 1]
                codes.append(nxt)
                ids = torch.cat([ids, torch.cat([nxt, nxt], dim=0)], dim=-1)    # both halves take the same token
                kw = self.model._update_model_kwargs_for_generation(out, kw, is_encoder_decoder=False)
        vq = torch.clamp(torch.cat(codes, dim=1) - self.code_offset, min=0, max=8191)
        return [self.vqgan.pil_from_img_toks(row) for row in vq]
