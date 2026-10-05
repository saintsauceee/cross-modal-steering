"""Activation steering: add alpha * v_hat_l to the output of transformer layer l during generation (paper §2.2).

One implementation serves every model and both modalities. A batch row carries its own steering, so a single
`generate()` call can mix concepts, settings, layer configurations and strengths:

    rows = [Steer(vector, layers=[16, 17, 18, 19], alpha=40.0), None, ...]    # None = unsteered row
    with steering(model.layers(), rows, prompt_mask=mask):
        model.generate(...)

Which positions are steered:
  - generated tokens: every decoding step (a forward over one new position) adds the vector at that position;
  - the prompt pass (the forward over the whole prompt) adds it only where `prompt_mask` is 1 (for text: the
    subject phrase); with `prompt_mask=None` the prompt is not steered at all (images: generated tokens only).

Rows past `len(rows)` are never steered. Models that batch classifier-free guidance as [cond rows; uncond rows] pass
the conditional rows only, so the unconditional stream stays unsteered.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch

from cmsteer.vectors import unit


@dataclass
class Steer:
    """Steering for one batch row: add alpha * unit(vector[l]) at every layer l in `layers`."""
    vector: torch.Tensor            # [n_layers, hidden], raw or unit (normalized per layer here)
    layers: list[int]
    alpha: float
    _unit: dict = field(default_factory=dict, repr=False, compare=False)

    def unit_vector(self, eps: float = 1e-8) -> torch.Tensor:
        if eps not in self._unit:
            self._unit[eps] = unit(self.vector, eps)
        return self._unit[eps]


class SteeringHandle:
    """Returned by `steering()`; `paused()` turns steering off for forwards that must stay clean (for example a
    separate unconditional CFG forward)."""

    def __init__(self):
        self.enabled = True
        self.steered_forwards = 0

    @contextmanager
    def paused(self):
        self.enabled = False
        try:
            yield
        finally:
            self.enabled = True


def _layer_adds(layers, rows: list[Steer | None], eps: float, scale_in_fp32: bool) -> dict[int, torch.Tensor]:
    """Per layer index: the [B, hidden] tensor to add (zero rows where a row is unsteered or off that layer)."""
    p = next(layers[0].parameters())
    hidden = next(r for r in rows if r is not None).vector.shape[-1]
    adds: dict[int, torch.Tensor] = {}
    for li in sorted({li for r in rows if r is not None and r.alpha != 0 for li in r.layers}):
        if not 0 <= li < len(layers):
            raise IndexError(f"layer {li} out of range for a {len(layers)}-layer model")
        v = torch.zeros(len(rows), hidden, dtype=torch.float32)
        a = torch.zeros(len(rows), 1, dtype=torch.float32)
        for b, r in enumerate(rows):
            if r is not None and li in r.layers:
                v[b] = r.unit_vector(eps)[li]
                a[b] = r.alpha
        # alpha * v_hat is either formed in float32 and then cast, or cast first and scaled in the model's dtype;
        # the two differ only in bf16 rounding, and each model uses the order of the paper's runs
        adds[li] = (a * v).to(p.device, p.dtype) if scale_in_fp32 else a.to(p.device, p.dtype) * v.to(p.device, p.dtype)
    return adds


@contextmanager
def steering(layers, rows: list[Steer | None], prompt_mask: torch.Tensor | None = None, *, eps: float = 1e-8,
             scale_in_fp32: bool = False):
    """Register forward hooks on `layers` (the model's transformer blocks) for the duration of the block.

    prompt_mask: [B, S] (0/1) over the padded prompt, or None. A forward whose sequence length equals S is the prompt
    pass; a forward over a single position is a decoding step. Anything else (none of our models produces it) is
    an error rather than silently unsteered. `eps` and `scale_in_fp32` only change bf16 rounding (see _layer_adds)."""
    handle = SteeringHandle()
    if not rows or all(r is None or r.alpha == 0 for r in rows):
        yield handle
        return
    adds = _layer_adds(layers, rows, eps, scale_in_fp32)
    B = len(rows)
    S = None if prompt_mask is None else prompt_mask.shape[1]
    mask = None
    if prompt_mask is not None:
        if prompt_mask.shape[0] != B:
            raise ValueError(f"prompt_mask has {prompt_mask.shape[0]} rows for {B} steered rows")
        p = next(layers[0].parameters())
        mask = prompt_mask.to(p.device, p.dtype).unsqueeze(-1)            # [B, S, 1]

    def make_hook(add):
        def hook(_module, _inputs, output):
            if not handle.enabled:
                return None
            h = output[0] if isinstance(output, tuple) else output
            if h.shape[0] < B:
                raise RuntimeError(f"hidden state has {h.shape[0]} rows, steering expects at least {B}")
            if h.shape[1] == 1:                                           # one generated position
                h[:B, -1, :] += add
                handle.steered_forwards += 1
            elif S is not None and h.shape[1] == S:                       # the prompt pass
                h[:B] += add[:, None, :] * mask
                handle.steered_forwards += 1
            elif S is None:                                               # prompt pass, prompt not steered
                pass
            else:
                raise RuntimeError(f"unexpected forward over {h.shape[1]} positions (prompt has {S})")
            return None
        return hook

    hooks = [layers[li].register_forward_hook(make_hook(add)) for li, add in adds.items()]
    try:
        yield handle
    finally:
        for h in hooks:
            h.remove()
    if handle.steered_forwards == 0:
        # e.g. a model that re-runs the whole sequence at every step instead of decoding one position with a cache
        warnings.warn("steering was requested but no forward pass was steered: this model does not decode one "
                      "position per forward, so the hook in cmsteer.steering needs adapting to it", stacklevel=3)


def span_mask(tokenizer, text: str, span: str | None, add_special_tokens: bool = False):
    """Token ids of `text` and a 0/1 mask marking every token whose characters overlap the last occurrence of `span`
    (by character offsets). If a token at the span's edge also covers characters outside it, the masked tokens no
    longer decode to the span and this raises rather than steer extra characters. span None -> all zeros."""
    try:
        enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=add_special_tokens)
        ids, offsets = list(enc["input_ids"]), [tuple(o) for o in enc["offset_mapping"]]
    except (NotImplementedError, TypeError, ValueError):       # slow tokenizers (Emu3.5): offsets by decoding
        ids = list(tokenizer(text, add_special_tokens=add_special_tokens)["input_ids"])
        offsets, prev = [], ""
        for i in range(len(ids)):
            cur = tokenizer.decode(ids[:i + 1])
            offsets.append((len(prev), len(cur)))
            prev = cur
        if prev != text:
            raise ValueError("decoding the tokens does not give back the prompt; cannot locate the span") from None
    if span is None:
        return ids, [0] * len(ids)
    cs = text.rfind(span)
    if cs < 0:
        raise ValueError(f"span {span!r} not found in the prompt")
    ce = cs + len(span)
    mask = [1 if (e > s and e > cs and s < ce) else 0 for s, e in offsets]
    got = tokenizer.decode([t for t, m in zip(ids, mask, strict=True) if m]).strip()
    if got != span.strip():
        raise ValueError(f"masked tokens decode to {got!r}, not the span {span!r}")
    return ids, mask


def left_pad(seqs: list[list[int]], masks: list[list[int]] | None, pad_id: int):
    """Left-pad token rows (and their masks) into tensors: ids [B, S], attention [B, S], mask [B, S] or None."""
    S = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), S), pad_id, dtype=torch.long)
    attn = torch.zeros((len(seqs), S), dtype=torch.long)
    m = None if masks is None else torch.zeros((len(seqs), S), dtype=torch.float32)
    for i, s in enumerate(seqs):
        ids[i, S - len(s):] = torch.tensor(s)
        attn[i, S - len(s):] = 1
        if m is not None:
            m[i, S - len(s):] = torch.tensor(masks[i], dtype=torch.float32)
    return ids, attn, m
