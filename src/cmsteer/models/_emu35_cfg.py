"""Classifier-free guidance for Emu3.5 with the unconditional stream batched into the SAME forward pass.

The reference `UnbatchedClassifierFreeGuidance...Processor` runs a second forward (with its own KV cache) for
the unconditional prompt on every in-image step, so each decode step reads the 68 GB of weights twice. Here the
generate() batch is laid out as [cond_0..cond_{b-1}, unc_0..unc_{b-1}]: one forward serves both halves, and the
unconditional rows are kept on the identical token continuation by sampling the next token in this processor
and returning one-hot scores for both halves (HF's sampler then picks that token deterministically for each row).

Everything else (phase tracking on row 0, h*w forcing, EOL/EOI forcing, visual-vocab mask, log-softmax CFG mix,
differential top-k) is inherited from the reference class unchanged; only `get_unconditional_logits` is redirected
to the logits the batched forward already produced. Distributionally identical to the reference; not bit-identical,
because the RNG is consumed in a different order.
"""
import math

import torch
from src.utils.logits_processor import (  # from the Emu3.5 repository (on sys.path via EMU35_REPO)
    UnbatchedClassifierFreeGuidanceLogitsForVisualTokenWithDifferentialTopKProcessor as _Ref,
)


class BatchedCFGProcessor(_Ref):
    def __init__(self, *args, do_sample: bool = True, generator=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.do_sample = do_sample
        self.generator = generator
        self._u = None

    def get_unconditional_logits(self, input_ids):
        # the reference calls this on every in-image step (and once at exit_image); serve the batched half
        return self._u

    def __call__(self, input_ids, scores):
        n = scores.shape[0]
        assert n % 2 == 0, "batched CFG expects [cond; uncond] rows"
        b = n // 2
        self._u = scores[b:]
        s = super().__call__(input_ids[:b], scores[:b].clone())          # reference logic on the conditional half
        if self.do_sample:
            tok = torch.multinomial(torch.softmax(s.float(), dim=-1), 1, generator=self.generator).squeeze(1)
        else:
            tok = s.argmax(-1)
        out = torch.full_like(scores, -math.inf)
        rows = torch.arange(b, device=scores.device)
        out[rows, tok] = 0.0
        out[rows + b, tok] = 0.0                                          # unconditional rows follow the same token
        return out
