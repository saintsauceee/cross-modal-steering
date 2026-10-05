"""The steering hook on a toy stack of layers: what gets added where."""

import pytest
import torch

from cmsteer.steering import Steer, left_pad, steering


class Block(torch.nn.Module):
    """Identity block returning a tuple, like HF decoder layers."""

    def __init__(self, d):
        super().__init__()
        self.lin = torch.nn.Linear(d, d)

    def forward(self, h):
        return (h.clone(),)


def run(layers, h):
    outs = []
    for layer in layers:
        h = layer(h)[0]
        outs.append(h)
    return outs


L, D = 4, 8


@pytest.fixture
def layers():
    return torch.nn.ModuleList([Block(D) for _ in range(L)])


def test_decode_step_adds_alpha_times_unit_vector(layers):
    v = torch.randn(L, D) * 7
    with steering(layers, [Steer(v, [1, 2], 3.0)]):
        outs = run(layers, torch.zeros(1, 1, D))
    unit = v / v.norm(dim=-1, keepdim=True)
    assert torch.allclose(outs[0], torch.zeros(1, 1, D))                       # layer 0 not steered
    assert torch.allclose(outs[1][0, 0], 3.0 * unit[1], atol=1e-5)
    assert torch.allclose(outs[2][0, 0], 3.0 * (unit[1] + unit[2]), atol=1e-5)  # identity blocks accumulate
    assert torch.allclose(outs[3], outs[2])


def test_prompt_pass_uses_mask_and_rows_are_independent(layers):
    v = torch.randn(L, D)
    rows = [Steer(v, [0], 2.0), None]
    mask = torch.tensor([[0, 1, 1], [1, 1, 1]], dtype=torch.float32)
    with steering(layers, rows, prompt_mask=mask):
        out = run(layers, torch.zeros(2, 3, D))[0]
    add = 2.0 * v[0] / v[0].norm()
    assert torch.allclose(out[0, 0], torch.zeros(D))
    assert torch.allclose(out[0, 1], add, atol=1e-5) and torch.allclose(out[0, 2], add, atol=1e-5)
    assert torch.allclose(out[1], torch.zeros(3, D))                          # unsteered row


def test_prompt_not_steered_without_mask_and_extra_rows_untouched(layers):
    v = torch.randn(L, D)
    with steering(layers, [Steer(v, [0], 1.0)]):                              # 1 conditional row
        prefill = run(layers, torch.zeros(2, 5, D))[0]                        # [cond; uncond] prompt pass
        step = run(layers, torch.zeros(2, 1, D))[0]                           # one generated token
    assert torch.allclose(prefill, torch.zeros(2, 5, D))
    assert step[0].abs().sum() > 0 and torch.allclose(step[1], torch.zeros(1, D))


def test_pause_and_hooks_removed(layers):
    v = torch.randn(L, D)
    with steering(layers, [Steer(v, [0], 1.0)]) as handle:
        with handle.paused():
            assert torch.allclose(run(layers, torch.zeros(1, 1, D))[0], torch.zeros(1, 1, D))
    assert torch.allclose(run(layers, torch.zeros(1, 1, D))[0], torch.zeros(1, 1, D))
    assert all(len(layer._forward_hooks) == 0 for layer in layers)


def test_unexpected_sequence_length_is_an_error(layers):
    with steering(layers, [Steer(torch.randn(L, D), [0], 1.0)], prompt_mask=torch.ones(1, 3)):
        with pytest.raises(RuntimeError):
            run(layers, torch.zeros(1, 2, D))


def test_left_pad():
    ids, attn, mask = left_pad([[1, 2, 3], [4]], [[0, 1, 1], [1]], pad_id=0)
    assert ids.tolist() == [[1, 2, 3], [0, 0, 4]]
    assert attn.tolist() == [[1, 1, 1], [0, 0, 1]]
    assert mask.tolist() == [[0, 1, 1], [0, 0, 1]]


class WordTokenizer:
    """Splits before each space; `fast=False` refuses offset mappings like a slow HF tokenizer."""

    def __init__(self, fast):
        self.fast, self.vocab = fast, {}

    def _pieces(self, text):
        import re
        return re.findall(r"\s*\S+", text)

    def __call__(self, text, return_offsets_mapping=False, add_special_tokens=False):
        if return_offsets_mapping and not self.fast:
            raise NotImplementedError
        pieces = self._pieces(text)
        ids = [self.vocab.setdefault(p, len(self.vocab)) for p in pieces]
        out = {"input_ids": ids}
        if return_offsets_mapping:
            pos, offs = 0, []
            for p in pieces:
                offs.append((pos, pos + len(p)))
                pos += len(p)
            out["offset_mapping"] = offs
        return out

    def decode(self, ids):
        inv = {v: k for k, v in self.vocab.items()}
        return "".join(inv[i] for i in ids)


def test_span_mask_fast_and_slow_agree():
    from cmsteer.steering import span_mask
    text = "USER: Describe your day. Your description must include a red car ASSISTANT:"
    ids_f, m_f = span_mask(WordTokenizer(True), text, "a red car")
    ids_s, m_s = span_mask(WordTokenizer(False), text, "a red car")
    assert m_f == m_s and sum(m_f) == 3
    with pytest.raises(ValueError):
        span_mask(WordTokenizer(True), text, "a blue car")


def test_warns_when_nothing_is_steered(layers):
    with pytest.warns(UserWarning, match="no forward pass was steered"):
        with steering(layers, [Steer(torch.randn(L, D), [0], 1.0)]):
            run(layers, torch.zeros(1, 5, D))                 # only full-sequence forwards, never one position
