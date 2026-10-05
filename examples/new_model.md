# Adding a model

A model needs to expose its transformer blocks, read text and images, and generate text and images. Subclass
`cmsteer.models.SteerableModel`:

```python
import torch
from cmsteer.models import SteerableModel, register
from cmsteer.steering import left_pad, span_mask, steering


class MyModel(SteerableModel):
    name = "mymodel"

    def __init__(self, device="cuda", **kwargs):
        ...  # load weights, tokenizer, image tokenizer

    @property
    def layers(self):
        return list(self.model.model.layers)          # the blocks whose outputs get the steering vector

    def encode_text(self, sentences):                 # [N, n_layers, hidden]: last token of the bare sentence
        ...

    def encode_images(self, images):                  # [N, n_layers, hidden]: mean over the image's tokens
        ...

    def generate_text(self, prompts, spans, steer, max_new_tokens=96):
        seqs, masks = zip(*(span_mask(self.tokenizer, self.chat(p), s) for p, s in zip(prompts, spans)))
        ids, attn, mask = left_pad(list(seqs), list(masks), self.tokenizer.pad_token_id)
        with steering(self.layers, steer, prompt_mask=mask):      # subject phrase + every generated token
            out = self.model.generate(input_ids=ids.cuda(), attention_mask=attn.cuda(), do_sample=False,
                                      max_new_tokens=max_new_tokens)
        return [self.tokenizer.decode(r[ids.shape[1]:], skip_special_tokens=True).strip() for r in out]

    def generate_images(self, prompts, steer, seed=42):
        torch.manual_seed(seed)
        # Put the conditional rows first ([cond; uncond]) and pass prompt_mask=None: only generated image
        # tokens of the conditional rows are steered.
        with steering(self.layers, steer, prompt_mask=None):
            ...
        return images                                  # list of PIL images


register("mymodel", "my_package.my_module:MyModel")
```

Then write `configs/mymodel.yaml` (copy one of the existing configs): layer configurations, a strength grid per
setting, decoding arguments for `__init__` under `model_args`. Pick the strength range on a few prompts at the middle
layer range so that the largest one or two strengths per sign break the outputs. This is how the paper chose its
ranges (Appendix A.4).

The hook in `cmsteer.steering` treats a forward over one position as a generated token and a forward over the whole
padded prompt as the prompt pass. If your model generates several tokens per forward, or re-encodes the prompt during
generation, adapt the hook rather than the model.
