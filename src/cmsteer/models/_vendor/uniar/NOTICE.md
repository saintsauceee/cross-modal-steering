# Vendored UniAR inference code

`uniar/` and `inference/visual_inputs.py` are the inference subset of UniAR (ShareLab-SII, released under the
Apache License 2.0 with the `ShareLab-SII/UniAR-RL` checkpoint). They are vendored so the model loads without a
separate checkout. Three small local changes are marked `[local patch]`:

- `modeling_uniar.py`: the refine stack expands the 2D padding mask to a 4D causal mask, so it runs under SDPA
  (upstream assumes flash-attention);
- `modeling_uniar.py`: `generate_visual(step_hook=...)` reports the index of the image token being generated;
- `modeling_vision_decoder.py`: the decode size is rounded to a multiple of 16, as SD3 requires.
