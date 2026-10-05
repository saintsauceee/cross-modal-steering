"""The interface every model implements. Adding a model = subclassing SteerableModel and registering it
(see cmsteer/models/__init__.py and examples/new_model.md)."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from PIL import Image

from cmsteer.steering import Steer


class SteerableModel(ABC):
    """A unified model that generates text and images from one transformer stack.

    Representations for vector extraction are per-layer hidden states (the output of each transformer block, the
    same place steering adds its vector):
      - text: the last token of the bare sentence (no chat template), as in the paper;
      - images: the mean over the image's token positions, with the image read as model input.

    Generation takes one `Steer` (or None) per prompt, so a batch may mix concepts, settings, layers and strengths.
    """

    name: str = ""

    @property
    @abstractmethod
    def layers(self) -> list[torch.nn.Module]:
        """The transformer blocks steering hooks attach to, in order."""

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    @property
    def device(self) -> torch.device:
        return next(self.layers[0].parameters()).device

    # ---- representations (vector extraction) ----
    @abstractmethod
    def encode_text(self, sentences: list[str]) -> torch.Tensor:
        """[N, n_layers, hidden] last-token hidden states, float32 on CPU."""

    @abstractmethod
    def encode_images(self, images: list[Image.Image]) -> torch.Tensor:
        """[N, n_layers, hidden] hidden states mean-pooled over each image's tokens, float32 on CPU."""

    # ---- generation ----
    @abstractmethod
    def generate_text(self, prompts: list[str], spans: list[str | None], steer: list[Steer | None],
                      max_new_tokens: int = 96) -> list[str]:
        """Greedy text generation from user messages. The prompt pass is steered on `spans[i]` (a substring of
        prompts[i], the subject phrase; None = not at all), every generated token is steered."""

    @abstractmethod
    def generate_images(self, prompts: list[str], steer: list[Steer | None], seed: int = 42) -> list[Image.Image]:
        """Image generation from descriptions; only generated image tokens are steered, never the prompt or the
        unconditional guidance stream. The RNG is seeded once per call."""
