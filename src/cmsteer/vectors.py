"""Steering vectors: a [n_layers, hidden] tensor, one direction per transformer layer.

The vector of a concept is the mean difference between paired representations (paper §2.2),
    v_l = 1/N sum_i (h+_{i,l} - h-_{i,l}),
and steering always uses its per-layer unit version v_hat_l = v_l / ||v_l||, so the strength is set by alpha alone.

Files are saved as {"vector": raw [L, d] float32, "meta": {...}}. `load_vector` also reads the formats of the
released files (a bare tensor, or a dict with "vector" and flat metadata).
"""

from __future__ import annotations

from pathlib import Path

import torch


def diff_of_means(neg: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """neg, pos: [N, L, d] paired representations -> [L, d] mean(pos - neg)."""
    if neg.shape != pos.shape:
        raise ValueError(f"paired representations differ in shape: {tuple(neg.shape)} vs {tuple(pos.shape)}")
    return (pos.float() - neg.float()).mean(0)


def unit(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-layer unit vectors."""
    v = v.float()
    return v / (v.norm(dim=-1, keepdim=True) + eps)


def split_half_cosine(neg: torch.Tensor, pos: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """Per-layer cosine between the vectors of two random halves of the pairs (a stability check)."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(neg.shape[0], generator=g)
    a, b = idx[: len(idx) // 2], idx[len(idx) // 2:]
    return (unit(diff_of_means(neg[a], pos[a])) * unit(diff_of_means(neg[b], pos[b]))).sum(-1)


def save_vector(path: str | Path, vector: torch.Tensor, **meta) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"vector": vector.detach().float().cpu(), "meta": meta}, path)
    return path


def load_vector(path: str | Path) -> torch.Tensor:
    """Load a [L, d] float32 vector (raw or unit, as stored)."""
    d = torch.load(path, map_location="cpu", weights_only=False)
    v = d["vector"] if isinstance(d, dict) else d
    if not isinstance(v, torch.Tensor) or v.dim() != 2:
        raise ValueError(f"{path}: expected a [n_layers, hidden] tensor")
    return v.float()


def load_meta(path: str | Path) -> dict:
    d = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(d, dict):
        return {}
    return d.get("meta") or {k: v for k, v in d.items() if not isinstance(v, torch.Tensor)}
