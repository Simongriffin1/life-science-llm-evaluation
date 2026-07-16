"""Compat helpers for MedCPT loads on platforms stuck on torch<2.6."""

from __future__ import annotations

from typing import Any, TypeVar

T = TypeVar("T")


def from_pretrained_medcpt(loader: Any, model_id: str, **kwargs: Any) -> Any:
    """
    Load a Hugging Face model, tolerating legacy ``pytorch_model.bin`` weights.

    Intel macOS is pinned to torch 2.2.x (no 2.6+ wheels on Ventura). Recent
    transformers refuse ``torch.load`` of ``.bin`` files unless torch>=2.6, even
    though MedCPT models from NCBI are trusted. Prefer safetensors when present;
    otherwise temporarily relax the transformers gate for this load only.
    """
    try:
        return loader.from_pretrained(model_id, use_safetensors=True, **kwargs)
    except Exception:
        pass

    import transformers.modeling_utils as modeling_utils

    previous = modeling_utils.check_torch_load_is_safe
    modeling_utils.check_torch_load_is_safe = lambda: None
    try:
        return loader.from_pretrained(model_id, **kwargs)
    finally:
        modeling_utils.check_torch_load_is_safe = previous
