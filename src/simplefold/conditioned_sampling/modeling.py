#!/usr/bin/env python3
"""Utilities extracted from scripts/sample_with_conditioning.py."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Any

import torch

import hydra  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402


def resolve_checkpoint_path(args: argparse.Namespace) -> Path:
    ckpt_path = args.checkpoint_path or (args.checkpoint_dir / "last.ckpt")
    ckpt_path = ckpt_path.expanduser().resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    return ckpt_path


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)
    if not torch.cuda.is_available():
        return torch.device("cpu")

    best_idx = 0
    best_free = -1
    for idx in range(torch.cuda.device_count()):
        with torch.cuda.device(idx):
            free_bytes, _ = torch.cuda.mem_get_info()
        if free_bytes > best_free:
            best_free = free_bytes
            best_idx = idx
    return torch.device(f"cuda:{best_idx}")


def load_checkpoint(path: Path, use_mmap: bool) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "map_location": "cpu",
        "weights_only": False,
    }
    if use_mmap:
        kwargs["mmap"] = True
    try:
        checkpoint = torch.load(path, **kwargs)
    except TypeError:
        kwargs.pop("mmap", None)
        checkpoint = torch.load(path, **kwargs)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected checkpoint dict, got {type(checkpoint)!r}.")
    return checkpoint


def strip_model_prefix(
    state_dict: dict[str, torch.Tensor],
    prefix: str,
) -> dict[str, torch.Tensor]:
    return {
        key[len(prefix) :]: value
        for key, value in state_dict.items()
        if key.startswith(prefix) and "esm_model." not in key
    }


def select_model_state_dict(
    state_dict: dict[str, torch.Tensor],
    *,
    prefer_ema: bool,
) -> tuple[dict[str, torch.Tensor], str]:
    prefixes = (
        (
            ("model_ema.module.", "EMA model"),
            ("model.", "non-EMA model"),
        )
        if prefer_ema
        else (
            ("model.", "non-EMA model"),
            ("model_ema.module.", "EMA model"),
        )
    )
    for prefix, name in prefixes:
        stripped = strip_model_prefix(state_dict, prefix)
        if stripped:
            return stripped, name

    unprefixed = {
        key: value
        for key, value in state_dict.items()
        if isinstance(key, str) and "esm_model." not in key
    }
    return unprefixed, "unprefixed model"


def configure_conditioning_masking_from_checkpoint(
    model_cfg: Any,
    checkpoint_state: dict[str, torch.Tensor],
) -> None:
    """Make the conditioning vocabulary match the checkpoint semantics.

    ``FoldingDiT`` creates both the mask-token embedding row and the cluster MLM
    head when conditioning masking is enabled. The MLM head in the state dict is
    therefore an unambiguous indication that the checkpoint has the extra mask
    token, even though masking itself is disabled in eval mode.
    """
    embedding = checkpoint_state.get("cluster_embeddings.weight")
    has_cluster_weights = embedding is not None or any(
        key.startswith("cluster_mlm_head.") for key in checkpoint_state
    )
    if not has_cluster_weights:
        return

    checkpoint_uses_masking = any(
        key.startswith("cluster_mlm_head.") for key in checkpoint_state
    )
    configured_uses_masking = bool(
        model_cfg.get("use_conditioning_masking", False)
    )
    if configured_uses_masking == checkpoint_uses_masking:
        return

    model_cfg.use_conditioning_masking = checkpoint_uses_masking
    embedding_rows = (
        int(embedding.shape[0])
        if isinstance(embedding, torch.Tensor) and embedding.ndim >= 1
        else "unknown"
    )
    print(
        "Adjusted architecture use_conditioning_masking="
        f"{checkpoint_uses_masking} to match the checkpoint "
        f"(cluster embedding rows={embedding_rows})."
    )


def assert_checkpoint_weights_loaded_exactly(
    model: torch.nn.Module,
    checkpoint_state: dict[str, torch.Tensor],
) -> None:
    """Verify every selected checkpoint tensor was copied without changes."""
    loaded_state = model.state_dict()
    checkpoint_keys = set(checkpoint_state)
    loaded_keys = set(loaded_state)
    if checkpoint_keys != loaded_keys:
        missing = sorted(loaded_keys - checkpoint_keys)
        unexpected = sorted(checkpoint_keys - loaded_keys)
        raise RuntimeError(
            "Cannot verify exact checkpoint loading because state-dict keys differ. "
            f"Missing checkpoint keys: {missing[:10]}; "
            f"unexpected checkpoint keys: {unexpected[:10]}."
        )

    mismatched: list[str] = []
    for key, expected in checkpoint_state.items():
        actual = loaded_state[key]
        if (
            actual.shape != expected.shape
            or actual.dtype != expected.dtype
            or not torch.equal(actual.detach().cpu(), expected.detach().cpu())
        ):
            mismatched.append(key)

    if mismatched:
        raise RuntimeError(
            f"{len(mismatched)} checkpoint tensor(s) were not loaded exactly; "
            f"first 10: {mismatched[:10]}."
        )


def instantiate_and_load_model(
    architecture_config: Path,
    checkpoint_path: Path,
    device: torch.device,
    prefer_ema: bool,
    use_mmap: bool,
) -> torch.nn.Module:
    checkpoint = load_checkpoint(checkpoint_path, use_mmap=use_mmap)
    state_dict = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint does not contain a dict-like state_dict.")

    stripped, chosen_name = select_model_state_dict(
        state_dict,
        prefer_ema=prefer_ema,
    )
    if not stripped:
        raise RuntimeError("Checkpoint does not contain any loadable model weights.")

    model_cfg = OmegaConf.load(architecture_config)
    configure_conditioning_masking_from_checkpoint(model_cfg, stripped)
    model = hydra.utils.instantiate(model_cfg)

    model.load_state_dict(stripped, strict=True)
    assert_checkpoint_weights_loaded_exactly(model, stripped)
    print(f"Loaded {chosen_name} weights from {checkpoint_path}")
    print(f"Verified exact loading of all {len(stripped)} checkpoint tensor(s).")

    del checkpoint, state_dict, stripped
    gc.collect()
    model = model.to(device)
    model.eval()
    return model
