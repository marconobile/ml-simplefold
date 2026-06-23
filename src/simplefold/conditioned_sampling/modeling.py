#!/usr/bin/env python3
"""Utilities extracted from scripts/sample_with_conditioning.py."""

from __future__ import annotations

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

def instantiate_and_load_model(
    architecture_config: Path,
    checkpoint_path: Path,
    device: torch.device,
    prefer_ema: bool,
    use_mmap: bool,
) -> torch.nn.Module:
    model_cfg = OmegaConf.load(architecture_config)
    model = hydra.utils.instantiate(model_cfg)

    checkpoint = load_checkpoint(checkpoint_path, use_mmap=use_mmap)
    state_dict = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint does not contain a dict-like state_dict.")

    primary_prefix = (
        ("model_ema.module.", "EMA model")
        if prefer_ema
        else ("model.", "non-EMA model")
    )
    fallback_prefixes = (
        ("model.", "non-EMA model"),
        ("model_ema.module.", "EMA model"),
    )

    chosen_name = None
    stripped: dict[str, torch.Tensor] = {}
    for prefix, name in [primary_prefix, *fallback_prefixes]:
        stripped = strip_model_prefix(state_dict, prefix)
        if stripped:
            chosen_name = name
            break

    if not stripped:
        stripped = {
            key: value
            for key, value in state_dict.items()
            if isinstance(key, str) and "esm_model." not in key
        }
        chosen_name = "unprefixed model"

    incompatible = model.load_state_dict(stripped, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    if missing:
        print(f"Warning: {len(missing)} missing model key(s); first 10: {missing[:10]}")
    if unexpected:
        print(f"Warning: {len(unexpected)} unexpected checkpoint key(s); first 10: {unexpected[:10]}")
    print(f"Loaded {chosen_name} weights from {checkpoint_path}")

    del checkpoint
    gc.collect()
    model = model.to(device)
    model.eval()
    return model
