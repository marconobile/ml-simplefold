#!/usr/bin/env python3
"""Utilities extracted from scripts/sample_with_conditioning.py."""

from __future__ import annotations

import numpy as np


def kabsch_align(
    mobile: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray]:
    valid = mask.astype(bool) & np.isfinite(mobile).all(axis=-1) & np.isfinite(target).all(axis=-1)
    if valid.sum() < 3:
        raise ValueError("Need at least 3 valid atoms to align structures.")

    mobile_valid = mobile[valid].astype(np.float64, copy=False)
    target_valid = target[valid].astype(np.float64, copy=False)
    mobile_center = mobile_valid.mean(axis=0)
    target_center = target_valid.mean(axis=0)

    mobile_centered = mobile_valid - mobile_center
    target_centered = target_valid - target_center
    covariance = mobile_centered.T @ target_centered
    u, _, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(u @ vt))
    rotation = u @ correction @ vt

    aligned = (mobile.astype(np.float64, copy=False) - mobile_center) @ rotation + target_center
    atomwise_rmsd = np.full(mobile.shape[0], np.nan, dtype=np.float32)
    atomwise_rmsd[valid] = np.linalg.norm(aligned[valid] - target[valid], axis=-1).astype(np.float32)
    global_rmsd = float(np.sqrt(np.nanmean(atomwise_rmsd[valid] ** 2)))
    return aligned.astype(np.float32), global_rmsd, atomwise_rmsd
