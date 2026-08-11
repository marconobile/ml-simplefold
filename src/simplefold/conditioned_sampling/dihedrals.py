#!/usr/bin/env python3
"""Utilities extracted from scripts/sample_with_conditioning.py."""

from __future__ import annotations

from typing import Any

import numpy as np


SYMMETRIC_SIDECHAIN_DIHEDRALS = {
    "ASP": frozenset(("chi2",)),
    "PHE": frozenset(("chi2",)),
    "TYR": frozenset(("chi2",)),
    "VAL": frozenset(("chi1",)),
}


def symmetry_correct_dihedral_errors(
    dihedral_diff_rad: np.ndarray,
    residue_names: np.ndarray,
    dihedral_keys: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Return signed/absolute degree errors with 180-degree symmetries applied."""
    dihedral_diff_rad = np.asarray(dihedral_diff_rad)
    residue_names = np.asarray(residue_names).astype(str)
    if dihedral_diff_rad.ndim != 2:
        raise ValueError(
            "`dihedral_diff_rad` must have shape (n_residues, n_dihedrals), "
            f"got {dihedral_diff_rad.shape}."
        )
    if residue_names.shape != (dihedral_diff_rad.shape[0],):
        raise ValueError(
            "`residue_names` must have one entry per residue, got "
            f"{residue_names.shape} for dihedral shape {dihedral_diff_rad.shape}."
        )
    if len(dihedral_keys) != dihedral_diff_rad.shape[1]:
        raise ValueError(
            "`dihedral_keys` must have one entry per dihedral column, got "
            f"{len(dihedral_keys)} for dihedral shape {dihedral_diff_rad.shape}."
        )

    signed_error_deg = np.degrees(dihedral_diff_rad).astype(np.float32, copy=False)
    abs_error_deg = np.abs(signed_error_deg)
    normalized_names = np.char.upper(np.char.strip(residue_names))

    for residue_name, symmetric_keys in SYMMETRIC_SIDECHAIN_DIHEDRALS.items():
        residue_mask = normalized_names == residue_name
        if not np.any(residue_mask):
            continue
        for key in symmetric_keys:
            if key not in dihedral_keys:
                continue
            key_idx = dihedral_keys.index(key)
            raw_abs = np.clip(abs_error_deg[residue_mask, key_idx], 0.0, 180.0)
            corrected_abs = np.minimum(raw_abs, 180.0 - raw_abs)
            abs_error_deg[residue_mask, key_idx] = corrected_abs
            signed_error_deg[residue_mask, key_idx] = np.copysign(
                corrected_abs,
                signed_error_deg[residue_mask, key_idx],
            )

    return signed_error_deg, abs_error_deg


def compute_dihedral_angles(
    coords: np.ndarray,
    atom_indices: np.ndarray,
    dihedral_mask: np.ndarray,
) -> np.ndarray:
    if atom_indices.ndim != 3 or atom_indices.shape[-1] != 4:
        raise ValueError(
            f"`dihedral_atom_indices` must have shape (n_res, n_dihedrals, 4), "
            f"got {atom_indices.shape}."
        )

    angles = np.full(atom_indices.shape[:2], np.nan, dtype=np.float32)
    valid = dihedral_mask.astype(bool) & (atom_indices >= 0).all(axis=-1)
    valid &= (atom_indices < coords.shape[0]).all(axis=-1)
    if not np.any(valid):
        return angles

    idx = atom_indices[valid]
    p0 = coords[idx[:, 0]].astype(np.float64)
    p1 = coords[idx[:, 1]].astype(np.float64)
    p2 = coords[idx[:, 2]].astype(np.float64)
    p3 = coords[idx[:, 3]].astype(np.float64)

    b0 = -(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    b1_norm = np.linalg.norm(b1, axis=-1, keepdims=True)
    nonzero = b1_norm[:, 0] > 0.0
    computed = np.full(idx.shape[0], np.nan, dtype=np.float64)
    if np.any(nonzero):
        b0_nz = b0[nonzero]
        b1_nz = b1[nonzero] / b1_norm[nonzero]
        b2_nz = b2[nonzero]

        v = b0_nz - (b0_nz * b1_nz).sum(axis=-1, keepdims=True) * b1_nz
        w = b2_nz - (b2_nz * b1_nz).sum(axis=-1, keepdims=True) * b1_nz

        x = (v * w).sum(axis=-1)
        y = (np.cross(b1_nz, v) * w).sum(axis=-1)
        computed[nonzero] = np.arctan2(y, x)

    angles[valid] = computed.astype(np.float32)
    return angles

def circular_difference(sampled: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(sampled - reference), np.cos(sampled - reference)).astype(np.float32)

def summarize_dihedrals(
    original_dihedrals: np.ndarray,
    sampled_dihedrals: np.ndarray,
    original_recomputed_dihedrals: np.ndarray,
    dihedral_mask: np.ndarray,
    dihedral_keys: list[str],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    valid = (
        dihedral_mask.astype(bool)
        & np.isfinite(original_dihedrals)
        & np.isfinite(sampled_dihedrals)
    )
    diff = np.full_like(sampled_dihedrals, np.nan, dtype=np.float32)
    diff[valid] = circular_difference(sampled_dihedrals[valid], original_dihedrals[valid])
    abs_diff_deg = np.abs(np.degrees(diff))

    recompute_valid = (
        dihedral_mask.astype(bool)
        & np.isfinite(original_dihedrals)
        & np.isfinite(original_recomputed_dihedrals)
    )
    recompute_diff = np.full_like(sampled_dihedrals, np.nan, dtype=np.float32)
    recompute_diff[recompute_valid] = circular_difference(
        original_recomputed_dihedrals[recompute_valid],
        original_dihedrals[recompute_valid],
    )

    summary: dict[str, Any] = {
        "count": int(valid.sum()),
        "mae_deg": float(np.nanmean(abs_diff_deg[valid])) if np.any(valid) else None,
        "rmse_deg": (
            float(np.sqrt(np.nanmean(np.degrees(diff[valid]) ** 2)))
            if np.any(valid)
            else None
        ),
        "max_abs_error_deg": float(np.nanmax(abs_diff_deg[valid])) if np.any(valid) else None,
        "stored_vs_recomputed_original_mae_deg": (
            float(np.nanmean(np.abs(np.degrees(recompute_diff[recompute_valid]))))
            if np.any(recompute_valid)
            else None
        ),
        "by_key": {},
    }

    for i, key in enumerate(dihedral_keys):
        key_valid = valid[:, i]
        key_abs = abs_diff_deg[:, i]
        key_diff_deg = np.degrees(diff[:, i])
        summary["by_key"][key] = {
            "count": int(key_valid.sum()),
            "mae_deg": float(np.nanmean(key_abs[key_valid])) if np.any(key_valid) else None,
            "rmse_deg": (
                float(np.sqrt(np.nanmean(key_diff_deg[key_valid] ** 2)))
                if np.any(key_valid)
                else None
            ),
            "max_abs_error_deg": (
                float(np.nanmax(key_abs[key_valid])) if np.any(key_valid) else None
            ),
        }

    return summary, diff, abs_diff_deg

def add_dihedral_validation_summary(
    summary: dict[str, Any],
    name: str,
    a: np.ndarray,
    b: np.ndarray,
    dihedral_mask: np.ndarray,
) -> None:
    valid = dihedral_mask.astype(bool) & np.isfinite(a) & np.isfinite(b)
    diff = np.full_like(a, np.nan, dtype=np.float32)
    diff[valid] = circular_difference(a[valid], b[valid])
    diff_deg = np.degrees(diff)
    abs_diff_deg = np.abs(diff_deg)
    summary[name] = {
        "count": int(valid.sum()),
        "mae_deg": float(np.nanmean(abs_diff_deg[valid])) if np.any(valid) else None,
        "rmse_deg": (
            float(np.sqrt(np.nanmean(diff_deg[valid] ** 2)))
            if np.any(valid)
            else None
        ),
        "max_abs_error_deg": (
            float(np.nanmax(abs_diff_deg[valid])) if np.any(valid) else None
        ),
    }
