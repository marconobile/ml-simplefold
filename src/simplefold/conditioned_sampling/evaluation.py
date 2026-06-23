#!/usr/bin/env python3
"""Coordinate and dihedral evaluation for conditioned samples."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .dihedrals import (
    add_dihedral_validation_summary,
    compute_dihedral_angles,
    summarize_dihedrals,
)
from .geometry import kabsch_align


def evaluate_coordinate_alignment(
    sampled_coords: np.ndarray,
    original_coords: np.ndarray,
) -> dict[str, Any]:
    atom_mask = np.ones(original_coords.shape[0], dtype=bool)
    aligned_sampled_coords, global_rmsd, atomwise_rmsd = kabsch_align(
        sampled_coords,
        original_coords,
        atom_mask,
    )
    return {
        "atom_mask": atom_mask,
        "aligned_sampled_coords": aligned_sampled_coords,
        "global_rmsd": global_rmsd,
        "atomwise_rmsd": atomwise_rmsd,
    }


def evaluate_sample_dihedrals(
    *,
    sampled_pdb_coords: np.ndarray,
    sampled_coords: np.ndarray,
    aligned_sampled_coords: np.ndarray,
    selected_sample: np.ndarray,
    original_dihedrals: np.ndarray,
    dihedral_atom_indices: np.ndarray,
    dihedral_mask: np.ndarray,
    dihedral_keys: list[str],
    sampled_pdb_path: Path,
) -> dict[str, Any]:
    sampled_dihedrals = compute_dihedral_angles(
        sampled_pdb_coords,
        dihedral_atom_indices,
        dihedral_mask,
    )
    sampled_raw_dihedrals = compute_dihedral_angles(
        sampled_coords,
        dihedral_atom_indices,
        dihedral_mask,
    )
    sampled_aligned_dihedrals = compute_dihedral_angles(
        aligned_sampled_coords,
        dihedral_atom_indices,
        dihedral_mask,
    )
    original_recomputed_dihedrals = compute_dihedral_angles(
        selected_sample,
        dihedral_atom_indices,
        dihedral_mask,
    )
    dihedral_summary, dihedral_diff_rad, dihedral_abs_error_deg = summarize_dihedrals(
        original_dihedrals=original_dihedrals,
        sampled_dihedrals=sampled_dihedrals,
        original_recomputed_dihedrals=original_recomputed_dihedrals,
        dihedral_mask=dihedral_mask,
        dihedral_keys=dihedral_keys,
    )
    dihedral_summary["sampled_dihedral_source"] = "converted_conditioned_eval_sampled_pdb"
    dihedral_summary["sampled_pdb_path"] = str(sampled_pdb_path)
    add_dihedral_validation_summary(
        dihedral_summary,
        "sampled_raw_vs_pdb",
        sampled_raw_dihedrals,
        sampled_dihedrals,
        dihedral_mask,
    )
    add_dihedral_validation_summary(
        dihedral_summary,
        "sampled_raw_vs_aligned",
        sampled_raw_dihedrals,
        sampled_aligned_dihedrals,
        dihedral_mask,
    )
    add_dihedral_validation_summary(
        dihedral_summary,
        "sampled_aligned_vs_pdb",
        sampled_aligned_dihedrals,
        sampled_dihedrals,
        dihedral_mask,
    )
    return {
        "dihedral_summary": dihedral_summary,
        "sampled_dihedrals": sampled_dihedrals,
        "sampled_raw_dihedrals": sampled_raw_dihedrals,
        "sampled_aligned_dihedrals": sampled_aligned_dihedrals,
        "original_recomputed_dihedrals": original_recomputed_dihedrals,
        "dihedral_diff_rad": dihedral_diff_rad,
        "dihedral_abs_error_deg": dihedral_abs_error_deg,
    }
