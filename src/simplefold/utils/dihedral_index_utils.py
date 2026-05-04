#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

from __future__ import annotations

import numpy as np


def _broadcast_dihedral_mask(dihedral_atom_indices: np.ndarray, dihedral_mask: np.ndarray | None) -> np.ndarray:
    if dihedral_mask is None:
        dihedral_mask = np.ones(dihedral_atom_indices.shape[:-1], dtype=bool)
    else:
        dihedral_mask = np.asarray(dihedral_mask, dtype=bool)

    if dihedral_atom_indices.ndim != 3 or dihedral_atom_indices.shape[-1] != 4:
        raise ValueError(
            f"Expected dihedral_atom_indices to have shape (R, D, 4), got {dihedral_atom_indices.shape}."
        )
    if dihedral_mask.shape != dihedral_atom_indices.shape[:-1]:
        raise ValueError(
            "dihedral_mask must align with dihedral_atom_indices on residue/dihedral dimensions."
        )
    return dihedral_mask


def normalize_dihedral_atom_indices(
    dihedral_atom_indices: np.ndarray,
    num_atoms: int,
    dihedral_mask: np.ndarray | None = None,
    *,
    context: str = "Target",
) -> np.ndarray:
    dihedral_atom_indices = np.asarray(dihedral_atom_indices, dtype=np.int64)
    dihedral_mask = _broadcast_dihedral_mask(dihedral_atom_indices, dihedral_mask)
    valid_dihedral_atoms = np.broadcast_to(dihedral_mask[..., None], dihedral_atom_indices.shape)

    active_values = dihedral_atom_indices[valid_dihedral_atoms]
    use_one_based = False
    if active_values.size > 0:
        zero_based_valid = np.all((active_values >= 0) & (active_values < num_atoms))
        one_based_valid = np.all((active_values >= 1) & (active_values <= num_atoms))
        if not zero_based_valid and one_based_valid:
            use_one_based = True

    normalized = dihedral_atom_indices - 1 if use_one_based else dihedral_atom_indices.copy()
    indices_in_range = (normalized >= 0) & (normalized < num_atoms)
    invalid_active = valid_dihedral_atoms & (~indices_in_range)
    if np.any(invalid_active):
        bad_idx = np.argwhere(invalid_active)[0]
        bad_position = tuple(int(x) for x in bad_idx.tolist())
        bad_value = int(dihedral_atom_indices[bad_position])
        raise ValueError(
            f"{context} dihedral atom indices contain out-of-range values for active entries. "
            f"First invalid position={bad_position}, value={bad_value}, num_atoms={num_atoms}. "
            "This usually means the atom array was filtered or reordered without updating dihedral_atom_indices."
        )

    normalized[~valid_dihedral_atoms] = -1
    return normalized


def remap_dihedral_atom_indices_to_input(
    dihedral_atom_indices: np.ndarray,
    target_index_for_input_index: np.ndarray,
    dihedral_mask: np.ndarray | None = None,
    *,
    context: str = "Target",
) -> np.ndarray:
    target_index_for_input_index = np.asarray(target_index_for_input_index, dtype=np.int64)
    target_num_atoms = target_index_for_input_index.shape[0]
    normalized = normalize_dihedral_atom_indices(
        dihedral_atom_indices,
        num_atoms=target_num_atoms,
        dihedral_mask=dihedral_mask,
        context=context,
    )
    dihedral_mask = _broadcast_dihedral_mask(normalized, dihedral_mask)

    input_index_for_target_index = np.empty(target_num_atoms, dtype=np.int64)
    input_index_for_target_index[target_index_for_input_index] = np.arange(
        target_num_atoms,
        dtype=np.int64,
    )

    remapped = np.full_like(normalized, fill_value=-1)
    valid_dihedral_atoms = np.broadcast_to(dihedral_mask[..., None], normalized.shape)
    remapped[valid_dihedral_atoms] = input_index_for_target_index[normalized[valid_dihedral_atoms]]
    return remapped


def remap_dihedral_atom_indices_after_atom_filter(
    dihedral_atom_indices: np.ndarray,
    keep_indices: np.ndarray,
    original_num_atoms: int,
    dihedral_mask: np.ndarray | None = None,
    *,
    context: str = "Target",
) -> tuple[np.ndarray, np.ndarray]:
    dihedral_mask = _broadcast_dihedral_mask(np.asarray(dihedral_atom_indices, dtype=np.int64), dihedral_mask)
    normalized = normalize_dihedral_atom_indices(
        dihedral_atom_indices,
        num_atoms=original_num_atoms,
        dihedral_mask=dihedral_mask,
        context=context,
    )

    keep_indices = np.asarray(keep_indices, dtype=np.int64)
    filtered_index_for_original_index = np.full(original_num_atoms, fill_value=-1, dtype=np.int64)
    filtered_index_for_original_index[keep_indices] = np.arange(keep_indices.shape[0], dtype=np.int64)

    valid_dihedral_atoms = np.broadcast_to(dihedral_mask[..., None], normalized.shape)
    mapped = np.full_like(normalized, fill_value=-1)
    mapped[valid_dihedral_atoms] = filtered_index_for_original_index[normalized[valid_dihedral_atoms]]

    dropped_active_dihedrals = dihedral_mask & np.any(mapped < 0, axis=-1)
    updated_mask = dihedral_mask & (~dropped_active_dihedrals)

    remapped = np.full_like(normalized, fill_value=-1)
    valid_after_filter = np.broadcast_to(updated_mask[..., None], normalized.shape)
    remapped[valid_after_filter] = mapped[valid_after_filter]
    return remapped, updated_mask
