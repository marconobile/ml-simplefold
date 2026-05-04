#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src" / "simplefold"))

from simplefold.utils.dihedral_index_utils import (
    normalize_dihedral_atom_indices,
    remap_dihedral_atom_indices_after_atom_filter,
    remap_dihedral_atom_indices_to_input,
)


class TestDihedralIndexUtils(unittest.TestCase):
    def test_normalize_dihedral_atom_indices_handles_one_based_active_entries(self):
        dihedral_atom_indices = np.array(
            [
                [[1, 2, 3, 4], [-1, -1, -1, -1]],
            ],
            dtype=np.int64,
        )
        dihedral_mask = np.array([[True, False]])

        normalized = normalize_dihedral_atom_indices(
            dihedral_atom_indices,
            num_atoms=4,
            dihedral_mask=dihedral_mask,
        )

        np.testing.assert_array_equal(
            normalized,
            np.array([[[0, 1, 2, 3], [-1, -1, -1, -1]]], dtype=np.int64),
        )

    def test_remap_dihedral_atom_indices_to_input_uses_input_atom_order(self):
        dihedral_atom_indices = np.array(
            [[[0, 2, 3, 1]]],
            dtype=np.int64,
        )
        dihedral_mask = np.array([[True]])
        # input index -> target index
        target_index_for_input_index = np.array([2, 0, 3, 1], dtype=np.int64)

        remapped = remap_dihedral_atom_indices_to_input(
            dihedral_atom_indices,
            target_index_for_input_index=target_index_for_input_index,
            dihedral_mask=dihedral_mask,
        )

        np.testing.assert_array_equal(
            remapped,
            np.array([[[1, 0, 2, 3]]], dtype=np.int64),
        )

    def test_remap_dihedral_atom_indices_after_atom_filter_compresses_indices(self):
        dihedral_atom_indices = np.array(
            [
                [[0, 2, 3, 5], [1, 2, 3, 4]],
            ],
            dtype=np.int64,
        )
        dihedral_mask = np.array([[True, True]])
        keep_indices = np.array([0, 2, 3, 5], dtype=np.int64)

        remapped, updated_mask = remap_dihedral_atom_indices_after_atom_filter(
            dihedral_atom_indices,
            keep_indices=keep_indices,
            original_num_atoms=6,
            dihedral_mask=dihedral_mask,
        )

        np.testing.assert_array_equal(
            remapped,
            np.array([[[0, 1, 2, 3], [-1, -1, -1, -1]]], dtype=np.int64),
        )
        np.testing.assert_array_equal(updated_mask, np.array([[True, False]]))


if __name__ == "__main__":
    unittest.main()
