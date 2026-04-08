#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

import sys
import unittest
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src" / "simplefold"))

from simplefold.inference import attach_aligned_target_to_batch


def _encode_atom_names(atom_names):
    encoded = torch.zeros((1, len(atom_names), 4, 64), dtype=torch.float32)
    for atom_idx, atom_name in enumerate(atom_names):
        for char_idx, char in enumerate(atom_name.strip().upper()[:4]):
            encoded[0, atom_idx, char_idx, ord(char) - 32] = 1.0
    return encoded


class TestInferenceAlignment(unittest.TestCase):
    def test_attach_aligned_target_to_batch_preserves_dihedral_index_shape(self):
        batch = {
            "coords": torch.zeros((1, 4, 3), dtype=torch.float32),
            "atom_pad_mask": torch.ones((1, 4), dtype=torch.float32),
            "ref_atom_name_chars": _encode_atom_names(["N", "CA", "C", "O"]),
        }
        target_data = {
            "target_atom_coords": np.zeros((4, 3), dtype=np.float32),
            "target_atom_names": np.asarray(["N", "CA", "C", "O"]),
            "target_atom_residue_index": None,
            "dihedrals": np.zeros((1, 1), dtype=np.float32),
            "dihedral_atom_indices": np.asarray([[[0, 1, 2, 3]]], dtype=np.int64),
            "dihedral_mask": np.asarray([[True]], dtype=bool),
        }

        aligned = attach_aligned_target_to_batch(batch, target_data, coord_scale=1.0)

        self.assertEqual(tuple(aligned["dihedrals"].shape), (1, 1, 1))
        self.assertEqual(tuple(aligned["dihedral_atom_indices"].shape), (1, 1, 1, 4))
        self.assertEqual(tuple(aligned["dihedral_mask"].shape), (1, 1, 1))
        self.assertTrue(
            torch.equal(
                aligned["dihedral_atom_indices"],
                torch.tensor([[[[0, 1, 2, 3]]]], dtype=torch.long),
            )
        )


if __name__ == "__main__":
    unittest.main()
