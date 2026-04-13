#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src" / "simplefold"))

from simplefold.inference import attach_aligned_target_to_batch, load_external_conditioning_npz


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

    def test_load_external_conditioning_npz_random_coords_uses_target_frame_dihedrals(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            npz_path = tmpdir_path / "conditioning.npz"

            trajectory = np.asarray(
                [
                    [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                    [[10.0, 0.0, 0.0], [20.0, 0.0, 0.0]],
                    [[100.0, 0.0, 0.0], [200.0, 0.0, 0.0]],
                ],
                dtype=np.float32,
            )
            dihedrals = np.asarray(
                [
                    [[0.0, 0.0, 0.0, 0.0, 0.0]],
                    [[1.0, 1.0, 1.0, 1.0, 1.0]],
                    [[2.0, 2.0, 2.0, 2.0, 2.0]],
                ],
                dtype=np.float32,
            )
            dihedral_atom_indices = np.broadcast_to(
                np.asarray([0, 1, 0, 1], dtype=np.int64),
                (1, 5, 4),
            ).copy()
            dihedral_mask = np.ones((1, 5), dtype=bool)
            np.savez(
                npz_path,
                trajectory=trajectory,
                atom_names=np.asarray(["N", "CA"]),
                atom_residue_index=np.asarray([0, 0], dtype=np.int64),
                dihedrals=dihedrals,
                dihedral_atom_indices=dihedral_atom_indices,
                dihedral_mask=dihedral_mask,
            )

            target_data = load_external_conditioning_npz(
                npz_path,
                target_frame_idx=1,
                randomize_coords=True,
                random_seed=42,
                output_dir=tmpdir_path,
            )

            # random_seed=42 deterministically picks coordinate frame 0.
            np.testing.assert_allclose(target_data["target_atom_coords"], trajectory[0])
            np.testing.assert_allclose(target_data["dihedrals"], dihedrals[1])
            self.assertEqual(target_data["target_frame_idx"], 1)
            self.assertEqual(target_data["target_coords_frame_idx"], 0)

            target_frame_pdb = tmpdir_path / "target_conditioning_target_frame_1.pdb"
            random_frame_pdb = tmpdir_path / "target_conditioning_random_coords_frame_0.pdb"
            self.assertTrue(target_frame_pdb.exists())
            self.assertTrue(random_frame_pdb.exists())


if __name__ == "__main__":
    unittest.main()
