#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

import math
import sys
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src" / "simplefold"))

from simplefold.model.torch.sampler import A_fn


class TestAFn(unittest.TestCase):
    def test_a_fn_matches_standard_dihedral_examples(self):
        atom_indices = torch.tensor([[[0, 1, 2, 3]]])
        dihedral_mask = torch.tensor([[True]])

        coplanar = torch.tensor(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
            ]
        )
        self.assertTrue(
            torch.allclose(
                A_fn(coplanar, atom_indices, dihedral_mask),
                torch.zeros((1, 1)),
                atol=1e-6,
            )
        )

        quarter_turn = torch.tensor(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0],
            ]
        )
        self.assertTrue(
            torch.allclose(
                A_fn(quarter_turn, atom_indices, dihedral_mask),
                torch.tensor([[math.pi / 2]], dtype=quarter_turn.dtype),
                atol=1e-6,
            )
        )

    def test_a_fn_broadcasts_indices_and_zeroes_masked_entries(self):
        coords = torch.tensor(
            [
                [
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 1.0],
                ],
                [
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                ],
            ]
        )
        atom_indices = torch.tensor([[[0, 1, 2, 3], [-1, -1, -1, -1]]])
        dihedral_mask = torch.tensor([[[True, False]], [[True, False]]])

        angles = A_fn(coords, atom_indices, dihedral_mask)

        self.assertTrue(
            torch.allclose(
                angles,
                torch.tensor([[[math.pi / 2, 0.0]], [[0.0, 0.0]]], dtype=coords.dtype),
                atol=1e-6,
            )
        )

    def test_a_fn_zeroes_degenerate_active_dihedrals(self):
        coords = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ]
        )

        angles = A_fn(
            coords,
            torch.tensor([[[0, 1, 2, 3]]]),
            torch.tensor([[True]]),
        )

        self.assertTrue(torch.allclose(angles, torch.zeros((1, 1)), atol=1e-6))

    def test_a_fn_rejects_active_out_of_bounds_indices(self):
        with self.assertRaisesRegex(IndexError, "Active dihedral atom index out of bounds."):
            A_fn(
                torch.zeros(4, 3),
                torch.tensor([[[0, 1, 2, 4]]]),
                torch.tensor([[True]]),
            )


if __name__ == "__main__":
    unittest.main()
