from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from simplefold.conditioned_sampling.dihedrals import (
    symmetry_correct_dihedral_errors,
)
from simplefold.conditioned_sampling.outputs import write_dihedral_histograms


class TestSymmetryCorrectDihedralErrors(unittest.TestCase):
    def test_corrects_only_the_requested_residue_dihedral_pairs(self):
        residue_names = np.asarray(["PHE", "TYR", "ASP", "VAL", "LEU"])
        dihedral_keys = ["chi1", "chi2"]
        differences = np.radians(
            np.asarray(
                [
                    [120.0, 120.0],
                    [120.0, -120.0],
                    [120.0, 100.0],
                    [-120.0, 120.0],
                    [120.0, 120.0],
                ],
                dtype=np.float32,
            )
        )

        signed, absolute = symmetry_correct_dihedral_errors(
            differences,
            residue_names,
            dihedral_keys,
        )

        np.testing.assert_allclose(
            absolute,
            np.asarray(
                [
                    [120.0, 60.0],
                    [120.0, 60.0],
                    [120.0, 80.0],
                    [60.0, 120.0],
                    [120.0, 120.0],
                ]
            ),
            atol=1e-4,
        )
        self.assertAlmostEqual(float(signed[1, 1]), -60.0, places=4)
        self.assertAlmostEqual(float(signed[3, 0]), -60.0, places=4)

    def test_rejects_residue_shape_mismatch(self):
        with self.assertRaisesRegex(ValueError, "one entry per residue"):
            symmetry_correct_dihedral_errors(
                np.zeros((2, 1), dtype=np.float32),
                np.asarray(["PHE"]),
                ["chi2"],
            )

    def test_standard_error_histogram_uses_corrected_error(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            artifacts = write_dihedral_histograms(
                output_dir=output_dir,
                output_stem="sample",
                original_dihedrals=np.zeros((1, 1), dtype=np.float32),
                sampled_dihedrals=np.radians(
                    np.asarray([[120.0]], dtype=np.float32)
                ),
                dihedral_diff_rad=np.radians(
                    np.asarray([[120.0]], dtype=np.float32)
                ),
                dihedral_mask=np.ones((1, 1), dtype=bool),
                dihedral_keys=["chi2"],
                residue_names=np.asarray(["PHE"]),
                angle_bins=4,
                error_bins=4,
            )
            with Path(artifacts["error_histogram_csv"]).open() as handle:
                rows = list(csv.DictReader(handle))

        corrected_bin = next(
            row
            for row in rows
            if row["dihedral_key"] == "chi2"
            and row["series"] == "abs_error_deg"
            and float(row["bin_left_deg"]) == 45.0
        )
        self.assertEqual(int(corrected_bin["count"]), 1)


if __name__ == "__main__":
    unittest.main()
