from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import convert_conditioned_eval_cifs_to_pdb as converter_script
from simplefold.conditioned_sampling import pdb_conversion


class TestEnsureConditionedEvalPdbs(unittest.TestCase):
    def test_reads_residue_names_in_topology_order(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            pdb_path = Path(tmp_dir) / "target.pdb"
            pdb_path.write_text(
                "ATOM      1  N   PHE A   1      10.000  11.000  12.000\n"
                "ATOM      2  CA  PHE A   1      11.000  12.000  13.000\n"
                "ATOM      3  N   VAL A   2      12.000  13.000  14.000\n"
                "ATOM      4  CA  VAL A   2      13.000  14.000  15.000\n"
            )

            residue_names = pdb_conversion.read_pdb_residue_names(pdb_path)

        self.assertEqual(residue_names.tolist(), ["PHE", "VAL"])

    def test_current_file_only_also_converts_and_requires_target_reference(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            sampled_cif_path = output_dir / "example_conditioned_eval_sampled.cif"
            target_cif_path = output_dir / "example_conditioned_eval_target_reference.cif"
            sampled_cif_path.touch()
            target_cif_path.touch()

            def create_pdbs(
                base_path: Path,
                additional_cif_paths: tuple[Path, ...] = (),
                chirality_log_path: Path | None = None,
            ) -> None:
                for cif_path in (base_path, *additional_cif_paths):
                    cif_path.with_suffix(".pdb").touch()

            with mock.patch.object(
                pdb_conversion,
                "run_conditioned_eval_cif_to_pdb_converter",
                side_effect=create_pdbs,
            ) as converter:
                sampled_pdb_path = pdb_conversion.ensure_conditioned_eval_sampled_pdb(
                    sampled_cif_path=sampled_cif_path,
                    configured_base_path=output_dir,
                    output_dir=output_dir,
                    current_file_only=True,
                    additional_required_cif_paths=(target_cif_path,),
                )

            self.assertEqual(sampled_pdb_path, sampled_cif_path.with_suffix(".pdb"))
            self.assertTrue(target_cif_path.with_suffix(".pdb").is_file())
            converter.assert_called_once_with(
                sampled_cif_path,
                additional_cif_paths=(target_cif_path,),
                chirality_log_path=None,
            )

    def test_missing_required_target_pdb_raises(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            sampled_cif_path = output_dir / "example_conditioned_eval_sampled.cif"
            target_cif_path = output_dir / "example_conditioned_eval_target_reference.cif"
            sampled_cif_path.touch()
            target_cif_path.touch()

            def create_only_sampled_pdb(
                base_path: Path,
                additional_cif_paths: tuple[Path, ...] = (),
                chirality_log_path: Path | None = None,
            ) -> None:
                if base_path == sampled_cif_path:
                    sampled_cif_path.with_suffix(".pdb").touch()

            with mock.patch.object(
                pdb_conversion,
                "run_conditioned_eval_cif_to_pdb_converter",
                side_effect=create_only_sampled_pdb,
            ):
                with self.assertRaisesRegex(
                    FileNotFoundError,
                    "target_reference\\.pdb",
                ):
                    pdb_conversion.ensure_conditioned_eval_sampled_pdb(
                        sampled_cif_path=sampled_cif_path,
                        configured_base_path=output_dir,
                        output_dir=output_dir,
                        current_file_only=True,
                        additional_required_cif_paths=(target_cif_path,),
                    )


class TestChiralityReports(unittest.TestCase):
    def test_records_pre_and_post_flip_counts(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            pdb_path = Path(tmp_dir) / "sample.pdb"
            pdb_path.touch()
            before = [
                {"chirality": "D"},
                {"chirality": "D"},
                {"chirality": "L"},
            ]
            after = [
                {"chirality": "L"},
                {"chirality": "L"},
                {"chirality": "D"},
            ]

            with (
                mock.patch.object(
                    converter_script,
                    "detect_chirality",
                    side_effect=[before, after],
                ),
                mock.patch.object(
                    converter_script,
                    "flip_pdb_coordinates",
                ) as flip,
            ):
                result = converter_script.ensure_l_chirality(pdb_path)

        self.assertTrue(result.activated)
        self.assertEqual((result.l_count_before, result.d_count_before), (1, 2))
        self.assertEqual((result.l_count_after, result.d_count_after), (2, 1))
        flip.assert_called_once_with(str(pdb_path), str(pdb_path))

    def test_upserts_structure_rows_and_writes_summary(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_root = Path(tmp_dir)
            first_path = output_root / "first_type" / "first.pdb"
            second_path = output_root / "second_type" / "second.pdb"
            log_path = output_root / "ensure_l_chirality.tsv"
            initial_results = [
                converter_script.ChiralityResult(
                    first_path,
                    True,
                    1,
                    2,
                    2,
                    1,
                ),
                converter_script.ChiralityResult(
                    second_path,
                    False,
                    3,
                    0,
                    3,
                    0,
                ),
            ]

            counts = converter_script.write_chirality_reports(
                initial_results,
                log_path,
            )
            replacement = converter_script.ChiralityResult(
                first_path,
                False,
                4,
                0,
                4,
                0,
            )
            updated_counts = converter_script.write_chirality_reports(
                [replacement],
                log_path,
            )

            with log_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            summary = converter_script.chirality_summary_path(log_path).read_text()

        self.assertEqual(counts, (1, 2))
        self.assertEqual(updated_counts, (0, 2))
        self.assertEqual(
            [row["structure"] for row in rows],
            ["first_type/first.pdb", "second_type/second.pdb"],
        )
        self.assertEqual(len(rows), 2)
        self.assertIn("activated_over_total\t0/2\n", summary)


if __name__ == "__main__":
    unittest.main()
