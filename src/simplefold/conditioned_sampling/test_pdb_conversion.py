from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

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


if __name__ == "__main__":
    unittest.main()
