from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from scripts import compare_conditioning_to_oracle as comparison


def check_labels_npz_mode_uses_indexed_labels_and_skips_rmsd(tmp_path: Path) -> None:
    n_residues = 300
    cluster_counts = np.full(n_residues, 2, dtype=np.int64)
    offsets = np.arange(n_residues, dtype=np.int64) * 2
    residue_label_rows = np.stack(
        [
            np.zeros(n_residues, dtype=np.int64),
            np.ones(n_residues, dtype=np.int64),
        ]
    )
    atom_label_rows = residue_label_rows + offsets[None, :]
    labels_npz_path = tmp_path / "conditioning_labels.npz"
    np.savez(
        labels_npz_path,
        labels=residue_label_rows,
        atom_idx_and_glob_cluster_id_per_frame=atom_label_rows,
    )

    sample_stem = tmp_path / "following_label_1"
    conditioned_eval_path = Path(f"{sample_stem}_conditioned_eval.npz")
    np.savez(
        conditioned_eval_path,
        conditioning_cluster_labels=atom_label_rows[1],
        atom_resids=np.arange(n_residues, dtype=np.int64),
        atom_names=np.full(n_residues, "CA"),
        residue_keys=np.asarray(
            [f"res_{resid}" for resid in range(1, n_residues + 1)]
        ),
        label_sample_index=np.asarray(1, dtype=np.int64),
    )
    Path(f"{sample_stem}_conditioned_eval.json").write_text(
        json.dumps(
            {
                "labels_npz_path": str(labels_npz_path),
                "label_sample_index": 1,
                "sample_index": 1,
                "target_reference_cif_path": None,
            }
        )
    )

    oracle_labels = residue_label_rows[1].copy()
    oracle_labels[1] = 0
    assigned_path = Path(f"{sample_stem}_assigned_clusters.npz")
    np.savez(
        assigned_path,
        labels_assigned=oracle_labels[None, :],
        cluster_counts=cluster_counts,
    )

    rmsd_guard = AssertionError("labels-NPZ mode must not execute RMSD logic")
    with (
        mock.patch.object(
            comparison,
            "strict_rmsd_pdb_paths",
            side_effect=rmsd_guard,
        ) as strict_paths,
        mock.patch.object(
            comparison,
            "evaluate_atom_selections",
            side_effect=rmsd_guard,
        ) as atom_selections,
        mock.patch.object(
            comparison,
            "load_mismatch_dihedral_data",
            side_effect=rmsd_guard,
        ) as mismatch_dihedrals,
    ):
        row, residue_rows, selection_rows, confusion_rows, dihedral_data = (
            comparison.compare_file(
                assigned_path,
                tmp_path,
                reference_pdb_path=tmp_path / "missing-reference.pdb",
            )
        )

    strict_paths.assert_not_called()
    atom_selections.assert_not_called()
    mismatch_dihedrals.assert_not_called()
    assert row["label_sample_index"] == 1
    assert row["labels_npz_path"] == str(labels_npz_path)
    assert row["sampled_pdb"] is None
    assert row["target_pdb"] is None
    assert row["mismatch_count"] == 1
    assert len(residue_rows) == len(comparison.VMD_RESIDS)
    assert len(confusion_rows) == n_residues
    assert dihedral_data is None

    selections = {item["selection_name"]: item for item in selection_rows}
    assert set(selections) == set(comparison.ERROR_SELECTION_KEYS)
    assert selections["vmd_ca_residues"]["mismatch_count"] == 1
    assert selections["all_atoms"]["mismatch_count"] == 1
    assert all(item["pdb_rmsd_angstrom"] is None for item in selection_rows)


class TestLabelsNpzComparison(unittest.TestCase):
    def test_uses_indexed_labels_and_skips_rmsd(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            check_labels_npz_mode_uses_indexed_labels_and_skips_rmsd(Path(tmp_dir))
