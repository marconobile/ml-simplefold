from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from simplefold.conditioned_sampling import workflow
from simplefold.conditioned_sampling.constants import CLUSTER_KEY
from simplefold.conditioned_sampling.data import RawNpzFrameLoader


def write_raw_npz(path: Path) -> None:
    np.savez_compressed(
        path,
        trajectory=np.arange(18, dtype=np.float32).reshape(3, 2, 3),
        frame_indices=np.asarray([20, 10, 30], dtype=np.int64),
        sample_id=np.asarray("ordered-test"),
        atom_names=np.asarray(["N", "CA"]),
        atom_resids=np.asarray([1, 1], dtype=np.int64),
        **{
            CLUSTER_KEY: np.asarray([[0, 1], [2, 3], [4, 5]], dtype=np.int64),
            "dihedrals": np.arange(6, dtype=np.float32).reshape(3, 1, 2),
            "dihedral_atom_indices": np.zeros((1, 2, 4), dtype=np.int64),
            "dihedral_mask": np.ones((1, 2), dtype=bool),
            "dihedral_keys": np.asarray(["phi", "psi"]),
        },
    )


class FakeEsmModel:
    def to(self, _device):
        return self

    def eval(self):
        return self


class TestRawNpzFrameLoader(unittest.TestCase):
    def test_loads_frames_by_npz_position_without_sorting_frame_indices(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            raw_npz_path = Path(tmp_dir) / "raw.npz"
            write_raw_npz(raw_npz_path)
            loader = RawNpzFrameLoader(raw_npz_path)
            try:
                frames = [loader.load_frame(position) for position in range(len(loader))]
            finally:
                loader.close()

        self.assertEqual([frame["frame_position"] for frame in frames], [0, 1, 2])
        self.assertEqual([frame["frame_index"] for frame in frames], [20, 10, 30])
        np.testing.assert_array_equal(
            frames[1]["original_coords"],
            np.arange(18, dtype=np.float32).reshape(3, 2, 3)[1],
        )


class TestOrderedSamplingWorkflow(unittest.TestCase):
    def test_minus_one_samples_every_raw_frame_in_file_order(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            raw_npz_path = tmp_path / "raw.npz"
            write_raw_npz(raw_npz_path)
            args = argparse.Namespace(
                num_steps=1,
                guidance_scale=1.0,
                num_samples=-1,
                dihedral_angle_bins=4,
                dihedral_error_bins=4,
                data_path=raw_npz_path,
                raw_npz_path=raw_npz_path,
                processed_dir=None,
                labels_npz_path=None,
                output_dir=tmp_path / "output",
                seed=100,
                device="cpu",
                esm_model="fake",
                scale=16.0,
                ref_scale=5.0,
                ref_pos_mode="zero",
                architecture_config=tmp_path / "architecture.yaml",
                use_non_ema_weights=False,
                no_mmap_checkpoint=False,
                tau=0.3,
                frame_index=None,
                conditioned_eval_pdb_base_path=tmp_path,
            )
            batch = {
                "coords": torch.zeros((1, 2, 3)),
                CLUSTER_KEY: torch.zeros((1, 2), dtype=torch.long),
            }

            with (
                mock.patch.object(workflow, "parse_args", return_value=args),
                mock.patch.object(workflow, "resolve_data_path", return_value=raw_npz_path),
                mock.patch.object(workflow, "resolve_processed_dir", return_value=None),
                mock.patch.object(workflow, "resolve_raw_npz_path", return_value=raw_npz_path),
                mock.patch.object(workflow, "resolve_labels_npz_path", return_value=None),
                mock.patch.object(
                    workflow,
                    "resolve_checkpoint_path",
                    return_value=tmp_path / "last.ckpt",
                ),
                mock.patch.object(
                    workflow,
                    "resolve_device",
                    return_value=torch.device("cpu"),
                ),
                mock.patch.dict(
                    workflow.esm_registry,
                    {"fake": lambda: (FakeEsmModel(), {})},
                ),
                mock.patch.object(
                    workflow,
                    "_af2_to_esm",
                    return_value=torch.zeros(1),
                ),
                mock.patch.object(workflow, "ProteinDataProcessor", return_value=mock.Mock()),
                mock.patch.object(workflow, "BoltzTokenizer", return_value=mock.Mock()),
                mock.patch.object(workflow, "BoltzFeaturizer", return_value=mock.Mock()),
                mock.patch.object(
                    workflow,
                    "prepare_conditioned_batch",
                    return_value=(batch, object(), object()),
                ) as prepare_batch,
                mock.patch.object(
                    workflow,
                    "instantiate_and_load_model",
                    return_value=mock.Mock(),
                ),
                mock.patch.object(workflow, "LinearPath", return_value=mock.Mock()),
                mock.patch.object(workflow, "EMSampler", return_value=mock.Mock()),
                mock.patch.object(workflow, "sample_conditioned_structure") as sample,
            ):
                workflow.main()

        self.assertEqual(prepare_batch.call_count, 1)
        self.assertEqual(sample.call_count, 3)
        self.assertEqual(
            [call.kwargs["frame_data"]["frame_position"] for call in sample.call_args_list],
            [0, 1, 2],
        )
        self.assertEqual(
            [call.kwargs["frame_data"]["frame_index"] for call in sample.call_args_list],
            [20, 10, 30],
        )
        self.assertEqual(
            [call.kwargs["sample_seed"] for call in sample.call_args_list],
            [100, 101, 102],
        )


if __name__ == "__main__":
    unittest.main()
