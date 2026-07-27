from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

import torch
from omegaconf import OmegaConf

from simplefold.conditioned_sampling import modeling


class TinyConditionedModel(torch.nn.Module):
    def __init__(self, use_conditioning_masking: bool) -> None:
        super().__init__()
        self.max_possible_global_clu_idx = 2
        self.pad_idx = 3
        self.use_conditioning_masking = use_conditioning_masking
        self.mask_idx = 4 if use_conditioning_masking else None
        num_embeddings = 5 if use_conditioning_masking else 4
        self.cluster_embeddings = torch.nn.Embedding(
            num_embeddings,
            3,
            padding_idx=self.pad_idx,
        )
        self.backbone = torch.nn.Linear(3, 2)
        self.cluster_mlm_head = (
            torch.nn.Linear(2, 3) if use_conditioning_masking else None
        )


def make_checkpoint_state(
    *,
    use_conditioning_masking: bool,
) -> dict[str, torch.Tensor]:
    source = TinyConditionedModel(use_conditioning_masking)
    with torch.no_grad():
        for tensor_index, tensor in enumerate(source.state_dict().values()):
            values = torch.arange(tensor.numel(), dtype=tensor.dtype).reshape(
                tensor.shape
            )
            tensor.copy_(values + tensor_index * 100)
    return {
        key: value.detach().clone()
        for key, value in source.state_dict().items()
    }


class TestConditionedModelLoading(unittest.TestCase):
    def test_mask_token_architecture_is_inferred_and_every_weight_is_exact(self):
        expected_state = make_checkpoint_state(use_conditioning_masking=True)
        prefixed_state = {
            f"model_ema.module.{key}": value.clone()
            for key, value in expected_state.items()
        }
        model_cfg = OmegaConf.create({"use_conditioning_masking": False})

        def instantiate(cfg):
            return TinyConditionedModel(bool(cfg.use_conditioning_masking))

        with (
            mock.patch.object(modeling.OmegaConf, "load", return_value=model_cfg),
            mock.patch.object(
                modeling,
                "load_checkpoint",
                return_value={"state_dict": prefixed_state},
            ),
            mock.patch.object(
                modeling.hydra.utils,
                "instantiate",
                side_effect=instantiate,
            ),
        ):
            loaded = modeling.instantiate_and_load_model(
                architecture_config=Path("unused.yaml"),
                checkpoint_path=Path("checkpoint.ckpt"),
                device=torch.device("cpu"),
                prefer_ema=True,
                use_mmap=True,
            )

        self.assertTrue(loaded.use_conditioning_masking)
        self.assertEqual(loaded.mask_idx, 4)
        self.assertEqual(loaded.cluster_embeddings.num_embeddings, 5)
        self.assertFalse(loaded.training)
        for key, expected in expected_state.items():
            actual = loaded.state_dict()[key]
            self.assertEqual(actual.dtype, expected.dtype)
            self.assertTrue(torch.equal(actual, expected), key)

    def test_exact_weight_verification_rejects_any_changed_tensor(self):
        checkpoint_state = make_checkpoint_state(use_conditioning_masking=True)
        model = TinyConditionedModel(use_conditioning_masking=True)
        model.load_state_dict(checkpoint_state, strict=True)
        with torch.no_grad():
            model.cluster_embeddings.weight[0, 0].add_(1)

        with self.assertRaisesRegex(RuntimeError, "not loaded exactly"):
            modeling.assert_checkpoint_weights_loaded_exactly(
                model,
                checkpoint_state,
            )

    def test_unmasked_checkpoint_disables_mask_token_architecture(self):
        checkpoint_state = make_checkpoint_state(use_conditioning_masking=False)
        model_cfg = OmegaConf.create({"use_conditioning_masking": True})

        modeling.configure_conditioning_masking_from_checkpoint(
            model_cfg,
            checkpoint_state,
        )

        self.assertFalse(model_cfg.use_conditioning_masking)


if __name__ == "__main__":
    unittest.main()
