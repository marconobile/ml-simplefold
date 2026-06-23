#!/usr/bin/env python3
"""Utilities extracted from scripts/sample_with_conditioning.py."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .constants import CLUSTER_KEY
from .data import build_structure_and_record_from_raw, find_processed_paths

from boltz_data_pipeline.feature.featurizer import BoltzFeaturizer  # noqa: E402
from boltz_data_pipeline.tokenize.boltz_protein import BoltzTokenizer  # noqa: E402
from boltz_data_pipeline.types import Input, Record, Structure  # noqa: E402
from processor.protein_processor import ProteinDataProcessor  # noqa: E402
from utils.datamodule_utils import collate, extract_sequence_from_tokens  # noqa: E402
from utils.trajectory_npz_utils import assert_no_conformer_coordinate_leak  # noqa: E402


def pad_cluster_labels(
    cluster_labels: np.ndarray | torch.Tensor,
    num_model_atoms: int,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    labels_tensor = cluster_labels_to_tensor(cluster_labels, device=device)
    if labels_tensor.shape[0] > num_model_atoms:
        raise ValueError(
            f"Conditioning labels have {labels_tensor.shape[0]} atoms, but the "
            f"featurized model input has only {num_model_atoms} atoms."
        )

    padded_cluster_labels = torch.full(
        (num_model_atoms,),
        -1,
        dtype=torch.long,
        device=device,
    )
    padded_cluster_labels[: labels_tensor.shape[0]] = labels_tensor
    return padded_cluster_labels

def cluster_labels_to_tensor(
    cluster_labels: np.ndarray | torch.Tensor,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    if isinstance(cluster_labels, torch.Tensor):
        labels_tensor = cluster_labels.detach().to(dtype=torch.long, device=device)
    else:
        labels_tensor = torch.as_tensor(
            np.asarray(cluster_labels, dtype=np.int64),
            dtype=torch.long,
            device=device,
        )

    if labels_tensor.ndim != 1:
        raise ValueError(
            f"Conditioning labels must be one-dimensional per sample, got "
            f"{tuple(labels_tensor.shape)}."
        )
    if labels_tensor.numel() > 0 and int(labels_tensor.min().item()) < -1:
        raise ValueError("Conditioning labels contain labels below -1.")
    return labels_tensor.clone()

def exact_cluster_labels(
    cluster_labels: np.ndarray | torch.Tensor,
    num_model_atoms: int,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    labels_tensor = cluster_labels_to_tensor(cluster_labels, device=device)
    if labels_tensor.shape[0] != num_model_atoms:
        raise ValueError(
            f"Labels NPZ row has {labels_tensor.shape[0]} atom labels, but the "
            f"featurized model input has {num_model_atoms} atoms. In "
            "--labels-npz-path mode, the model receives the labels row exactly, "
            "so the lengths must match."
        )
    return labels_tensor

def set_batch_cluster_labels(
    batch: dict[str, Any],
    cluster_labels: np.ndarray,
    *,
    exact: bool = False,
) -> None:
    num_model_atoms = int(batch["coords"].shape[1])
    batch_size = int(batch["coords"].shape[0])
    if batch_size != 1:
        raise ValueError(f"Expected a single-structure batch, got batch size {batch_size}.")
    if exact:
        labels_tensor = cluster_labels_to_tensor(
            cluster_labels,
            device=batch["coords"].device,
        )
        if labels_tensor.shape[0] == num_model_atoms:
            label_tensor = labels_tensor
        else:
            atom_pad_mask = batch.get("atom_pad_mask")
            num_valid_atoms = (
                int(atom_pad_mask[0].sum().item())
                if atom_pad_mask is not None
                else num_model_atoms
            )
            if labels_tensor.shape[0] != num_valid_atoms:
                raise ValueError(
                    f"Labels NPZ row has {labels_tensor.shape[0]} atom labels, "
                    f"but the featurized model input has {num_valid_atoms} valid "
                    f"atoms and {num_model_atoms} padded atoms."
                )
            label_tensor = pad_cluster_labels(
                labels_tensor,
                num_model_atoms,
                device=batch["coords"].device,
            )
    else:
        label_tensor = pad_cluster_labels(
            cluster_labels,
            num_model_atoms,
            device=batch["coords"].device,
        )
    batch[CLUSTER_KEY] = label_tensor.unsqueeze(0)

def validate_cluster_labels_for_model(
    cluster_labels: np.ndarray,
    model: torch.nn.Module,
) -> None:
    non_padding = cluster_labels[cluster_labels >= 0]
    if non_padding.size == 0:
        return
    max_supported = getattr(model, "max_possible_global_clu_idx", None)
    if max_supported is None:
        return
    max_label = int(non_padding.max())
    if max_label > int(max_supported):
        raise ValueError(
            f"Conditioning label {max_label} exceeds the model-supported maximum "
            f"global cluster id {int(max_supported)}."
        )

def prepare_conditioned_batch(
    frame_data: dict[str, Any],
    processed_dir: Path | None,
    tokenizer: BoltzTokenizer,
    featurizer: BoltzFeaturizer,
    processor: ProteinDataProcessor,
    esm_model: torch.nn.Module,
    esm_dict: dict[str, Any],
    af2_to_esm: torch.Tensor,
) -> tuple[dict[str, Any], Structure, Record]:
    structure_path, record_path, _tokenized_path = find_processed_paths(
        processed_dir,
        frame_data["record_id"],
        int(frame_data["frame_index"]),
    )

    if structure_path is not None and record_path is not None:
        structure = Structure.load(structure_path)
        assert_no_conformer_coordinate_leak(structure.atoms, str(structure_path))
        with record_path.open() as f:
            record_dict = json.load(f)
        # Re-tokenize from the loaded structure so stale tokens cannot carry
        # leaked frame coordinates in atom["conformer"].
        tokenized = tokenizer.tokenize(Input(structure, {}))
    else:
        structure, record_dict = build_structure_and_record_from_raw(frame_data)
        tokenized = tokenizer.tokenize(Input(structure, {}))

    sequence = extract_sequence_from_tokens(tokenized)
    features = featurizer.process(tokenized)
    features["aa_seq"] = sequence
    features["record"] = record_dict
    features["num_repeats"] = torch.tensor(1)
    features["max_num_tokens"] = torch.tensor(len(tokenized.tokens), dtype=torch.long)
    features["cropped_num_tokens"] = torch.tensor(len(tokenized.tokens), dtype=torch.long)
    num_model_atoms = int(features["ref_pos"].shape[0])
    features[CLUSTER_KEY] = pad_cluster_labels(
        frame_data["original_cluster_labels"],
        num_model_atoms,
    )

    batch = collate([features])
    batch = processor.preprocess_inference(
        batch,
        esm_model=esm_model,
        esm_dict=esm_dict,
        af2_to_esm=af2_to_esm,
    )
    return batch, structure, Record(**record_dict)
