#!/usr/bin/env python3
"""Utilities extracted from scripts/sample_with_conditioning.py."""

from __future__ import annotations

import gc

import numpy as np
import torch

from .cli import parse_args
from .conditioning import prepare_conditioned_batch
from .constants import CLUSTER_KEY
from .data import (
    load_conditioning_label_rows,
    load_processed_frame,
    load_raw_frame,
    match_processed_record_id,
    resolve_data_path,
    resolve_labels_npz_path,
    resolve_processed_dir,
    resolve_raw_npz_path,
)
from .modeling import instantiate_and_load_model, resolve_checkpoint_path, resolve_device
from .sampling import sample_conditioned_structure

from boltz_data_pipeline.feature.featurizer import BoltzFeaturizer  # noqa: E402
from boltz_data_pipeline.tokenize.boltz_protein import BoltzTokenizer  # noqa: E402
from model.flow import LinearPath  # noqa: E402
from model.torch.sampler import EMSampler  # noqa: E402
from processor.protein_processor import ProteinDataProcessor  # noqa: E402
from utils.esm_utils import _af2_to_esm, esm_registry  # noqa: E402


def move_batch_tensors(batch: dict, device: torch.device) -> dict:
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device)
    return batch


def seed_torch_sampling(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_seed_for_index(base_seed: int | None, sample_index: int) -> int | None:
    if base_seed is None:
        return None
    return int(base_seed) + int(sample_index)


def make_sample_rng(base_seed: int | None, sample_index: int) -> np.random.Generator:
    sample_seed = sample_seed_for_index(base_seed, sample_index)
    return np.random.default_rng(sample_seed)


def load_input_frame(
    *,
    raw_npz_path,
    processed_dir,
    frame_index,
    rng,
) -> dict:
    if raw_npz_path is not None:
        print(f"Loading raw trajectory NPZ: {raw_npz_path}")
        frame_data = load_raw_frame(raw_npz_path, frame_index, rng)
    elif processed_dir is not None:
        print(
            "Raw trajectory NPZ was not found; loading coordinates and cluster labels "
            f"from processed directory: {processed_dir}"
        )
        frame_data = load_processed_frame(processed_dir, frame_index, rng)
    else:
        raise FileNotFoundError(
            "Could not resolve either a raw trajectory NPZ or a processed SimpleFold directory."
        )

    frame_data["raw_record_id"] = frame_data["record_id"]
    frame_data["record_id"] = match_processed_record_id(
        processed_dir,
        frame_data["record_id"],
        int(frame_data["frame_index"]),
    )
    return frame_data


def log_frame_selection(frame_data: dict, *, template: bool) -> None:
    original_coords = frame_data["original_coords"]
    original_cluster_labels = frame_data["original_cluster_labels"]
    frame_log_label = "Template frame" if template else "Selected frame"
    print(
        f"{frame_log_label} "
        f"position={frame_data['frame_position']} frame_index={frame_data['frame_index']} "
        f"record_id={frame_data['record_id']} atoms={original_coords.shape[0]}"
    )
    if not template:
        print(
            "Cluster label range: "
            f"{int(original_cluster_labels.min())}..{int(original_cluster_labels.max())}"
        )


def main() -> None:
    args = parse_args()
    if args.num_steps <= 0:
        raise ValueError("--num-steps must be > 0")
    if args.guidance_scale < 0:
        raise ValueError("--guidance-scale must be >= 0")
    if args.num_samples is not None and args.num_samples <= 0:
        raise ValueError("-N/--num-samples must be > 0")
    if args.dihedral_angle_bins <= 0:
        raise ValueError("--dihedral-angle-bins must be > 0")
    if args.dihedral_error_bins <= 0:
        raise ValueError("--dihedral-error-bins must be > 0")

    data_path = resolve_data_path(args.data_path)
    raw_npz_was_explicit = args.raw_npz_path is not None
    processed_dir = resolve_processed_dir(
        data_path,
        args.processed_dir,
        allow_data_path_dir=not raw_npz_was_explicit or args.data_path is not None,
        allow_default_fallback=not raw_npz_was_explicit,
    )
    raw_npz_path = resolve_raw_npz_path(data_path, args.raw_npz_path)
    labels_npz_path = resolve_labels_npz_path(args.labels_npz_path)
    checkpoint_path = resolve_checkpoint_path(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.seed is not None:
        seed_torch_sampling(args.seed)

    conditioning_label_rows = None
    if labels_npz_path is not None:
        print(f"Loading conditioning labels NPZ: {labels_npz_path}")
        conditioning_label_rows = load_conditioning_label_rows(labels_npz_path)
        if args.num_samples is not None:
            conditioning_label_rows = conditioning_label_rows[: args.num_samples]
        print(
            f"Loaded {conditioning_label_rows.shape[0]} conditioning label row(s) "
            f"with {conditioning_label_rows.shape[1]} label(s) each."
        )
    num_samples = (
        int(conditioning_label_rows.shape[0])
        if conditioning_label_rows is not None
        else int(args.num_samples or 1)
    )

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    print(f"Loading ESM model: {args.esm_model}")
    esm_model, esm_dict = esm_registry[args.esm_model]()
    esm_model = esm_model.to(device)
    esm_model.eval()
    af2_to_esm = _af2_to_esm(esm_dict).to(device)

    processor = ProteinDataProcessor(
        device=device,
        scale=args.scale,
        ref_scale=args.ref_scale,
        multiplicity=1,
        inference_multiplicity=1,
        backend="torch",
        ref_pos_mode=args.ref_pos_mode,
    )
    tokenizer = BoltzTokenizer()
    featurizer = BoltzFeaturizer()

    sample_contexts = []
    if conditioning_label_rows is not None:
        frame_data = load_input_frame(
            raw_npz_path=raw_npz_path,
            processed_dir=processed_dir,
            frame_index=args.frame_index,
            rng=make_sample_rng(args.seed, 0),
        )
        log_frame_selection(frame_data, template=True)
        batch, structure, record = prepare_conditioned_batch(
            frame_data=frame_data,
            processed_dir=processed_dir,
            tokenizer=tokenizer,
            featurizer=featurizer,
            processor=processor,
            esm_model=esm_model,
            esm_dict=esm_dict,
            af2_to_esm=af2_to_esm,
        )
        sample_contexts.append(
            {
                "sample_index": None,
                "frame_data": frame_data,
                "batch": batch,
                "structure": structure,
                "record": record,
            }
        )
    else:
        print(f"Preparing {num_samples} conditioned sample input(s).")
        for sample_index in range(num_samples):
            sample_seed = sample_seed_for_index(args.seed, sample_index)
            frame_data = load_input_frame(
                raw_npz_path=raw_npz_path,
                processed_dir=processed_dir,
                frame_index=args.frame_index,
                rng=make_sample_rng(args.seed, sample_index),
            )
            log_frame_selection(frame_data, template=False)
            batch, structure, record = prepare_conditioned_batch(
                frame_data=frame_data,
                processed_dir=processed_dir,
                tokenizer=tokenizer,
                featurizer=featurizer,
                processor=processor,
                esm_model=esm_model,
                esm_dict=esm_dict,
                af2_to_esm=af2_to_esm,
            )
            if batch[CLUSTER_KEY].shape[1] != batch["coords"].shape[1]:
                raise ValueError(
                    f"Conditioning labels have {batch[CLUSTER_KEY].shape[1]} atoms after batching, "
                    f"but model coordinates have {batch['coords'].shape[1]} atoms."
                )
            sample_contexts.append(
                {
                    "sample_index": sample_index,
                    "sample_seed": sample_seed,
                    "frame_data": frame_data,
                    "batch": move_batch_tensors(batch, torch.device("cpu")),
                    "structure": structure,
                    "record": record,
                }
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ESM features are now materialized in the batch, so the ESM model can be released
    # before the flow model and checkpoint are loaded.
    del esm_model, esm_dict, af2_to_esm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if conditioning_label_rows is not None:
        batch = sample_contexts[0]["batch"]
        labels_natoms = batch[CLUSTER_KEY].shape[1]
        coords_natoms = batch["coords"].shape[1]
        if labels_natoms != coords_natoms:
            raise ValueError(
                f"Conditioning labels have {labels_natoms} atoms after batching, "
                f"but model coordinates have {coords_natoms} atoms."
            )

    print(f"Loading flow model checkpoint: {checkpoint_path}")
    model = instantiate_and_load_model(
        architecture_config=args.architecture_config,
        checkpoint_path=checkpoint_path,
        device=device,
        prefer_ema=not args.use_non_ema_weights,
        use_mmap=not args.no_mmap_checkpoint,
    )
    flow = LinearPath()
    sampler = EMSampler(
        num_timesteps=args.num_steps,
        t_start=1e-4,
        tau=args.tau,
        log_timesteps=True,
        w_cutoff=0.99,
        guidance_scale=args.guidance_scale,
        conditioning_key=CLUSTER_KEY,
    )

    if conditioning_label_rows is None:
        print(f"Running raw/processed sampling mode: generating {num_samples} sample(s).")
        for context in sample_contexts:
            sample_seed = context["sample_seed"]
            if sample_seed is not None:
                seed_torch_sampling(sample_seed)
            batch = move_batch_tensors(context["batch"], device)
            sample_conditioned_structure(
                args=args,
                frame_data=context["frame_data"],
                raw_npz_path=raw_npz_path,
                labels_npz_path=None,
                processed_dir=processed_dir,
                checkpoint_path=checkpoint_path,
                model=model,
                flow=flow,
                sampler=sampler,
                processor=processor,
                batch=batch,
                structure=context["structure"],
                record=context["record"],
                conditioning_cluster_labels=context["frame_data"]["original_cluster_labels"],
                label_sample_index=None,
                sample_index=context["sample_index"],
                sample_seed=sample_seed,
                evaluate_against_original=True,
            )
            context["batch"] = move_batch_tensors(batch, torch.device("cpu"))
        return

    print(
        "Running labels-NPZ sampling mode: generating "
        f"{conditioning_label_rows.shape[0]} sample(s) and skipping original-structure "
        "coordinate/dihedral evaluation."
    )
    for label_sample_index, conditioning_cluster_labels in enumerate(
        conditioning_label_rows
    ):
        sample_seed = None if args.seed is None else args.seed + label_sample_index
        if sample_seed is not None:
            seed_torch_sampling(sample_seed)
        context = sample_contexts[0]
        sample_conditioned_structure(
            args=args,
            frame_data=context["frame_data"],
            raw_npz_path=raw_npz_path,
            labels_npz_path=labels_npz_path,
            processed_dir=processed_dir,
            checkpoint_path=checkpoint_path,
            model=model,
            flow=flow,
            sampler=sampler,
            processor=processor,
            batch=context["batch"],
            structure=context["structure"],
            record=context["record"],
            conditioning_cluster_labels=conditioning_cluster_labels,
            label_sample_index=label_sample_index,
            sample_index=label_sample_index,
            sample_seed=sample_seed,
            evaluate_against_original=False,
        )
