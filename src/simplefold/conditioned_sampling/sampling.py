#!/usr/bin/env python3
"""Utilities extracted from scripts/sample_with_conditioning.py."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .conditioning import set_batch_cluster_labels, validate_cluster_labels_for_model
from .constants import CLUSTER_KEY
from .evaluation import evaluate_coordinate_alignment, evaluate_sample_dihedrals
from .outputs import (
    format_optional_float,
    output_stem_for_sample,
    write_atomwise_csv,
    write_dihedral_csv,
    write_dihedral_histograms,
    write_report,
)
from .pdb_conversion import (
    ensure_conditioned_eval_sampled_pdb,
    validate_sampled_pdb_matches_coords,
)

from boltz_data_pipeline.types import Record, Structure  # noqa: E402
from model.flow import LinearPath  # noqa: E402
from model.torch.sampler import EMSampler  # noqa: E402
from processor.protein_processor import ProteinDataProcessor  # noqa: E402
from utils.boltz_utils import process_structure, save_structure  # noqa: E402


def prepare_sampling_conditioning(
    *,
    batch: dict[str, Any],
    model: torch.nn.Module,
    conditioning_cluster_labels: np.ndarray,
    exact_labels: bool,
) -> tuple[np.ndarray, np.ndarray]:
    conditioning_cluster_labels = np.asarray(
        conditioning_cluster_labels,
        dtype=np.int64,
    ).copy()
    validate_cluster_labels_for_model(conditioning_cluster_labels, model)
    set_batch_cluster_labels(
        batch,
        conditioning_cluster_labels,
        exact=exact_labels,
    )
    model_conditioning_cluster_labels = (
        batch[CLUSTER_KEY][0].detach().cpu().numpy().astype(np.int64, copy=False)
    )
    return conditioning_cluster_labels, model_conditioning_cluster_labels


def run_model_sampler(
    *,
    model: torch.nn.Module,
    flow: LinearPath,
    sampler: EMSampler,
    processor: ProteinDataProcessor,
    batch: dict[str, Any],
) -> dict[str, Any]:
    with torch.no_grad():
        noise = torch.randn_like(batch["coords"])
        out_dict = sampler.sample(model, flow, noise, batch)
        out_dict = processor.postprocess(out_dict, batch)
        sampled_coords_full_tensor = out_dict["denoised_coords"][0].detach().cpu()
        atom_mask_full_tensor = batch["atom_pad_mask"][0].detach().cpu()
        sampled_coords_full = sampled_coords_full_tensor.numpy().astype(np.float32)
        atom_mask_full = atom_mask_full_tensor.numpy().astype(bool)
        sampled_coords = sampled_coords_full[atom_mask_full]

    return {
        "sampled_coords_full_tensor": sampled_coords_full_tensor,
        "atom_mask_full_tensor": atom_mask_full_tensor,
        "sampled_coords_full": sampled_coords_full,
        "atom_mask_full": atom_mask_full,
        "sampled_coords": sampled_coords,
    }


def sample_conditioned_structure(
    *,
    args: argparse.Namespace,
    frame_data: dict[str, Any],
    raw_npz_path: Path | None,
    labels_npz_path: Path | None,
    processed_dir: Path | None,
    checkpoint_path: Path,
    model: torch.nn.Module,
    flow: LinearPath,
    sampler: EMSampler,
    processor: ProteinDataProcessor,
    batch: dict[str, Any],
    structure: Structure,
    record: Record,
    conditioning_cluster_labels: np.ndarray,
    label_sample_index: int | None,
    sample_index: int | None,
    sample_seed: int | None,
    evaluate_against_original: bool,
) -> dict[str, Any]:
    conditioning_cluster_labels, model_conditioning_cluster_labels = (
        prepare_sampling_conditioning(
            batch=batch,
            model=model,
            conditioning_cluster_labels=conditioning_cluster_labels,
            exact_labels=labels_npz_path is not None,
        )
    )
    template_coords = frame_data["original_coords"]
    print(
        "Sampling conditioned structure"
        + (
            ""
            if label_sample_index is None
            else f" for labels row {label_sample_index + 1}"
        )
        + f" with label range {int(conditioning_cluster_labels.min())}.."
        f"{int(conditioning_cluster_labels.max())}"
    )

    sample_outputs = run_model_sampler(
        model=model,
        flow=flow,
        sampler=sampler,
        processor=processor,
        batch=batch,
    )
    sampled_coords_full_tensor = sample_outputs["sampled_coords_full_tensor"]
    atom_mask_full_tensor = sample_outputs["atom_mask_full_tensor"]
    sampled_coords_full = sample_outputs["sampled_coords_full"]
    atom_mask_full = sample_outputs["atom_mask_full"]
    sampled_coords = sample_outputs["sampled_coords"]

    if sampled_coords.shape != template_coords.shape:
        raise ValueError(
            f"Unpadded sampled coordinates have shape {sampled_coords.shape}, "
            f"but the template structure has shape {template_coords.shape}."
        )

    global_rmsd = None
    atomwise_rmsd = None
    aligned_sampled_coords = None
    atom_mask = None
    target_cif_path = None
    target_pdb_path = None
    raw_sampled_cif_path = None
    atom_csv_path = None
    original_dihedrals = None
    selected_sample = None

    if evaluate_against_original:
        original_coords = frame_data["original_coords"]
        selected_sample = frame_data["selected_sample"]
        original_dihedrals = frame_data["original_dihedrals"]
        alignment = evaluate_coordinate_alignment(
            sampled_coords,
            original_coords,
        )
        atom_mask = alignment["atom_mask"]
        aligned_sampled_coords = alignment["aligned_sampled_coords"]
        global_rmsd = alignment["global_rmsd"]
        atomwise_rmsd = alignment["atomwise_rmsd"]
        print(f"Aligned RMSD: {global_rmsd:.4f} A")
        sampled_output_coord = torch.as_tensor(
            aligned_sampled_coords,
            dtype=torch.float32,
        )
        sampled_output_mask = torch.ones(
            aligned_sampled_coords.shape[0],
            dtype=torch.bool,
        )
    else:
        original_coords = None
        print(
            "Skipping coordinate and dihedral comparison because --labels-npz-path "
            "samples do not have an original structure target."
        )
        sampled_output_coord = sampled_coords_full_tensor
        sampled_output_mask = atom_mask_full_tensor

    dihedral_summary = None
    sampled_dihedrals = None
    sampled_raw_dihedrals = None
    sampled_aligned_dihedrals = None
    original_recomputed_dihedrals = None
    dihedral_diff_rad = None
    dihedral_abs_error_deg = None
    sampled_pdb_coords = None

    output_stem = output_stem_for_sample(
        frame_data["record_id"],
        label_sample_index,
        sample_index,
    )
    metrics_path = args.output_dir / f"{output_stem}.json"
    arrays_path = args.output_dir / f"{output_stem}.npz"
    sampled_cif_path = args.output_dir / f"{output_stem}_sampled.cif"
    report_path = args.output_dir / f"{output_stem}_report.txt"
    dihedral_csv_path = None
    dihedral_histogram_artifacts = None

    if evaluate_against_original:
        atom_csv_path = args.output_dir / f"{output_stem}_atomwise_rmsd.csv"
        dihedral_csv_path = args.output_dir / f"{output_stem}_dihedrals.csv"
        raw_sampled_cif_path = args.output_dir / f"{output_stem}_sampled_raw.cif"
        target_cif_path = args.output_dir / f"{output_stem}_target_reference.cif"

    sampled_structure = process_structure(
        deepcopy(structure),
        sampled_output_coord,
        sampled_output_mask,
        record,
    )
    save_structure(
        sampled_structure,
        args.output_dir,
        f"{output_stem}_sampled",
        output_format="mmcif",
    )

    if evaluate_against_original:
        raw_sampled_structure = process_structure(
            deepcopy(structure),
            sampled_coords_full_tensor,
            atom_mask_full_tensor,
            record,
        )
        save_structure(
            raw_sampled_structure,
            args.output_dir,
            f"{output_stem}_sampled_raw",
            output_format="mmcif",
        )
        target_structure = process_structure(
            deepcopy(structure),
            torch.as_tensor(original_coords, dtype=torch.float32),
            torch.ones(original_coords.shape[0], dtype=torch.bool),
            record,
        )
        save_structure(
            target_structure,
            args.output_dir,
            f"{output_stem}_target_reference",
            output_format="mmcif",
        )

    sampled_pdb_path = ensure_conditioned_eval_sampled_pdb(
        sampled_cif_path=sampled_cif_path,
        configured_base_path=args.conditioned_eval_pdb_base_path,
        output_dir=args.output_dir,
        current_file_only=(labels_npz_path is not None or args.num_samples == -1),
        additional_required_cif_paths=(
            (target_cif_path,) if target_cif_path is not None else ()
        ),
    )
    if target_cif_path is not None:
        target_pdb_path = target_cif_path.with_suffix(".pdb")
    sampled_coords_for_pdb_validation = (
        aligned_sampled_coords if evaluate_against_original else sampled_coords
    )
    sampled_pdb_coords = validate_sampled_pdb_matches_coords(
        sampled_pdb_path,
        sampled_coords_for_pdb_validation,
    )

    if evaluate_against_original and original_dihedrals is not None:
        dihedral_evaluation = evaluate_sample_dihedrals(
            sampled_pdb_coords=sampled_pdb_coords,
            sampled_coords=sampled_coords,
            aligned_sampled_coords=aligned_sampled_coords,
            selected_sample=selected_sample,
            original_dihedrals=original_dihedrals,
            dihedral_atom_indices=frame_data["dihedral_atom_indices"],
            dihedral_mask=frame_data["dihedral_mask"],
            dihedral_keys=frame_data["dihedral_keys"],
            sampled_pdb_path=sampled_pdb_path,
        )
        dihedral_summary = dihedral_evaluation["dihedral_summary"]
        sampled_dihedrals = dihedral_evaluation["sampled_dihedrals"]
        sampled_raw_dihedrals = dihedral_evaluation["sampled_raw_dihedrals"]
        sampled_aligned_dihedrals = dihedral_evaluation["sampled_aligned_dihedrals"]
        original_recomputed_dihedrals = dihedral_evaluation["original_recomputed_dihedrals"]
        dihedral_diff_rad = dihedral_evaluation["dihedral_diff_rad"]
        dihedral_abs_error_deg = dihedral_evaluation["dihedral_abs_error_deg"]
        print(
            "Dihedral MAE from converted sampled PDB: "
            f"{format_optional_float(dihedral_summary['mae_deg'], ' deg')} "
            f"over {dihedral_summary['count']} angles"
        )
    elif evaluate_against_original:
        print("Skipping dihedral comparison because no raw trajectory dihedrals were available.")

    if evaluate_against_original and original_dihedrals is not None and dihedral_summary is not None:
        dihedral_histogram_artifacts = write_dihedral_histograms(
            output_dir=args.output_dir,
            output_stem=output_stem,
            original_dihedrals=original_dihedrals,
            sampled_dihedrals=sampled_dihedrals,
            dihedral_diff_rad=dihedral_diff_rad,
            dihedral_mask=frame_data["dihedral_mask"],
            dihedral_keys=frame_data["dihedral_keys"],
            angle_bins=args.dihedral_angle_bins,
            error_bins=args.dihedral_error_bins,
        )
        dihedral_summary["histograms"] = dihedral_histogram_artifacts

    metrics = {
        "raw_npz_path": str(raw_npz_path) if raw_npz_path is not None else None,
        "labels_npz_path": str(labels_npz_path) if labels_npz_path is not None else None,
        "processed_dir": str(processed_dir) if processed_dir is not None else None,
        "checkpoint_path": str(checkpoint_path),
        "sampled_cif_path": str(sampled_cif_path),
        "sampled_pdb_path": str(sampled_pdb_path),
        "raw_sampled_cif_path": (
            str(raw_sampled_cif_path) if raw_sampled_cif_path is not None else None
        ),
        "target_reference_cif_path": (
            str(target_cif_path) if target_cif_path is not None else None
        ),
        "target_reference_pdb_path": (
            str(target_pdb_path) if target_pdb_path is not None else None
        ),
        "record_id": frame_data["record_id"],
        "raw_record_id": frame_data.get("raw_record_id"),
        "sample_id": frame_data["sample_id"],
        "frame_position": int(frame_data["frame_position"]),
        "frame_index": int(frame_data["frame_index"]),
        "label_sample_index": label_sample_index,
        "sample_index": sample_index,
        "seed": sample_seed,
        "base_seed": args.seed,
        "num_steps": args.num_steps,
        "tau": args.tau,
        "num_atoms": int(sampled_coords.shape[0]),
        "global_rmsd": global_rmsd,
        "atomwise_rmsd_mean": (
            float(np.nanmean(atomwise_rmsd)) if atomwise_rmsd is not None else None
        ),
        "atomwise_rmsd_median": (
            float(np.nanmedian(atomwise_rmsd)) if atomwise_rmsd is not None else None
        ),
        "atomwise_rmsd_max": (
            float(np.nanmax(atomwise_rmsd)) if atomwise_rmsd is not None else None
        ),
        "dihedrals": dihedral_summary,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2))

    arrays = {
        "sampled_coords": sampled_coords,
        "sampled_coords_full": sampled_coords_full,
        "model_atom_pad_mask": atom_mask_full,
        "conditioning_cluster_labels": conditioning_cluster_labels,
        "conditioning_cluster_labels_model_input": model_conditioning_cluster_labels,
        "label_sample_index": np.asarray(
            -1 if label_sample_index is None else label_sample_index,
            dtype=np.int64,
        ),
        "sample_index": np.asarray(-1 if sample_index is None else sample_index, dtype=np.int64),
        "seed": np.asarray(-1 if sample_seed is None else sample_seed, dtype=np.int64),
        "base_seed": np.asarray(-1 if args.seed is None else args.seed, dtype=np.int64),
        "frame_position": np.asarray(frame_data["frame_position"], dtype=np.int64),
        "frame_index": np.asarray(frame_data["frame_index"], dtype=np.int64),
    }
    if frame_data["atom_resids"] is not None:
        arrays["atom_resids"] = frame_data["atom_resids"]
    if evaluate_against_original:
        arrays.update(
            {
                "original_coords": original_coords,
                "selected_sample": selected_sample,
                "aligned_sampled_coords": aligned_sampled_coords,
                "atomwise_rmsd": atomwise_rmsd,
                "atom_mask": atom_mask,
                "original_cluster_labels": frame_data["original_cluster_labels"],
            }
        )
    if original_dihedrals is not None and dihedral_summary is not None:
        arrays.update(
            {
                "original_dihedrals": original_dihedrals,
                "sampled_dihedrals": sampled_dihedrals,
                "sampled_raw_dihedrals": sampled_raw_dihedrals,
                "sampled_aligned_dihedrals": sampled_aligned_dihedrals,
                "sampled_pdb_coords": sampled_pdb_coords,
                "original_recomputed_dihedrals": original_recomputed_dihedrals,
                "dihedral_diff_rad": dihedral_diff_rad,
                "dihedral_abs_error_deg": dihedral_abs_error_deg,
                "dihedral_atom_indices": frame_data["dihedral_atom_indices"],
                "dihedral_mask": frame_data["dihedral_mask"],
                "dihedral_keys": np.asarray(frame_data["dihedral_keys"]),
            }
        )
    np.savez_compressed(arrays_path, **arrays)

    if evaluate_against_original and atom_csv_path is not None:
        write_atomwise_csv(
            atom_csv_path,
            atomwise_rmsd,
            frame_data["atom_names"],
            frame_data["atom_resids"],
        )
    if original_dihedrals is not None and dihedral_summary is not None:
        write_dihedral_csv(
            dihedral_csv_path,
            original_dihedrals,
            sampled_dihedrals,
            dihedral_diff_rad,
            frame_data["dihedral_keys"],
        )
    else:
        dihedral_csv_path = None

    write_report(
        report_path,
        metrics,
        sampled_cif_path,
        sampled_pdb_path,
        target_cif_path,
        arrays_path,
        atom_csv_path,
        dihedral_csv_path,
        raw_sampled_cif_path,
    )

    print(f"Wrote report:  {report_path}")
    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote arrays:  {arrays_path}")
    print(f"Wrote sampled CIF: {sampled_cif_path}")
    print(f"Wrote sampled PDB: {sampled_pdb_path}")
    if target_cif_path is not None:
        print(f"Wrote target CIF:  {target_cif_path}")
    if target_pdb_path is not None:
        print(f"Wrote target PDB:  {target_pdb_path}")
    if raw_sampled_cif_path is not None:
        print(f"Wrote raw sampled CIF: {raw_sampled_cif_path}")
    if atom_csv_path is not None:
        print(f"Wrote atom RMSD CSV: {atom_csv_path}")
    if dihedral_csv_path is not None:
        print(f"Wrote dihedral CSV: {dihedral_csv_path}")
    if dihedral_histogram_artifacts is not None:
        print(f"Wrote dihedral angle histogram CSV: {dihedral_histogram_artifacts['angle_histogram_csv']}")
        print(f"Wrote dihedral error histogram CSV: {dihedral_histogram_artifacts['error_histogram_csv']}")
        if dihedral_histogram_artifacts["angle_histogram_png"] is not None:
            print(f"Wrote dihedral angle histogram PNG: {dihedral_histogram_artifacts['angle_histogram_png']}")
        if dihedral_histogram_artifacts["error_histogram_png"] is not None:
            print(f"Wrote dihedral error histogram PNG: {dihedral_histogram_artifacts['error_histogram_png']}")

    return metrics
