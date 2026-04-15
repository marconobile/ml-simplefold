#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
import argparse
import csv
from datetime import datetime
import importlib.util
import subprocess
import numpy as np
from simplefold import __version__
from simplefold.inference import predict_structures_from_fastas
from simplefold.utils.dihedral_index_utils import normalize_dihedral_atom_indices


_DIHEDRAL_NAMES = ("phi", "psi", "omega", "chi1", "chi2")


def _load_target_dihedral_payload(target_conditioning_npz: Path, target_frame_idx: int):
    with np.load(target_conditioning_npz, allow_pickle=False) as npz_data:
        if "trajectory" not in npz_data:
            raise ValueError(
                "Target conditioning NPZ does not contain `trajectory`, "
                "required to determine atom indexing."
            )

        all_target_coords = np.asarray(npz_data["trajectory"], dtype=np.float32)
        if all_target_coords.ndim == 3:
            if target_frame_idx < 0 or target_frame_idx >= all_target_coords.shape[0]:
                raise ValueError(
                    f"target_frame_idx={target_frame_idx} is out of bounds for "
                    f"`trajectory` with shape {tuple(all_target_coords.shape)}."
                )
            num_target_atoms = int(all_target_coords.shape[1])
        elif all_target_coords.ndim == 2:
            num_target_atoms = int(all_target_coords.shape[0])
        else:
            raise ValueError(
                "Target atom coordinates must be shape (N, 3) or (T, N, 3), "
                f"got {tuple(all_target_coords.shape)}."
            )

        dihedral_key = None
        for key in ("dihedrals", "target_dihedrals"):
            if key in npz_data:
                dihedral_key = key
                break
        if dihedral_key is None:
            raise ValueError(
                "Target conditioning NPZ is missing dihedrals. "
                "Expected `dihedrals` or `target_dihedrals`."
            )

        target_dihedrals = np.asarray(npz_data[dihedral_key], dtype=np.float32)
        if target_dihedrals.ndim == 3:
            if target_frame_idx < 0 or target_frame_idx >= target_dihedrals.shape[0]:
                raise ValueError(
                    f"target_frame_idx={target_frame_idx} is out of bounds for "
                    f"`{dihedral_key}` with shape {tuple(target_dihedrals.shape)}."
                )
            target_dihedrals = target_dihedrals[target_frame_idx]
        elif target_dihedrals.ndim != 2:
            raise ValueError(
                "Target dihedrals must be shape (R, D) or (T, R, D), "
                f"got {tuple(target_dihedrals.shape)}."
            )

        index_key = None
        for key in ("dihedral_atom_indices", "target_dihedral_atom_indices"):
            if key in npz_data:
                index_key = key
                break
        if index_key is None:
            raise ValueError(
                "Target conditioning NPZ is missing dihedral atom indices. "
                "Expected `dihedral_atom_indices` or `target_dihedral_atom_indices`."
            )
        target_dihedral_atom_indices = np.asarray(npz_data[index_key], dtype=np.int64)
        if (
            target_dihedral_atom_indices.ndim != 3
            or target_dihedral_atom_indices.shape[-1] != 4
        ):
            raise ValueError(
                "Target dihedral atom indices must have shape (R, D, 4), "
                f"got {tuple(target_dihedral_atom_indices.shape)}."
            )

        mask_key = None
        for key in ("dihedral_mask", "target_dihedral_mask"):
            if key in npz_data:
                mask_key = key
                break
        if mask_key is None:
            raise ValueError(
                "Target conditioning NPZ is missing dihedral mask. "
                "Expected `dihedral_mask` or `target_dihedral_mask`."
            )
        target_dihedral_mask = np.asarray(npz_data[mask_key]).astype(bool)
        if target_dihedral_mask.ndim != 2:
            raise ValueError(
                f"Target dihedral mask must have shape (R, D), got {tuple(target_dihedral_mask.shape)}."
            )

    if target_dihedrals.shape != target_dihedral_mask.shape:
        raise ValueError(
            "Target dihedrals and dihedral mask must have matching shape: "
            f"{tuple(target_dihedrals.shape)} vs {tuple(target_dihedral_mask.shape)}."
        )
    if target_dihedral_atom_indices.shape[:2] != target_dihedrals.shape:
        raise ValueError(
            "Target dihedral atom indices must match target dihedrals shape on (R, D): "
            f"{tuple(target_dihedral_atom_indices.shape[:2])} vs {tuple(target_dihedrals.shape)}."
        )

    target_dihedral_atom_indices = normalize_dihedral_atom_indices(
        target_dihedral_atom_indices,
        num_atoms=num_target_atoms,
        dihedral_mask=target_dihedral_mask,
        context="target_conditioning_npz",
    )
    return (
        target_dihedrals,
        target_dihedral_atom_indices,
        target_dihedral_mask,
        num_target_atoms,
    )


def _compute_masked_dihedral_angles(
    coords: np.ndarray,
    dihedral_atom_indices: np.ndarray,
    dihedral_mask: np.ndarray,
    eps: float = 1e-8,
):
    coords = np.asarray(coords, dtype=np.float64)
    dihedral_atom_indices = np.asarray(dihedral_atom_indices, dtype=np.int64)
    dihedral_mask = np.asarray(dihedral_mask, dtype=bool)

    if coords.ndim != 2 or coords.shape[-1] != 3:
        raise ValueError(f"Expected coordinates with shape (N, 3), got {tuple(coords.shape)}.")
    if dihedral_atom_indices.shape[-1] != 4:
        raise ValueError(
            "dihedral_atom_indices must have last dimension 4, "
            f"got {tuple(dihedral_atom_indices.shape)}."
        )
    if dihedral_atom_indices.shape[:-1] != dihedral_mask.shape:
        raise ValueError(
            "dihedral_atom_indices and dihedral_mask must align on (R, D): "
            f"{tuple(dihedral_atom_indices.shape[:-1])} vs {tuple(dihedral_mask.shape)}."
        )

    active_atom_mask = np.broadcast_to(dihedral_mask[..., None], dihedral_atom_indices.shape)
    active_atom_indices = dihedral_atom_indices[active_atom_mask]
    if active_atom_indices.size > 0:
        if np.any(active_atom_indices < 0) or np.any(active_atom_indices >= coords.shape[0]):
            raise IndexError("Active dihedral atom index out of bounds.")

    safe_indices = np.where(active_atom_mask, dihedral_atom_indices, 0)
    atoms = coords[safe_indices]
    p0 = atoms[..., 0, :]
    p1 = atoms[..., 1, :]
    p2 = atoms[..., 2, :]
    p3 = atoms[..., 3, :]

    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2

    b1_norm = np.linalg.norm(b1, axis=-1, keepdims=True)
    b1_hat = b1 / np.clip(b1_norm, eps, None)

    v = b0 - np.sum(b0 * b1_hat, axis=-1, keepdims=True) * b1_hat
    w = b2 - np.sum(b2 * b1_hat, axis=-1, keepdims=True) * b1_hat

    x_comp = np.sum(v * w, axis=-1)
    y_comp = np.sum(np.cross(b1_hat, v, axis=-1) * w, axis=-1)
    angles = np.arctan2(y_comp, x_comp)

    degenerate = (np.linalg.norm(v, axis=-1) < eps) | (np.linalg.norm(w, axis=-1) < eps)
    valid_mask = dihedral_mask & (~degenerate)
    angles = np.where(valid_mask, angles, np.nan)
    return angles, valid_mask


def _build_target_to_reference_mapping(
    per_atom_csv_path: Path,
    num_target_atoms: int,
    num_reference_atoms: int,
) -> np.ndarray:
    target_to_reference = np.full(num_target_atoms, -1, dtype=np.int64)
    required_columns = {"reference_atom_index", "target_atom_index"}

    with per_atom_csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        if not required_columns.issubset(columns):
            raise ValueError(
                f"Per-atom CSV {per_atom_csv_path} is missing required columns: "
                f"{sorted(required_columns - columns)}."
            )

        for row in reader:
            ref_idx = int(row["reference_atom_index"])
            tgt_idx = int(row["target_atom_index"])
            if ref_idx < 0 or ref_idx >= num_reference_atoms:
                raise ValueError(
                    f"Reference atom index out of range in {per_atom_csv_path}: {ref_idx}."
                )
            if tgt_idx < 0 or tgt_idx >= num_target_atoms:
                raise ValueError(
                    f"Target atom index out of range in {per_atom_csv_path}: {tgt_idx}."
                )
            previous_ref_idx = target_to_reference[tgt_idx]
            if previous_ref_idx >= 0 and previous_ref_idx != ref_idx:
                raise ValueError(
                    "Ambiguous target-to-reference mapping in per-atom CSV: "
                    f"target index {tgt_idx} maps to both {previous_ref_idx} and {ref_idx}."
                )
            target_to_reference[tgt_idx] = ref_idx
    return target_to_reference


def _load_target_conditioning_analysis_module(analysis_script: Path):
    module_name = "target_conditioning_atomwise_rmsd_module"
    spec = importlib.util.spec_from_file_location(
        module_name,
        str(analysis_script),
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import analysis helper module from {analysis_script}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _save_final_dihedral_error_histogram(
    output_dir: Path,
    target_conditioning_npz: Path,
    target_frame_idx: int,
    analysis_script: Path,
):
    summary_path = output_dir / "target_conditioning_atomwise_rmsd_summary.csv"
    if not summary_path.exists():
        print(
            "Skipping final dihedral error histogram because atomwise RMSD summary "
            f"was not found: {summary_path}"
        )
        return None

    try:
        (
            target_dihedrals,
            target_dihedral_atom_indices,
            target_dihedral_mask,
            num_target_atoms,
        ) = _load_target_dihedral_payload(target_conditioning_npz, target_frame_idx)
    except Exception as exc:  # noqa: BLE001
        print(
            "Skipping final dihedral error histogram because target dihedral data "
            f"could not be loaded: {exc}"
        )
        return None

    rows = []
    with summary_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("target_kind") != "target_frame":
                continue
            if row.get("status") != "success":
                continue
            rows.append(row)

    if len(rows) == 0:
        print(
            "Skipping final dihedral error histogram because no successful "
            "`target_frame` rows were found in the atomwise RMSD summary."
        )
        return None

    try:
        analysis_module = _load_target_conditioning_analysis_module(analysis_script)
    except Exception as exc:  # noqa: BLE001
        print(f"Skipping final dihedral error histogram because analysis module import failed: {exc}")
        return None

    num_dihedrals = int(target_dihedrals.shape[1])
    collected_errors: list[list[np.ndarray]] = [[] for _ in range(num_dihedrals)]

    for row in rows:
        cif_path = Path(row["cif_path"])
        per_atom_csv_path = Path(row["per_atom_csv"])
        if not cif_path.exists():
            print(f"Skipping row because generated CIF does not exist: {cif_path}")
            continue
        if not per_atom_csv_path.exists():
            print(f"Skipping row because per-atom CSV does not exist: {per_atom_csv_path}")
            continue

        try:
            cif_atoms = analysis_module.load_structure_atoms(
                structure_path=cif_path,
                model_index=0,
                chain_filter=None,
                include_hydrogens=False,
                include_non_polymer=False,
            )
            reference_coords = np.asarray(cif_atoms.coords, dtype=np.float64)

            target_to_reference = _build_target_to_reference_mapping(
                per_atom_csv_path=per_atom_csv_path,
                num_target_atoms=num_target_atoms,
                num_reference_atoms=reference_coords.shape[0],
            )

            active_dihedral_atom_mask = np.broadcast_to(
                target_dihedral_mask[..., None],
                target_dihedral_atom_indices.shape,
            )
            mapped_dihedral_atom_indices = np.full_like(target_dihedral_atom_indices, -1)
            mapped_dihedral_atom_indices[active_dihedral_atom_mask] = target_to_reference[
                target_dihedral_atom_indices[active_dihedral_atom_mask]
            ]
            mapped_dihedral_mask = target_dihedral_mask & np.all(
                mapped_dihedral_atom_indices >= 0,
                axis=-1,
            )

            pred_dihedrals, pred_valid_mask = _compute_masked_dihedral_angles(
                reference_coords,
                mapped_dihedral_atom_indices,
                mapped_dihedral_mask,
            )

            wrapped_error = np.abs(
                np.arctan2(
                    np.sin(pred_dihedrals - target_dihedrals),
                    np.cos(pred_dihedrals - target_dihedrals),
                )
            )
            valid_error_mask = (
                pred_valid_mask
                & np.isfinite(pred_dihedrals)
                & np.isfinite(target_dihedrals)
            )

            for dihedral_idx in range(num_dihedrals):
                channel_values = wrapped_error[valid_error_mask[:, dihedral_idx], dihedral_idx]
                if channel_values.size > 0:
                    collected_errors[dihedral_idx].append(channel_values.astype(np.float64))
        except Exception as exc:  # noqa: BLE001
            print(
                "Skipping row while building final dihedral error histogram due to "
                f"processing error for CIF {cif_path}: {exc}"
            )

    final_errors = [
        np.concatenate(values) if len(values) > 0 else np.asarray([], dtype=np.float64)
        for values in collected_errors
    ]
    if all(values.size == 0 for values in final_errors):
        print("Skipping final dihedral error histogram because no valid dihedral errors were computed.")
        return None

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not available; skipping final dihedral error histogram plotting.")
        return None

    ncols = min(3, num_dihedrals)
    nrows = (num_dihedrals + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.8 * ncols, 3.6 * nrows),
        sharex=False,
    )
    axes = np.atleast_1d(axes).reshape(-1)

    for dihedral_idx in range(num_dihedrals):
        ax = axes[dihedral_idx]
        dihedral_errors = final_errors[dihedral_idx]
        dihedral_name = (
            _DIHEDRAL_NAMES[dihedral_idx]
            if dihedral_idx < len(_DIHEDRAL_NAMES)
            else f"dihedral_{dihedral_idx}"
        )
        ax.set_title(f"{dihedral_name} (n={int(dihedral_errors.size)})")

        if dihedral_errors.size > 0:
            ax.hist(
                dihedral_errors,
                bins=30,
                range=(0.0, float(np.pi)),
                edgecolor="black",
                linewidth=0.5,
                alpha=0.8,
            )
        else:
            ax.text(
                0.5,
                0.5,
                "no valid data",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=9,
            )
        ax.set_xlabel("final wrapped error [rad]")
        ax.set_ylabel("count")
        ax.set_xlim(0.0, float(np.pi))
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)

    for ax in axes[num_dihedrals:]:
        ax.set_visible(False)

    fig.suptitle("Per-dihedral final absolute wrapped error distribution", fontsize=12)
    fig.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_path = output_dir / f"dihedral_final_error_hist_{timestamp}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved final dihedral error histogram to: {out_path}")
    count_parts = []
    for dihedral_idx, dihedral_errors in enumerate(final_errors):
        dihedral_name = (
            _DIHEDRAL_NAMES[dihedral_idx]
            if dihedral_idx < len(_DIHEDRAL_NAMES)
            else f"dihedral_{dihedral_idx}"
        )
        count_parts.append(f"{dihedral_name}={int(dihedral_errors.size)}")
    print("Per-dihedral valid counts used for final histogram: " + ", ".join(count_parts))
    return out_path


def main():
    parser = argparse.ArgumentParser(
        prog="simplefold",
        description="Folding proteins with SimpleFold."
    )
    parser.add_argument("--simplefold_model", type=str, default="simplefold_3B", help="Name of the model to load.")
    parser.add_argument("--ckpt_dir", type=str, default="artifacts", help="Directory to save the checkpoint.")
    parser.add_argument("--output_dir", type=str, default="artifacts/debug_samples", help="Directory to save the output structure.")
    parser.add_argument("--num_steps", type=int, default=1600, help="Number of steps in inference.")
    parser.add_argument("--tau", type=float, default=0.1, help="Diffusion coefficient scaling factor.")
    parser.add_argument("--no_log_timesteps", action="store_true", help="Disable logarithmic timesteps.")
    parser.add_argument("--fasta_path", required=True, type=str, help="Path to the input FASTA file/directory.")
    parser.add_argument("--nsample_per_protein", type=int, default=1, help="Number of samples to generate per protein.")
    parser.add_argument("--plddt", action="store_true", help="Enable pLDDT prediction.")
    parser.add_argument("--output_format", type=str, default="mmcif", choices=["pdb", "mmcif"], help="Output file format.")
    parser.add_argument("--backend", type=str, default='torch', choices=['torch', 'mlx'], help="Backend to run inference either torch or mlx")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--target_conditioning_npz",
        type=str,
        default=None,
        help=(
            "Optional NPZ path with target atom coordinates and atom names "
            "for external conditioning. Expected keys include "
            "`target_atom_coords` (or `xyz`/`trajectory`) and `atom_names`."
        ),
    )
    parser.add_argument(
        "--target_frame_idx",
        type=int,
        default=0,
        help=(
            "Reference frame index in the conditioning NPZ. This frame is always "
            "used for target dihedrals and is exported to output_dir as a PDB."
        ),
    )
    parser.add_argument(
        "--random_target_coords",
        dest="random_target_coords",
        action="store_true",
        help=(
            "Sample target_atom_coords from a random conditioning NPZ frame. "
            "target_frame_idx is still used for target dihedrals."
        ),
    )
    parser.add_argument(
        "--no_random_target_coords",
        dest="random_target_coords",
        action="store_false",
        help=(
            "Disable random target coordinate frame sampling and use "
            "target_frame_idx for target_atom_coords."
        ),
    )
    parser.set_defaults(random_target_coords=True)
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}"
    )
    args = parser.parse_args()

    print(f"Running protein folding with SimpleFold ...")
    predict_structures_from_fastas(args)

    if args.target_conditioning_npz is not None:
        repo_root = Path(__file__).resolve().parents[2]
        analysis_script = repo_root / "scripts" / "cif_target_conditioning_atomwise_rmsd.py"
        if not analysis_script.exists():
            raise FileNotFoundError(
                f"Could not find analysis script: {analysis_script}"
            )
        print("Running target-conditioning atomwise RMSD analysis ...")
        subprocess.run(
            [
                sys.executable,
                str(analysis_script),
                "--run-dir",
                str(Path(args.output_dir).resolve()),
            ],
            check=True,
        )
        _save_final_dihedral_error_histogram(
            output_dir=Path(args.output_dir).resolve(),
            target_conditioning_npz=Path(args.target_conditioning_npz).resolve(),
            target_frame_idx=int(args.target_frame_idx),
            analysis_script=analysis_script,
        )

main()
