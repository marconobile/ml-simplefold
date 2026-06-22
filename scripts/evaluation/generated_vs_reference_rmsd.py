#!/usr/bin/env python3
"""Evaluate generated PDB samples against randomly sampled NPZ reference frames.

The protein system is assumed to be fixed across inputs. Generated PDB atoms are
matched to NPZ atoms by residue ordinal and atom name, so the comparison is
robust to PDB-vs-NPZ atom-order differences.

conda run -n simplefold python /home/nobilm@usi.ch/ml-simplefold/scripts/evaluation/generated_vs_reference_rmsd.py \
  --generated_samples /home/nobilm@usi.ch/ml-simplefold/artifacts/debug_samples/predictions_simplefold_100M_pdbs \
  --reference_structures /home/nobilm@usi.ch/ml-simplefold/test_new_data_with_clusters/pas_without_hs.npz \
  --output_dir /home/nobilm@usi.ch/ml-simplefold/artifacts/debug_samples/predictions_simplefold_100M_pdbs/pas
  
  
  

"""

from __future__ import annotations

import argparse
import csv
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib
import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


REQUIRED_REFERENCE_KEYS = ("trajectory", "atom_names", "atom_residue_index")
RESULT_COLUMNS = [
    "generated_file",
    "generated_path",
    "alignment_mode",
    "status",
    "error",
    "n_atoms_in_generated",
    "n_reference_atoms",
    "n_common_atoms",
    "n_atoms_used",
    "n_reference_samples",
    "closest_reference_sample_index",
    "closest_reference_trajectory_position",
    "closest_reference_frame_index",
    "min_rmsd_A",
]


@dataclass(frozen=True, order=True)
class AtomKey:
    """Stable atom identity shared between NPZ metadata and generated PDBs."""

    residue_index: int
    atom_name: str


@dataclass(frozen=True)
class AlignmentMode:
    """Atom selection used for both rigid alignment and RMSD calculation."""

    name: str
    atom_selection: str
    plot_title: str
    output_name: str


ALIGNMENT_MODES: tuple[AlignmentMode, ...] = (
    AlignmentMode(
        name="ca",
        atom_selection="ca",
        plot_title="C-alpha aligned minimum RMSD",
        output_name="minimum_rmsd_hist_ca.png",
    ),
    AlignmentMode(
        name="all_atoms",
        atom_selection="all",
        plot_title="All-atom aligned minimum RMSD",
        output_name="minimum_rmsd_hist_all_atoms.png",
    ),
)
ALIGNMENT_MODE_BY_NAME = {mode.name: mode for mode in ALIGNMENT_MODES}


@dataclass
class ProteinSample:
    """Generated PDB coordinates keyed by residue ordinal and atom name."""

    name: str
    path: Path
    atoms: dict[AtomKey, np.ndarray]
    ordered_keys: list[AtomKey]
    duplicate_keys: int = 0


@dataclass
class ReferenceSamples:
    """Randomly sampled reference trajectory frames and their atom metadata."""

    path: Path
    coords: np.ndarray
    trajectory_positions: np.ndarray
    frame_indices: np.ndarray
    atom_keys: list[AtomKey]
    atom_index_by_key: dict[AtomKey, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load generated PDB samples, randomly sample reference structures from "
            "an NPZ trajectory, compute each generated sample's minimum aligned "
            "RMSD to the reference set, and save separate histograms for C-alpha "
            "and all-atom alignment modes."
        )
    )
    parser.add_argument(
        "--generated_samples",
        "--generated-samples",
        dest="generated_samples",
        type=Path,
        required=True,
        help="Directory containing generated .pdb files.",
    )
    parser.add_argument(
        "--reference_structures",
        "--reference-structures",
        dest="reference_structures",
        type=Path,
        required=True,
        help="Reference NPZ containing trajectory and atom metadata.",
    )
    parser.add_argument(
        "--output_dir",
        "--output-dir",
        dest="output_dir",
        type=Path,
        required=True,
        help="Directory where CSVs and histogram PNGs will be written.",
    )
    parser.add_argument(
        "--reference-sample-count",
        type=int,
        default=5000,
        help="Number of trajectory frames to sample from the reference NPZ. Default: 5000.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="Seed used for reproducible reference-frame sampling. Default: 0.",
    )
    parser.add_argument(
        "--alignment-mode",
        dest="alignment_modes",
        action="append",
        choices=sorted(ALIGNMENT_MODE_BY_NAME),
        default=None,
        help=(
            "Alignment mode to run. Can be passed multiple times. "
            "Defaults to both ca and all_atoms."
        ),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Number of reference frames processed per vectorized Kabsch chunk. Default: 512.",
    )
    parser.add_argument(
        "--model-index",
        type=int,
        default=0,
        help="Zero-based PDB model index to read from each generated sample. Default: 0.",
    )
    parser.add_argument(
        "--include-hetero",
        action="store_true",
        help="Include HETATM residues from generated PDBs. By default only ATOM residues are used.",
    )
    parser.add_argument(
        "--min-atoms",
        type=int,
        default=3,
        help="Minimum matched atoms required for a rigid-body alignment. Default: 3.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="Print progress every N generated samples. Use 0 to disable. Default: 1.",
    )
    return parser.parse_args()


def expand_path(path: Path) -> Path:
    return path.expanduser().resolve()


def find_pdb_files(generated_samples: Path) -> list[Path]:
    return sorted(
        path
        for path in generated_samples.rglob("*.pdb")
        if path.is_file()
    )


def normalize_atom_name(atom_name: object) -> str:
    return str(atom_name).strip().upper()


def build_reference_atom_keys(atom_names: np.ndarray, residue_indices: np.ndarray) -> list[AtomKey]:
    if atom_names.ndim != 1 or residue_indices.ndim != 1:
        raise ValueError("`atom_names` and `atom_residue_index` must be 1D arrays.")
    if atom_names.shape[0] != residue_indices.shape[0]:
        raise ValueError(
            "`atom_names` and `atom_residue_index` must have the same length, got "
            f"{atom_names.shape[0]} and {residue_indices.shape[0]}."
        )

    atom_keys = [
        AtomKey(int(residue_index), normalize_atom_name(atom_name))
        for residue_index, atom_name in zip(residue_indices, atom_names, strict=True)
    ]
    if len(set(atom_keys)) != len(atom_keys):
        raise ValueError(
            "Reference atom metadata contains duplicate residue-index/atom-name keys; "
            "cannot match generated PDB atoms unambiguously."
        )
    return atom_keys


def sample_reference_positions(
    nframes: int,
    sample_count: int,
    random_seed: int,
) -> np.ndarray:
    if sample_count <= 0:
        raise ValueError("--reference-sample-count must be positive.")
    if nframes <= 0:
        raise ValueError("Reference trajectory contains no frames.")

    rng = np.random.default_rng(random_seed)
    actual_count = min(sample_count, nframes)
    if actual_count < sample_count:
        print(
            "Warning: requested "
            f"{sample_count} reference samples, but the trajectory only has "
            f"{nframes} frame(s). Using all frames.",
            flush=True,
        )
    return np.sort(rng.choice(nframes, size=actual_count, replace=False)).astype(np.int64)


def load_reference_samples(
    reference_path: Path,
    sample_count: int,
    random_seed: int,
) -> ReferenceSamples:
    reference_path = expand_path(reference_path)
    if not reference_path.is_file():
        raise FileNotFoundError(f"Reference NPZ not found: {reference_path}")

    with np.load(reference_path, allow_pickle=False) as data:
        missing = sorted(set(REQUIRED_REFERENCE_KEYS) - set(data.files))
        if missing:
            raise KeyError(
                f"Reference NPZ is missing required key(s): {', '.join(missing)}"
            )

        trajectory = data["trajectory"]
        if trajectory.ndim != 3 or trajectory.shape[-1] != 3:
            raise ValueError(
                "`trajectory` must have shape (n_frames, n_atoms, 3), "
                f"got {trajectory.shape}."
            )

        nframes, natoms, _ = trajectory.shape
        atom_names = data["atom_names"]
        residue_indices = data["atom_residue_index"]
        if atom_names.shape[0] != natoms or residue_indices.shape[0] != natoms:
            raise ValueError(
                "`trajectory`, `atom_names`, and `atom_residue_index` must agree "
                f"on atom count. Got trajectory atoms={natoms}, "
                f"atom_names={atom_names.shape[0]}, "
                f"atom_residue_index={residue_indices.shape[0]}."
            )

        trajectory_positions = sample_reference_positions(
            nframes=nframes,
            sample_count=sample_count,
            random_seed=random_seed,
        )
        coords = np.asarray(trajectory[trajectory_positions], dtype=np.float32)
        if not np.isfinite(coords).all():
            raise ValueError("Sampled reference coordinates contain non-finite values.")

        if "frame_indices" in data.files:
            frame_indices = np.asarray(data["frame_indices"][trajectory_positions], dtype=np.int64)
        else:
            frame_indices = trajectory_positions.copy()

        atom_keys = build_reference_atom_keys(atom_names, residue_indices)

    atom_index_by_key = {key: index for index, key in enumerate(atom_keys)}
    return ReferenceSamples(
        path=reference_path,
        coords=coords,
        trajectory_positions=trajectory_positions,
        frame_indices=frame_indices,
        atom_keys=atom_keys,
        atom_index_by_key=atom_index_by_key,
    )


def residue_identity(atom) -> tuple[str, tuple[str, int, str]]:
    residue = atom.get_parent()
    chain = residue.get_parent()
    hetero_flag, residue_number, insertion_code = residue.id
    return str(chain.id), (str(hetero_flag), int(residue_number), str(insertion_code))


def load_generated_pdb(
    pdb_path: Path,
    parser: PDBParser,
    model_index: int,
    include_hetero: bool,
) -> ProteinSample:
    pdb_path = expand_path(pdb_path)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=PDBConstructionWarning)
        structure = parser.get_structure(pdb_path.stem, str(pdb_path))

    models = list(structure.get_models())
    if not models:
        raise ValueError(f"{pdb_path}: PDB contains no models.")
    if model_index < 0 or model_index >= len(models):
        raise ValueError(
            f"{pdb_path}: requested model index {model_index}, but PDB contains "
            f"{len(models)} model(s)."
        )

    residue_ordinal_by_id: dict[tuple[str, tuple[str, int, str]], int] = {}
    atoms: dict[AtomKey, np.ndarray] = {}
    occupancy_by_key: dict[AtomKey, float] = {}
    ordered_keys: list[AtomKey] = []
    duplicate_keys = 0

    for atom in models[model_index].get_atoms():
        residue = atom.get_parent()
        if not include_hetero and residue.id[0] != " ":
            continue

        residue_id = residue_identity(atom)
        if residue_id not in residue_ordinal_by_id:
            residue_ordinal_by_id[residue_id] = len(residue_ordinal_by_id)

        key = AtomKey(
            residue_index=residue_ordinal_by_id[residue_id],
            atom_name=normalize_atom_name(atom.get_name()),
        )
        coord = np.asarray(atom.get_coord(), dtype=np.float32)
        if coord.shape != (3,) or not np.isfinite(coord).all():
            continue

        occupancy = atom.get_occupancy()
        occupancy_value = float(occupancy) if occupancy is not None else -np.inf
        if key in atoms:
            duplicate_keys += 1
            if occupancy_value <= occupancy_by_key[key]:
                continue
        else:
            ordered_keys.append(key)

        atoms[key] = coord
        occupancy_by_key[key] = occupancy_value

    if not atoms:
        raise ValueError(f"{pdb_path}: no usable atoms found.")

    return ProteinSample(
        name=pdb_path.name,
        path=pdb_path,
        atoms=atoms,
        ordered_keys=ordered_keys,
        duplicate_keys=duplicate_keys,
    )


def load_generated_samples(
    generated_dir: Path,
    model_index: int,
    include_hetero: bool,
) -> list[ProteinSample]:
    generated_dir = expand_path(generated_dir)
    if not generated_dir.is_dir():
        raise NotADirectoryError(f"Generated samples directory not found: {generated_dir}")

    pdb_paths = find_pdb_files(generated_dir)
    if not pdb_paths:
        raise FileNotFoundError(f"No .pdb files found under {generated_dir}")

    parser = PDBParser(QUIET=True)
    return [
        load_generated_pdb(
            pdb_path=pdb_path,
            parser=parser,
            model_index=model_index,
            include_hetero=include_hetero,
        )
        for pdb_path in pdb_paths
    ]


def select_atom_keys(keys: Iterable[AtomKey], atom_selection: str) -> list[AtomKey]:
    if atom_selection == "all":
        return list(keys)
    if atom_selection == "ca":
        return [key for key in keys if key.atom_name == "CA"]
    raise ValueError(f"Unsupported atom selection: {atom_selection}")


def coords_for_keys(sample: ProteinSample, keys: list[AtomKey]) -> np.ndarray:
    return np.stack([sample.atoms[key] for key in keys], axis=0).astype(np.float32, copy=False)


def reference_indices_for_keys(reference: ReferenceSamples, keys: list[AtomKey]) -> np.ndarray:
    return np.asarray([reference.atom_index_by_key[key] for key in keys], dtype=np.int64)


def batched_aligned_rmsd(mobile: np.ndarray, targets: np.ndarray, min_atoms: int) -> np.ndarray:
    """Compute Kabsch-aligned RMSDs from one mobile structure to many targets."""

    if mobile.ndim != 2 or mobile.shape[-1] != 3:
        raise ValueError(f"Mobile coordinates must have shape (n_atoms, 3), got {mobile.shape}.")
    if targets.ndim != 3 or targets.shape[1:] != mobile.shape:
        raise ValueError(
            "Target coordinates must have shape (n_targets, n_atoms, 3), got "
            f"{targets.shape} for mobile shape {mobile.shape}."
        )
    if mobile.shape[0] < min_atoms:
        raise ValueError(
            f"Need at least {min_atoms} atoms for alignment, got {mobile.shape[0]}."
        )
    if not np.isfinite(mobile).all() or not np.isfinite(targets).all():
        raise ValueError("Alignment coordinates contain non-finite values.")

    mobile64 = mobile.astype(np.float64, copy=False)
    targets64 = targets.astype(np.float64, copy=False)

    mobile_center = mobile64.mean(axis=0)
    target_centers = targets64.mean(axis=1)
    mobile_centered = mobile64 - mobile_center
    target_centered = targets64 - target_centers[:, None, :]

    covariance = np.einsum("ni,mnj->mij", mobile_centered, target_centered)
    u, _, vt = np.linalg.svd(covariance)
    det = np.linalg.det(np.matmul(u, vt))
    correction = np.repeat(np.eye(3, dtype=np.float64)[None, :, :], targets.shape[0], axis=0)
    correction[:, -1, -1] = np.where(det >= 0.0, 1.0, -1.0)
    rotation = np.matmul(np.matmul(u, correction), vt)

    aligned = np.einsum("ni,mij->mnj", mobile_centered, rotation) + target_centers[:, None, :]
    squared_distances = np.sum((aligned - targets64) ** 2, axis=-1)
    return np.sqrt(np.mean(squared_distances, axis=1))


def min_aligned_rmsd_to_references(
    mobile: np.ndarray,
    reference_coords: np.ndarray,
    chunk_size: int,
    min_atoms: int,
) -> tuple[float, int]:
    if chunk_size <= 0:
        raise ValueError("--chunk-size must be positive.")
    if reference_coords.shape[0] == 0:
        raise ValueError("No reference coordinates were provided.")

    best_rmsd = np.inf
    best_index = -1
    for start in range(0, reference_coords.shape[0], chunk_size):
        end = min(start + chunk_size, reference_coords.shape[0])
        rmsds = batched_aligned_rmsd(
            mobile=mobile,
            targets=reference_coords[start:end],
            min_atoms=min_atoms,
        )
        local_index = int(np.argmin(rmsds))
        local_rmsd = float(rmsds[local_index])
        if local_rmsd < best_rmsd:
            best_rmsd = local_rmsd
            best_index = start + local_index

    return best_rmsd, best_index


def blank_result_row(
    sample: ProteinSample,
    mode: AlignmentMode,
    reference: ReferenceSamples,
    error: str,
) -> dict[str, object]:
    return {
        "generated_file": sample.name,
        "generated_path": str(sample.path),
        "alignment_mode": mode.name,
        "status": "error",
        "error": error,
        "n_atoms_in_generated": len(sample.atoms),
        "n_reference_atoms": len(reference.atom_keys),
        "n_common_atoms": "",
        "n_atoms_used": "",
        "n_reference_samples": reference.coords.shape[0],
        "closest_reference_sample_index": "",
        "closest_reference_trajectory_position": "",
        "closest_reference_frame_index": "",
        "min_rmsd_A": "",
    }


def analyze_sample_mode(
    sample: ProteinSample,
    reference: ReferenceSamples,
    mode: AlignmentMode,
    chunk_size: int,
    min_atoms: int,
) -> dict[str, object]:
    common_keys = [key for key in reference.atom_keys if key in sample.atoms]
    selected_keys = select_atom_keys(common_keys, mode.atom_selection)
    if len(selected_keys) < min_atoms:
        return blank_result_row(
            sample=sample,
            mode=mode,
            reference=reference,
            error=(
                f"Only {len(selected_keys)} matched atom(s) available for mode "
                f"{mode.name}; require at least {min_atoms}."
            ),
        )

    try:
        reference_atom_indices = reference_indices_for_keys(reference, selected_keys)
        reference_coords = reference.coords[:, reference_atom_indices, :]
        mobile_coords = coords_for_keys(sample, selected_keys)
        min_rmsd, best_reference_index = min_aligned_rmsd_to_references(
            mobile=mobile_coords,
            reference_coords=reference_coords,
            chunk_size=chunk_size,
            min_atoms=min_atoms,
        )
    except Exception as exc:
        return blank_result_row(
            sample=sample,
            mode=mode,
            reference=reference,
            error=str(exc),
        )

    return {
        "generated_file": sample.name,
        "generated_path": str(sample.path),
        "alignment_mode": mode.name,
        "status": "ok",
        "error": "",
        "n_atoms_in_generated": len(sample.atoms),
        "n_reference_atoms": len(reference.atom_keys),
        "n_common_atoms": len(common_keys),
        "n_atoms_used": len(selected_keys),
        "n_reference_samples": reference.coords.shape[0],
        "closest_reference_sample_index": best_reference_index,
        "closest_reference_trajectory_position": int(reference.trajectory_positions[best_reference_index]),
        "closest_reference_frame_index": int(reference.frame_indices[best_reference_index]),
        "min_rmsd_A": min_rmsd,
    }


def analyze_samples(
    samples: list[ProteinSample],
    reference: ReferenceSamples,
    modes: list[AlignmentMode],
    chunk_size: int,
    min_atoms: int,
    progress_every: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for sample_index, sample in enumerate(samples, start=1):
        for mode in modes:
            rows.append(
                analyze_sample_mode(
                    sample=sample,
                    reference=reference,
                    mode=mode,
                    chunk_size=chunk_size,
                    min_atoms=min_atoms,
                )
            )
        if progress_every and sample_index % progress_every == 0:
            print(
                f"Processed {sample_index}/{len(samples)} generated sample(s)",
                flush=True,
            )
    return rows


def csv_value(value: object) -> object:
    if isinstance(value, (float, np.floating)):
        if np.isfinite(value):
            return f"{float(value):.6f}"
        return ""
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return int(value)
    return value


def write_results_csv(rows: list[dict[str, object]], csv_path: Path) -> None:
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: csv_value(row.get(column, "")) for column in RESULT_COLUMNS})


def write_reference_sample_csv(reference: ReferenceSamples, csv_path: Path) -> None:
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "reference_sample_index",
                "trajectory_position",
                "frame_index",
            ],
        )
        writer.writeheader()
        for index, (trajectory_position, frame_index) in enumerate(
            zip(reference.trajectory_positions, reference.frame_indices, strict=True)
        ):
            writer.writerow(
                {
                    "reference_sample_index": index,
                    "trajectory_position": int(trajectory_position),
                    "frame_index": int(frame_index),
                }
            )


def successful_min_rmsds(rows: list[dict[str, object]], mode_name: str) -> np.ndarray:
    values = [
        float(row["min_rmsd_A"])
        for row in rows
        if row.get("alignment_mode") == mode_name
        and row.get("status") == "ok"
        and row.get("min_rmsd_A") != ""
    ]
    return np.asarray(values, dtype=np.float64)


def histogram_bins(values: np.ndarray) -> int:
    if values.size <= 1:
        return 1
    return int(np.clip(np.ceil(np.sqrt(values.size)), 5, 40))


def plot_mode_histogram(
    rows: list[dict[str, object]],
    mode: AlignmentMode,
    output_path: Path,
) -> None:
    values = successful_min_rmsds(rows, mode.name)
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)

    if values.size:
        ax.hist(
            values,
            bins=histogram_bins(values),
            color="#4C78A8",
            edgecolor="black",
            alpha=0.85,
        )
        ax.axvline(float(np.mean(values)), color="#F58518", linestyle="--", linewidth=1.5)
        ax.set_title(mode.plot_title)
        ax.set_xlabel("Minimum aligned RMSD (A)")
        ax.set_ylabel("Generated sample count")
    else:
        ax.text(
            0.5,
            0.5,
            "No valid RMSD results",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()

    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def write_summary_json(
    summary_path: Path,
    *,
    args: argparse.Namespace,
    generated_samples: list[ProteinSample],
    reference: ReferenceSamples,
    rows: list[dict[str, object]],
    modes: list[AlignmentMode],
) -> None:
    ok_counts = {
        mode.name: sum(
            1
            for row in rows
            if row.get("alignment_mode") == mode.name and row.get("status") == "ok"
        )
        for mode in modes
    }
    min_rmsd_summary = {}
    for mode in modes:
        values = successful_min_rmsds(rows, mode.name)
        min_rmsd_summary[mode.name] = {
            "count": int(values.size),
            "mean_A": float(np.mean(values)) if values.size else None,
            "median_A": float(np.median(values)) if values.size else None,
            "min_A": float(np.min(values)) if values.size else None,
            "max_A": float(np.max(values)) if values.size else None,
        }

    summary = {
        "generated_samples": str(expand_path(args.generated_samples)),
        "reference_structures": str(reference.path),
        "output_dir": str(expand_path(args.output_dir)),
        "reference_sample_count_requested": int(args.reference_sample_count),
        "reference_sample_count_used": int(reference.coords.shape[0]),
        "random_seed": int(args.random_seed),
        "chunk_size": int(args.chunk_size),
        "generated_pdb_count": len(generated_samples),
        "alignment_modes": [mode.name for mode in modes],
        "ok_counts": ok_counts,
        "min_rmsd_summary": min_rmsd_summary,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def selected_modes(mode_names: list[str] | None) -> list[AlignmentMode]:
    if mode_names is None:
        return list(ALIGNMENT_MODES)
    return [ALIGNMENT_MODE_BY_NAME[name] for name in mode_names]


def validate_args(args: argparse.Namespace) -> None:
    if args.min_atoms < 3:
        raise ValueError("--min-atoms must be at least 3 for rigid-body alignment.")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive.")
    if args.progress_every < 0:
        raise ValueError("--progress-every must be >= 0.")


def main() -> None:
    args = parse_args()
    validate_args(args)

    output_dir = expand_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    modes = selected_modes(args.alignment_modes)

    print(f"Loading generated PDB samples from: {expand_path(args.generated_samples)}", flush=True)
    generated_samples = load_generated_samples(
        generated_dir=args.generated_samples,
        model_index=args.model_index,
        include_hetero=args.include_hetero,
    )
    print(f"Loaded {len(generated_samples)} generated PDB sample(s).", flush=True)

    print(f"Loading and sampling reference structures from: {expand_path(args.reference_structures)}", flush=True)
    reference = load_reference_samples(
        reference_path=args.reference_structures,
        sample_count=args.reference_sample_count,
        random_seed=args.random_seed,
    )
    print(
        f"Sampled {reference.coords.shape[0]} reference frame(s) with "
        f"{len(reference.atom_keys)} atom(s) each.",
        flush=True,
    )

    rows = analyze_samples(
        samples=generated_samples,
        reference=reference,
        modes=modes,
        chunk_size=args.chunk_size,
        min_atoms=args.min_atoms,
        progress_every=args.progress_every,
    )

    results_csv = output_dir / "generated_vs_reference_min_rmsd.csv"
    reference_csv = output_dir / "reference_sampled_frames.csv"
    summary_json = output_dir / "generated_vs_reference_rmsd_summary.json"
    write_results_csv(rows, results_csv)
    write_reference_sample_csv(reference, reference_csv)
    write_summary_json(
        summary_path=summary_json,
        args=args,
        generated_samples=generated_samples,
        reference=reference,
        rows=rows,
        modes=modes,
    )

    for mode in modes:
        plot_path = output_dir / mode.output_name
        plot_mode_histogram(rows=rows, mode=mode, output_path=plot_path)
        values = successful_min_rmsds(rows, mode.name)
        if values.size:
            print(
                f"{mode.name}: {values.size} valid sample(s), "
                f"median min RMSD={np.median(values):.6f} A, "
                f"mean min RMSD={np.mean(values):.6f} A. "
                f"Wrote {plot_path}",
                flush=True,
            )
        else:
            print(f"{mode.name}: no valid RMSD values. Wrote {plot_path}", flush=True)

    error_count = sum(1 for row in rows if row.get("status") != "ok")
    print(f"Wrote results CSV: {results_csv}")
    print(f"Wrote reference sampled-frame CSV: {reference_csv}")
    print(f"Wrote summary JSON: {summary_json}")
    if error_count:
        raise RuntimeError(f"Completed with {error_count} failed mode/sample result(s).")


if __name__ == "__main__":
    main()
