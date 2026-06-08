#!/usr/bin/env python3
"""Assign PDB files to the closest reference structure by aligned RMSD."""

from __future__ import annotations

import argparse
import csv
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import matplotlib
import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DEFAULT_ACTIVE_PDB = Path(
    "/home/nobilm@usi.ch/ml-simplefold/data/pdb_for_sampling_jupyter/FApo_no_caps.pdb"
)
DEFAULT_INACTIVE_PDB = Path(
    "/home/nobilm@usi.ch/ml-simplefold/data/pdb_for_sampling_jupyter/INApo_no_caps.pdb"
)
DEFAULT_PAS_PDB = Path(
    "/home/nobilm@usi.ch/ml-simplefold/test_new_data_with_clusters/"
    "pas_without_hs_frame_19168.pdb"
)
DEFAULT_OUTPUT_DIR = Path(
    "/storage_common/nobilm/backmapping_pots_model/pots_samples/simplefold_samples"
)

REFERENCE_LABELS = ("active", "inactive", "pas")
BACKBONE_ATOMS = frozenset({"N", "CA", "C", "O"})
RESULT_COLUMNS = [
    "filename",
    "pdb_path",
    "status",
    "error",
    "n_atoms_in_file",
    "n_atoms_used",
    "n_reference_common_atoms",
    "n_missing_reference_common_atoms",
    "match_method",
    "rmsd_active",
    "rmsd_inactive",
    "rmsd_pas",
    "closest_reference",
    "min_rmsd",
]


@dataclass(frozen=True, order=True)
class AtomKey:
    """Stable identity for matching atoms across structures."""

    chain_id: str
    hetero_flag: str
    residue_number: int
    insertion_code: str
    atom_name: str


@dataclass
class StructureAtoms:
    """Selected atom coordinates keyed by structural identity."""

    path: Path
    atoms: dict[AtomKey, np.ndarray]
    ordered_keys: list[AtomKey]
    duplicate_keys: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "For each PDB directly inside base_directory, align it independently "
            "to active, inactive, and pas references, compute aligned RMSDs, "
            "write a CSV, and save a summary figure."
        )
    )
    parser.add_argument(
        "base_directory",
        type=Path,
        help="Directory containing input .pdb files. Only files directly inside it are used.",
    )
    parser.add_argument(
        "--active-pdb",
        type=Path,
        default=DEFAULT_ACTIVE_PDB,
        help=f"Active reference PDB. Default: {DEFAULT_ACTIVE_PDB}",
    )
    parser.add_argument(
        "--inactive-pdb",
        type=Path,
        default=DEFAULT_INACTIVE_PDB,
        help=f"Inactive reference PDB. Default: {DEFAULT_INACTIVE_PDB}",
    )
    parser.add_argument(
        "--pas-pdb",
        type=Path,
        default=DEFAULT_PAS_PDB,
        help=f"PAS reference PDB. Default: {DEFAULT_PAS_PDB}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for the CSV and figure. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--csv-name",
        type=str,
        default="pdb_reference_rmsd_assignments.csv",
        help="Output CSV filename.",
    )
    parser.add_argument(
        "--figure-name",
        type=str,
        default="pdb_reference_rmsd_summary.png",
        help="Output figure filename.",
    )
    parser.add_argument(
        "--atom-selection",
        choices=("heavy", "backbone", "ca", "all"),
        default="heavy",
        help=(
            "Atoms considered for matching, alignment, and RMSD. "
            "The final set is the intersection shared by all references and each sample. "
            "Default: heavy."
        ),
    )
    parser.add_argument(
        "--match-mode",
        choices=("auto", "identity", "order"),
        default="auto",
        help=(
            "How sample atoms are matched to reference atoms. `identity` uses "
            "chain/residue/atom ids, `order` uses selected atom file order, and "
            "`auto` tries identity first then order. Default: auto."
        ),
    )
    parser.add_argument(
        "--include-hetero",
        action="store_true",
        help="Include HETATM residues. By default only standard ATOM residues are used.",
    )
    parser.add_argument(
        "--model-index",
        type=int,
        default=0,
        help="Zero-based model index to read from each PDB. Default: 0.",
    )
    parser.add_argument(
        "--min-atoms",
        type=int,
        default=3,
        help="Minimum number of matched atoms required for alignment. Default: 3.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=500,
        help="Print progress after every N processed PDB files. Use 0 to disable. Default: 500.",
    )
    parser.add_argument(
        "--reference-duplicate-threshold",
        type=float,
        default=0.05,
        help=(
            "Warn when two references have aligned RMSD below this value in Angstrom. "
            "Use 0 to disable. Default: 0.05."
        ),
    )
    return parser.parse_args()


def is_hydrogen_atom(atom) -> bool:
    element = str(getattr(atom, "element", "") or "").strip().upper()
    if element in {"H", "D"}:
        return True
    if element and element not in {"X", "?"}:
        return False

    atom_name = atom.get_name().strip().upper()
    return atom_name.startswith(("H", "D")) or (
        len(atom_name) >= 2 and atom_name[0].isdigit() and atom_name[1] in {"H", "D"}
    )


def atom_matches_selection(atom, atom_selection: str) -> bool:
    atom_name = atom.get_name().strip().upper()

    if atom_selection == "all":
        return True
    if atom_selection == "heavy":
        return not is_hydrogen_atom(atom)
    if atom_selection == "backbone":
        return atom_name in BACKBONE_ATOMS
    if atom_selection == "ca":
        return atom_name == "CA"

    raise ValueError(f"Unsupported atom selection: {atom_selection}")


def make_atom_key(atom) -> AtomKey:
    residue = atom.get_parent()
    chain = residue.get_parent()
    hetero_flag, residue_number, insertion_code = residue.id

    return AtomKey(
        chain_id=str(chain.id).strip() or "_",
        hetero_flag=str(hetero_flag).strip(),
        residue_number=int(residue_number),
        insertion_code=str(insertion_code).strip(),
        atom_name=atom.get_name().strip().upper(),
    )


def load_structure_atoms(
    pdb_path: Path,
    parser: PDBParser,
    model_index: int,
    atom_selection: str,
    include_hetero: bool,
) -> StructureAtoms:
    pdb_path = pdb_path.expanduser().resolve()
    if not pdb_path.is_file():
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=PDBConstructionWarning)
        structure = parser.get_structure(pdb_path.stem, str(pdb_path))

    models = list(structure.get_models())
    if not models:
        raise ValueError("PDB contains no models.")
    if model_index < 0 or model_index >= len(models):
        raise ValueError(
            f"Requested model index {model_index}, but PDB contains {len(models)} model(s)."
        )

    atom_map: dict[AtomKey, np.ndarray] = {}
    occupancy_by_key: dict[AtomKey, float] = {}
    ordered_keys: list[AtomKey] = []
    duplicate_keys = 0

    for atom in models[model_index].get_atoms():
        residue = atom.get_parent()
        if not include_hetero and residue.id[0] != " ":
            continue
        if not atom_matches_selection(atom, atom_selection):
            continue

        coord = np.asarray(atom.get_coord(), dtype=np.float64)
        if coord.shape != (3,) or not np.isfinite(coord).all():
            continue

        key = make_atom_key(atom)
        occupancy = atom.get_occupancy()
        occupancy_value = float(occupancy) if occupancy is not None else -np.inf

        if key in atom_map:
            duplicate_keys += 1
            if occupancy_value <= occupancy_by_key[key]:
                continue
        else:
            ordered_keys.append(key)

        atom_map[key] = coord
        occupancy_by_key[key] = occupancy_value

    if not atom_map:
        raise ValueError(f"No atoms selected with atom selection `{atom_selection}`.")

    return StructureAtoms(
        path=pdb_path,
        atoms=atom_map,
        ordered_keys=ordered_keys,
        duplicate_keys=duplicate_keys,
    )


def common_reference_keys(
    references: Mapping[str, StructureAtoms],
    min_atoms: int,
) -> list[AtomKey]:
    common_keys: set[AtomKey] | None = None
    for label in REFERENCE_LABELS:
        keys = set(references[label].atoms)
        common_keys = keys if common_keys is None else common_keys & keys

    ordered_common_keys = [
        key for key in references[REFERENCE_LABELS[0]].ordered_keys if key in (common_keys or set())
    ]
    if len(ordered_common_keys) < min_atoms:
        raise ValueError(
            "References do not share enough atoms for alignment: "
            f"found {len(ordered_common_keys)}, require at least {min_atoms}."
        )
    return ordered_common_keys


def coords_for_keys(atom_map: Mapping[AtomKey, np.ndarray], keys: list[AtomKey]) -> np.ndarray:
    return np.stack([atom_map[key] for key in keys], axis=0)


def print_reference_diagnostics(
    references: Mapping[str, StructureAtoms],
    reference_common_keys: list[AtomKey],
    min_atoms: int,
    duplicate_threshold: float,
) -> None:
    reference_coords = {
        label: coords_for_keys(references[label].atoms, reference_common_keys)
        for label in REFERENCE_LABELS
    }

    print("Pairwise reference aligned RMSDs:", flush=True)
    for first_index, first_label in enumerate(REFERENCE_LABELS):
        for second_label in REFERENCE_LABELS[first_index + 1 :]:
            rmsd = aligned_rmsd(
                reference_coords[first_label],
                reference_coords[second_label],
                min_atoms=min_atoms,
            )
            print(
                f"  {first_label} vs {second_label}: {rmsd:.6f} A",
                flush=True,
            )
            if duplicate_threshold > 0 and rmsd < duplicate_threshold:
                print(
                    "  WARNING: these references are nearly identical at the "
                    f"{duplicate_threshold:.6f} A threshold. Closest-reference "
                    "assignments between them may be decided by tiny coordinate "
                    "rounding differences.",
                    flush=True,
                )


def comparison_key_sets(
    mobile_atoms: StructureAtoms,
    reference_common_keys: list[AtomKey],
    match_mode: str,
    min_atoms: int,
) -> tuple[list[AtomKey], list[AtomKey], str, int]:
    identity_mobile_keys = [
        key for key in reference_common_keys if key in mobile_atoms.atoms
    ]
    if match_mode in {"auto", "identity"} and len(identity_mobile_keys) >= min_atoms:
        return (
            identity_mobile_keys,
            identity_mobile_keys,
            "identity",
            len(reference_common_keys) - len(identity_mobile_keys),
        )

    if match_mode == "identity":
        raise ValueError(
            "Not enough atoms shared with the references by chain/residue/atom identity: "
            f"found {len(identity_mobile_keys)}, require at least {min_atoms}."
        )

    if len(mobile_atoms.ordered_keys) != len(reference_common_keys):
        raise ValueError(
            "Cannot use order-based matching because selected atom counts differ: "
            f"sample has {len(mobile_atoms.ordered_keys)}, references have "
            f"{len(reference_common_keys)} common selected atoms. "
            f"Identity overlap was {len(identity_mobile_keys)} atom(s)."
        )

    if len(mobile_atoms.ordered_keys) < min_atoms:
        raise ValueError(
            "Not enough selected atoms for order-based alignment: "
            f"found {len(mobile_atoms.ordered_keys)}, require at least {min_atoms}."
        )

    return (
        list(mobile_atoms.ordered_keys),
        list(reference_common_keys),
        "order",
        0,
    )


def aligned_rmsd(mobile: np.ndarray, target: np.ndarray, min_atoms: int) -> float:
    valid = np.isfinite(mobile).all(axis=1) & np.isfinite(target).all(axis=1)
    if int(valid.sum()) < min_atoms:
        raise ValueError(
            f"Need at least {min_atoms} finite matched atoms, found {int(valid.sum())}."
        )

    mobile_valid = mobile[valid].astype(np.float64, copy=False)
    target_valid = target[valid].astype(np.float64, copy=False)

    mobile_center = mobile_valid.mean(axis=0)
    target_center = target_valid.mean(axis=0)
    mobile_centered = mobile_valid - mobile_center
    target_centered = target_valid - target_center

    covariance = mobile_centered.T @ target_centered
    u, _, vt = np.linalg.svd(covariance)
    det = np.linalg.det(u @ vt)
    correction = np.eye(3)
    correction[-1, -1] = 1.0 if det >= 0.0 else -1.0
    rotation = u @ correction @ vt

    aligned = (mobile_valid - mobile_center) @ rotation + target_center
    squared_distances = np.sum((aligned - target_valid) ** 2, axis=1)
    return float(np.sqrt(np.mean(squared_distances)))


def blank_result_row(pdb_path: Path, error: str) -> dict[str, object]:
    row: dict[str, object] = {column: np.nan for column in RESULT_COLUMNS}
    row.update(
        {
            "filename": pdb_path.name,
            "pdb_path": str(pdb_path),
            "status": "error",
            "error": error,
            "closest_reference": "",
        }
    )
    return row


def analyze_pdb_file(
    pdb_path: Path,
    parser: PDBParser,
    references: Mapping[str, StructureAtoms],
    reference_common_keys: list[AtomKey],
    model_index: int,
    atom_selection: str,
    match_mode: str,
    include_hetero: bool,
    min_atoms: int,
) -> dict[str, object]:
    try:
        mobile_atoms = load_structure_atoms(
            pdb_path=pdb_path,
            parser=parser,
            model_index=model_index,
            atom_selection=atom_selection,
            include_hetero=include_hetero,
        )

        mobile_keys, reference_keys, match_method, missing_reference_atoms = comparison_key_sets(
            mobile_atoms=mobile_atoms,
            reference_common_keys=reference_common_keys,
            match_mode=match_mode,
            min_atoms=min_atoms,
        )

        mobile_coords = coords_for_keys(mobile_atoms.atoms, mobile_keys)
        rmsds = {}
        for label in REFERENCE_LABELS:
            target_coords = coords_for_keys(references[label].atoms, reference_keys)
            rmsds[label] = aligned_rmsd(
                mobile=mobile_coords,
                target=target_coords,
                min_atoms=min_atoms,
            )

        closest_reference = min(
            REFERENCE_LABELS,
            key=lambda label: (rmsds[label], REFERENCE_LABELS.index(label)),
        )
        min_rmsd = rmsds[closest_reference]

        return {
            "filename": pdb_path.name,
            "pdb_path": str(pdb_path),
            "status": "ok",
            "error": "",
            "n_atoms_in_file": len(mobile_atoms.atoms),
            "n_atoms_used": len(mobile_keys),
            "n_reference_common_atoms": len(reference_common_keys),
            "n_missing_reference_common_atoms": missing_reference_atoms,
            "match_method": match_method,
            "rmsd_active": rmsds["active"],
            "rmsd_inactive": rmsds["inactive"],
            "rmsd_pas": rmsds["pas"],
            "closest_reference": closest_reference,
            "min_rmsd": min_rmsd,
        }
    except Exception as exc:
        return blank_result_row(pdb_path, str(exc))


def direct_pdb_files(base_directory: Path) -> list[Path]:
    return sorted(
        path
        for path in base_directory.iterdir()
        if path.is_file() and path.suffix.lower() == ".pdb"
    )


def finite_float(value: object) -> float | None:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric_value):
        return None
    return numeric_value


def csv_value(value: object) -> object:
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (float, np.floating)):
        numeric_value = finite_float(value)
        if numeric_value is not None:
            return f"{numeric_value:.6f}"
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return value


def write_results_csv(rows: list[dict[str, object]], csv_path: Path) -> None:
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: csv_value(row.get(column, "")) for column in RESULT_COLUMNS})


def valid_result_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    valid_rows = []
    for row in rows:
        if row.get("status") != "ok":
            continue
        if row.get("closest_reference") not in REFERENCE_LABELS:
            continue
        if finite_float(row.get("min_rmsd")) is None:
            continue
        valid_rows.append(row)
    return valid_rows


def plot_summary(rows: list[dict[str, object]], figure_path: Path) -> None:
    valid = valid_result_rows(rows)

    colors = {
        "active": "#4C78A8",
        "inactive": "#F58518",
        "pas": "#54A24B",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    counts = {
        label: sum(1 for row in valid if row.get("closest_reference") == label)
        for label in REFERENCE_LABELS
    }
    axes[0].bar(
        REFERENCE_LABELS,
        [counts[label] for label in REFERENCE_LABELS],
        color=[colors[label] for label in REFERENCE_LABELS],
    )
    axes[0].set_title("Closest reference assignments")
    axes[0].set_xlabel("Reference label")
    axes[0].set_ylabel("Number of structures")

    grouped_values = [
        np.array(
            [
                float(row["min_rmsd"])
                for row in valid
                if row.get("closest_reference") == label
            ],
            dtype=np.float64,
        )
        for label in REFERENCE_LABELS
    ]
    non_empty_values = [values for values in grouped_values if values.size > 0]

    if non_empty_values:
        boxplot_result = axes[1].boxplot(
            grouped_values,
            tick_labels=REFERENCE_LABELS,
            showmeans=True,
            patch_artist=True,
            medianprops={"color": "black"},
            meanprops={
                "marker": "D",
                "markerfacecolor": "black",
                "markeredgecolor": "black",
                "markersize": 4,
            },
        )
        for patch, label in zip(boxplot_result["boxes"], REFERENCE_LABELS):
            patch.set_facecolor(colors[label])
            patch.set_alpha(0.35)

        rng = np.random.default_rng(0)
        for x_position, (label, values) in enumerate(zip(REFERENCE_LABELS, grouped_values), start=1):
            if values.size == 0:
                continue
            jitter = rng.normal(loc=0.0, scale=0.04, size=values.size)
            axes[1].scatter(
                np.full(values.size, x_position) + jitter,
                values,
                color=colors[label],
                alpha=0.55,
                s=18,
                linewidths=0,
            )
    else:
        axes[1].text(
            0.5,
            0.5,
            "No valid RMSD results",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
        )
        axes[1].set_xticks(range(1, len(REFERENCE_LABELS) + 1), REFERENCE_LABELS)

    axes[1].set_title("Minimum RMSD by assigned reference")
    axes[1].set_xlabel("Assigned reference label")
    axes[1].set_ylabel("Minimum aligned RMSD (A)")

    fig.tight_layout()
    fig.savefig(figure_path, dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    if args.min_atoms < 3:
        raise ValueError("--min-atoms must be at least 3 for rigid-body alignment.")
    if args.progress_every < 0:
        raise ValueError("--progress-every must be >= 0.")
    if args.reference_duplicate_threshold < 0:
        raise ValueError("--reference-duplicate-threshold must be >= 0.")

    base_directory = args.base_directory.expanduser().resolve()
    if not base_directory.is_dir():
        raise NotADirectoryError(f"Base directory not found: {base_directory}")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / args.csv_name
    figure_path = output_dir / args.figure_name

    parser = PDBParser(QUIET=True)
    reference_paths = {
        "active": args.active_pdb,
        "inactive": args.inactive_pdb,
        "pas": args.pas_pdb,
    }
    references = {
        label: load_structure_atoms(
            pdb_path=path,
            parser=parser,
            model_index=args.model_index,
            atom_selection=args.atom_selection,
            include_hetero=args.include_hetero,
        )
        for label, path in reference_paths.items()
    }
    reference_common_keys = common_reference_keys(
        references=references,
        min_atoms=args.min_atoms,
    )
    print_reference_diagnostics(
        references=references,
        reference_common_keys=reference_common_keys,
        min_atoms=args.min_atoms,
        duplicate_threshold=args.reference_duplicate_threshold,
    )

    pdb_paths = direct_pdb_files(base_directory)
    rows = []
    for index, pdb_path in enumerate(pdb_paths, start=1):
        rows.append(
            analyze_pdb_file(
                pdb_path=pdb_path,
                parser=parser,
                references=references,
                reference_common_keys=reference_common_keys,
                model_index=args.model_index,
                atom_selection=args.atom_selection,
                match_mode=args.match_mode,
                include_hetero=args.include_hetero,
                min_atoms=args.min_atoms,
            )
        )
        if args.progress_every and index % args.progress_every == 0:
            print(f"Processed {index}/{len(pdb_paths)} PDB file(s)", flush=True)

    write_results_csv(rows, csv_path)
    plot_summary(rows, figure_path)

    valid_count = sum(1 for row in rows if row.get("status") == "ok")
    error_count = sum(1 for row in rows if row.get("status") == "error")
    print(f"Found {len(pdb_paths)} PDB file(s) in {base_directory}")
    print(f"Valid RMSD assignments: {valid_count}; errors: {error_count}")
    print(f"Reference common atoms used as candidates: {len(reference_common_keys)}")
    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote figure: {figure_path}")


if __name__ == "__main__":
    main()
