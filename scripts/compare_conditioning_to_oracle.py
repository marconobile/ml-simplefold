#!/usr/bin/env python3
"""Compare conditioning labels, RMSDs, and mismatch-associated dihedral errors."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from simplefold.conditioned_sampling.outputs import (  # noqa: E402
    write_dihedral_error_histogram_png,
)
from simplefold.conditioned_sampling.dihedrals import (  # noqa: E402
    symmetry_correct_dihedral_errors,
)


ASSIGNED_TOKEN = "_assigned_clusters.npz"
CONDITIONED_EVAL_TOKEN = "_conditioned_eval.npz"
CONDITIONED_EVAL_JSON_TOKEN = "_conditioned_eval.json"
CLUSTER_KEY = "atom_idx_and_glob_cluster_id_per_frame"
SOURCE_SUFFIX = "_samples"
DEFAULT_REFERENCE_PDB = Path(
    "/home/nobilm@usi.ch/ml-simplefold/data/pdb_for_sampling_jupyter/"
    "INApo_no_caps.pdb"
)
VMD_SELECTION = (
    "name CA and resid 2 to 30 35 to 65 69 to 104 113 to 138 "
    "169 to 209 215 to 255 261 to 287 288 to 300"
)
VMD_RESID_RANGES = (
    (2, 30),
    (35, 65),
    (69, 104),
    (113, 138),
    (169, 209),
    (215, 255),
    (261, 287),
    (288, 300),
)
VMD_RESIDS = tuple(
    resid
    for first_resid, last_resid in VMD_RESID_RANGES
    for resid in range(first_resid, last_resid + 1)
)
RESIDUE_KEY_PATTERN = re.compile(r"^res_(-?\d+)$")
PDB_ATOM_RECORDS = ("ATOM  ", "HETATM")
BACKBONE_ATOM_NAMES = frozenset(("N", "CA", "C", "O"))
RMSD_SELECTIONS = (
    (
        "vmd_ca_residues",
        VMD_SELECTION,
    ),
    (
        "ca",
        "name CA",
    ),
    (
        "backbone",
        "protein backbone atoms (name N CA C O)",
    ),
    (
        "protein_not_backbone",
        "protein and not backbone",
    ),
    (
        "all_atoms",
        "all ATOM/HETATM records",
    ),
)
RMSD_SELECTION_KEYS = tuple(key for key, _ in RMSD_SELECTIONS)
ERROR_SELECTION_KEYS = ("vmd_ca_residues", "all_atoms")
RMSD_SELECTION_DEFINITIONS = dict(RMSD_SELECTIONS)
RMSD_SELECTION_PLOT_TITLES = {
    "vmd_ca_residues": "Strict VMD CA/resid selection",
    "ca": "All CA atoms",
    "backbone": "Protein backbone (N, CA, C, O)",
    "protein_not_backbone": "Protein and not backbone",
    "all_atoms": "All atoms",
}
RMSD_SELECTION_COLORS = {
    "vmd_ca_residues": "#0072B2",
    "ca": "#E69F00",
    "backbone": "#009E73",
    "protein_not_backbone": "#CC79A7",
    "all_atoms": "#D55E00",
}
VIOLIN_Y_AXIS_THRESHOLD_FRACTION = 0.05
WRONG_CLUSTER_DIHEDRAL_PLOT_TOKEN = (
    "_conditioned_eval_wrong_cluster_dihedral_error_histograms.png"
)


@dataclass(frozen=True)
class PdbAtom:
    record_name: str
    atom_name: str
    alternate_location: str
    residue_name: str
    chain_id: str
    residue_number: int
    insertion_code: str
    coords: tuple[float, float, float]
    line_number: int

    @property
    def atom_identity(self) -> tuple[str, str, str, str, str, int, str]:
        return (
            self.record_name,
            self.atom_name,
            self.alternate_location,
            self.residue_name,
            self.chain_id,
            self.residue_number,
            self.insertion_code,
        )

    @property
    def residue_identity(self) -> tuple[str, int, str, str]:
        return (
            self.chain_id,
            self.residue_number,
            self.insertion_code,
            self.residue_name,
        )


@dataclass(frozen=True)
class MismatchDihedralData:
    sample_name: str
    output_path: Path
    dihedral_diff_rad: np.ndarray
    dihedral_mask: np.ndarray
    dihedral_keys: tuple[str, ...]
    residue_names: np.ndarray
    mismatch_count: int
    comparable_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the conditioning vector saved by sample_with_conditioning.py "
            "against oracle labels written by assign_conditioned_eval_sample_clusters.py, "
            "and calculate strictly paired sampled-PDB-vs-target-PDB RMSDs for five "
            "atom selections, per-residue cluster accuracy matrices, and "
            "mismatch-associated dihedral-error distributions."
        )
    )
    parser.add_argument(
        "--base-path",
        "--base_path",
        dest="base_path",
        type=Path,
        required=True,
        help="Directory to recursively search for *_assigned_clusters.npz files.",
    )
    parser.add_argument(
        "--out-dir",
        "--out_dir",
        dest="out_dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to --base-path.",
    )
    parser.add_argument(
        "--output-prefix",
        default="conditioning_vs_oracle",
        help="Prefix for CSV, TXT, and PNG outputs.",
    )
    parser.add_argument(
        "--reference-pdb",
        type=Path,
        default=DEFAULT_REFERENCE_PDB,
        help=(
            "Fixed reference PDB used for the per-selection sampled-vs-reference "
            f"RMSDs reported in the CSV and text outputs. Defaults to "
            f"{DEFAULT_REFERENCE_PDB}."
        ),
    )
    parser.add_argument(
        "--all-res",
        "--all_res",
        dest="all_res",
        action="store_true",
        help=(
            "Use all residues for the primary cluster-label comparison. By "
            f"default, use the VMD selection {VMD_SELECTION!r}. All five PDB "
            "RMSD atom selections are always evaluated."
        ),
    )
    parser.add_argument(
        "--dihedral-error-bins",
        "--dihedral_error_bins",
        type=int,
        default=72,
        help="Histogram bins for mismatch-only dihedral-error plots. Defaults to 72.",
    )
    return parser.parse_args()


def find_assigned_cluster_files(base_path: Path) -> list[Path]:
    return sorted(path for path in base_path.rglob(f"*{ASSIGNED_TOKEN}") if path.is_file())


def conditioned_eval_path_for_assigned(path: Path) -> Path:
    if not path.name.endswith(ASSIGNED_TOKEN):
        raise ValueError(f"Unexpected assigned-clusters filename: {path}")
    return path.with_name(path.name.replace(ASSIGNED_TOKEN, CONDITIONED_EVAL_TOKEN, 1))


def conditioned_eval_json_path_for_assigned(path: Path) -> Path:
    if not path.name.endswith(ASSIGNED_TOKEN):
        raise ValueError(f"Unexpected assigned-clusters filename: {path}")
    return path.with_name(path.name.replace(ASSIGNED_TOKEN, CONDITIONED_EVAL_JSON_TOKEN, 1))


def load_npz_array(path: Path, key: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        if key not in data.files:
            available = ", ".join(data.files)
            raise KeyError(f"{path} is missing {key!r}. Available keys: {available}")
        return np.asarray(data[key])


def optional_npz_array(path: Path, key: str) -> np.ndarray | None:
    with np.load(path, allow_pickle=False) as data:
        if key not in data.files:
            return None
        return np.asarray(data[key])


def one_dimensional(array: np.ndarray, name: str, path: Path) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim == 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 1:
        raise ValueError(f"{name} for {path} must be 1D, got shape {array.shape}.")
    return array


def load_metrics(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open() as f:
        return json.load(f)


def source_type_for_path(path: Path, base_path: Path) -> str:
    candidates = []
    try:
        candidates.extend(path.relative_to(base_path).parts[:-1])
    except ValueError:
        pass
    candidates.extend(parent.name for parent in path.parents)
    candidates.append(base_path.name)

    for part in candidates:
        if part.endswith(SOURCE_SUFFIX):
            return part[: -len(SOURCE_SUFFIX)]

    name = path.name.lower()
    for source in ("active", "inactive", "pas"):
        if source in name:
            return source
    return "unknown"


def sample_name_for_assigned(path: Path) -> str:
    return path.name.replace(ASSIGNED_TOKEN, "")


def load_conditioning_labels(conditioned_eval_path: Path) -> np.ndarray:
    labels = optional_npz_array(conditioned_eval_path, "conditioning_cluster_labels")
    if labels is None:
        labels = optional_npz_array(conditioned_eval_path, "original_cluster_labels")
    if labels is None:
        raise KeyError(
            f"{conditioned_eval_path} is missing both 'conditioning_cluster_labels' "
            "and 'original_cluster_labels'."
        )
    return one_dimensional(labels, "conditioning labels", conditioned_eval_path).astype(
        np.int64,
        copy=False,
    )


def load_cluster_counts(assigned_path: Path) -> np.ndarray:
    cluster_counts = optional_npz_array(assigned_path, "cluster_counts")
    if cluster_counts is None:
        cluster_counts = optional_npz_array(assigned_path, "merged__cluster_counts")
    if cluster_counts is None:
        raise KeyError(
            f"{assigned_path} is missing both 'cluster_counts' and "
            "'merged__cluster_counts'."
        )
    return one_dimensional(cluster_counts, "cluster_counts", assigned_path).astype(
        np.int64,
        copy=False,
    )


def load_atom_resids(conditioned_eval_path: Path, metrics: dict[str, Any]) -> np.ndarray:
    atom_resids = optional_npz_array(conditioned_eval_path, "atom_resids")
    if atom_resids is not None:
        return one_dimensional(atom_resids, "atom_resids", conditioned_eval_path).astype(
            np.int64,
            copy=False,
        )

    raw_npz = metrics.get("raw_npz_path")
    if raw_npz:
        raw_npz_path = Path(raw_npz).expanduser()
        if raw_npz_path.is_file():
            raw_atom_resids = optional_npz_array(raw_npz_path, "atom_resids")
            if raw_atom_resids is not None:
                return one_dimensional(raw_atom_resids, "atom_resids", raw_npz_path).astype(
                    np.int64,
                    copy=False,
                )

    raise KeyError(
        f"Could not load atom_resids for {conditioned_eval_path}. "
        "Expected atom_resids in the conditioned NPZ or in metrics['raw_npz_path']."
    )


def metadata_npz_paths(
    conditioned_eval_path: Path,
    metrics: dict[str, Any],
) -> list[Path]:
    paths = [conditioned_eval_path] if conditioned_eval_path.is_file() else []
    raw_npz = metrics.get("raw_npz_path")
    if raw_npz:
        raw_npz_path = Path(raw_npz).expanduser()
        if raw_npz_path.is_file() and raw_npz_path != conditioned_eval_path:
            paths.append(raw_npz_path)
    return paths


def load_optional_metadata_array(
    conditioned_eval_path: Path,
    metrics: dict[str, Any],
    key: str,
) -> tuple[np.ndarray | None, Path | None]:
    for candidate in metadata_npz_paths(conditioned_eval_path, metrics):
        value = optional_npz_array(candidate, key)
        if value is not None:
            return value, candidate
    return None, None


def text_value(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def load_vmd_resids_by_residue_index(
    conditioned_eval_path: Path,
    metrics: dict[str, Any],
    n_residues: int,
) -> np.ndarray:
    residue_keys, source_path = load_optional_metadata_array(
        conditioned_eval_path,
        metrics,
        "residue_keys",
    )
    if residue_keys is None or source_path is None:
        raise KeyError(
            "The strict VMD-residue evaluation requires 'residue_keys' in the "
            f"conditioned NPZ or metrics['raw_npz_path'] for {conditioned_eval_path}. "
            "Use --all-res only if evaluating every residue is intended."
        )

    residue_keys = one_dimensional(residue_keys, "residue_keys", source_path)
    if residue_keys.shape[0] != n_residues:
        raise ValueError(
            f"residue_keys for {source_path} has {residue_keys.shape[0]} entries, "
            f"but the cluster arrays have {n_residues} residues."
        )

    vmd_resids = []
    for residue_index, raw_key in enumerate(residue_keys):
        key = text_value(raw_key)
        match = RESIDUE_KEY_PATTERN.fullmatch(key)
        if match is None:
            raise ValueError(
                f"Invalid residue key {key!r} at residue index {residue_index} in "
                f"{source_path}; strict VMD selection requires keys of the form "
                "'res_<integer>'."
            )
        vmd_resids.append(int(match.group(1)))

    result = np.asarray(vmd_resids, dtype=np.int64)
    unique, counts = np.unique(result, return_counts=True)
    duplicates = unique[counts != 1]
    if duplicates.size:
        raise ValueError(
            f"Duplicate VMD residue IDs in {source_path}: {duplicates.tolist()}."
        )
    return result


def validate_selected_ca_atoms(
    conditioned_eval_path: Path,
    metrics: dict[str, Any],
    atom_resids: np.ndarray,
    vmd_resids_by_index: np.ndarray,
    selection_mask: np.ndarray,
) -> None:
    atom_names, source_path = load_optional_metadata_array(
        conditioned_eval_path,
        metrics,
        "atom_names",
    )
    if atom_names is None or source_path is None:
        raise KeyError(
            "The strict VMD selection includes 'name CA', but 'atom_names' was not "
            f"found for {conditioned_eval_path}. Use --all-res only if evaluating "
            "every residue is intended."
        )
    atom_names = one_dimensional(atom_names, "atom_names", source_path)
    if atom_names.shape != atom_resids.shape:
        raise ValueError(
            f"atom_names/atom_resids shape mismatch for {source_path}: "
            f"{atom_names.shape} vs {atom_resids.shape}."
        )

    normalized_names = np.asarray(
        [text_value(name).strip().upper() for name in atom_names],
        dtype=str,
    )
    missing_ca_resids = []
    for residue_index in np.flatnonzero(selection_mask):
        has_ca = np.any((atom_resids == residue_index) & (normalized_names == "CA"))
        if not has_ca:
            missing_ca_resids.append(int(vmd_resids_by_index[residue_index]))
    if missing_ca_resids:
        raise ValueError(
            f"The requested VMD residues without a CA atom in {source_path} are: "
            f"{missing_ca_resids}."
        )


def strict_vmd_selection_mask(
    conditioned_eval_path: Path,
    metrics: dict[str, Any],
    atom_resids: np.ndarray,
    n_residues: int,
) -> tuple[np.ndarray, np.ndarray]:
    vmd_resids_by_index = load_vmd_resids_by_residue_index(
        conditioned_eval_path,
        metrics,
        n_residues,
    )
    selection_mask = np.isin(vmd_resids_by_index, VMD_RESIDS)
    selected_resids = set(vmd_resids_by_index[selection_mask].tolist())
    requested_resids = set(VMD_RESIDS)
    if selected_resids != requested_resids:
        missing = sorted(requested_resids - selected_resids)
        unexpected = sorted(selected_resids - requested_resids)
        raise ValueError(
            "Residue metadata does not exactly satisfy the requested VMD selection "
            f"for {conditioned_eval_path}. Missing={missing}; unexpected={unexpected}."
        )
    if int(selection_mask.sum()) != len(VMD_RESIDS):
        raise ValueError(
            f"Expected exactly {len(VMD_RESIDS)} selected residues for "
            f"{conditioned_eval_path}, got {int(selection_mask.sum())}."
        )
    validate_selected_ca_atoms(
        conditioned_eval_path,
        metrics,
        atom_resids,
        vmd_resids_by_index,
        selection_mask,
    )
    return selection_mask, vmd_resids_by_index


def resolve_recorded_path(raw_path: Any, metrics_path: Path, field_name: str) -> Path:
    if not raw_path:
        raise KeyError(f"{metrics_path} is missing required field {field_name!r}.")
    path = Path(str(raw_path)).expanduser()
    if not path.is_absolute():
        path = metrics_path.parent / path
    return path.resolve()


def expected_rmsd_pdb_paths(conditioned_eval_path: Path) -> tuple[Path, Path, Path]:
    if not conditioned_eval_path.name.endswith(CONDITIONED_EVAL_TOKEN):
        raise ValueError(f"Unexpected conditioned-evaluation filename: {conditioned_eval_path}")
    sampled_pdb = conditioned_eval_path.with_name(
        conditioned_eval_path.name.replace(
            CONDITIONED_EVAL_TOKEN,
            "_conditioned_eval_sampled.pdb",
            1,
        )
    )
    target_cif = conditioned_eval_path.with_name(
        conditioned_eval_path.name.replace(
            CONDITIONED_EVAL_TOKEN,
            "_conditioned_eval_target_reference.cif",
            1,
        )
    )
    return sampled_pdb, target_cif, target_cif.with_suffix(".pdb")


def strict_rmsd_pdb_paths(
    conditioned_eval_path: Path,
    metrics_path: Path,
    metrics: dict[str, Any],
) -> tuple[Path, Path]:
    expected_sampled_pdb, expected_target_cif, expected_target_pdb = (
        expected_rmsd_pdb_paths(conditioned_eval_path)
    )
    recorded_sampled_pdb = resolve_recorded_path(
        metrics.get("sampled_pdb_path"),
        metrics_path,
        "sampled_pdb_path",
    )
    recorded_target_cif = resolve_recorded_path(
        metrics.get("target_reference_cif_path"),
        metrics_path,
        "target_reference_cif_path",
    )

    expected_sampled_pdb = expected_sampled_pdb.resolve()
    expected_target_cif = expected_target_cif.resolve()
    expected_target_pdb = expected_target_pdb.resolve()
    if recorded_sampled_pdb != expected_sampled_pdb:
        raise ValueError(
            "Refusing RMSD evaluation because metrics['sampled_pdb_path'] does not "
            f"match the conditioned-evaluation sample. Recorded={recorded_sampled_pdb}; "
            f"expected={expected_sampled_pdb}."
        )
    if recorded_target_cif != expected_target_cif:
        raise ValueError(
            "Refusing RMSD evaluation because metrics['target_reference_cif_path'] "
            "does not match the conditioned-evaluation sample. "
            f"Recorded={recorded_target_cif}; expected={expected_target_cif}."
        )
    if recorded_sampled_pdb == expected_target_pdb:
        raise ValueError(
            f"Sampled and target PDB paths unexpectedly resolve to the same file: "
            f"{recorded_sampled_pdb}."
        )
    if not recorded_sampled_pdb.is_file():
        raise FileNotFoundError(f"Sampled PDB not found: {recorded_sampled_pdb}")
    if not expected_target_pdb.is_file():
        raise FileNotFoundError(
            "Target PDB not found beside the recorded target-reference CIF: "
            f"{expected_target_pdb}"
        )
    return recorded_sampled_pdb, expected_target_pdb


def read_pdb_atoms(path: Path) -> list[PdbAtom]:
    atoms = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.startswith(PDB_ATOM_RECORDS):
                continue
            if len(line) < 54:
                raise ValueError(
                    f"Malformed ATOM/HETATM record at {path}:{line_number}; "
                    "the coordinate columns are missing."
                )
            try:
                residue_number = int(line[22:26])
                coords = (
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                )
            except ValueError as exc:
                raise ValueError(
                    f"Malformed residue number or coordinates at {path}:{line_number}."
                ) from exc
            if not np.isfinite(coords).all():
                raise ValueError(f"Non-finite coordinates at {path}:{line_number}.")
            atoms.append(
                PdbAtom(
                    record_name=line[0:6].strip(),
                    atom_name=line[12:16].strip().upper(),
                    alternate_location=line[16].strip(),
                    residue_name=line[17:20].strip().upper(),
                    chain_id=line[21].strip(),
                    residue_number=residue_number,
                    insertion_code=line[26].strip(),
                    coords=coords,
                    line_number=line_number,
                )
            )

    if not atoms:
        raise ValueError(f"No ATOM/HETATM records found in {path}.")
    identities = [atom.atom_identity for atom in atoms]
    if len(set(identities)) != len(identities):
        seen = set()
        duplicate = None
        for identity in identities:
            if identity in seen:
                duplicate = identity
                break
            seen.add(identity)
        raise ValueError(f"Duplicate atom identity in {path}: {duplicate}.")
    return atoms


def pdb_residue_ordinals(atoms: list[PdbAtom], path: Path) -> np.ndarray:
    ordinal_by_identity: dict[tuple[str, int, str, str], int] = {}
    completed_residues: set[tuple[str, int, str, str]] = set()
    ordinals = []
    previous_identity = None
    for atom in atoms:
        identity = atom.residue_identity
        if identity != previous_identity:
            if identity in completed_residues:
                raise ValueError(
                    f"Residue {identity} is non-contiguous in {path}; atom order is "
                    "not safe for NPZ-to-PDB validation."
                )
            if previous_identity is not None:
                completed_residues.add(previous_identity)
            ordinal_by_identity.setdefault(identity, len(ordinal_by_identity))
            previous_identity = identity
        ordinals.append(ordinal_by_identity[identity])
    return np.asarray(ordinals, dtype=np.int64)


def validate_pdb_coordinates_against_npz(
    pdb_coords: np.ndarray,
    npz_coords: np.ndarray,
    pdb_path: Path,
    conditioned_eval_path: Path,
    npz_key: str,
    atol: float = 1e-2,
) -> str:
    npz_coords = np.asarray(npz_coords, dtype=np.float64)
    if pdb_coords.shape != npz_coords.shape:
        raise ValueError(
            f"Refusing RMSD evaluation because {pdb_path} has coordinate shape "
            f"{pdb_coords.shape}, while {conditioned_eval_path}[{npz_key!r}] has "
            f"shape {npz_coords.shape}."
        )
    direct_delta = np.abs(pdb_coords - npz_coords)
    direct_max = float(np.max(direct_delta))
    if direct_max <= atol:
        return "direct"

    inverted_delta = np.abs(pdb_coords + npz_coords)
    inverted_max = float(np.max(inverted_delta))
    if inverted_max <= atol:
        return "global_inversion"

    direct_rmsd = float(np.sqrt(np.mean(np.sum(direct_delta**2, axis=1))))
    inverted_rmsd = float(np.sqrt(np.mean(np.sum(inverted_delta**2, axis=1))))
    raise ValueError(
        "Refusing RMSD evaluation because the selected PDB does not reproduce its "
        f"stored conditioned-evaluation coordinates: pdb={pdb_path}; "
        f"npz={conditioned_eval_path}; key={npz_key!r}; "
        f"direct_max_abs_delta={direct_max:.6f}; direct_rmsd={direct_rmsd:.6f}; "
        f"inverted_max_abs_delta={inverted_max:.6f}; "
        f"inverted_rmsd={inverted_rmsd:.6f}."
    )


def validate_pdb_pair_and_metadata(
    sampled_pdb_path: Path,
    target_pdb_path: Path,
    conditioned_eval_path: Path,
    metrics: dict[str, Any],
    atom_resids: np.ndarray,
    n_residues: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, str]:
    sampled_atoms = read_pdb_atoms(sampled_pdb_path)
    target_atoms = read_pdb_atoms(target_pdb_path)
    sampled_identities = [atom.atom_identity for atom in sampled_atoms]
    target_identities = [atom.atom_identity for atom in target_atoms]
    if sampled_identities != target_identities:
        first_mismatch = next(
            (
                idx
                for idx, pair in enumerate(
                    zip(sampled_identities, target_identities, strict=False)
                )
                if pair[0] != pair[1]
            ),
            min(len(sampled_identities), len(target_identities)),
        )
        raise ValueError(
            "Refusing RMSD evaluation because sampled/target PDB atom identities "
            f"do not match exactly. sampled={sampled_pdb_path}; target={target_pdb_path}; "
            f"sampled_atoms={len(sampled_atoms)}; target_atoms={len(target_atoms)}; "
            f"first_mismatch_index={first_mismatch}."
        )

    atom_names, atom_names_path = load_optional_metadata_array(
        conditioned_eval_path,
        metrics,
        "atom_names",
    )
    if atom_names is None or atom_names_path is None:
        raise KeyError(
            f"Cannot validate PDB atom identities for {conditioned_eval_path}: "
            "'atom_names' is unavailable."
        )
    atom_names = one_dimensional(atom_names, "atom_names", atom_names_path)
    normalized_atom_names = np.asarray(
        [text_value(name).strip().upper() for name in atom_names],
        dtype=str,
    )
    if normalized_atom_names.shape != atom_resids.shape:
        raise ValueError(
            f"NPZ atom_names/atom_resids shape mismatch: {normalized_atom_names.shape} "
            f"vs {atom_resids.shape}."
        )
    if len(target_atoms) != atom_resids.shape[0]:
        raise ValueError(
            "Refusing RMSD evaluation because PDB and NPZ atom counts differ: "
            f"PDB={len(target_atoms)}, NPZ={atom_resids.shape[0]}."
        )

    pdb_atom_names = np.asarray([atom.atom_name for atom in target_atoms], dtype=str)
    if not np.array_equal(pdb_atom_names, normalized_atom_names):
        mismatch = np.flatnonzero(pdb_atom_names != normalized_atom_names)[:20].tolist()
        raise ValueError(
            "Refusing RMSD evaluation because target PDB atom names do not match "
            f"the raw NPZ atom order at indices {mismatch}."
        )
    target_residue_ordinals = pdb_residue_ordinals(target_atoms, target_pdb_path)
    sampled_residue_ordinals = pdb_residue_ordinals(sampled_atoms, sampled_pdb_path)
    if not np.array_equal(sampled_residue_ordinals, target_residue_ordinals):
        raise ValueError("Sampled and target PDB residue ordering differs.")
    if not np.array_equal(target_residue_ordinals, atom_resids):
        mismatch = np.flatnonzero(target_residue_ordinals != atom_resids)[:20].tolist()
        raise ValueError(
            "Refusing RMSD evaluation because target PDB residue order does not "
            f"match atom_resids at atom indices {mismatch}."
        )
    observed_residues = np.unique(target_residue_ordinals)
    if not np.array_equal(observed_residues, np.arange(n_residues, dtype=np.int64)):
        raise ValueError(
            f"PDB residues do not cover exactly 0..{n_residues - 1} by ordinal: "
            f"{target_pdb_path}."
        )

    sampled_coords = np.asarray([atom.coords for atom in sampled_atoms], dtype=np.float64)
    target_coords = np.asarray([atom.coords for atom in target_atoms], dtype=np.float64)
    sampled_npz_coords = load_npz_array(conditioned_eval_path, "aligned_sampled_coords")
    target_npz_coords = load_npz_array(conditioned_eval_path, "original_coords")
    sampled_coordinate_relation = validate_pdb_coordinates_against_npz(
        sampled_coords,
        sampled_npz_coords,
        sampled_pdb_path,
        conditioned_eval_path,
        "aligned_sampled_coords",
    )
    target_coordinate_relation = validate_pdb_coordinates_against_npz(
        target_coords,
        target_npz_coords,
        target_pdb_path,
        conditioned_eval_path,
        "original_coords",
    )
    return (
        sampled_coords,
        target_coords,
        pdb_atom_names,
        sampled_coordinate_relation,
        target_coordinate_relation,
    )


def is_hydrogen_atom_name(atom_name: str) -> bool:
    return re.fullmatch(r"[0-9]?H.*", atom_name.upper()) is not None


def reference_coords_in_sample_atom_order(
    reference_pdb_path: Path,
    atom_names: np.ndarray,
    atom_resids: np.ndarray,
    vmd_resids_by_index: np.ndarray,
) -> np.ndarray:
    reference_atoms = read_pdb_atoms(reference_pdb_path)
    reference_chains = {atom.chain_id for atom in reference_atoms}
    if len(reference_chains) != 1:
        raise ValueError(
            f"Reference PDB must contain exactly one atom-bearing chain, got "
            f"{sorted(reference_chains)} in {reference_pdb_path}."
        )
    if any(atom.alternate_location for atom in reference_atoms):
        raise ValueError(
            f"Reference PDB contains alternate locations and cannot be mapped "
            f"unambiguously: {reference_pdb_path}."
        )
    if any(atom.insertion_code for atom in reference_atoms):
        raise ValueError(
            f"Reference PDB contains insertion codes and cannot be matched strictly "
            f"to VMD residue IDs: {reference_pdb_path}."
        )

    expected_vmd_resids = set(vmd_resids_by_index.tolist())
    observed_vmd_resids = {atom.residue_number for atom in reference_atoms}
    if observed_vmd_resids != expected_vmd_resids:
        missing = sorted(expected_vmd_resids - observed_vmd_resids)
        unexpected = sorted(observed_vmd_resids - expected_vmd_resids)
        raise ValueError(
            "Reference PDB residue IDs do not exactly match the sampled topology: "
            f"reference={reference_pdb_path}; missing={missing}; unexpected={unexpected}."
        )

    reference_by_key: dict[tuple[int, str], PdbAtom] = {}
    for atom in reference_atoms:
        key = (atom.residue_number, atom.atom_name)
        if key in reference_by_key:
            raise ValueError(
                f"Reference PDB has duplicate (resid, atom name) key {key}: "
                f"{reference_pdb_path}."
            )
        reference_by_key[key] = atom

    sample_keys = [
        (int(vmd_resids_by_index[residue_index]), str(atom_name))
        for residue_index, atom_name in zip(atom_resids, atom_names, strict=True)
    ]
    if len(set(sample_keys)) != len(sample_keys):
        raise ValueError("Sampled topology has duplicate (VMD resid, atom name) keys.")
    sample_key_set = set(sample_keys)
    reference_heavy_keys = {
        key
        for key in reference_by_key
        if not is_hydrogen_atom_name(key[1])
    }
    if reference_heavy_keys != sample_key_set:
        missing = sorted(sample_key_set - reference_heavy_keys)[:20]
        unexpected = sorted(reference_heavy_keys - sample_key_set)[:20]
        raise ValueError(
            "Reference/sample heavy-atom identities do not match exactly by "
            f"(VMD resid, atom name): reference={reference_pdb_path}; "
            f"missing={missing}; unexpected={unexpected}."
        )

    return np.asarray(
        [reference_by_key[key].coords for key in sample_keys],
        dtype=np.float64,
    )


def selection_fitted_rmsd(
    sampled_coords: np.ndarray,
    target_coords: np.ndarray,
    atom_mask: np.ndarray,
    selection_name: str,
) -> float:
    atom_mask = np.asarray(atom_mask, dtype=bool)
    if sampled_coords.shape != target_coords.shape:
        raise ValueError(
            f"Sampled/target coordinate shape mismatch: {sampled_coords.shape} vs "
            f"{target_coords.shape}."
        )
    if atom_mask.shape != (sampled_coords.shape[0],):
        raise ValueError(
            f"Atom mask for selection {selection_name!r} has shape {atom_mask.shape}; "
            f"expected {(sampled_coords.shape[0],)}."
        )
    if int(atom_mask.sum()) < 3:
        raise ValueError(
            f"Selection {selection_name!r} contains fewer than three atoms."
        )

    mobile = sampled_coords[atom_mask]
    target = target_coords[atom_mask]
    mobile_centered = mobile - mobile.mean(axis=0)
    target_centered = target - target.mean(axis=0)
    covariance = mobile_centered.T @ target_centered
    u, _, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(u @ vt))
    rotation = u @ correction @ vt
    aligned = mobile_centered @ rotation + target.mean(axis=0)
    squared_distances = np.sum((aligned - target) ** 2, axis=1)
    return float(np.sqrt(np.mean(squared_distances)))


def evaluate_atom_selections(
    *,
    sampled_pdb_path: Path,
    target_pdb_path: Path,
    reference_pdb_path: Path,
    conditioned_eval_path: Path,
    metrics: dict[str, Any],
    atom_resids: np.ndarray,
    vmd_resids_by_index: np.ndarray,
    expected_local: np.ndarray,
    oracle_labels: np.ndarray,
    assigned_path: Path,
) -> list[dict[str, Any]]:
    (
        sampled_coords,
        target_coords,
        atom_names,
        sampled_coordinate_relation,
        target_coordinate_relation,
    ) = validate_pdb_pair_and_metadata(
        sampled_pdb_path,
        target_pdb_path,
        conditioned_eval_path,
        metrics,
        atom_resids,
        expected_local.shape[0],
    )
    reference_coords = reference_coords_in_sample_atom_order(
        reference_pdb_path,
        atom_names,
        atom_resids,
        vmd_resids_by_index,
    )
    ca_mask = atom_names == "CA"
    ca_counts = np.bincount(
        atom_resids[ca_mask],
        minlength=expected_local.shape[0],
    )
    invalid_ca_residues = np.flatnonzero(ca_counts != 1)
    if invalid_ca_residues.size:
        raise ValueError(
            "Strict CA selections require exactly one CA per residue; invalid "
            f"residue indices: {invalid_ca_residues[:20].tolist()}."
        )
    vmd_resids_per_atom = vmd_resids_by_index[atom_resids]
    backbone_mask = np.isin(atom_names, tuple(BACKBONE_ATOM_NAMES))
    protein_residue_mask = np.zeros(expected_local.shape, dtype=bool)
    for residue_index in range(expected_local.shape[0]):
        residue_atom_names = set(atom_names[atom_resids == residue_index].tolist())
        protein_residue_mask[residue_index] = BACKBONE_ATOM_NAMES.issubset(
            residue_atom_names
        )
    protein_atom_mask = protein_residue_mask[atom_resids]
    atom_masks = {
        "vmd_ca_residues": ca_mask & np.isin(vmd_resids_per_atom, VMD_RESIDS),
        "ca": ca_mask,
        "backbone": backbone_mask,
        "protein_not_backbone": protein_atom_mask & ~backbone_mask,
        "all_atoms": np.ones(atom_names.shape, dtype=bool),
    }

    evaluations = []
    for selection_name in RMSD_SELECTION_KEYS:
        atom_mask = atom_masks[selection_name]
        residue_mask = np.zeros(expected_local.shape, dtype=bool)
        residue_mask[np.unique(atom_resids[atom_mask])] = True
        comparison = compare_labels(
            expected_local,
            oracle_labels,
            assigned_path,
            selection_mask=residue_mask,
        )
        evaluations.append(
            {
                "selection_name": selection_name,
                "selection_definition": RMSD_SELECTION_DEFINITIONS[selection_name],
                "selected_atom_count": int(atom_mask.sum()),
                "selected_residue_count": int(residue_mask.sum()),
                "sampled_pdb_npz_relation": sampled_coordinate_relation,
                "target_pdb_npz_relation": target_coordinate_relation,
                "reference_pdb": str(reference_pdb_path),
                "pdb_rmsd_angstrom": selection_fitted_rmsd(
                    sampled_coords,
                    target_coords,
                    atom_mask,
                    selection_name,
                ),
                "reference_pdb_rmsd_angstrom": selection_fitted_rmsd(
                    sampled_coords,
                    reference_coords,
                    atom_mask,
                    f"{selection_name} against fixed reference",
                ),
                "n_compared": comparison["n_compared"],
                "match_count": comparison["match_count"],
                "mismatch_count": comparison["mismatch_count"],
                "accuracy": comparison["accuracy"],
                "mismatch_fraction": comparison["mismatch_fraction"],
            }
        )

    vmd_evaluation = evaluations[0]
    if vmd_evaluation["selected_atom_count"] != len(VMD_RESIDS):
        raise ValueError(
            f"The strict VMD CA selection must contain {len(VMD_RESIDS)} atoms, "
            f"got {vmd_evaluation['selected_atom_count']}."
        )
    return evaluations


def labels_to_residue_global(
    atom_global_labels: np.ndarray,
    atom_resids: np.ndarray,
    path: Path,
) -> np.ndarray:
    if atom_global_labels.shape != atom_resids.shape:
        raise ValueError(
            f"Atom label/residue shape mismatch for {path}: "
            f"{atom_global_labels.shape} vs {atom_resids.shape}."
        )
    if atom_resids.size == 0:
        raise ValueError(f"No atom_resids for {path}.")
    if int(atom_resids.min()) < 0:
        raise ValueError(f"atom_resids contains negative residue indices for {path}.")

    n_residues = int(atom_resids.max()) + 1
    residue_global = np.full(n_residues, -1, dtype=np.int64)
    for residue_idx in range(n_residues):
        atom_mask = atom_resids == residue_idx
        if not np.any(atom_mask):
            raise ValueError(f"Residue {residue_idx} has no atoms for {path}.")
        labels = atom_global_labels[atom_mask]
        labels = labels[labels >= 0]
        if labels.size == 0:
            continue
        unique = np.unique(labels)
        if unique.size != 1:
            raise ValueError(
                f"Conditioning labels are not residue-consistent for {path} "
                f"at residue {residue_idx}: {unique[:20].tolist()}."
            )
        residue_global[residue_idx] = int(unique[0])
    return residue_global


def global_to_local_labels(
    residue_global_labels: np.ndarray,
    cluster_counts: np.ndarray,
    path: Path,
) -> np.ndarray:
    cluster_counts = one_dimensional(cluster_counts, "cluster_counts", path).astype(
        np.int64,
        copy=False,
    )
    if residue_global_labels.shape != cluster_counts.shape:
        raise ValueError(
            f"Residue global label shape {residue_global_labels.shape} does not match "
            f"cluster_counts shape {cluster_counts.shape} for {path}."
        )
    offsets = np.concatenate(
        [np.zeros((1,), dtype=np.int64), np.cumsum(cluster_counts[:-1], dtype=np.int64)]
    )
    local = residue_global_labels - offsets
    valid = residue_global_labels >= 0
    invalid = valid & ((local < 0) | (local >= cluster_counts))
    if np.any(invalid):
        bad = np.flatnonzero(invalid)[:20].tolist()
        raise ValueError(
            f"Global-to-local conversion is invalid for {path} at residues {bad}."
        )
    local[~valid] = -1
    return local.astype(np.int64, copy=False)


def compare_labels(
    expected: np.ndarray,
    oracle: np.ndarray,
    path: Path,
    selection_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    expected = one_dimensional(expected, "expected labels", path).astype(np.int64, copy=False)
    oracle = one_dimensional(oracle, "oracle labels", path).astype(np.int64, copy=False)
    if expected.shape != oracle.shape:
        raise ValueError(
            f"Expected/oracle label shape mismatch for {path}: "
            f"{expected.shape} vs {oracle.shape}."
        )
    if selection_mask is None:
        selection_mask = np.ones(expected.shape, dtype=bool)
    else:
        selection_mask = one_dimensional(
            selection_mask,
            "residue selection mask",
            path,
        ).astype(bool, copy=False)
        if selection_mask.shape != expected.shape:
            raise ValueError(
                f"Residue selection mask shape mismatch for {path}: "
                f"{selection_mask.shape} vs {expected.shape}."
            )

    valid = (expected >= 0) & (oracle >= 0) & selection_mask
    if not np.any(valid):
        raise ValueError(f"No comparable residue labels for {path}.")
    diff = oracle - expected
    mismatches = valid & (diff != 0)
    abs_diff = np.abs(diff[valid])
    n_compared = int(valid.sum())
    mismatch_count = int(mismatches.sum())
    return {
        "n_residues": int(expected.shape[0]),
        "n_selected_residues": int(selection_mask.sum()),
        "n_compared": n_compared,
        "match_count": int(n_compared - mismatch_count),
        "mismatch_count": mismatch_count,
        "accuracy": float((n_compared - mismatch_count) / n_compared),
        "mismatch_fraction": float(mismatch_count / n_compared),
        "mean_abs_error": float(np.mean(abs_diff)),
        "max_abs_error": int(np.max(abs_diff)),
        "diff": diff,
        "valid": valid,
        "mismatches": mismatches,
    }


def write_per_sample_report(
    path: Path,
    row: dict[str, Any],
    selection_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "Conditioning vs oracle comparison",
        "",
        "Sample",
        f"  sample_name: {row['sample_name']}",
        f"  source_type: {row['source_type']}",
        f"  assigned_npz: {row['assigned_npz']}",
        f"  conditioned_eval_npz: {row['conditioned_eval_npz']}",
        f"  raw_npz_path: {row.get('raw_npz_path')}",
        f"  frame_index: {row.get('frame_index')}",
        f"  sample_index: {row.get('sample_index')}",
        f"  seed: {row.get('seed')}",
        "",
        "Evaluation scope",
        f"  selection: {row['selection']}",
        f"  selected_residues: {row['n_selected_residues']}",
        "",
        "Metrics",
        f"  total_residues: {row['n_residues']}",
        f"  compared: {row['n_compared']}",
        f"  matches: {row['match_count']}",
        f"  mismatches: {row['mismatch_count']}",
        f"  accuracy: {row['accuracy']:.6f}",
        f"  mismatch_fraction: {row['mismatch_fraction']:.6f}",
        f"  mean_abs_error: {row['mean_abs_error']:.6f}",
        f"  max_abs_error: {row['max_abs_error']}",
        "",
        "Sampled PDB vs target PDB selection-fitted RMSDs",
        f"  sampled_pdb: {row['sampled_pdb']}",
        f"  target_pdb: {row['target_pdb']}",
        f"  fixed_reference_pdb: {row['reference_pdb']}",
        f"  sampled_pdb_npz_relation: {row['sampled_pdb_npz_relation']}",
        f"  target_pdb_npz_relation: {row['target_pdb_npz_relation']}",
    ]
    for selection_row in selection_rows:
        lines.extend(
            [
                f"  {selection_row['selection_name']}",
                f"    definition: {selection_row['selection_definition']}",
                f"    atoms: {selection_row['selected_atom_count']}",
                f"    residues: {selection_row['selected_residue_count']}",
                f"    rmsd_angstrom: {selection_row['pdb_rmsd_angstrom']:.6f}",
                (
                    "    fixed_reference_rmsd_angstrom: "
                    f"{selection_row['reference_pdb_rmsd_angstrom']:.6f}"
                ),
                (
                    f"    cluster_matches: {selection_row['match_count']}/"
                    f"{selection_row['n_compared']}"
                ),
            ]
        )
    path.write_text("\n".join(lines) + "\n")


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {"overall": rows}
    for row in rows:
        groups.setdefault(row["source_type"], []).append(row)

    summary = {}
    for name, group_rows in groups.items():
        n_compared = sum(int(row["n_compared"]) for row in group_rows)
        mismatches = sum(int(row["mismatch_count"]) for row in group_rows)
        matches = sum(int(row["match_count"]) for row in group_rows)
        perfect_samples = sum(1 for row in group_rows if int(row["mismatch_count"]) == 0)
        summary[name] = {
            "sample_count": len(group_rows),
            "perfect_sample_count": perfect_samples,
            "perfect_sample_fraction": float(perfect_samples / len(group_rows)),
            "n_compared": n_compared,
            "match_count": matches,
            "mismatch_count": mismatches,
            "accuracy": float(matches / n_compared) if n_compared else None,
            "mismatch_fraction": float(mismatches / n_compared) if n_compared else None,
            "mean_sample_mismatch_fraction": float(
                np.mean([row["mismatch_fraction"] for row in group_rows])
            ),
            "max_sample_mismatches": int(max(row["mismatch_count"] for row in group_rows)),
        }
    return summary


def write_summary_report(
    path: Path,
    rows: list[dict[str, Any]],
    summary: dict[str, dict[str, Any]],
    selection_rows: list[dict[str, Any]],
) -> None:
    selections = sorted({str(row["selection"]) for row in rows})
    lines = [
        "Conditioning vs oracle aggregate report",
        "",
        "Evaluation scope",
        *[f"  {selection}" for selection in selections],
        "",
        "Groups",
    ]
    for group_name in sorted(summary, key=lambda name: (name != "overall", name)):
        stats = summary[group_name]
        accuracy = stats["accuracy"]
        mismatch_fraction = stats["mismatch_fraction"]
        lines.extend(
            [
                f"  {group_name}",
                f"    sample_count: {stats['sample_count']}",
                f"    perfect_sample_count: {stats['perfect_sample_count']}",
                f"    perfect_sample_fraction: {stats['perfect_sample_fraction']:.6f}",
                f"    compared: {stats['n_compared']}",
                f"    matches: {stats['match_count']}",
                f"    mismatches: {stats['mismatch_count']}",
                f"    accuracy: {accuracy:.6f}" if accuracy is not None else "    accuracy: n/a",
                (
                    f"    mismatch_fraction: {mismatch_fraction:.6f}"
                    if mismatch_fraction is not None
                    else "    mismatch_fraction: n/a"
                ),
                f"    mean_sample_mismatch_fraction: {stats['mean_sample_mismatch_fraction']:.6f}",
                f"    max_sample_mismatches: {stats['max_sample_mismatches']}",
                "",
            ]
        )

    lines.append("Sampled PDB vs Target PDB RMSD by Selection")
    for selection_name in RMSD_SELECTION_KEYS:
        group = [
            row for row in selection_rows if row["selection_name"] == selection_name
        ]
        values = np.asarray(
            [row["pdb_rmsd_angstrom"] for row in group],
            dtype=np.float64,
        )
        reference_values = np.asarray(
            [row["reference_pdb_rmsd_angstrom"] for row in group],
            dtype=np.float64,
        )
        n_compared = sum(int(row["n_compared"]) for row in group)
        matches = sum(int(row["match_count"]) for row in group)
        mismatches = sum(int(row["mismatch_count"]) for row in group)
        lines.extend(
            [
                f"  {selection_name}",
                f"    definition: {RMSD_SELECTION_DEFINITIONS[selection_name]}",
                f"    sample_count: {values.size}",
                f"    compared_residue_labels: {n_compared}",
                f"    matching_residue_labels: {matches}",
                f"    mismatching_residue_labels: {mismatches}",
                f"    residue_label_accuracy: {matches / n_compared:.6f}",
                f"    mean_angstrom: {float(np.mean(values)):.6f}",
                f"    median_angstrom: {float(np.median(values)):.6f}",
                f"    min_angstrom: {float(np.min(values)):.6f}",
                f"    max_angstrom: {float(np.max(values)):.6f}",
                f"    fixed_reference_pdb: {group[0]['reference_pdb']}",
                (
                    "    fixed_reference_mean_angstrom: "
                    f"{float(np.mean(reference_values)):.6f}"
                ),
                (
                    "    fixed_reference_median_angstrom: "
                    f"{float(np.median(reference_values)):.6f}"
                ),
                "",
            ]
        )

    lines.append("Per Sample")
    for row in rows:
        lines.append(
            "  "
            f"{row['sample_name']}: source={row['source_type']} "
            f"mismatches={row['mismatch_count']}/{row['n_compared']} "
            f"accuracy={row['accuracy']:.6f}"
        )
    path.write_text("\n".join(lines) + "\n")


def write_sample_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "source_type",
        "sample_name",
        "assigned_npz",
        "conditioned_eval_npz",
        "raw_npz_path",
        "frame_index",
        "sample_index",
        "seed",
        "selection",
        "sampled_pdb",
        "target_pdb",
        "reference_pdb",
        "sampled_pdb_npz_relation",
        "target_pdb_npz_relation",
        "n_residues",
        "n_selected_residues",
        "n_compared",
        "match_count",
        "mismatch_count",
        "accuracy",
        "mismatch_fraction",
        "mean_abs_error",
        "max_abs_error",
    ]
    for selection_name in RMSD_SELECTION_KEYS:
        columns.extend(
            [
                f"rmsd_{selection_name}_angstrom",
                f"rmsd_{selection_name}_atom_count",
                f"reference_rmsd_{selection_name}_angstrom",
            ]
        )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def write_selection_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "source_type",
        "sample_name",
        "sampled_pdb",
        "target_pdb",
        "reference_pdb",
        "selection_name",
        "selection_definition",
        "selected_atom_count",
        "selected_residue_count",
        "sampled_pdb_npz_relation",
        "target_pdb_npz_relation",
        "pdb_rmsd_angstrom",
        "reference_pdb_rmsd_angstrom",
        "n_compared",
        "match_count",
        "mismatch_count",
        "accuracy",
        "mismatch_fraction",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def write_residue_csv(path: Path, residue_rows: list[dict[str, Any]]) -> None:
    columns = [
        "source_type",
        "sample_name",
        "residue_index",
        "vmd_resid",
        "expected_local_label",
        "oracle_label",
        "match",
        "label_error",
        "abs_label_error",
        "expected_global_label",
        "cluster_count",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in residue_rows:
            writer.writerow({column: row.get(column) for column in columns})


def draw_violin(
    ax: Any,
    values: np.ndarray,
    position: int,
    color: Any,
    width: float = 0.78,
) -> None:
    """Draw a violin without sample markers, with a constant-data fallback."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Violin data must be a non-empty one-dimensional array.")
    if not np.isfinite(values).all():
        raise ValueError("Cannot plot non-finite violin values.")

    scale = max(1.0, float(np.max(np.abs(values))))
    has_density = (
        values.size >= 2
        and float(np.ptp(values)) > np.finfo(np.float64).eps * scale
    )
    if has_density:
        artists = ax.violinplot(
            [values],
            positions=[position],
            widths=width,
            showmeans=False,
            showmedians=True,
            showextrema=True,
        )
        for body in artists["bodies"]:
            body.set_facecolor(color)
            body.set_edgecolor("black")
            body.set_alpha(0.9)
            body.set_linewidth(0.8)
        for artist_name in ("cmins", "cmaxes", "cbars", "cmedians"):
            artists[artist_name].set_color("black")
            artists[artist_name].set_linewidth(1.0)
    else:
        ax.hlines(
            float(values[0]),
            position - width / 2,
            position + width / 2,
            color=color,
            linewidth=4.0,
            alpha=0.9,
        )


def values_by_selection(
    rows: list[dict[str, Any]],
    value_key: str,
    selection_keys: tuple[str, ...] = RMSD_SELECTION_KEYS,
) -> dict[str, np.ndarray]:
    values = {
        selection_name: np.asarray(
            [
                row[value_key]
                for row in rows
                if row["selection_name"] == selection_name
            ],
            dtype=np.float64,
        )
        for selection_name in selection_keys
    }
    missing = [
        selection_name
        for selection_name, data in values.items()
        if not data.size
    ]
    if missing:
        raise ValueError(
            f"No {value_key!r} values available for selections: {missing}."
        )
    if not np.isfinite(np.concatenate(list(values.values()))).all():
        raise ValueError(f"Cannot plot non-finite {value_key!r} values.")
    return values


def mismatch_percentage_values(
    selection_rows: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    return {
        selection_name: 100.0 * values
        for selection_name, values in values_by_selection(
            selection_rows,
            "mismatch_fraction",
            ERROR_SELECTION_KEYS,
        ).items()
    }


def finite_distribution_statistics(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "count": 0,
            "mean": None,
            "standard_deviation": None,
        }
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "standard_deviation": float(np.std(values, ddof=0)),
    }


def write_violin_statistics(
    path: Path,
    *,
    plot_name: str,
    value_name: str,
    unit: str,
    values_by_group: dict[str, np.ndarray],
) -> None:
    columns = [
        "plot_name",
        "selection_name",
        "selection_title",
        "value_name",
        "unit",
        "count",
        "mean",
        "standard_deviation",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for selection_name, values in values_by_group.items():
            writer.writerow(
                {
                    "plot_name": plot_name,
                    "selection_name": selection_name,
                    "selection_title": RMSD_SELECTION_PLOT_TITLES[selection_name],
                    "value_name": value_name,
                    "unit": unit,
                    **finite_distribution_statistics(values),
                }
            )


def set_selection_axis(
    ax: Any,
    positions: np.ndarray,
    selection_keys: tuple[str, ...] = RMSD_SELECTION_KEYS,
) -> None:
    ax.set_xlabel("Selection type")
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [RMSD_SELECTION_PLOT_TITLES[name] for name in selection_keys],
        rotation=12,
        ha="right",
    )
    ax.grid(axis="y", alpha=0.2)


def set_violin_y_limits(
    ax: Any,
    values_by_group: dict[str, np.ndarray],
    threshold_fraction: float = VIOLIN_Y_AXIS_THRESHOLD_FRACTION,
) -> None:
    values = np.concatenate(list(values_by_group.values()))
    value_min = float(np.min(values))
    value_max = float(np.max(values))
    value_range = value_max - value_min
    if value_range > 0:
        threshold = threshold_fraction * value_range
    else:
        threshold = threshold_fraction * max(abs(value_min), 1.0)
    ax.set_ylim(value_min - threshold, value_max + threshold)


def plot_summary(path: Path, selection_rows: list[dict[str, Any]]) -> None:
    mismatch_percentages = mismatch_percentage_values(selection_rows)
    sample_count = mismatch_percentages[ERROR_SELECTION_KEYS[0]].size
    violin_positions = np.arange(1, len(ERROR_SELECTION_KEYS) + 1)
    fig, ax = plt.subplots(figsize=(12.0, 6.0), constrained_layout=True)

    for position, selection_name in zip(
        violin_positions,
        ERROR_SELECTION_KEYS,
        strict=True,
    ):
        draw_violin(
            ax,
            mismatch_percentages[selection_name],
            int(position),
            RMSD_SELECTION_COLORS[selection_name],
        )
    ax.set_title(
        "Conditioning vs Oracle Mismatch Percentages Across Sampled Structures "
        f"(n={sample_count})"
    )
    ax.set_ylabel("Mismatching residues per structure (%)")
    set_violin_y_limits(ax, mismatch_percentages)
    set_selection_axis(ax, violin_positions, ERROR_SELECTION_KEYS)

    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_rmsd_violins(path: Path, selection_rows: list[dict[str, Any]]) -> None:
    rmsd_values = values_by_selection(selection_rows, "pdb_rmsd_angstrom")
    sample_count = rmsd_values[RMSD_SELECTION_KEYS[0]].size
    violin_positions = np.arange(1, len(RMSD_SELECTION_KEYS) + 1)
    fig, ax = plt.subplots(figsize=(12.0, 6.0), constrained_layout=True)

    for position, selection_name in zip(
        violin_positions,
        RMSD_SELECTION_KEYS,
        strict=True,
    ):
        draw_violin(
            ax,
            rmsd_values[selection_name],
            int(position),
            RMSD_SELECTION_COLORS[selection_name],
        )
    ax.set_title(
        f"Sampled PDB vs Target PDB RMSD Distributions (n={sample_count})"
    )
    ax.set_ylabel("Selection-fitted RMSD (Å)")
    set_violin_y_limits(ax, rmsd_values)
    set_selection_axis(ax, violin_positions)

    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_residue_confusion_matrices(
    path: Path,
    confusion_rows: list[dict[str, Any]],
    *,
    selected_vmd_resids: frozenset[int] | None = None,
    selection_title: str = "All residues",
) -> None:
    if not confusion_rows:
        raise ValueError("Cannot plot residue confusion matrices without residue rows.")
    sample_count = len(
        {
            (str(row["source_type"]), str(row["sample_name"]))
            for row in confusion_rows
        }
    )

    rows_by_residue: dict[int, list[dict[str, Any]]] = {}
    for row in confusion_rows:
        if int(row["cluster_count"]) <= 1:
            continue
        if (
            selected_vmd_resids is not None
            and int(row["vmd_resid"]) not in selected_vmd_resids
        ):
            continue
        rows_by_residue.setdefault(int(row["residue_index"]), []).append(row)

    residue_indices = sorted(rows_by_residue)
    if not residue_indices:
        raise ValueError(
            f"No multi-cluster residues are available for {selection_title}."
        )

    matrices = []
    for residue_index in residue_indices:
        residue_rows = rows_by_residue[residue_index]
        vmd_resids = {int(row["vmd_resid"]) for row in residue_rows}
        cluster_counts = {int(row["cluster_count"]) for row in residue_rows}
        if len(vmd_resids) != 1 or len(cluster_counts) != 1:
            raise ValueError(
                f"Inconsistent metadata for residue index {residue_index}: "
                f"vmd_resids={sorted(vmd_resids)}, "
                f"cluster_counts={sorted(cluster_counts)}."
            )
        vmd_resid = next(iter(vmd_resids))
        cluster_count = next(iter(cluster_counts))

        counts = np.zeros((cluster_count, cluster_count), dtype=np.int64)
        for row in residue_rows:
            conditioning_cluster = int(row["expected_local_label"])
            oracle_cluster = int(row["oracle_label"])
            if conditioning_cluster < 0 or oracle_cluster < 0:
                continue
            if (
                conditioning_cluster >= cluster_count
                or oracle_cluster >= cluster_count
            ):
                raise ValueError(
                    f"Cluster label outside 0..{cluster_count - 1} for residue "
                    f"{vmd_resid}: conditioning={conditioning_cluster}, "
                    f"oracle={oracle_cluster}."
                )
            counts[conditioning_cluster, oracle_cluster] += 1
        matrices.append((vmd_resid, counts))

    n_columns = int(np.ceil(np.sqrt(len(residue_indices))))
    n_rows = int(np.ceil(len(residue_indices) / n_columns))
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(max(10.0, 2.0 * n_columns), max(5.5, 2.0 * n_rows)),
        squeeze=False,
    )
    axes_flat = axes.reshape(-1)
    image = None
    max_count = max(int(np.max(counts)) for _, counts in matrices)
    color_max = max(max_count, 1)

    for ax, (vmd_resid, counts) in zip(axes_flat, matrices, strict=False):
        cluster_count = counts.shape[0]
        image = ax.imshow(
            counts,
            cmap="Blues",
            interpolation="nearest",
            vmin=0,
            vmax=color_max,
        )
        ax.set_title(
            f"Residue {vmd_resid} (n={int(counts.sum())})",
            fontsize=8,
        )
        ticks = np.arange(cluster_count)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.tick_params(labelsize=6, length=2)
        for conditioning_cluster in range(cluster_count):
            for oracle_cluster in range(cluster_count):
                count = int(counts[conditioning_cluster, oracle_cluster])
                if count == 0:
                    continue
                ax.text(
                    oracle_cluster,
                    conditioning_cluster,
                    str(count),
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white" if count >= 0.55 * color_max else "black",
                )

    for unused_ax in axes_flat[len(residue_indices) :]:
        unused_ax.remove()
    if image is None:
        raise ValueError("No residue confusion matrix was rendered.")

    fig.suptitle(
        "Per-Residue Conditioning vs Oracle Confusion Matrices\n"
        f"{selection_title} · multi-cluster residues only · n={sample_count}",
        fontsize=14,
        y=0.96,
    )
    fig.supxlabel("Oracle cluster")
    fig.supylabel("Conditioning cluster")
    compact_layout = n_rows <= 2
    fig.subplots_adjust(
        left=0.08 if compact_layout else 0.045,
        right=0.9,
        bottom=0.12 if compact_layout else 0.05,
        top=0.82 if compact_layout else 0.94,
        wspace=0.5,
        hspace=0.65,
    )
    colorbar_ax = fig.add_axes(
        (0.925, 0.12, 0.012, 0.72 if compact_layout else 0.8)
    )
    colorbar = fig.colorbar(image, cax=colorbar_ax)
    colorbar.set_label("Generated structures")
    fig.savefig(path, dpi=140)
    plt.close(fig)


# Backward-compatible callable name for code that imported the old plotter.
plot_rmsd_histograms = plot_rmsd_violins


def pdb_residue_names(path: Path) -> np.ndarray:
    names = []
    previous_identity = None
    for atom in read_pdb_atoms(path):
        if atom.residue_identity != previous_identity:
            names.append(atom.residue_name)
            previous_identity = atom.residue_identity
    return np.asarray(names, dtype=str)


def load_mismatch_dihedral_data(
    *,
    conditioned_eval_path: Path,
    target_pdb_path: Path,
    sample_name: str,
    expected_local: np.ndarray,
    oracle_labels: np.ndarray,
) -> MismatchDihedralData | None:
    arrays = {
        key: optional_npz_array(conditioned_eval_path, key)
        for key in ("dihedral_diff_rad", "dihedral_mask", "dihedral_keys")
    }
    if all(value is None for value in arrays.values()):
        return None
    missing = [key for key, value in arrays.items() if value is None]
    if missing:
        raise KeyError(
            f"{conditioned_eval_path} has incomplete dihedral data; missing {missing}."
        )

    dihedral_diff_rad = np.asarray(arrays["dihedral_diff_rad"], dtype=np.float64)
    dihedral_mask = np.asarray(arrays["dihedral_mask"], dtype=bool)
    raw_dihedral_keys = one_dimensional(
        np.asarray(arrays["dihedral_keys"]),
        "dihedral_keys",
        conditioned_eval_path,
    )
    dihedral_keys = tuple(text_value(key) for key in raw_dihedral_keys)
    if dihedral_diff_rad.ndim != 2 or dihedral_mask.shape != dihedral_diff_rad.shape:
        raise ValueError(
            f"Invalid dihedral shapes in {conditioned_eval_path}: "
            f"differences={dihedral_diff_rad.shape}, mask={dihedral_mask.shape}."
        )
    if len(dihedral_keys) != dihedral_diff_rad.shape[1]:
        raise ValueError(
            f"{conditioned_eval_path} has {len(dihedral_keys)} dihedral keys for "
            f"{dihedral_diff_rad.shape[1]} dihedral columns."
        )
    if expected_local.shape != (dihedral_diff_rad.shape[0],):
        raise ValueError(
            f"Residue/dihedral shape mismatch for {conditioned_eval_path}: "
            f"labels={expected_local.shape}, dihedrals={dihedral_diff_rad.shape}."
        )

    stored_residue_names = optional_npz_array(conditioned_eval_path, "residue_names")
    if stored_residue_names is None:
        residue_names = pdb_residue_names(target_pdb_path)
    else:
        residue_names = np.asarray(
            [
                text_value(name)
                for name in one_dimensional(
                    stored_residue_names,
                    "residue_names",
                    conditioned_eval_path,
                )
            ],
            dtype=str,
        )
    if residue_names.shape != expected_local.shape:
        raise ValueError(
            f"Residue-name shape mismatch for {conditioned_eval_path}: "
            f"{residue_names.shape} vs {expected_local.shape}."
        )

    comparable = (expected_local >= 0) & (oracle_labels >= 0)
    mismatches = comparable & (expected_local != oracle_labels)
    output_path = conditioned_eval_path.with_name(
        conditioned_eval_path.name.replace(
            CONDITIONED_EVAL_TOKEN,
            WRONG_CLUSTER_DIHEDRAL_PLOT_TOKEN,
            1,
        )
    )
    return MismatchDihedralData(
        sample_name=sample_name,
        output_path=output_path,
        dihedral_diff_rad=dihedral_diff_rad[mismatches],
        dihedral_mask=dihedral_mask[mismatches],
        dihedral_keys=dihedral_keys,
        residue_names=residue_names[mismatches],
        mismatch_count=int(mismatches.sum()),
        comparable_count=int(comparable.sum()),
    )


def statistics_path_for_plot(plot_path: Path) -> Path:
    return plot_path.with_name(f"{plot_path.stem}_statistics.csv")


def write_dihedral_error_statistics(
    path: Path,
    *,
    scope: str,
    sample_name: str,
    dihedral_diff_rad: np.ndarray,
    dihedral_mask: np.ndarray,
    dihedral_keys: tuple[str, ...],
    residue_names: np.ndarray,
    mismatch_count: int,
    comparable_count: int,
) -> None:
    signed_error_deg, absolute_error_deg = symmetry_correct_dihedral_errors(
        dihedral_diff_rad,
        residue_names,
        list(dihedral_keys),
    )
    valid = np.asarray(dihedral_mask, dtype=bool) & np.isfinite(dihedral_diff_rad)
    columns = [
        "scope",
        "sample_name",
        "mismatch_residue_count",
        "comparable_residue_count",
        "dihedral_key",
        "error_type",
        "unit",
        "count",
        "mean",
        "standard_deviation",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for key_index, dihedral_key in [
            (None, "all"),
            *enumerate(dihedral_keys),
        ]:
            if key_index is None:
                key_valid = valid.reshape(-1)
                signed_values = signed_error_deg.reshape(-1)[key_valid]
                absolute_values = absolute_error_deg.reshape(-1)[key_valid]
            else:
                key_valid = valid[:, key_index]
                signed_values = signed_error_deg[:, key_index][key_valid]
                absolute_values = absolute_error_deg[:, key_index][key_valid]
            for error_type, values in (
                ("signed_error", signed_values),
                ("absolute_error", absolute_values),
            ):
                writer.writerow(
                    {
                        "scope": scope,
                        "sample_name": sample_name,
                        "mismatch_residue_count": mismatch_count,
                        "comparable_residue_count": comparable_count,
                        "dihedral_key": dihedral_key,
                        "error_type": error_type,
                        "unit": "degrees",
                        **finite_distribution_statistics(values),
                    }
                )


def write_mismatch_dihedral_plots(
    data_by_sample: list[MismatchDihedralData],
    aggregate_path: Path,
    error_bins: int,
) -> list[Path]:
    if not data_by_sample:
        return []

    expected_keys = data_by_sample[0].dihedral_keys
    inconsistent = [
        data.sample_name
        for data in data_by_sample
        if data.dihedral_keys != expected_keys
    ]
    if inconsistent:
        raise ValueError(
            "Cannot aggregate mismatch dihedrals with inconsistent dihedral keys; "
            f"samples={inconsistent}."
        )

    written_paths = []
    for data in data_by_sample:
        write_dihedral_error_histogram_png(
            output_path=data.output_path,
            dihedral_diff_rad=data.dihedral_diff_rad,
            dihedral_mask=data.dihedral_mask,
            dihedral_keys=list(data.dihedral_keys),
            residue_names=data.residue_names,
            error_bins=error_bins,
            figure_title=(
                f"{data.sample_name}: dihedral errors at cluster mismatches "
                f"(n=1 structure; {data.mismatch_count}/"
                f"{data.comparable_count} residues)"
            ),
        )
        write_dihedral_error_statistics(
            statistics_path_for_plot(data.output_path),
            scope="per_structure",
            sample_name=data.sample_name,
            dihedral_diff_rad=data.dihedral_diff_rad,
            dihedral_mask=data.dihedral_mask,
            dihedral_keys=data.dihedral_keys,
            residue_names=data.residue_names,
            mismatch_count=data.mismatch_count,
            comparable_count=data.comparable_count,
        )
        written_paths.append(data.output_path)

    aggregate_diff = np.concatenate(
        [data.dihedral_diff_rad for data in data_by_sample],
        axis=0,
    )
    aggregate_mask = np.concatenate(
        [data.dihedral_mask for data in data_by_sample],
        axis=0,
    )
    aggregate_residue_names = np.concatenate(
        [data.residue_names for data in data_by_sample],
        axis=0,
    )
    total_mismatches = sum(data.mismatch_count for data in data_by_sample)
    total_comparable = sum(data.comparable_count for data in data_by_sample)
    write_dihedral_error_histogram_png(
        output_path=aggregate_path,
        dihedral_diff_rad=aggregate_diff,
        dihedral_mask=aggregate_mask,
        dihedral_keys=list(expected_keys),
        residue_names=aggregate_residue_names,
        error_bins=error_bins,
        figure_title=(
            "Dihedral errors at cluster mismatches across "
            f"generated structures (n={len(data_by_sample)}; "
            f"{total_mismatches}/{total_comparable} residue events)"
        ),
    )
    write_dihedral_error_statistics(
        statistics_path_for_plot(aggregate_path),
        scope="aggregate",
        sample_name="all_generated_structures",
        dihedral_diff_rad=aggregate_diff,
        dihedral_mask=aggregate_mask,
        dihedral_keys=expected_keys,
        residue_names=aggregate_residue_names,
        mismatch_count=total_mismatches,
        comparable_count=total_comparable,
    )
    written_paths.append(aggregate_path)
    return written_paths


def compare_file(
    assigned_path: Path,
    base_path: Path,
    all_res: bool = False,
    reference_pdb_path: Path = DEFAULT_REFERENCE_PDB,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    MismatchDihedralData | None,
]:
    conditioned_eval_path = conditioned_eval_path_for_assigned(assigned_path)
    metrics_path = conditioned_eval_json_path_for_assigned(assigned_path)
    metrics = load_metrics(metrics_path)

    if not conditioned_eval_path.is_file():
        raise FileNotFoundError(f"Conditioned eval NPZ not found: {conditioned_eval_path}")

    oracle_labels = one_dimensional(
        load_npz_array(assigned_path, "labels_assigned"),
        "labels_assigned",
        assigned_path,
    ).astype(np.int64, copy=False)
    cluster_counts = load_cluster_counts(assigned_path)
    atom_conditioning = load_conditioning_labels(conditioned_eval_path)
    atom_resids = load_atom_resids(conditioned_eval_path, metrics)

    residue_global = labels_to_residue_global(atom_conditioning, atom_resids, conditioned_eval_path)
    expected_local = global_to_local_labels(residue_global, cluster_counts, assigned_path)
    vmd_selection_mask, vmd_resids_by_index = strict_vmd_selection_mask(
        conditioned_eval_path,
        metrics,
        atom_resids,
        expected_local.shape[0],
    )
    if all_res:
        selection_mask = np.ones(expected_local.shape, dtype=bool)
        selection = "all residues"
    else:
        selection_mask = vmd_selection_mask
        selection = VMD_SELECTION
    comparison = compare_labels(
        expected_local,
        oracle_labels,
        assigned_path,
        selection_mask=selection_mask,
    )

    sample_name = sample_name_for_assigned(assigned_path)
    source_type = source_type_for_path(assigned_path, base_path)
    sampled_pdb_path, target_pdb_path = strict_rmsd_pdb_paths(
        conditioned_eval_path,
        metrics_path,
        metrics,
    )
    selection_evaluations = evaluate_atom_selections(
        sampled_pdb_path=sampled_pdb_path,
        target_pdb_path=target_pdb_path,
        reference_pdb_path=reference_pdb_path,
        conditioned_eval_path=conditioned_eval_path,
        metrics=metrics,
        atom_resids=atom_resids,
        vmd_resids_by_index=vmd_resids_by_index,
        expected_local=expected_local,
        oracle_labels=oracle_labels,
        assigned_path=assigned_path,
    )
    mismatch_dihedral_data = load_mismatch_dihedral_data(
        conditioned_eval_path=conditioned_eval_path,
        target_pdb_path=target_pdb_path,
        sample_name=sample_name,
        expected_local=expected_local,
        oracle_labels=oracle_labels,
    )
    row = {
        "source_type": source_type,
        "sample_name": sample_name,
        "assigned_npz": str(assigned_path),
        "conditioned_eval_npz": str(conditioned_eval_path),
        "raw_npz_path": metrics.get("raw_npz_path"),
        "frame_index": metrics.get("frame_index"),
        "sample_index": metrics.get("sample_index"),
        "seed": metrics.get("seed"),
        "selection": selection,
        "sampled_pdb": str(sampled_pdb_path),
        "target_pdb": str(target_pdb_path),
        "reference_pdb": str(reference_pdb_path),
        "sampled_pdb_npz_relation": selection_evaluations[0][
            "sampled_pdb_npz_relation"
        ],
        "target_pdb_npz_relation": selection_evaluations[0][
            "target_pdb_npz_relation"
        ],
        "n_residues": comparison["n_residues"],
        "n_selected_residues": comparison["n_selected_residues"],
        "n_compared": comparison["n_compared"],
        "match_count": comparison["match_count"],
        "mismatch_count": comparison["mismatch_count"],
        "accuracy": comparison["accuracy"],
        "mismatch_fraction": comparison["mismatch_fraction"],
        "mean_abs_error": comparison["mean_abs_error"],
        "max_abs_error": comparison["max_abs_error"],
    }
    for evaluation in selection_evaluations:
        selection_name = evaluation["selection_name"]
        row[f"rmsd_{selection_name}_angstrom"] = evaluation["pdb_rmsd_angstrom"]
        row[f"rmsd_{selection_name}_atom_count"] = evaluation[
            "selected_atom_count"
        ]
        row[f"reference_rmsd_{selection_name}_angstrom"] = evaluation[
            "reference_pdb_rmsd_angstrom"
        ]

    selection_rows = []
    for evaluation in selection_evaluations:
        selection_rows.append(
            {
                "source_type": source_type,
                "sample_name": sample_name,
                "sampled_pdb": str(sampled_pdb_path),
                "target_pdb": str(target_pdb_path),
                "reference_pdb": str(reference_pdb_path),
                **evaluation,
            }
        )

    residue_rows = []
    diff = comparison["diff"]
    valid = comparison["valid"]
    mismatches = comparison["mismatches"]
    for residue_idx in range(expected_local.shape[0]):
        if not bool(valid[residue_idx]):
            continue
        residue_rows.append(
            {
                "source_type": source_type,
                "sample_name": sample_name,
                "residue_index": residue_idx,
                "vmd_resid": int(vmd_resids_by_index[residue_idx]),
                "expected_local_label": int(expected_local[residue_idx]),
                "oracle_label": int(oracle_labels[residue_idx]),
                "match": int(not bool(mismatches[residue_idx])),
                "label_error": int(diff[residue_idx]),
                "abs_label_error": int(abs(diff[residue_idx])),
                "expected_global_label": int(residue_global[residue_idx]),
                "cluster_count": int(cluster_counts[residue_idx]),
            }
        )

    confusion_rows = [
        {
            "source_type": source_type,
            "sample_name": sample_name,
            "residue_index": residue_idx,
            "vmd_resid": int(vmd_resids_by_index[residue_idx]),
            "expected_local_label": int(expected_local[residue_idx]),
            "oracle_label": int(oracle_labels[residue_idx]),
            "cluster_count": int(cluster_counts[residue_idx]),
        }
        for residue_idx in range(expected_local.shape[0])
    ]

    report_path = assigned_path.with_name(
        assigned_path.name.replace(ASSIGNED_TOKEN, "_conditioning_vs_oracle_report.txt", 1)
    )
    write_per_sample_report(report_path, row, selection_rows)
    return (
        row,
        residue_rows,
        selection_rows,
        confusion_rows,
        mismatch_dihedral_data,
    )


def main() -> None:
    args = parse_args()
    if args.dihedral_error_bins <= 0:
        raise ValueError("--dihedral-error-bins must be positive.")
    base_path = args.base_path.expanduser().resolve()
    out_dir = (args.out_dir or base_path).expanduser().resolve()
    reference_pdb_path = args.reference_pdb.expanduser().resolve()
    if not base_path.is_dir():
        raise NotADirectoryError(f"Base path is not a directory: {base_path}")
    if not reference_pdb_path.is_file():
        raise FileNotFoundError(f"Reference PDB is not a file: {reference_pdb_path}")

    assigned_files = find_assigned_cluster_files(base_path)
    if not assigned_files:
        raise FileNotFoundError(f"No files matching '*{ASSIGNED_TOKEN}' under {base_path}.")

    rows = []
    residue_rows = []
    selection_rows = []
    confusion_rows = []
    mismatch_dihedral_data = []
    missing_dihedral_samples = []
    for assigned_path in assigned_files:
        row, per_residue, per_selection, per_confusion, per_dihedrals = compare_file(
            assigned_path,
            base_path,
            all_res=args.all_res,
            reference_pdb_path=reference_pdb_path,
        )
        rows.append(row)
        residue_rows.extend(per_residue)
        selection_rows.extend(per_selection)
        confusion_rows.extend(per_confusion)
        if per_dihedrals is None:
            missing_dihedral_samples.append(row["sample_name"])
        else:
            mismatch_dihedral_data.append(per_dihedrals)

    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix
    sample_csv = out_dir / f"{prefix}_per_structure.csv"
    residue_csv = out_dir / f"{prefix}_per_residue.csv"
    selection_csv = out_dir / f"{prefix}_per_selection.csv"
    report_path = out_dir / f"{prefix}_report.txt"
    plot_path = out_dir / f"{prefix}_errors.png"
    error_statistics_path = statistics_path_for_plot(plot_path)
    residue_confusion_plot_path = (
        out_dir / f"{prefix}_per_residue_confusion_matrices.png"
    )
    vmd_residue_confusion_plot_path = (
        out_dir / f"{prefix}_vmd_selection_residue_confusion_matrices.png"
    )
    aggregate_dihedral_plot_path = (
        out_dir / f"{prefix}_wrong_cluster_dihedral_error_histograms.png"
    )
    # Keep the historical filename so existing workflows can find the artifact.
    rmsd_plot_path = out_dir / f"{prefix}_rmsd_histograms.png"
    rmsd_statistics_path = statistics_path_for_plot(rmsd_plot_path)

    rows = sorted(rows, key=lambda row: (row["source_type"], row["sample_name"]))
    residue_rows = sorted(
        residue_rows,
        key=lambda row: (row["source_type"], row["sample_name"], row["residue_index"]),
    )
    selection_rows = sorted(
        selection_rows,
        key=lambda row: (
            row["source_type"],
            row["sample_name"],
            RMSD_SELECTION_KEYS.index(row["selection_name"]),
        ),
    )
    summary = summarize_rows(rows)
    write_sample_csv(sample_csv, rows)
    write_residue_csv(residue_csv, residue_rows)
    write_selection_csv(selection_csv, selection_rows)
    write_summary_report(report_path, rows, summary, selection_rows)
    plot_summary(plot_path, selection_rows)
    write_violin_statistics(
        error_statistics_path,
        plot_name="conditioning_vs_oracle_errors",
        value_name="mismatching_residues_per_structure",
        unit="percent",
        values_by_group=mismatch_percentage_values(selection_rows),
    )
    plot_rmsd_violins(rmsd_plot_path, selection_rows)
    write_violin_statistics(
        rmsd_statistics_path,
        plot_name="conditioning_vs_oracle_rmsd",
        value_name="selection_fitted_rmsd",
        unit="angstrom",
        values_by_group=values_by_selection(
            selection_rows,
            "pdb_rmsd_angstrom",
        ),
    )
    plot_residue_confusion_matrices(
        residue_confusion_plot_path,
        confusion_rows,
        selection_title="All residues",
    )
    plot_residue_confusion_matrices(
        vmd_residue_confusion_plot_path,
        confusion_rows,
        selected_vmd_resids=frozenset(VMD_RESIDS),
        selection_title="Strict VMD CA/residue selection",
    )
    dihedral_plot_paths = write_mismatch_dihedral_plots(
        mismatch_dihedral_data,
        aggregate_dihedral_plot_path,
        args.dihedral_error_bins,
    )

    scope = "all residues" if args.all_res else VMD_SELECTION
    print(f"Processed {len(rows)} assigned cluster file(s) using: {scope}")
    print(f"Wrote per-structure CSV: {sample_csv}")
    print(f"Wrote per-residue CSV: {residue_csv}")
    print(f"Wrote per-selection CSV: {selection_csv}")
    print(f"Wrote report: {report_path}")
    print(f"Wrote error plot: {plot_path}")
    print(f"Wrote error violin statistics: {error_statistics_path}")
    print(f"Wrote RMSD violin plot: {rmsd_plot_path}")
    print(f"Wrote RMSD violin statistics: {rmsd_statistics_path}")
    print(f"Wrote all-residue confusion matrices: {residue_confusion_plot_path}")
    print(
        "Wrote strict-VMD-selection confusion matrices: "
        f"{vmd_residue_confusion_plot_path}"
    )
    if dihedral_plot_paths:
        print(
            f"Wrote {len(dihedral_plot_paths) - 1} per-structure dihedral-error "
            f"plot(s) and aggregate plot: {aggregate_dihedral_plot_path}"
        )
        print(
            "Wrote a mean/standard-deviation CSV beside every dihedral-error "
            "plot."
        )
    if missing_dihedral_samples:
        print(
            "Skipped mismatch-dihedral plots for "
            f"{len(missing_dihedral_samples)} sample(s) without dihedral arrays."
        )


if __name__ == "__main__":
    main()
