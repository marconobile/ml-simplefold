#!/usr/bin/env python3
"""Plot oracle-assigned cluster labels against the sampling conditioning labels."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402


DEFAULT_REF_NPZ = Path(
    "/storage_common/angiod/phase-data/projects/a2a/systems/a2a_small/clusters/6d4a7baa-c096-494e-b417-c8014437d37d/cluster.npz"
)
MATCH_TOKEN = "_assigned_clusters.npz"
CONDITIONED_EVAL_TOKEN = "_conditioned_eval.npz"
IDX_RE = re.compile(r"(?:^|_)(\d+)_assigned_clusters\.npz$")
FOLLOWING_LABEL_IDX_RE = re.compile(
    r"(?:^|_)following_label_(\d+)_assigned_clusters\.npz$"
)
DEFAULT_LABELS_KEY = "labels"
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
SOURCE_LABELS = {
    "active_samples": "active",
    "inactive_samples": "inactive",
    "pas_samples": "pas",
    "anecag_samples": "anecag",
    "theo_samples": "theo",
    "inzma_samples": "inzma",
    "all_structures_samples": "all_structures",
}
MERGED_SOURCE = "all_structures"
NEW_CATEGORY_ORDER = ("anecag", "inactive", "inzma", "theo")
LEGACY_CATEGORY_ORDER = ("active", "inactive", "pas")
CATEGORY_SCHEMAS = (NEW_CATEGORY_ORDER, LEGACY_CATEGORY_ORDER)
SOURCE_COLORS = {
    "active": "#1f77b4",
    "anecag": "#1f77b4",
    "theo": "#ff7f0e",
    "inzma": "#2ca02c",
    "inactive": "#d62728",
    "pas": "#2ca02c",
    MERGED_SOURCE: "#6f4e9c",
}
SOURCE_DISPLAY_NAMES = {
    "active": "ACTIVE",
    "anecag": "ANECAG",
    "theo": "THEO",
    "inzma": "INZMA",
    "inactive": "INACTIVE",
    "pas": "PAS",
    MERGED_SOURCE: "All structures",
}


def normalize_source_type(source_type: str) -> str:
    return str(source_type).split("_", 1)[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find *_assigned_clusters.npz files, compare labels_assigned to "
            "the conditioning labels saved with each sample, and plot the "
            "per-structure mismatch-fraction distributions as violin plots. "
            "A merged plot is always written. "
            "A categorized plot is also written when every category in a supported "
            "category set is present. Legacy samples without a companion "
            "conditioned-evaluation NPZ fall back to the reference cluster NPZ. "
            "Every available dataset is evaluated once across all residues and "
            f"once using the strict VMD selection {VMD_SELECTION!r}."
        )
    )
    parser.add_argument(
        "--base_path",
        type=Path,
        required=True,
        help="Directory to recursively search for files containing '_assigned_clusters.npz'.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        required=True,
        help="Directory where the output PNG will be written.",
    )
    parser.add_argument(
        "--ref_npz",
        type=Path,
        default=DEFAULT_REF_NPZ,
        help="Reference cluster NPZ containing merged__labels_assigned.",
    )
    parser.add_argument(
        "--output_name",
        default="assigned_cluster_match_histograms.png",
        help=(
            "Merged output PNG filename. The historical default filename is "
            "retained for workflow compatibility, but the plot is a violin plot."
        ),
    )
    parser.add_argument(
        "--category_output_name",
        "--category-output-name",
        dest="category_output_name",
        default=None,
        help=(
            "Categorized output PNG filename. By default, '_by_category' is "
            "inserted before the extension of --output_name. The file is only "
            "written when a complete category set is present."
        ),
    )
    parser.add_argument(
        "--vmd_output_name",
        "--vmd-output-name",
        dest="vmd_output_name",
        default=None,
        help=(
            "Merged strict-VMD-selection output PNG filename. By default, "
            "'_vmd_selection' is inserted before the extension of --output_name."
        ),
    )
    parser.add_argument(
        "--vmd_category_output_name",
        "--vmd-category-output-name",
        dest="vmd_category_output_name",
        default=None,
        help=(
            "Categorized strict-VMD-selection output PNG filename. By default, "
            "'_by_category' is inserted into the VMD output filename."
        ),
    )
    parser.add_argument(
        "--labels_npz",
        "--labels-npz",
        dest="labels_npz",
        type=Path,
        default=None,
        help=(
            "Optional labels NPZ used for following_label_* samples. If omitted, "
            "the script reads labels_npz_path from the companion JSON file."
        ),
    )
    parser.add_argument(
        "--labels_key",
        "--labels-key",
        dest="labels_key",
        default=DEFAULT_LABELS_KEY,
        help="Key in --labels_npz containing residue-level labels.",
    )
    return parser.parse_args()


def find_sampled_structures(base_path: Path) -> list[Path]:
    return sorted(
        path
        for path in base_path.rglob("*")
        if path.is_file() and MATCH_TOKEN in path.name
    )


def extract_idx(sampled_npz: Path) -> int:
    match = IDX_RE.search(sampled_npz.name)
    if match is None:
        raise ValueError(
            f"Could not extract frame index from {sampled_npz}. Expected a name like "
            "active_without_hs_000771_assigned_clusters.npz."
        )
    return int(match.group(1))


def extract_following_label_idx(sampled_npz: Path) -> int | None:
    match = FOLLOWING_LABEL_IDX_RE.search(sampled_npz.name)
    if match is None:
        return None
    return int(match.group(1))


def extract_source_type(sampled_npz: Path, base_path: Path) -> str:
    try:
        relative_parts = sampled_npz.relative_to(base_path).parts[:-1]
    except ValueError:
        relative_parts = ()

    for part in relative_parts:
        normalized_part = part.lower()
        if normalized_part in SOURCE_LABELS:
            return SOURCE_LABELS[normalized_part]

    for parent in sampled_npz.parents:
        normalized_parent = parent.name.lower()
        if normalized_parent in SOURCE_LABELS:
            return SOURCE_LABELS[normalized_parent]

    expected_dirs = ", ".join(sorted(SOURCE_LABELS))
    raise ValueError(
        f"Could not determine source type for {sampled_npz}. Expected the file "
        f"or one of its parent directories to be under one of: {expected_dirs}."
    )


def explicit_category_for_sample(sampled_npz: Path, base_path: Path) -> str | None:
    """Return a category only when the sample lives in a category-specific tree."""
    try:
        source_type = extract_source_type(sampled_npz, base_path)
    except ValueError:
        return None
    if source_type == MERGED_SOURCE:
        return None
    return source_type


def companion_metrics_path(sampled_npz: Path) -> Path:
    return sampled_npz.with_name(
        sampled_npz.name.replace(MATCH_TOKEN, "_conditioned_eval.json", 1)
    )


def companion_conditioned_eval_path(sampled_npz: Path) -> Path:
    return sampled_npz.with_name(
        sampled_npz.name.replace(MATCH_TOKEN, CONDITIONED_EVAL_TOKEN, 1)
    )


def optional_npz_array(path: Path, key: str) -> np.ndarray | None:
    with np.load(path, allow_pickle=False) as data:
        if key not in data.files:
            return None
        return np.asarray(data[key])


def load_conditioning_labels(conditioned_eval_path: Path) -> np.ndarray:
    labels = optional_npz_array(conditioned_eval_path, "conditioning_cluster_labels")
    if labels is None:
        labels = optional_npz_array(conditioned_eval_path, "original_cluster_labels")
    if labels is None:
        raise KeyError(
            f"{conditioned_eval_path} is missing both 'conditioning_cluster_labels' "
            "and 'original_cluster_labels'."
        )
    return one_dimensional_labels(
        labels,
        "Conditioning labels",
        conditioned_eval_path,
    ).astype(np.int64, copy=False)


def load_cluster_counts(sampled_npz: Path) -> np.ndarray:
    cluster_counts = optional_npz_array(sampled_npz, "cluster_counts")
    if cluster_counts is None:
        cluster_counts = optional_npz_array(sampled_npz, "merged__cluster_counts")
    if cluster_counts is None:
        raise KeyError(
            f"{sampled_npz} is missing both 'cluster_counts' and "
            "'merged__cluster_counts'."
        )
    return one_dimensional_labels(
        cluster_counts,
        "Cluster counts",
        sampled_npz,
    ).astype(np.int64, copy=False)


def load_metrics(metrics_path: Path) -> dict[str, object]:
    if not metrics_path.is_file():
        return {}
    with metrics_path.open() as handle:
        return json.load(handle)


def load_atom_resids(
    conditioned_eval_path: Path,
    metrics_path: Path,
) -> np.ndarray:
    atom_resids = optional_npz_array(conditioned_eval_path, "atom_resids")
    if atom_resids is not None:
        return one_dimensional_labels(
            atom_resids,
            "Atom residue indices",
            conditioned_eval_path,
        ).astype(np.int64, copy=False)

    metrics = load_metrics(metrics_path)
    raw_npz = metrics.get("raw_npz_path")
    if raw_npz:
        raw_npz_path = Path(str(raw_npz)).expanduser()
        if not raw_npz_path.is_absolute():
            raw_npz_path = metrics_path.parent / raw_npz_path
        if raw_npz_path.is_file():
            atom_resids = optional_npz_array(raw_npz_path, "atom_resids")
            if atom_resids is not None:
                return one_dimensional_labels(
                    atom_resids,
                    "Atom residue indices",
                    raw_npz_path,
                ).astype(np.int64, copy=False)

    raise KeyError(
        f"Could not load atom_resids for {conditioned_eval_path}. Expected "
        "'atom_resids' in that NPZ or in the raw NPZ recorded by its companion JSON."
    )


def metadata_npz_paths(
    sampled_npz: Path,
    conditioned_eval_path: Path,
    metrics_path: Path,
    metrics: dict[str, object],
    reference_npz: Path,
) -> list[Path]:
    paths = []
    if conditioned_eval_path.is_file():
        paths.append(conditioned_eval_path)

    raw_npz = metrics.get("raw_npz_path")
    if raw_npz:
        raw_npz_path = Path(str(raw_npz)).expanduser()
        if not raw_npz_path.is_absolute():
            raw_npz_path = metrics_path.parent / raw_npz_path
        raw_npz_path = raw_npz_path.resolve()
        if raw_npz_path.is_file() and raw_npz_path not in paths:
            paths.append(raw_npz_path)

    if sampled_npz.is_file() and sampled_npz not in paths:
        paths.append(sampled_npz)
    if reference_npz.is_file() and reference_npz not in paths:
        paths.append(reference_npz)
    return paths


def load_optional_metadata_array(
    sampled_npz: Path,
    conditioned_eval_path: Path,
    metrics_path: Path,
    metrics: dict[str, object],
    reference_npz: Path,
    key: str,
) -> tuple[np.ndarray | None, Path | None]:
    for candidate in metadata_npz_paths(
        sampled_npz,
        conditioned_eval_path,
        metrics_path,
        metrics,
        reference_npz,
    ):
        try:
            value = optional_npz_array(candidate, key)
        except ValueError as exc:
            if "Object arrays cannot be loaded" in str(exc):
                continue
            raise
        if value is not None:
            return value, candidate
    return None, None


def text_value(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def load_vmd_resids_by_residue_index(
    sampled_npz: Path,
    conditioned_eval_path: Path,
    metrics_path: Path,
    metrics: dict[str, object],
    reference_npz: Path,
    n_residues: int,
) -> np.ndarray:
    residue_keys, source_path = load_optional_metadata_array(
        sampled_npz,
        conditioned_eval_path,
        metrics_path,
        metrics,
        reference_npz,
        "residue_keys",
    )
    if residue_keys is None or source_path is None:
        raise KeyError(
            "The strict VMD-residue evaluation requires a non-object "
            f"'residue_keys' array for {sampled_npz}. Expected it in the "
            "conditioned-evaluation NPZ, its recorded raw NPZ, the assigned NPZ, "
            "or the reference NPZ."
        )

    residue_keys = one_dimensional_labels(
        residue_keys,
        "Residue keys",
        source_path,
    )
    if residue_keys.shape[0] != n_residues:
        raise ValueError(
            f"Residue keys for {source_path} contain {residue_keys.shape[0]} entries, "
            f"but the cluster-label arrays contain {n_residues} residues."
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
    duplicate_resids = unique[counts != 1]
    if duplicate_resids.size:
        raise ValueError(
            f"Duplicate VMD residue IDs in {source_path}: {duplicate_resids.tolist()}."
        )
    return result


def validate_selected_ca_atoms(
    sampled_npz: Path,
    conditioned_eval_path: Path,
    metrics_path: Path,
    metrics: dict[str, object],
    reference_npz: Path,
    atom_resids: np.ndarray,
    vmd_resids_by_index: np.ndarray,
    selection_mask: np.ndarray,
) -> None:
    atom_names, source_path = load_optional_metadata_array(
        sampled_npz,
        conditioned_eval_path,
        metrics_path,
        metrics,
        reference_npz,
        "atom_names",
    )
    if atom_names is None or source_path is None:
        return
    atom_names = one_dimensional_labels(atom_names, "Atom names", source_path)
    if atom_names.shape != atom_resids.shape:
        raise ValueError(
            f"Atom-name/residue-index shape mismatch for {source_path}: "
            f"{atom_names.shape} vs {atom_resids.shape}."
        )

    normalized_names = np.asarray(
        [text_value(name).strip().upper() for name in atom_names],
        dtype=str,
    )
    invalid_ca_counts = {}
    for residue_index in np.flatnonzero(selection_mask):
        ca_count = int(
            np.count_nonzero(
                (atom_resids == residue_index) & (normalized_names == "CA")
            )
        )
        if ca_count != 1:
            resid = int(vmd_resids_by_index[residue_index])
            invalid_ca_counts[resid] = ca_count
    if invalid_ca_counts:
        raise ValueError(
            "The strict VMD query requires exactly one CA atom for every requested "
            f"residue in {source_path}; invalid counts={invalid_ca_counts}."
        )


def strict_vmd_selection_mask(
    sampled_npz: Path,
    reference_npz: Path,
    n_residues: int,
) -> np.ndarray:
    conditioned_eval_path = companion_conditioned_eval_path(sampled_npz)
    metrics_path = companion_metrics_path(sampled_npz)
    metrics = load_metrics(metrics_path)
    vmd_resids_by_index = load_vmd_resids_by_residue_index(
        sampled_npz,
        conditioned_eval_path,
        metrics_path,
        metrics,
        reference_npz,
        n_residues,
    )

    selection_mask = np.isin(vmd_resids_by_index, VMD_RESIDS)
    selected_resids = set(vmd_resids_by_index[selection_mask].tolist())
    requested_resids = set(VMD_RESIDS)
    if selected_resids != requested_resids:
        missing = sorted(requested_resids - selected_resids)
        unexpected = sorted(selected_resids - requested_resids)
        raise ValueError(
            "Residue metadata does not exactly satisfy the requested VMD query "
            f"for {sampled_npz}. Missing={missing}; unexpected={unexpected}."
        )
    if int(selection_mask.sum()) != len(VMD_RESIDS):
        raise ValueError(
            f"Expected exactly {len(VMD_RESIDS)} selected residues for "
            f"{sampled_npz}, got {int(selection_mask.sum())}."
        )

    atom_resids, atom_resids_path = load_optional_metadata_array(
        sampled_npz,
        conditioned_eval_path,
        metrics_path,
        metrics,
        reference_npz,
        "atom_resids",
    )
    if atom_resids is not None and atom_resids_path is not None:
        atom_resids = one_dimensional_labels(
            atom_resids,
            "Atom residue indices",
            atom_resids_path,
        ).astype(np.int64, copy=False)
        validate_selected_ca_atoms(
            sampled_npz,
            conditioned_eval_path,
            metrics_path,
            metrics,
            reference_npz,
            atom_resids,
            vmd_resids_by_index,
            selection_mask,
        )
    return selection_mask


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
        raise ValueError(f"No atom residue indices found for {path}.")
    if int(atom_resids.min()) < 0:
        raise ValueError(f"Atom residue indices contain negative values for {path}.")

    n_residues = int(atom_resids.max()) + 1
    residue_global = np.full(n_residues, -1, dtype=np.int64)
    for residue_idx in range(n_residues):
        labels = atom_global_labels[atom_resids == residue_idx]
        if labels.size == 0:
            raise ValueError(f"Residue {residue_idx} has no atoms for {path}.")
        labels = labels[labels >= 0]
        if labels.size == 0:
            continue
        unique = np.unique(labels)
        if unique.size != 1:
            raise ValueError(
                f"Conditioning labels are not residue-consistent for {path} at "
                f"residue {residue_idx}: {unique[:20].tolist()}."
            )
        residue_global[residue_idx] = int(unique[0])
    return residue_global


def global_to_local_labels(
    residue_global_labels: np.ndarray,
    cluster_counts: np.ndarray,
    path: Path,
) -> np.ndarray:
    if residue_global_labels.shape != cluster_counts.shape:
        raise ValueError(
            f"Residue global label shape {residue_global_labels.shape} does not "
            f"match cluster-count shape {cluster_counts.shape} for {path}."
        )
    if np.any(cluster_counts <= 0):
        raise ValueError(f"Cluster counts must all be positive for {path}.")

    offsets = np.concatenate(
        [np.zeros(1, dtype=np.int64), np.cumsum(cluster_counts[:-1], dtype=np.int64)]
    )
    local = residue_global_labels - offsets
    valid = residue_global_labels >= 0
    invalid = valid & ((local < 0) | (local >= cluster_counts))
    if np.any(invalid):
        bad_residues = np.flatnonzero(invalid)[:20].tolist()
        raise ValueError(
            f"Global-to-local label conversion failed for {path} at residues "
            f"{bad_residues}."
        )
    local[~valid] = -1
    return local.astype(np.int64, copy=False)


def conditioning_labels_for_sample(sampled_npz: Path) -> np.ndarray:
    conditioned_eval_path = companion_conditioned_eval_path(sampled_npz)
    if not conditioned_eval_path.is_file():
        raise FileNotFoundError(
            f"Conditioned-evaluation NPZ not found for {sampled_npz}: "
            f"{conditioned_eval_path}"
        )

    atom_global_labels = load_conditioning_labels(conditioned_eval_path)
    atom_resids = load_atom_resids(
        conditioned_eval_path,
        companion_metrics_path(sampled_npz),
    )
    residue_global_labels = labels_to_residue_global(
        atom_global_labels,
        atom_resids,
        conditioned_eval_path,
    )
    return global_to_local_labels(
        residue_global_labels,
        load_cluster_counts(sampled_npz),
        sampled_npz,
    )


def labels_npz_path_for_sample(
    sampled_npz: Path,
    label_idx: int,
    labels_npz_arg: Path | None,
) -> Path:
    if labels_npz_arg is not None:
        return labels_npz_arg

    metrics_path = companion_metrics_path(sampled_npz)
    if not metrics_path.is_file():
        raise FileNotFoundError(
            f"Could not find companion metrics JSON for {sampled_npz}: "
            f"{metrics_path}. Pass --labels_npz explicitly."
        )

    with metrics_path.open() as f:
        metrics = json.load(f)

    metrics_label_idx = metrics.get("label_sample_index")
    if metrics_label_idx is not None and int(metrics_label_idx) != label_idx:
        raise ValueError(
            f"Filename label index {label_idx} does not match "
            f"label_sample_index={metrics_label_idx} in {metrics_path}."
        )

    labels_npz_path = metrics.get("labels_npz_path")
    if not labels_npz_path:
        raise KeyError(
            f"{metrics_path} does not contain labels_npz_path. Pass --labels_npz "
            "explicitly."
        )

    return Path(labels_npz_path).expanduser().resolve()


def load_labels_rows(labels_npz: Path, labels_key: str) -> np.ndarray:
    if not labels_npz.is_file():
        raise FileNotFoundError(f"Labels NPZ not found: {labels_npz}")

    with np.load(labels_npz, allow_pickle=False) as labels_data:
        if labels_key not in labels_data.files:
            available = ", ".join(labels_data.files)
            raise KeyError(
                f"Labels NPZ {labels_npz} is missing key {labels_key!r}. "
                f"Available keys: {available}"
            )
        labels = np.asarray(labels_data[labels_key], dtype=np.int64)

    if labels.ndim != 2:
        raise ValueError(
            f"{labels_key!r} in {labels_npz} must be 2D "
            f"(n_samples, n_residues), got shape {labels.shape}."
        )
    if labels.shape[0] == 0 or labels.shape[1] == 0:
        raise ValueError(f"{labels_key!r} in {labels_npz} has empty shape {labels.shape}.")
    return labels


def labels_row_for_sample(
    sampled_npz: Path,
    label_idx: int,
    labels_npz_arg: Path | None,
    labels_key: str,
    labels_cache: dict[tuple[Path, str], np.ndarray],
) -> np.ndarray:
    labels_npz = labels_npz_path_for_sample(sampled_npz, label_idx, labels_npz_arg)
    cache_key = (labels_npz, labels_key)
    if cache_key not in labels_cache:
        labels_cache[cache_key] = load_labels_rows(labels_npz, labels_key)

    labels = labels_cache[cache_key]
    if label_idx >= labels.shape[0]:
        raise IndexError(
            f"Extracted label index {label_idx} from {sampled_npz}, but "
            f"{labels_npz}:{labels_key} only contains {labels.shape[0]} row(s)."
        )
    return labels[label_idx]


def build_reference_row_lookup(ref_data: np.lib.npyio.NpzFile) -> dict[tuple[str, int], int]:
    required = {"merged__frame_state_ids", "merged__frame_indices"}
    missing = sorted(required - set(ref_data.files))
    if missing:
        raise KeyError(
            "Reference NPZ is missing key(s) needed to match samples by source "
            f"type and frame index: {', '.join(missing)}"
        )

    state_ids = np.asarray(ref_data["merged__frame_state_ids"]).astype(str)
    frame_indices = np.asarray(ref_data["merged__frame_indices"], dtype=np.int64)
    if state_ids.shape[0] != frame_indices.shape[0]:
        raise ValueError(
            "`merged__frame_state_ids` and `merged__frame_indices` must have the "
            f"same length, got {state_ids.shape[0]} and {frame_indices.shape[0]}."
        )

    lookup: dict[tuple[str, int], int] = {}
    for row_idx, (state_id, frame_idx) in enumerate(zip(state_ids, frame_indices, strict=True)):
        key = (normalize_source_type(str(state_id)), int(frame_idx))
        if key in lookup:
            raise ValueError(
                f"Reference NPZ contains duplicate rows for state={state_id!r}, "
                f"frame_index={int(frame_idx)}."
            )
        lookup[key] = int(row_idx)
    return lookup


def reference_row_for_sample(
    source_type: str,
    frame_index: int,
    reference_lookup: dict[tuple[str, int], int],
) -> int:
    key = (source_type, int(frame_index))
    try:
        return reference_lookup[key]
    except KeyError as exc:
        raise KeyError(
            f"Could not find reference labels for source={source_type!r}, "
            f"frame_index={int(frame_index)}."
        ) from exc


def load_sampled_labels(sampled_npz: Path) -> np.ndarray:
    with np.load(sampled_npz) as sampled_obs:
        if "labels_assigned" not in sampled_obs.files:
            raise KeyError(f"{sampled_npz} is missing required key 'labels_assigned'.")
        return np.asarray(sampled_obs["labels_assigned"], dtype=np.int64)


def one_dimensional_labels(labels: np.ndarray, label_name: str, path: Path) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.ndim == 2 and labels.shape[0] == 1:
        labels = labels[0]
    if labels.ndim != 1:
        raise ValueError(
            f"{label_name} for {path} must be 1D or a single-row 2D array, "
            f"got shape {labels.shape}."
        )
    return labels


def count_label_matches(
    reference_labels: np.ndarray,
    sampled_labels: np.ndarray,
    sampled_npz: Path,
    selection_mask: np.ndarray | None = None,
) -> tuple[int, int]:
    reference_labels = one_dimensional_labels(reference_labels, "Reference labels", sampled_npz)
    sampled_labels = one_dimensional_labels(sampled_labels, "Sampled labels", sampled_npz)

    if reference_labels.shape != sampled_labels.shape:
        raise ValueError(
            f"Label shape mismatch for {sampled_npz}: reference labels have shape "
            f"{reference_labels.shape}, sampled labels have shape {sampled_labels.shape}."
        )

    has_explicit_selection = selection_mask is not None
    if selection_mask is None:
        selection_mask = np.ones(reference_labels.shape, dtype=bool)
    else:
        selection_mask = one_dimensional_labels(
            selection_mask,
            "Residue selection mask",
            sampled_npz,
        ).astype(bool, copy=False)
        if selection_mask.shape != reference_labels.shape:
            raise ValueError(
                f"Residue selection mask for {sampled_npz} has shape "
                f"{selection_mask.shape}, expected {reference_labels.shape}."
            )

    comparable = (reference_labels >= 0) & (sampled_labels >= 0)
    if has_explicit_selection and np.any(selection_mask & ~comparable):
        invalid_residue_indices = np.flatnonzero(selection_mask & ~comparable).tolist()
        raise ValueError(
            "Every residue in the strict VMD selection must have two valid labels "
            f"for {sampled_npz}; invalid residue indices={invalid_residue_indices}."
        )
    valid = comparable & selection_mask
    if not np.any(valid):
        raise ValueError(f"No comparable residue labels for {sampled_npz}.")
    exact_match_count = int(
        np.count_nonzero(reference_labels[valid] == sampled_labels[valid])
    )
    difference_count = int(np.count_nonzero(valid) - exact_match_count)
    return exact_match_count, difference_count


def draw_violin(
    ax: Any,
    values: np.ndarray,
    position: int,
    color: Any,
    width: float = 0.78,
) -> None:
    """Draw a violin plus observations, with a safe fallback for constant data."""
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
            body.set_alpha(0.72)
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
            alpha=0.8,
        )

    sorted_values = np.sort(values)
    point_offsets = (
        np.zeros(1)
        if values.size == 1
        else np.linspace(-0.13 * width, 0.13 * width, values.size)
    )
    ax.scatter(
        position + point_offsets,
        sorted_values,
        s=15,
        color=color,
        edgecolor="black",
        linewidth=0.35,
        alpha=0.72,
        zorder=3,
    )
    ax.scatter(
        position,
        float(np.mean(values)),
        marker="D",
        s=30,
        color="black",
        edgecolor="white",
        linewidth=0.5,
        zorder=4,
    )


def save_cluster_error_violin_plot(
    exact_matches_by_source: dict[str, list[int]],
    differences_by_source: dict[str, list[int]],
    source_order: tuple[str, ...],
    output_path: Path,
    title_suffix: str,
) -> None:
    mismatch_fractions_by_source: dict[str, np.ndarray] = {}
    for source in source_order:
        source_exact = np.asarray(exact_matches_by_source[source], dtype=np.float64)
        source_differences = np.asarray(
            differences_by_source[source],
            dtype=np.float64,
        )
        if len(source_exact) != len(source_differences):
            raise ValueError(
                f"Exact-match and difference counts have different lengths for "
                f"{source!r}: {len(source_exact)} vs {len(source_differences)}."
            )
        if source_exact.size == 0:
            raise ValueError(f"Cannot create an empty violin for {source!r}.")
        totals = source_exact + source_differences
        if np.any(totals <= 0):
            raise ValueError(
                f"Every sample must compare at least one residue for {source!r}."
            )
        mismatch_fractions_by_source[source] = source_differences / totals

    x_positions = np.arange(1, len(source_order) + 1)
    fig_width = max(8.0, 2.0 * len(source_order))
    fig, ax = plt.subplots(figsize=(fig_width, 6), constrained_layout=True)
    for position, source in zip(x_positions, source_order, strict=True):
        draw_violin(
            ax,
            mismatch_fractions_by_source[source],
            int(position),
            SOURCE_COLORS.get(source, "#7f7f7f"),
        )

    sample_count = sum(
        values.size for values in mismatch_fractions_by_source.values()
    )
    ax.set_title(
        f"Assigned-cluster mismatch distribution — {title_suffix} "
        f"(n={sample_count})"
    )
    ax.set_xlabel("Sample type")
    ax.set_ylabel("Mismatching residue fraction")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [
            f"{SOURCE_DISPLAY_NAMES.get(source, source.upper())}\n"
            f"(n={mismatch_fractions_by_source[source].size})"
            for source in source_order
        ]
    )
    ax.set_xlim(0.4, len(source_order) + 0.6)
    max_mismatch_fraction = max(
        float(np.max(values)) for values in mismatch_fractions_by_source.values()
    )
    mismatch_axis_max = min(1.0, max(0.05, 1.15 * max_mismatch_fraction))
    ax.set_ylim(-0.02 * mismatch_axis_max, 1.02 * mismatch_axis_max)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.grid(axis="y", alpha=0.25)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


# Backward-compatible callable name for code that imported the old plotter.
save_stacked_match_plot = save_cluster_error_violin_plot


def output_name_with_suffix(output_name: str, suffix_to_add: str) -> str:
    output_path = Path(output_name)
    extension = output_path.suffix or ".png"
    return str(
        output_path.with_name(f"{output_path.stem}{suffix_to_add}{extension}")
    )


def categorized_output_name(merged_output_name: str, explicit_name: str | None) -> str:
    if explicit_name is not None:
        return explicit_name
    return output_name_with_suffix(merged_output_name, "_by_category")


def complete_category_schema(
    exact_matches_by_category: dict[str, list[int]],
) -> tuple[str, ...] | None:
    categories_with_values = {
        category for category, values in exact_matches_by_category.items() if values
    }
    for schema in CATEGORY_SCHEMAS:
        if set(schema).issubset(categories_with_values):
            return schema
    return None


def main() -> None:
    args = parse_args()
    base_path = args.base_path.resolve()
    out_dir = args.out_dir.resolve()
    ref_npz = args.ref_npz.resolve()
    labels_npz_arg = (
        args.labels_npz.expanduser().resolve()
        if args.labels_npz is not None
        else None
    )

    if not base_path.is_dir():
        raise NotADirectoryError(f"Base path is not a directory: {base_path}")

    sampled_structures = find_sampled_structures(base_path)
    if not sampled_structures:
        raise FileNotFoundError(
            f"No files containing {MATCH_TOKEN!r} were found under {base_path}"
        )

    merged_exact_matches: list[int] = []
    merged_differences: list[int] = []
    merged_vmd_exact_matches: list[int] = []
    merged_vmd_differences: list[int] = []
    exact_matches_by_category: dict[str, list[int]] = {
        source: [] for source in SOURCE_COLORS if source != MERGED_SOURCE
    }
    differences_by_category: dict[str, list[int]] = {
        source: [] for source in SOURCE_COLORS if source != MERGED_SOURCE
    }
    vmd_exact_matches_by_category: dict[str, list[int]] = {
        source: [] for source in SOURCE_COLORS if source != MERGED_SOURCE
    }
    vmd_differences_by_category: dict[str, list[int]] = {
        source: [] for source in SOURCE_COLORS if source != MERGED_SOURCE
    }
    labels_cache: dict[tuple[Path, str], np.ndarray] = {}

    needs_reference_npz = any(
        extract_following_label_idx(sampled_npz) is None
        and not companion_conditioned_eval_path(sampled_npz).is_file()
        for sampled_npz in sampled_structures
    )
    ref_data = None
    ref_labels_all = None
    reference_lookup = None
    if needs_reference_npz:
        if not ref_npz.is_file():
            raise FileNotFoundError(f"Reference NPZ not found: {ref_npz}")
        ref_data = np.load(ref_npz)
        if "merged__labels_assigned" not in ref_data.files:
            raise KeyError(
                f"Reference NPZ is missing key 'merged__labels_assigned': {ref_npz}"
            )
        ref_labels_all = ref_data["merged__labels_assigned"]
        reference_lookup = build_reference_row_lookup(ref_data)

    try:
        for sampled_npz in sampled_structures:
            sampled_labels = load_sampled_labels(sampled_npz)
            label_idx = extract_following_label_idx(sampled_npz)
            conditioned_eval_path = companion_conditioned_eval_path(sampled_npz)
            if label_idx is not None:
                reference_labels = labels_row_for_sample(
                    sampled_npz,
                    label_idx,
                    labels_npz_arg,
                    args.labels_key,
                    labels_cache,
                )
            elif conditioned_eval_path.is_file():
                reference_labels = conditioning_labels_for_sample(sampled_npz)
            else:
                source_type = extract_source_type(sampled_npz, base_path)
                idx = extract_idx(sampled_npz)
                if ref_labels_all is None or reference_lookup is None:
                    raise RuntimeError("Reference NPZ was not loaded.")
                ref_row_idx = reference_row_for_sample(source_type, idx, reference_lookup)
                if ref_row_idx >= ref_labels_all.shape[0]:
                    raise IndexError(
                        f"Matched reference row {ref_row_idx} for {sampled_npz}, but "
                        f"reference labels only contain {ref_labels_all.shape[0]} frame(s)."
                    )
                reference_labels = ref_labels_all[ref_row_idx]

            exact_match_count, difference_count = count_label_matches(
                reference_labels,
                sampled_labels,
                sampled_npz,
            )
            n_residues = one_dimensional_labels(
                sampled_labels,
                "Sampled labels",
                sampled_npz,
            ).shape[0]
            vmd_selection_mask = strict_vmd_selection_mask(
                sampled_npz,
                ref_npz,
                n_residues,
            )
            vmd_exact_match_count, vmd_difference_count = count_label_matches(
                reference_labels,
                sampled_labels,
                sampled_npz,
                selection_mask=vmd_selection_mask,
            )
            merged_exact_matches.append(exact_match_count)
            merged_differences.append(difference_count)
            merged_vmd_exact_matches.append(vmd_exact_match_count)
            merged_vmd_differences.append(vmd_difference_count)

            category = explicit_category_for_sample(sampled_npz, base_path)
            if category is not None:
                if category not in exact_matches_by_category:
                    exact_matches_by_category[category] = []
                    differences_by_category[category] = []
                    vmd_exact_matches_by_category[category] = []
                    vmd_differences_by_category[category] = []
                    SOURCE_COLORS[category] = "#7f7f7f"
                    SOURCE_DISPLAY_NAMES[category] = category.upper()
                exact_matches_by_category[category].append(exact_match_count)
                differences_by_category[category].append(difference_count)
                vmd_exact_matches_by_category[category].append(
                    vmd_exact_match_count
                )
                vmd_differences_by_category[category].append(vmd_difference_count)
    finally:
        if ref_data is not None:
            ref_data.close()

    out_dir.mkdir(parents=True, exist_ok=True)
    merged_output_path = out_dir / args.output_name
    save_cluster_error_violin_plot(
        {MERGED_SOURCE: merged_exact_matches},
        {MERGED_SOURCE: merged_differences},
        (MERGED_SOURCE,),
        merged_output_path,
        "All Structures — All Residues",
    )
    vmd_output_name = args.vmd_output_name or output_name_with_suffix(
        args.output_name,
        "_vmd_selection",
    )
    vmd_output_path = out_dir / vmd_output_name
    save_cluster_error_violin_plot(
        {MERGED_SOURCE: merged_vmd_exact_matches},
        {MERGED_SOURCE: merged_vmd_differences},
        (MERGED_SOURCE,),
        vmd_output_path,
        "All Structures — Strict VMD Selection",
    )

    print(f"Processed {len(sampled_structures)} file(s).")
    print(f"Saved all-residue merged violin plot to: {merged_output_path}")
    print(
        f"Saved strict-VMD-selection merged violin plot ({len(VMD_RESIDS)} "
        f"residues) to: {vmd_output_path}"
    )

    category_schema = complete_category_schema(exact_matches_by_category)
    if category_schema is not None:
        category_output_path = out_dir / categorized_output_name(
            args.output_name,
            args.category_output_name,
        )
        save_cluster_error_violin_plot(
            exact_matches_by_category,
            differences_by_category,
            category_schema,
            category_output_path,
            "By Category — All Residues",
        )
        vmd_category_output_path = out_dir / (
            args.vmd_category_output_name
            or categorized_output_name(vmd_output_name, None)
        )
        save_cluster_error_violin_plot(
            vmd_exact_matches_by_category,
            vmd_differences_by_category,
            category_schema,
            vmd_category_output_path,
            "By Category — Strict VMD Selection",
        )
        print(
            f"Saved all-residue categorized violin plot to: {category_output_path}"
        )
        print(
            "Saved strict-VMD-selection categorized violin plot to: "
            f"{vmd_category_output_path}"
        )
    else:
        categories_with_values = [
            SOURCE_DISPLAY_NAMES.get(category, category.upper())
            for category, values in exact_matches_by_category.items()
            if values
        ]
        if categories_with_values:
            found = ", ".join(categories_with_values)
            print(
                "Skipped categorized plot because no complete category set "
                f"was present. Found: {found}."
            )
        else:
            print(
                "Skipped categorized plot because the inputs are from the "
                "merged all_structures dataset."
            )


if __name__ == "__main__":
    main()
