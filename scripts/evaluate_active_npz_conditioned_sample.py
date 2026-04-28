#!/usr/bin/env python3
"""Evaluate one cluster-conditioned SimpleFold sample against an NPZ frame."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import pickle
import sys
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SIMPLEFOLD_ROOT = REPO_ROOT / "src" / "simplefold"
sys.path.insert(0, str(SIMPLEFOLD_ROOT))

import hydra  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from boltz_data_pipeline import const  # noqa: E402
from boltz_data_pipeline.feature.featurizer import BoltzFeaturizer  # noqa: E402
from boltz_data_pipeline.tokenize.boltz_protein import BoltzTokenizer  # noqa: E402
from boltz_data_pipeline.types import ChainInfo, Input, Record, Structure, StructureInfo  # noqa: E402
from model.flow import LinearPath  # noqa: E402
from model.torch.sampler import EMSampler  # noqa: E402
from processor.protein_processor import ProteinDataProcessor  # noqa: E402
from utils.datamodule_utils import collate, extract_sequence_from_tokens  # noqa: E402
from utils.boltz_utils import process_structure, save_structure  # noqa: E402
from utils.esm_utils import _af2_to_esm, esm_registry  # noqa: E402
from utils.trajectory_npz_utils import build_static_topology, sanitize_record_prefix  # noqa: E402


DEFAULT_DATA_PATH = Path("/scratch/nobilm/quantum_backmapping/training_data_active_npz")
DEFAULT_RAW_NPZ_CANDIDATES = (
    REPO_ROOT / "test_new_data_with_clusters" / "active_without_hs.npz",
    REPO_ROOT / "traj_with_cluster_labels.npz",
)
DEFAULT_CHECKPOINT_DIR = Path(
    "/storage_common/nobilm/ml-simplefold/"
    "fine_tune_with_clusters/inapo_ft_active_npz_from_simplefold100M_gpu0/checkpoints"
)
CLUSTER_KEY = "atom_idx_and_glob_cluster_id_per_frame"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample one random trajectory frame with the fine-tuned SimpleFold "
            "checkpoint, conditioning on the frame's original atom cluster labels, "
            "then compare sampled coordinates and dihedrals to the original frame."
        )
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=(
            "Raw trajectory NPZ, or the processed SimpleFold directory. The requested "
            "default is the processed directory at /scratch/nobilm/quantum_backmapping/"
            "training_data_active_npz."
        ),
    )
    parser.add_argument(
        "--raw-npz-path",
        type=Path,
        default=None,
        help=(
            "Raw trajectory NPZ containing `trajectory`, `dihedrals`, and cluster labels. "
            "Required only when --data-path points to a processed SimpleFold directory "
            "and auto-discovery does not find the raw NPZ."
        ),
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=None,
        help=(
            "Processed SimpleFold directory containing structures/, records/, and "
            "optionally tokens/. Defaults to --data-path when it is a directory."
        ),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help="Directory containing last.ckpt.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Checkpoint path. Defaults to --checkpoint-dir/last.ckpt.",
    )
    parser.add_argument(
        "--architecture-config",
        type=Path,
        default=REPO_ROOT / "configs/model/architecture/foldingdit_100M.yaml",
        help="Hydra YAML config for the FoldingDiT architecture.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts/active_npz_conditioned_eval",
        help="Directory for JSON, NPZ, and CSV evaluation outputs.",
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=None,
        help=(
            "Trajectory frame position to evaluate. If omitted, a random frame is "
            "sampled from the raw NPZ or processed manifest."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for frame selection and sampling.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device. Defaults to the visible CUDA device with most free memory, otherwise cpu.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=500,
        help="Number of Euler-Maruyama sampling steps.",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.3,
        help="EMSampler tau value.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=16.0,
        help="Coordinate scale used by ProteinDataProcessor.",
    )
    parser.add_argument(
        "--ref-scale",
        type=float,
        default=5.0,
        help="Reference-position scale used by ProteinDataProcessor.",
    )
    parser.add_argument(
        "--esm-model",
        type=str,
        default="esm2_3B",
        help="ESM model name used by the trained SimpleFold model.",
    )
    parser.add_argument(
        "--use-non-ema-weights",
        action="store_true",
        help="Load `model.` weights instead of `model_ema.module.` weights when both exist.",
    )
    parser.add_argument(
        "--no-mmap-checkpoint",
        action="store_true",
        help="Disable torch.load(..., mmap=True) for the checkpoint.",
    )
    parser.add_argument(
        "--dihedral-angle-bins",
        type=int,
        default=72,
        help="Number of bins for dihedral angle histograms (degrees).",
    )
    parser.add_argument(
        "--dihedral-error-bins",
        type=int,
        default=72,
        help="Number of bins for dihedral error histograms (degrees).",
    )
    return parser.parse_args()


def resolve_checkpoint_path(args: argparse.Namespace) -> Path:
    ckpt_path = args.checkpoint_path or (args.checkpoint_dir / "last.ckpt")
    ckpt_path = ckpt_path.expanduser().resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    return ckpt_path


def resolve_raw_npz_path(data_path: Path, raw_npz_path: Path | None) -> Path | None:
    if raw_npz_path is not None:
        raw_npz_path = raw_npz_path.expanduser().resolve()
        if not raw_npz_path.exists():
            raise FileNotFoundError(f"Raw NPZ not found: {raw_npz_path}")
        return raw_npz_path

    data_path = data_path.expanduser().resolve()
    if data_path.is_file():
        return data_path

    candidates = [
        data_path.with_suffix(".npz"),
        data_path / "trajectory.npz",
        data_path / "active_without_hs.npz",
        *DEFAULT_RAW_NPZ_CANDIDATES,
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()
    return None


def resolve_processed_dir(data_path: Path, processed_dir: Path | None) -> Path | None:
    if processed_dir is not None:
        processed_dir = processed_dir.expanduser().resolve()
        if not processed_dir.exists():
            raise FileNotFoundError(f"Processed SimpleFold directory not found: {processed_dir}")
        return processed_dir

    data_path = data_path.expanduser().resolve()
    if data_path.is_dir():
        return data_path
    if DEFAULT_DATA_PATH.exists() and DEFAULT_DATA_PATH.is_dir():
        return DEFAULT_DATA_PATH.resolve()
    return None


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)
    if not torch.cuda.is_available():
        return torch.device("cpu")

    best_idx = 0
    best_free = -1
    for idx in range(torch.cuda.device_count()):
        with torch.cuda.device(idx):
            free_bytes, _ = torch.cuda.mem_get_info()
        if free_bytes > best_free:
            best_free = free_bytes
            best_idx = idx
    return torch.device(f"cuda:{best_idx}")


def np_scalar_to_string(data: np.lib.npyio.NpzFile, key: str, default: str) -> str:
    if key not in data.files:
        return default
    value = data[key]
    if value.shape == ():
        return str(value.item())
    if value.size == 1:
        return str(value.reshape(-1)[0])
    return default


def choose_frame_position(
    nframes: int,
    frame_index: int | None,
    rng: np.random.Generator,
) -> int:
    if frame_index is None:
        return int(rng.integers(0, nframes))
    if frame_index < 0 or frame_index >= nframes:
        raise ValueError(f"--frame-index must be in [0, {nframes - 1}], got {frame_index}.")
    return int(frame_index)


def load_raw_frame(
    raw_npz_path: Path,
    frame_index: int | None,
    rng: np.random.Generator,
) -> dict[str, Any]:
    with np.load(raw_npz_path, allow_pickle=False) as data:
        required = {
            "trajectory",
            CLUSTER_KEY,
            "dihedrals",
            "dihedral_atom_indices",
            "dihedral_mask",
        }
        missing = sorted(required - set(data.files))
        if missing:
            raise KeyError(f"Raw NPZ is missing required key(s): {', '.join(missing)}")

        trajectory = data["trajectory"]
        if trajectory.ndim != 3 or trajectory.shape[-1] != 3:
            raise ValueError(
                f"`trajectory` must have shape (n_frames, n_atoms, 3), got {trajectory.shape}."
            )

        nframes, natoms, _ = trajectory.shape
        frame_position = choose_frame_position(nframes, frame_index, rng)

        frame_indices = (
            data["frame_indices"].astype(np.int64, copy=False)
            if "frame_indices" in data.files
            else np.arange(nframes, dtype=np.int64)
        )
        if frame_indices.shape[0] != nframes:
            frame_indices = np.arange(nframes, dtype=np.int64)
        frame_number = int(frame_indices[frame_position])

        sample_id = np_scalar_to_string(data, "sample_id", raw_npz_path.stem)
        record_prefix = sanitize_record_prefix(sample_id)
        record_id = f"{record_prefix}_{frame_number:06d}"

        cluster_labels = data[CLUSTER_KEY]
        dihedrals = data["dihedrals"]
        selected_sample = np.asarray(trajectory[frame_position], dtype=np.float32)
        original_cluster_labels = np.asarray(cluster_labels[frame_position], dtype=np.int64)
        original_dihedrals = np.asarray(dihedrals[frame_position], dtype=np.float32)

        if original_cluster_labels.shape != (natoms,):
            raise ValueError(
                f"`{CLUSTER_KEY}` frame must have shape ({natoms},), "
                f"got {original_cluster_labels.shape}."
            )
        if original_cluster_labels.min(initial=0) < -1:
            raise ValueError(f"`{CLUSTER_KEY}` contains labels below -1.")

        payload = {
            "selected_sample": selected_sample.copy(),
            "original_coords": selected_sample.copy(),
            "original_cluster_labels": original_cluster_labels.copy(),
            "original_dihedrals": original_dihedrals.copy(),
            "dihedral_atom_indices": np.asarray(data["dihedral_atom_indices"], dtype=np.int64),
            "dihedral_mask": np.asarray(data["dihedral_mask"], dtype=bool),
            "dihedral_keys": (
                np.asarray(data["dihedral_keys"]).astype(str).tolist()
                if "dihedral_keys" in data.files
                else [f"dihedral_{i}" for i in range(original_dihedrals.shape[-1])]
            ),
            "atom_names": (
                np.asarray(data["atom_names"]).astype(str)
                if "atom_names" in data.files
                else np.asarray([f"A{i}" for i in range(natoms)], dtype="U8")
            ),
            "atom_resids": (
                np.asarray(data["atom_resids"], dtype=np.int64)
                if "atom_resids" in data.files
                else None
            ),
            "frame_position": frame_position,
            "frame_index": frame_number,
            "record_id": record_id,
            "sample_id": sample_id,
        }
    return payload


def load_processed_frame(
    processed_dir: Path,
    frame_index: int | None,
    rng: np.random.Generator,
) -> dict[str, Any]:
    manifest_path = processed_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Processed manifest not found: {manifest_path}")

    with manifest_path.open() as f:
        manifest = json.load(f)
    if not manifest:
        raise ValueError(f"Processed manifest is empty: {manifest_path}")

    if frame_index is None:
        entry = manifest[int(rng.integers(0, len(manifest)))]
    else:
        suffix = f"_{frame_index:06d}"
        matches = [record for record in manifest if str(record["id"]).endswith(suffix)]
        if not matches:
            raise ValueError(
                f"No processed record ending in {suffix} found in {manifest_path}."
            )
        entry = matches[0]

    record_id = str(entry["id"])
    struct_path = processed_dir / "structures" / f"{record_id}.npz"
    if not struct_path.exists():
        raise FileNotFoundError(f"Processed structure not found: {struct_path}")

    with np.load(struct_path, allow_pickle=False) as data:
        atoms = data["atoms"]
        selected_sample = np.asarray(atoms["coords"], dtype=np.float32)
        if CLUSTER_KEY not in data.files:
            raise KeyError(f"Processed structure is missing `{CLUSTER_KEY}`: {struct_path}")
        original_cluster_labels = np.asarray(data[CLUSTER_KEY], dtype=np.int64)
        frame_number = int(data["frame_index"]) if "frame_index" in data.files else frame_index

    return {
        "selected_sample": selected_sample.copy(),
        "original_coords": selected_sample.copy(),
        "original_cluster_labels": original_cluster_labels.copy(),
        "original_dihedrals": None,
        "dihedral_atom_indices": None,
        "dihedral_mask": None,
        "dihedral_keys": [],
        "atom_names": None,
        "atom_resids": None,
        "frame_position": frame_number,
        "frame_index": frame_number,
        "record_id": record_id,
        "sample_id": processed_dir.name,
    }


def match_processed_record_id(
    processed_dir: Path | None,
    record_id: str,
    frame_index: int,
) -> str:
    if processed_dir is None:
        return record_id

    structures_dir = processed_dir / "structures"
    exact_path = structures_dir / f"{record_id}.npz"
    if exact_path.exists():
        return record_id

    suffix = f"_{frame_index:06d}"
    matches = sorted(structures_dir.glob(f"*{suffix}.npz"))
    if len(matches) == 1:
        return matches[0].stem

    manifest_path = processed_dir / "manifest.json"
    if manifest_path.exists():
        with manifest_path.open() as f:
            manifest = json.load(f)
        manifest_matches = [str(record["id"]) for record in manifest if str(record["id"]).endswith(suffix)]
        if len(manifest_matches) == 1:
            return manifest_matches[0]

    return record_id


def find_processed_paths(
    processed_dir: Path | None,
    record_id: str,
    frame_index: int,
) -> tuple[Path | None, Path | None, Path | None]:
    if processed_dir is None:
        return None, None, None

    structures_dir = processed_dir / "structures"
    records_dir = processed_dir / "records"
    tokens_dir = processed_dir / "tokens"

    structure_path = structures_dir / f"{record_id}.npz"
    record_path = records_dir / f"{record_id}.json"
    tokenized_path = tokens_dir / f"{record_id}.pkl"

    if not structure_path.exists():
        matches = sorted(structures_dir.glob(f"*_{frame_index:06d}.npz"))
        if matches:
            structure_path = matches[0]
            record_id = structure_path.stem
            record_path = records_dir / f"{record_id}.json"
            tokenized_path = tokens_dir / f"{record_id}.pkl"

    if not structure_path.exists() or not record_path.exists():
        return None, None, None
    return structure_path, record_path, tokenized_path if tokenized_path.exists() else None


def build_structure_and_record_from_raw(frame_data: dict[str, Any]) -> tuple[Structure, dict[str, Any]]:
    atom_names = frame_data["atom_names"]
    atom_resids = frame_data["atom_resids"]
    if atom_names is None or atom_resids is None:
        raise ValueError(
            "Cannot build a SimpleFold structure from raw data without `atom_names` "
            "and `atom_resids`."
        )

    atoms_template, bonds, residues, chains, connections, interfaces, mask = build_static_topology(
        atom_names,
        atom_resids,
    )
    atoms = atoms_template.copy()
    coords = frame_data["original_coords"].astype(np.float32, copy=False)
    atoms["coords"] = coords
    atoms["conformer"] = coords

    structure = Structure(
        atoms=atoms,
        bonds=bonds,
        residues=residues,
        chains=chains,
        connections=connections,
        interfaces=interfaces,
        mask=mask,
    )
    record = Record(
        id=frame_data["record_id"],
        structure=StructureInfo(
            resolution=0.0,
            method="simulation",
            deposited="",
            released="",
            revised="",
            num_chains=1,
            num_interfaces=0,
        ),
        chains=[
            ChainInfo(
                chain_id=0,
                chain_name="A",
                mol_type=const.chain_type_ids["PROTEIN"],
                cluster_id=-1,
                msa_id="",
                num_residues=int(residues.shape[0]),
                valid=True,
                entity_id=0,
            )
        ],
        interfaces=[],
    )
    return structure, asdict(record)


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
    structure_path, record_path, tokenized_path = find_processed_paths(
        processed_dir,
        frame_data["record_id"],
        int(frame_data["frame_index"]),
    )

    if structure_path is not None and record_path is not None:
        structure = Structure.load(structure_path)
        with record_path.open() as f:
            record_dict = json.load(f)
        if tokenized_path is not None:
            with tokenized_path.open("rb") as f:
                tokenized = pickle.load(f)
        else:
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
    cluster_labels = torch.as_tensor(frame_data["original_cluster_labels"], dtype=torch.long)
    num_model_atoms = int(features["ref_pos"].shape[0])
    if cluster_labels.shape[0] > num_model_atoms:
        raise ValueError(
            f"Cluster labels have {cluster_labels.shape[0]} atoms, but the featurized "
            f"model input has only {num_model_atoms} atoms."
        )
    padded_cluster_labels = torch.full((num_model_atoms,), -1, dtype=torch.long)
    padded_cluster_labels[: cluster_labels.shape[0]] = cluster_labels
    features[CLUSTER_KEY] = padded_cluster_labels

    batch = collate([features])
    batch = processor.preprocess_inference(
        batch,
        esm_model=esm_model,
        esm_dict=esm_dict,
        af2_to_esm=af2_to_esm,
    )
    return batch, structure, Record(**record_dict)


def load_checkpoint(path: Path, use_mmap: bool) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "map_location": "cpu",
        "weights_only": False,
    }
    if use_mmap:
        kwargs["mmap"] = True
    try:
        checkpoint = torch.load(path, **kwargs)
    except TypeError:
        kwargs.pop("mmap", None)
        checkpoint = torch.load(path, **kwargs)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected checkpoint dict, got {type(checkpoint)!r}.")
    return checkpoint


def strip_model_prefix(
    state_dict: dict[str, torch.Tensor],
    prefix: str,
) -> dict[str, torch.Tensor]:
    return {
        key[len(prefix) :]: value
        for key, value in state_dict.items()
        if key.startswith(prefix) and "esm_model." not in key
    }


def instantiate_and_load_model(
    architecture_config: Path,
    checkpoint_path: Path,
    device: torch.device,
    prefer_ema: bool,
    use_mmap: bool,
) -> torch.nn.Module:
    model_cfg = OmegaConf.load(architecture_config)
    model = hydra.utils.instantiate(model_cfg)

    checkpoint = load_checkpoint(checkpoint_path, use_mmap=use_mmap)
    state_dict = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint does not contain a dict-like state_dict.")

    primary_prefix = (
        ("model_ema.module.", "EMA model")
        if prefer_ema
        else ("model.", "non-EMA model")
    )
    fallback_prefixes = (
        ("model.", "non-EMA model"),
        ("model_ema.module.", "EMA model"),
    )

    chosen_name = None
    stripped: dict[str, torch.Tensor] = {}
    for prefix, name in [primary_prefix, *fallback_prefixes]:
        stripped = strip_model_prefix(state_dict, prefix)
        if stripped:
            chosen_name = name
            break

    if not stripped:
        stripped = {
            key: value
            for key, value in state_dict.items()
            if isinstance(key, str) and "esm_model." not in key
        }
        chosen_name = "unprefixed model"

    incompatible = model.load_state_dict(stripped, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    if missing:
        print(f"Warning: {len(missing)} missing model key(s); first 10: {missing[:10]}")
    if unexpected:
        print(f"Warning: {len(unexpected)} unexpected checkpoint key(s); first 10: {unexpected[:10]}")
    print(f"Loaded {chosen_name} weights from {checkpoint_path}")

    del checkpoint
    gc.collect()
    model = model.to(device)
    model.eval()
    return model


def kabsch_align(
    mobile: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray]:
    valid = mask.astype(bool) & np.isfinite(mobile).all(axis=-1) & np.isfinite(target).all(axis=-1)
    if valid.sum() < 3:
        raise ValueError("Need at least 3 valid atoms to align structures.")

    mobile_valid = mobile[valid].astype(np.float64, copy=False)
    target_valid = target[valid].astype(np.float64, copy=False)
    mobile_center = mobile_valid.mean(axis=0)
    target_center = target_valid.mean(axis=0)

    mobile_centered = mobile_valid - mobile_center
    target_centered = target_valid - target_center
    covariance = mobile_centered.T @ target_centered
    u, _, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(u @ vt))
    rotation = u @ correction @ vt

    aligned = (mobile.astype(np.float64, copy=False) - mobile_center) @ rotation + target_center
    atomwise_rmsd = np.full(mobile.shape[0], np.nan, dtype=np.float32)
    atomwise_rmsd[valid] = np.linalg.norm(aligned[valid] - target[valid], axis=-1).astype(np.float32)
    global_rmsd = float(np.sqrt(np.nanmean(atomwise_rmsd[valid] ** 2)))
    return aligned.astype(np.float32), global_rmsd, atomwise_rmsd


def compute_dihedral_angles(
    coords: np.ndarray,
    atom_indices: np.ndarray,
    dihedral_mask: np.ndarray,
) -> np.ndarray:
    if atom_indices.ndim != 3 or atom_indices.shape[-1] != 4:
        raise ValueError(
            f"`dihedral_atom_indices` must have shape (n_res, n_dihedrals, 4), "
            f"got {atom_indices.shape}."
        )

    angles = np.full(atom_indices.shape[:2], np.nan, dtype=np.float32)
    valid = dihedral_mask.astype(bool) & (atom_indices >= 0).all(axis=-1)
    valid &= (atom_indices < coords.shape[0]).all(axis=-1)
    if not np.any(valid):
        return angles

    idx = atom_indices[valid]
    p0 = coords[idx[:, 0]].astype(np.float64)
    p1 = coords[idx[:, 1]].astype(np.float64)
    p2 = coords[idx[:, 2]].astype(np.float64)
    p3 = coords[idx[:, 3]].astype(np.float64)

    b0 = -(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    b1_norm = np.linalg.norm(b1, axis=-1, keepdims=True)
    nonzero = b1_norm[:, 0] > 0.0
    computed = np.full(idx.shape[0], np.nan, dtype=np.float64)
    if np.any(nonzero):
        b0_nz = b0[nonzero]
        b1_nz = b1[nonzero] / b1_norm[nonzero]
        b2_nz = b2[nonzero]

        v = b0_nz - (b0_nz * b1_nz).sum(axis=-1, keepdims=True) * b1_nz
        w = b2_nz - (b2_nz * b1_nz).sum(axis=-1, keepdims=True) * b1_nz

        x = (v * w).sum(axis=-1)
        y = (np.cross(b1_nz, v) * w).sum(axis=-1)
        computed[nonzero] = np.arctan2(y, x)

    angles[valid] = computed.astype(np.float32)
    return angles


def circular_difference(sampled: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(sampled - reference), np.cos(sampled - reference)).astype(np.float32)


def summarize_dihedrals(
    original_dihedrals: np.ndarray,
    sampled_dihedrals: np.ndarray,
    original_recomputed_dihedrals: np.ndarray,
    dihedral_mask: np.ndarray,
    dihedral_keys: list[str],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    valid = (
        dihedral_mask.astype(bool)
        & np.isfinite(original_dihedrals)
        & np.isfinite(sampled_dihedrals)
    )
    diff = np.full_like(sampled_dihedrals, np.nan, dtype=np.float32)
    diff[valid] = circular_difference(sampled_dihedrals[valid], original_dihedrals[valid])
    abs_diff_deg = np.abs(np.degrees(diff))

    recompute_valid = (
        dihedral_mask.astype(bool)
        & np.isfinite(original_dihedrals)
        & np.isfinite(original_recomputed_dihedrals)
    )
    recompute_diff = np.full_like(sampled_dihedrals, np.nan, dtype=np.float32)
    recompute_diff[recompute_valid] = circular_difference(
        original_recomputed_dihedrals[recompute_valid],
        original_dihedrals[recompute_valid],
    )

    summary: dict[str, Any] = {
        "count": int(valid.sum()),
        "mae_deg": float(np.nanmean(abs_diff_deg[valid])) if np.any(valid) else None,
        "rmse_deg": (
            float(np.sqrt(np.nanmean(np.degrees(diff[valid]) ** 2)))
            if np.any(valid)
            else None
        ),
        "max_abs_error_deg": float(np.nanmax(abs_diff_deg[valid])) if np.any(valid) else None,
        "stored_vs_recomputed_original_mae_deg": (
            float(np.nanmean(np.abs(np.degrees(recompute_diff[recompute_valid]))))
            if np.any(recompute_valid)
            else None
        ),
        "by_key": {},
    }

    for i, key in enumerate(dihedral_keys):
        key_valid = valid[:, i]
        key_abs = abs_diff_deg[:, i]
        key_diff_deg = np.degrees(diff[:, i])
        summary["by_key"][key] = {
            "count": int(key_valid.sum()),
            "mae_deg": float(np.nanmean(key_abs[key_valid])) if np.any(key_valid) else None,
            "rmse_deg": (
                float(np.sqrt(np.nanmean(key_diff_deg[key_valid] ** 2)))
                if np.any(key_valid)
                else None
            ),
            "max_abs_error_deg": (
                float(np.nanmax(key_abs[key_valid])) if np.any(key_valid) else None
            ),
        }

    return summary, diff, abs_diff_deg


def add_dihedral_validation_summary(
    summary: dict[str, Any],
    name: str,
    a: np.ndarray,
    b: np.ndarray,
    dihedral_mask: np.ndarray,
) -> None:
    valid = dihedral_mask.astype(bool) & np.isfinite(a) & np.isfinite(b)
    diff = np.full_like(a, np.nan, dtype=np.float32)
    diff[valid] = circular_difference(a[valid], b[valid])
    diff_deg = np.degrees(diff)
    abs_diff_deg = np.abs(diff_deg)
    summary[name] = {
        "count": int(valid.sum()),
        "mae_deg": float(np.nanmean(abs_diff_deg[valid])) if np.any(valid) else None,
        "rmse_deg": (
            float(np.sqrt(np.nanmean(diff_deg[valid] ** 2)))
            if np.any(valid)
            else None
        ),
        "max_abs_error_deg": (
            float(np.nanmax(abs_diff_deg[valid])) if np.any(valid) else None
        ),
    }


def _histogram_counts(
    values: np.ndarray,
    bins: int,
    value_range: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(value_range[0], value_range[1], bins + 1, dtype=np.float32)
    if values.size == 0:
        return np.zeros(bins, dtype=np.int64), edges
    counts, _ = np.histogram(values, bins=bins, range=value_range)
    return counts.astype(np.int64, copy=False), edges


def write_dihedral_histograms(
    output_dir: Path,
    output_stem: str,
    original_dihedrals: np.ndarray,
    sampled_dihedrals: np.ndarray,
    dihedral_diff_rad: np.ndarray,
    dihedral_mask: np.ndarray,
    dihedral_keys: list[str],
    angle_bins: int,
    error_bins: int,
) -> dict[str, Any]:
    valid = (
        dihedral_mask.astype(bool)
        & np.isfinite(original_dihedrals)
        & np.isfinite(sampled_dihedrals)
        & np.isfinite(dihedral_diff_rad)
    )

    original_deg = np.degrees(original_dihedrals)
    sampled_deg = np.degrees(sampled_dihedrals)
    error_deg = np.degrees(dihedral_diff_rad)
    abs_error_deg = np.abs(error_deg)

    angle_csv_path = output_dir / f"{output_stem}_dihedral_angle_histograms.csv"
    error_csv_path = output_dir / f"{output_stem}_dihedral_error_histograms.csv"
    angle_png_path = output_dir / f"{output_stem}_dihedral_angle_histograms.png"
    error_png_path = output_dir / f"{output_stem}_dihedral_error_histograms.png"

    with angle_csv_path.open("w", newline="") as f_angle, error_csv_path.open("w", newline="") as f_error:
        angle_writer = csv.writer(f_angle)
        error_writer = csv.writer(f_error)

        angle_writer.writerow(
            [
                "dihedral_key",
                "series",
                "bin_left_deg",
                "bin_right_deg",
                "count",
            ]
        )
        error_writer.writerow(
            [
                "dihedral_key",
                "series",
                "bin_left_deg",
                "bin_right_deg",
                "count",
            ]
        )

        all_valid = valid.reshape(-1)
        all_orig = original_deg.reshape(-1)[all_valid]
        all_sampled = sampled_deg.reshape(-1)[all_valid]
        all_err = error_deg.reshape(-1)[all_valid]
        all_abs_err = abs_error_deg.reshape(-1)[all_valid]

        angle_specs = [
            ("all", "original_deg", all_orig),
            ("all", "sampled_deg", all_sampled),
        ]
        error_specs = [
            ("all", "signed_error_deg", all_err),
            ("all", "abs_error_deg", all_abs_err),
        ]

        for key_idx, key in enumerate(dihedral_keys):
            key_valid = valid[:, key_idx]
            angle_specs.extend(
                [
                    (key, "original_deg", original_deg[:, key_idx][key_valid]),
                    (key, "sampled_deg", sampled_deg[:, key_idx][key_valid]),
                ]
            )
            error_specs.extend(
                [
                    (key, "signed_error_deg", error_deg[:, key_idx][key_valid]),
                    (key, "abs_error_deg", abs_error_deg[:, key_idx][key_valid]),
                ]
            )

        for key, series, values in angle_specs:
            counts, edges = _histogram_counts(values, angle_bins, (-180.0, 180.0))
            for bin_idx, count in enumerate(counts):
                angle_writer.writerow([key, series, float(edges[bin_idx]), float(edges[bin_idx + 1]), int(count)])

        for key, series, values in error_specs:
            if series == "abs_error_deg":
                hist_range = (0.0, 180.0)
            else:
                hist_range = (-180.0, 180.0)
            counts, edges = _histogram_counts(values, error_bins, hist_range)
            for bin_idx, count in enumerate(counts):
                error_writer.writerow([key, series, float(edges[bin_idx]), float(edges[bin_idx + 1]), int(count)])

    histogram_artifacts: dict[str, Any] = {
        "angle_histogram_csv": str(angle_csv_path),
        "error_histogram_csv": str(error_csv_path),
        "angle_histogram_png": None,
        "error_histogram_png": None,
        "angle_bins": int(angle_bins),
        "error_bins": int(error_bins),
    }

    try:
        import matplotlib.pyplot as plt

        row_labels = ["all", *dihedral_keys]

        angle_fig, angle_axes = plt.subplots(
            nrows=len(row_labels),
            ncols=1,
            figsize=(9.0, 2.4 * len(row_labels)),
            constrained_layout=True,
        )
        if len(row_labels) == 1:
            angle_axes = [angle_axes]
        for row_idx, label in enumerate(row_labels):
            if label == "all":
                row_valid = all_valid
                orig_vals = all_orig
                samp_vals = all_sampled
            else:
                key_idx = dihedral_keys.index(label)
                row_valid = valid[:, key_idx]
                orig_vals = original_deg[:, key_idx][row_valid]
                samp_vals = sampled_deg[:, key_idx][row_valid]
            ax = angle_axes[row_idx]
            ax.hist(orig_vals, bins=angle_bins, range=(-180.0, 180.0), alpha=0.5, label="original")
            ax.hist(samp_vals, bins=angle_bins, range=(-180.0, 180.0), alpha=0.5, label="sampled")
            ax.set_xlim(-180.0, 180.0)
            ax.set_ylabel("count")
            ax.set_title(f"{label} angle distribution (n={int(row_valid.sum())})")
            if row_idx == 0:
                ax.legend(loc="upper right")
        angle_axes[-1].set_xlabel("dihedral angle (deg)")
        angle_fig.savefig(angle_png_path, dpi=180)
        plt.close(angle_fig)
        histogram_artifacts["angle_histogram_png"] = str(angle_png_path)

        error_fig, error_axes = plt.subplots(
            nrows=len(row_labels),
            ncols=2,
            figsize=(12.0, 2.6 * len(row_labels)),
            constrained_layout=True,
        )
        if len(row_labels) == 1:
            error_axes = np.asarray([error_axes])
        for row_idx, label in enumerate(row_labels):
            if label == "all":
                row_valid = all_valid
                signed_vals = all_err
                abs_vals = all_abs_err
            else:
                key_idx = dihedral_keys.index(label)
                row_valid = valid[:, key_idx]
                signed_vals = error_deg[:, key_idx][row_valid]
                abs_vals = abs_error_deg[:, key_idx][row_valid]
            signed_ax = error_axes[row_idx, 0]
            abs_ax = error_axes[row_idx, 1]
            signed_ax.hist(signed_vals, bins=error_bins, range=(-180.0, 180.0), color="tab:orange", alpha=0.8)
            abs_ax.hist(abs_vals, bins=error_bins, range=(0.0, 180.0), color="tab:red", alpha=0.8)
            signed_ax.set_xlim(-180.0, 180.0)
            abs_ax.set_xlim(0.0, 180.0)
            signed_ax.set_ylabel("count")
            signed_ax.set_title(f"{label} signed error (n={int(row_valid.sum())})")
            abs_ax.set_title(f"{label} absolute error")
        error_axes[-1, 0].set_xlabel("signed error (deg)")
        error_axes[-1, 1].set_xlabel("absolute error (deg)")
        error_fig.savefig(error_png_path, dpi=180)
        plt.close(error_fig)
        histogram_artifacts["error_histogram_png"] = str(error_png_path)
    except Exception as exc:
        print(f"Warning: failed to render dihedral histogram PNG files: {exc}")

    return histogram_artifacts


def write_atomwise_csv(
    path: Path,
    atomwise_rmsd: np.ndarray,
    atom_names: np.ndarray | None,
    atom_resids: np.ndarray | None,
) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["atom_index", "atom_name", "residue_index", "atomwise_rmsd"])
        for atom_idx, value in enumerate(atomwise_rmsd):
            atom_name = "" if atom_names is None else str(atom_names[atom_idx])
            residue_index = "" if atom_resids is None else int(atom_resids[atom_idx])
            writer.writerow([atom_idx, atom_name, residue_index, value])


def write_dihedral_csv(
    path: Path,
    original_dihedrals: np.ndarray,
    sampled_dihedrals: np.ndarray,
    dihedral_diff_rad: np.ndarray,
    dihedral_keys: list[str],
) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "residue_index",
                "dihedral_key",
                "original_rad",
                "sampled_rad",
                "diff_rad",
                "abs_diff_deg",
            ]
        )
        for res_idx in range(original_dihedrals.shape[0]):
            for dih_idx, key in enumerate(dihedral_keys):
                diff_rad = dihedral_diff_rad[res_idx, dih_idx]
                writer.writerow(
                    [
                        res_idx,
                        key,
                        original_dihedrals[res_idx, dih_idx],
                        sampled_dihedrals[res_idx, dih_idx],
                        diff_rad,
                        np.abs(np.degrees(diff_rad)) if np.isfinite(diff_rad) else np.nan,
                    ]
                )


def format_optional_float(value: float | None, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    return f"{value:.6f}{suffix}"


def write_report(
    path: Path,
    metrics: dict[str, Any],
    sampled_cif_path: Path,
    target_cif_path: Path,
    arrays_path: Path,
    atom_csv_path: Path,
    dihedral_csv_path: Path | None,
    raw_sampled_cif_path: Path | None = None,
) -> None:
    dihedrals = metrics.get("dihedrals") or {}
    lines = [
        "SimpleFold conditioned sampling evaluation",
        "",
        "Selected sample",
        f"  record_id: {metrics['record_id']}",
        f"  raw_record_id: {metrics.get('raw_record_id')}",
        f"  sample_id: {metrics['sample_id']}",
        f"  frame_position: {metrics['frame_position']}",
        f"  frame_index: {metrics['frame_index']}",
        f"  atoms: {metrics['num_atoms']}",
        "",
        "Inputs",
        f"  raw_npz_path: {metrics['raw_npz_path']}",
        f"  processed_dir: {metrics['processed_dir']}",
        f"  checkpoint_path: {metrics['checkpoint_path']}",
        f"  seed: {metrics['seed']}",
        f"  sampler: EMSampler, num_steps={metrics['num_steps']}, tau={metrics['tau']}",
        "",
        "Coordinate comparison after Kabsch alignment",
        f"  global_rmsd_A: {metrics['global_rmsd']:.6f}",
        f"  atomwise_rmsd_mean_A: {metrics['atomwise_rmsd_mean']:.6f}",
        f"  atomwise_rmsd_median_A: {metrics['atomwise_rmsd_median']:.6f}",
        f"  atomwise_rmsd_max_A: {metrics['atomwise_rmsd_max']:.6f}",
        "",
        "Dihedral comparison",
        f"  compared_angles: {dihedrals.get('count', 'n/a')}",
        f"  mae_deg: {format_optional_float(dihedrals.get('mae_deg'))}",
        f"  rmse_deg: {format_optional_float(dihedrals.get('rmse_deg'))}",
        f"  max_abs_error_deg: {format_optional_float(dihedrals.get('max_abs_error_deg'))}",
        "  stored_vs_recomputed_original_mae_deg: "
        f"{format_optional_float(dihedrals.get('stored_vs_recomputed_original_mae_deg'))}",
    ]
    raw_vs_aligned = dihedrals.get("sampled_raw_vs_aligned")
    if raw_vs_aligned is not None:
        lines.extend(
            [
                "  sampled_raw_vs_aligned_count: "
                f"{raw_vs_aligned.get('count', 'n/a')}",
                "  sampled_raw_vs_aligned_mae_deg: "
                f"{format_optional_float(raw_vs_aligned.get('mae_deg'))}",
                "  sampled_raw_vs_aligned_max_abs_error_deg: "
                f"{format_optional_float(raw_vs_aligned.get('max_abs_error_deg'))}",
            ]
        )

    by_key = dihedrals.get("by_key") or {}
    if by_key:
        lines.extend(["", "Dihedral breakdown"])
        for key, stats in by_key.items():
            lines.append(
                "  "
                f"{key}: count={stats['count']}, "
                f"mae_deg={format_optional_float(stats['mae_deg'])}, "
                f"rmse_deg={format_optional_float(stats['rmse_deg'])}, "
                f"max_abs_error_deg={format_optional_float(stats['max_abs_error_deg'])}"
            )

    histograms = dihedrals.get("histograms")
    if histograms:
        lines.extend(
            [
                "",
                "Dihedral histograms",
                f"  angle_bins: {histograms.get('angle_bins', 'n/a')}",
                f"  error_bins: {histograms.get('error_bins', 'n/a')}",
                f"  angle_histogram_csv: {histograms.get('angle_histogram_csv')}",
                f"  error_histogram_csv: {histograms.get('error_histogram_csv')}",
            ]
        )
        if histograms.get("angle_histogram_png") is not None:
            lines.append(f"  angle_histogram_png: {histograms.get('angle_histogram_png')}")
        if histograms.get("error_histogram_png") is not None:
            lines.append(f"  error_histogram_png: {histograms.get('error_histogram_png')}")

    lines.extend(
        [
            "",
            "Main artifacts",
            f"  sampled_model_cif_aligned_for_rmsd: {sampled_cif_path}",
            f"  target_reference_cif: {target_cif_path}",
            f"  detailed_arrays_npz: {arrays_path}",
            f"  atomwise_rmsd_csv: {atom_csv_path}",
        ]
    )
    if raw_sampled_cif_path is not None:
        lines.append(f"  sampled_model_raw_unaligned_cif: {raw_sampled_cif_path}")
    if dihedral_csv_path is not None:
        lines.append(f"  dihedral_csv: {dihedral_csv_path}")

    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    if args.num_steps <= 0:
        raise ValueError("--num-steps must be > 0")
    if args.dihedral_angle_bins <= 0:
        raise ValueError("--dihedral-angle-bins must be > 0")
    if args.dihedral_error_bins <= 0:
        raise ValueError("--dihedral-error-bins must be > 0")

    data_path = args.data_path.expanduser().resolve()
    processed_dir = resolve_processed_dir(data_path, args.processed_dir)
    raw_npz_path = resolve_raw_npz_path(data_path, args.raw_npz_path)
    checkpoint_path = resolve_checkpoint_path(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    if raw_npz_path is not None:
        print(f"Loading raw trajectory NPZ: {raw_npz_path}")
        frame_data = load_raw_frame(raw_npz_path, args.frame_index, rng)
    elif processed_dir is not None:
        print(
            "Raw trajectory NPZ was not found; loading coordinates and cluster labels "
            f"from processed directory: {processed_dir}"
        )
        frame_data = load_processed_frame(processed_dir, args.frame_index, rng)
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
    gc.collect()

    selected_sample = frame_data["selected_sample"]
    original_coords = frame_data["original_coords"]
    original_cluster_labels = frame_data["original_cluster_labels"]
    original_dihedrals = frame_data["original_dihedrals"]

    print(
        "Selected frame "
        f"position={frame_data['frame_position']} frame_index={frame_data['frame_index']} "
        f"record_id={frame_data['record_id']} atoms={original_coords.shape[0]}"
    )
    print(
        "Cluster label range: "
        f"{int(original_cluster_labels.min())}..{int(original_cluster_labels.max())}"
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
    )
    tokenizer = BoltzTokenizer()
    featurizer = BoltzFeaturizer()

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

    # ESM features are now materialized in the batch, so the ESM model can be released
    # before the flow model and checkpoint are loaded.
    del esm_model, esm_dict, af2_to_esm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if batch[CLUSTER_KEY].shape[1] != batch["coords"].shape[1]:
        raise ValueError(
            f"Conditioning labels have {batch[CLUSTER_KEY].shape[1]} atoms after batching, "
            f"but model coordinates have {batch['coords'].shape[1]} atoms."
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
    )

    print("Sampling conditioned structure with EMSampler")
    with torch.no_grad():
        noise = torch.randn_like(batch["coords"], device=device)
        out_dict = sampler.sample(model, flow, noise, batch)
        out_dict = processor.postprocess(out_dict, batch)
        sampled_coords_full_tensor = out_dict["denoised_coords"][0].detach().cpu()
        atom_mask_full_tensor = batch["atom_pad_mask"][0].detach().cpu()
        sampled_coords_full = sampled_coords_full_tensor.numpy().astype(np.float32)
        atom_mask_full = atom_mask_full_tensor.numpy().astype(bool)
        sampled_coords = sampled_coords_full[atom_mask_full]

    if sampled_coords.shape != original_coords.shape:
        raise ValueError(
            f"Unpadded sampled coordinates have shape {sampled_coords.shape}, "
            f"but original frame has shape {original_coords.shape}."
        )
    atom_mask = np.ones(original_coords.shape[0], dtype=bool)

    aligned_sampled_coords, global_rmsd, atomwise_rmsd = kabsch_align(
        sampled_coords,
        original_coords,
        atom_mask,
    )
    print(f"Aligned RMSD: {global_rmsd:.4f} A")

    dihedral_summary = None
    sampled_dihedrals = None
    original_recomputed_dihedrals = None
    dihedral_diff_rad = None
    dihedral_abs_error_deg = None
    if original_dihedrals is not None:
        sampled_dihedrals = compute_dihedral_angles(
            sampled_coords,
            frame_data["dihedral_atom_indices"],
            frame_data["dihedral_mask"],
        )
        sampled_aligned_dihedrals = compute_dihedral_angles(
            aligned_sampled_coords,
            frame_data["dihedral_atom_indices"],
            frame_data["dihedral_mask"],
        )
        original_recomputed_dihedrals = compute_dihedral_angles(
            selected_sample,
            frame_data["dihedral_atom_indices"],
            frame_data["dihedral_mask"],
        )
        dihedral_summary, dihedral_diff_rad, dihedral_abs_error_deg = summarize_dihedrals(
            original_dihedrals=original_dihedrals,
            sampled_dihedrals=sampled_dihedrals,
            original_recomputed_dihedrals=original_recomputed_dihedrals,
            dihedral_mask=frame_data["dihedral_mask"],
            dihedral_keys=frame_data["dihedral_keys"],
        )
        add_dihedral_validation_summary(
            dihedral_summary,
            "sampled_raw_vs_aligned",
            sampled_dihedrals,
            sampled_aligned_dihedrals,
            frame_data["dihedral_mask"],
        )
        print(
            "Dihedral MAE: "
            f"{dihedral_summary['mae_deg']:.4f} deg over {dihedral_summary['count']} angles"
        )
    else:
        print("Skipping dihedral comparison because no raw trajectory dihedrals were available.")

    output_stem = f"{frame_data['record_id']}_conditioned_eval"
    metrics_path = args.output_dir / f"{output_stem}.json"
    arrays_path = args.output_dir / f"{output_stem}.npz"
    atom_csv_path = args.output_dir / f"{output_stem}_atomwise_rmsd.csv"
    dihedral_csv_path = args.output_dir / f"{output_stem}_dihedrals.csv"
    sampled_cif_path = args.output_dir / f"{output_stem}_sampled.cif"
    raw_sampled_cif_path = args.output_dir / f"{output_stem}_sampled_raw.cif"
    target_cif_path = args.output_dir / f"{output_stem}_target_reference.cif"
    report_path = args.output_dir / f"{output_stem}_report.txt"
    dihedral_histogram_artifacts = None

    sampled_structure = process_structure(
        deepcopy(structure),
        torch.as_tensor(aligned_sampled_coords, dtype=torch.float32),
        torch.ones(aligned_sampled_coords.shape[0], dtype=torch.bool),
        record,
    )
    save_structure(
        sampled_structure,
        args.output_dir,
        f"{output_stem}_sampled",
        output_format="mmcif",
    )
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

    if original_dihedrals is not None and dihedral_summary is not None:
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
        "processed_dir": str(processed_dir) if processed_dir is not None else None,
        "checkpoint_path": str(checkpoint_path),
        "sampled_cif_path": str(sampled_cif_path),
        "raw_sampled_cif_path": str(raw_sampled_cif_path),
        "target_reference_cif_path": str(target_cif_path),
        "record_id": frame_data["record_id"],
        "raw_record_id": frame_data.get("raw_record_id"),
        "sample_id": frame_data["sample_id"],
        "frame_position": int(frame_data["frame_position"]),
        "frame_index": int(frame_data["frame_index"]),
        "seed": args.seed,
        "num_steps": args.num_steps,
        "tau": args.tau,
        "num_atoms": int(original_coords.shape[0]),
        "global_rmsd": global_rmsd,
        "atomwise_rmsd_mean": float(np.nanmean(atomwise_rmsd)),
        "atomwise_rmsd_median": float(np.nanmedian(atomwise_rmsd)),
        "atomwise_rmsd_max": float(np.nanmax(atomwise_rmsd)),
        "dihedrals": dihedral_summary,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2))

    arrays = {
        "original_coords": original_coords,
        "selected_sample": selected_sample,
        "sampled_coords": sampled_coords,
        "sampled_coords_full": sampled_coords_full,
        "aligned_sampled_coords": aligned_sampled_coords,
        "atomwise_rmsd": atomwise_rmsd,
        "atom_mask": atom_mask,
        "model_atom_pad_mask": atom_mask_full,
        "original_cluster_labels": original_cluster_labels,
        "frame_position": np.asarray(frame_data["frame_position"], dtype=np.int64),
        "frame_index": np.asarray(frame_data["frame_index"], dtype=np.int64),
    }
    if original_dihedrals is not None:
        arrays.update(
            {
                "original_dihedrals": original_dihedrals,
                "sampled_dihedrals": sampled_dihedrals,
                "sampled_aligned_dihedrals": sampled_aligned_dihedrals,
                "original_recomputed_dihedrals": original_recomputed_dihedrals,
                "dihedral_diff_rad": dihedral_diff_rad,
                "dihedral_abs_error_deg": dihedral_abs_error_deg,
                "dihedral_atom_indices": frame_data["dihedral_atom_indices"],
                "dihedral_mask": frame_data["dihedral_mask"],
                "dihedral_keys": np.asarray(frame_data["dihedral_keys"]),
            }
        )
    np.savez_compressed(arrays_path, **arrays)

    write_atomwise_csv(
        atom_csv_path,
        atomwise_rmsd,
        frame_data["atom_names"],
        frame_data["atom_resids"],
    )
    if original_dihedrals is not None:
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
    print(f"Wrote target CIF:  {target_cif_path}")
    print(f"Wrote atom RMSD CSV: {atom_csv_path}")
    if original_dihedrals is not None:
        print(f"Wrote dihedral CSV: {dihedral_csv_path}")
    if dihedral_histogram_artifacts is not None:
        print(f"Wrote dihedral angle histogram CSV: {dihedral_histogram_artifacts['angle_histogram_csv']}")
        print(f"Wrote dihedral error histogram CSV: {dihedral_histogram_artifacts['error_histogram_csv']}")
        if dihedral_histogram_artifacts["angle_histogram_png"] is not None:
            print(f"Wrote dihedral angle histogram PNG: {dihedral_histogram_artifacts['angle_histogram_png']}")
        if dihedral_histogram_artifacts["error_histogram_png"] is not None:
            print(f"Wrote dihedral error histogram PNG: {dihedral_histogram_artifacts['error_histogram_png']}")


if __name__ == "__main__":
    main()
