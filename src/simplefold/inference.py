#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

import os
import torch
import hydra
import omegaconf
import argparse
import numpy as np
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from itertools import starmap
import lightning.pytorch as pl
from importlib import resources

from model.flow import LinearPath
from model.torch.sampler import EMSampler

from processor.protein_processor import ProteinDataProcessor
from utils.datamodule_utils import process_one_inference_structure
from utils.esm_utils import _af2_to_esm, esm_registry
from utils.boltz_utils import process_structure, save_structure
from utils.dihedral_index_utils import (
    normalize_dihedral_atom_indices,
    remap_dihedral_atom_indices_to_input,
)
from utils.fasta_utils import process_fastas, download_fasta_utilities, check_fasta_inputs
from boltz_data_pipeline.feature.featurizer import BoltzFeaturizer
from boltz_data_pipeline.tokenize.boltz_protein import BoltzTokenizer

try:
    import mlx.core as mx
    from mlx.utils import tree_unflatten, tree_flatten
    from model.mlx.sampler import EMSampler as EMSamplerMLX
    from model.mlx.esm_network import ESM2 as ESM2MLX
    from utils.mlx_utils import map_torch_to_mlx, map_plddt_torch_to_mlx
    MLX_AVAILABLE = True
except:
    MLX_AVAILABLE = False
    print("MLX not installed, skip importing MLX related packages.")


ckpt_url_dict = {
    "simplefold_100M": "https://ml-site.cdn-apple.com/models/simplefold/simplefold_100M.ckpt",
    "simplefold_360M": "https://ml-site.cdn-apple.com/models/simplefold/simplefold_360M.ckpt",
    "simplefold_700M": "https://ml-site.cdn-apple.com/models/simplefold/simplefold_700M.ckpt",
    "simplefold_1.1B": "https://ml-site.cdn-apple.com/models/simplefold/simplefold_1.1B.ckpt",
    "simplefold_1.6B": "https://ml-site.cdn-apple.com/models/simplefold/simplefold_1.6B.ckpt",
    "simplefold_3B": "https://ml-site.cdn-apple.com/models/simplefold/simplefold_3B.ckpt",
}

plddt_ckpt_url = "https://ml-site.cdn-apple.com/models/simplefold/plddt_module_1.6B.ckpt"


def _decode_atom_name_code(code_row):
    # Atom names are stored as (ord(char) - 32) with zero-padding.
    chars = [chr(int(c) + 32) for c in code_row if int(c) != 0]
    return "".join(chars).strip().upper()


def _normalize_atom_name(name):
    if isinstance(name, (bytes, np.bytes_)):
        return name.decode("utf-8").strip().upper()
    return str(name).strip().upper()


def _decode_batch_atom_names(batch):
    ref_atom_name_chars = batch["ref_atom_name_chars"]
    atom_pad_mask = batch["atom_pad_mask"]

    # [B, N, 4, 64] -> [B, N, 4] encoded name chars
    atom_name_codes = torch.argmax(ref_atom_name_chars, dim=-1).detach().cpu().numpy()
    atom_pad_mask = atom_pad_mask.detach().cpu().numpy() > 0.5

    decoded_per_batch = []
    for b in range(atom_name_codes.shape[0]):
        num_valid_atoms = int(atom_pad_mask[b].sum())
        decoded = [
            _decode_atom_name_code(atom_name_codes[b, i])
            for i in range(num_valid_atoms)
        ]
        decoded_per_batch.append(decoded)
    return decoded_per_batch


def _decode_batch_atom_residue_indices(batch):
    atom_to_token_idx = batch["atom_to_token_idx"].long()
    residue_index = batch["residue_index"].long()
    atom_pad_mask = batch["atom_pad_mask"].detach().cpu().numpy() > 0.5

    atom_residue_index = torch.gather(residue_index, dim=1, index=atom_to_token_idx)
    atom_residue_index = atom_residue_index.detach().cpu().numpy()

    decoded_per_batch = []
    for b in range(atom_residue_index.shape[0]):
        num_valid_atoms = int(atom_pad_mask[b].sum())
        decoded_per_batch.append(atom_residue_index[b, :num_valid_atoms])
    return decoded_per_batch


def _load_target_atom_names(npz_data):
    name_key_candidates = ("atom_names", "target_atom_names", "names")
    name_key = None
    for key in name_key_candidates:
        if key in npz_data:
            name_key = key
            break
    if name_key is None:
        raise ValueError(
            "Target NPZ does not contain atom names. "
            f"Tried keys: {name_key_candidates}."
        )

    atom_names = npz_data[name_key]
    if atom_names.ndim == 2 and atom_names.shape[-1] == 4 and np.issubdtype(atom_names.dtype, np.integer):
        # Handle names encoded as 4 integer chars (SimpleFold structure format).
        decoded = [_decode_atom_name_code(row) for row in atom_names]
        return np.asarray(decoded, dtype=np.str_), name_key

    if atom_names.ndim != 1:
        raise ValueError(
            f"Target atom names must be 1D (or Nx4 int-encoded), got shape {atom_names.shape}."
        )

    decoded = np.asarray([_normalize_atom_name(name) for name in atom_names], dtype=np.str_)
    return decoded, name_key


def _load_target_atom_residue_index(npz_data):
    residue_key_candidates = ("atom_residue_index", "atom_resids")
    residue_key = None
    for key in residue_key_candidates:
        if key in npz_data:
            residue_key = key
            break
    if residue_key is None:
        return None, None

    atom_residue_index = np.asarray(npz_data[residue_key], dtype=np.int64)
    if atom_residue_index.ndim != 1:
        raise ValueError(
            f"Target atom residue indices must be 1D, got shape {atom_residue_index.shape}."
        )
    return atom_residue_index, residue_key


def _pdb_atom_name_field(atom_name):
    atom_name = _normalize_atom_name(atom_name)[:4]
    if len(atom_name) >= 4:
        return atom_name[:4]
    if atom_name and atom_name[0].isdigit():
        return atom_name.ljust(4)
    return f" {atom_name:<3}"


def _pdb_element_symbol(atom_name):
    atom_name = _normalize_atom_name(atom_name)
    letters = [char for char in atom_name if char.isalpha()]
    if not letters:
        return "X"
    return letters[0].upper()


def _write_coords_to_pdb(
    output_path,
    coords,
    atom_names,
    atom_residue_index=None,
    source_npz_path=None,
    frame_idx=None,
    label=None,
):
    output_path = Path(output_path)
    coords = np.asarray(coords, dtype=np.float32)
    atom_names = np.asarray(atom_names)

    if coords.ndim != 2 or coords.shape[-1] != 3:
        raise ValueError(f"Expected coordinates with shape (N, 3), got {coords.shape}.")
    if atom_names.ndim != 1:
        raise ValueError(f"Expected atom names with shape (N,), got {atom_names.shape}.")
    if coords.shape[0] != atom_names.shape[0]:
        raise ValueError(
            "Coordinates and atom names lengths must match: "
            f"{coords.shape[0]} vs {atom_names.shape[0]}."
        )

    residue_numbers = np.ones(coords.shape[0], dtype=np.int64)
    if atom_residue_index is not None:
        residue_numbers = np.asarray(atom_residue_index, dtype=np.int64).copy()
        if residue_numbers.ndim != 1 or residue_numbers.shape[0] != coords.shape[0]:
            raise ValueError(
                "atom_residue_index must have shape (N,) matching coordinates. "
                f"Got {residue_numbers.shape} for {coords.shape[0]} atoms."
            )
        min_residue = int(np.min(residue_numbers))
        if min_residue <= 0:
            residue_numbers = residue_numbers - min_residue + 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        if source_npz_path is not None:
            handle.write(f"REMARK source_npz={source_npz_path}\n")
        if frame_idx is not None:
            handle.write(f"REMARK frame_idx={frame_idx}\n")
        if label is not None:
            handle.write(f"REMARK label={label}\n")
        for atom_idx, (xyz, atom_name, residue_number) in enumerate(
            zip(coords, atom_names, residue_numbers, strict=True),
            start=1,
        ):
            atom_field = _pdb_atom_name_field(atom_name)
            element = _pdb_element_symbol(atom_name)
            x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])
            handle.write(
                f"ATOM  {atom_idx:5d} {atom_field} UNK A{int(residue_number):4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2}\n"
            )
        handle.write("END\n")


def _save_target_conditioning_reference_structures(
    output_dir,
    npz_path,
    target_atom_names,
    target_atom_residue_index,
    target_frame_coords,
    target_frame_idx,
    target_coords,
    target_coords_frame_idx,
):
    if output_dir is None:
        return

    output_dir = Path(output_dir)
    target_idx_output_path = output_dir / f"target_conditioning_target_frame_{target_frame_idx}.pdb"
    _write_coords_to_pdb(
        target_idx_output_path,
        coords=target_frame_coords,
        atom_names=target_atom_names,
        atom_residue_index=target_atom_residue_index,
        source_npz_path=npz_path,
        frame_idx=target_frame_idx,
        label="target_frame_idx_reference",
    )
    print(
        "Saved target conditioning reference structure from target_frame_idx: "
        f"path={target_idx_output_path}, frame_idx={target_frame_idx}."
    )

    random_output_path = output_dir / f"target_conditioning_random_coords_frame_{target_coords_frame_idx}.pdb"
    _write_coords_to_pdb(
        random_output_path,
        coords=target_coords,
        atom_names=target_atom_names,
        atom_residue_index=target_atom_residue_index,
        source_npz_path=npz_path,
        frame_idx=target_coords_frame_idx,
        label="random_target_atom_coords_frame",
    )
    print(
        "Saved target conditioning structure used for target_atom_coords: "
        f"path={random_output_path}, frame_idx={target_coords_frame_idx}."
    )


def _load_target_coords(npz_data, target_frame_idx, randomize_coords=False, rng=None):
    coord_key = "trajectory"
    all_target_atom_coords = np.asarray(npz_data[coord_key], dtype=np.float32)
    target_coords_frame_idx = target_frame_idx
    target_frame_coords = None
    num_coordinate_frames = 1

    if all_target_atom_coords.ndim == 3:
        num_coordinate_frames = all_target_atom_coords.shape[0]
        if target_frame_idx < 0 or target_frame_idx >= num_coordinate_frames:
            raise ValueError(
                f"target_frame_idx={target_frame_idx} is out of bounds for "
                f"coordinates with shape {all_target_atom_coords.shape}."
            )
        target_frame_coords = all_target_atom_coords[target_frame_idx]
        if randomize_coords:
            if rng is None:
                rng = np.random.default_rng()
            target_coords_frame_idx = int(rng.integers(num_coordinate_frames))
        target_atom_coords = all_target_atom_coords[target_coords_frame_idx]
    elif all_target_atom_coords.ndim == 2:
        target_atom_coords = all_target_atom_coords
        target_frame_coords = target_atom_coords
        target_coords_frame_idx = 0
    else:
        raise ValueError(
            f"Target atom coordinates must be shape (N, 3) or (T, N, 3), got {all_target_atom_coords.shape}."
        )

    if target_atom_coords.shape[-1] != 3:
        raise ValueError(
            f"Expected target atom coordinates to have last dimension 3, got shape {target_atom_coords.shape}."
        )
    return (
        target_atom_coords,
        coord_key,
        target_coords_frame_idx,
        target_frame_coords,
        num_coordinate_frames,
    )


def _load_target_dihedrals(npz_data, _target_frame_idx):
    dihedral_key_candidates = ("dihedrals", "target_dihedrals")
    dihedral_key = None
    for key in dihedral_key_candidates:
        if key in npz_data:
            dihedral_key = key
            break
    if dihedral_key is None:
        return None, None

    target_dihedrals = np.asarray(npz_data[dihedral_key], dtype=np.float32)
    if target_dihedrals.ndim not in (2, 3):
        raise ValueError(
            f"Target dihedrals must be shape (R, D) or (T, R, D), got {target_dihedrals.shape}."
        )

    if target_dihedrals.shape[-1] != 5:
        raise ValueError(
            f"Expected target dihedrals to have last dimension 5, got shape {target_dihedrals.shape}."
        )
    # (frames, residues, values of the 5 dihedrals: ['phi', 'psi', 'omega', 'chi1', 'chi2'])
    return target_dihedrals[_target_frame_idx], dihedral_key


def _load_target_dihedral_atom_indices(npz_data):
    index_key_candidates = ("dihedral_atom_indices", "target_dihedral_atom_indices")
    index_key = None
    for key in index_key_candidates:
        if key in npz_data:
            index_key = key
            break
    if index_key is None:
        return None, None

    target_dihedral_atom_indices = np.asarray(npz_data[index_key], dtype=np.int64)
    if target_dihedral_atom_indices.ndim != 3 or target_dihedral_atom_indices.shape[-1] != 4:
        raise ValueError(
            "Target dihedral atom indices must have shape (R, D, 4), "
            f"got {target_dihedral_atom_indices.shape}."
        )
    return target_dihedral_atom_indices, index_key


def _load_target_dihedral_mask(npz_data):
    mask_key_candidates = ("dihedral_mask", "target_dihedral_mask")
    mask_key = None
    for key in mask_key_candidates:
        if key in npz_data:
            mask_key = key
            break
    if mask_key is None:
        return None, None

    target_dihedral_mask = np.asarray(npz_data[mask_key]).astype(bool)
    if target_dihedral_mask.ndim != 2:
        raise ValueError(
            f"Target dihedral mask must have shape (R, D), got {target_dihedral_mask.shape}."
        )
    return target_dihedral_mask, mask_key


def _build_index_mapping(input_keys, target_keys, key_label):
    if len(input_keys) != len(target_keys):
        raise ValueError(
            f"Atom count mismatch between input protein and target NPZ: "
            f"{len(input_keys)} vs {len(target_keys)}."
        )

    input_idx_by_key = defaultdict(list)
    target_idx_by_key = defaultdict(list)
    for idx, key in enumerate(input_keys):
        input_idx_by_key[key].append(idx)
    for idx, key in enumerate(target_keys):
        target_idx_by_key[key].append(idx)

    all_keys = set(input_idx_by_key.keys()) | set(target_idx_by_key.keys())
    mismatched_counts = []
    for key in all_keys:
        n_input = len(input_idx_by_key.get(key, []))
        n_target = len(target_idx_by_key.get(key, []))
        if n_input != n_target:
            mismatched_counts.append((key, n_input, n_target))

    if mismatched_counts:
        mismatch_details = ", ".join(
            f"{key}: input={n_input}, target={n_target}"
            for key, n_input, n_target in mismatched_counts[:12]
        )
        raise ValueError(
            f"{key_label} multiplicities do not match between input and target NPZ. "
            f"Examples: {mismatch_details}."
        )

    target_index_for_input_index = np.empty(len(input_keys), dtype=np.int64)
    for key, input_indices in input_idx_by_key.items():
        target_indices = target_idx_by_key[key]
        # Deterministic 1:1 assignment by occurrence order within each key.
        for occurrence_idx, input_index in enumerate(input_indices):
            target_index_for_input_index[input_index] = target_indices[occurrence_idx]

    return target_index_for_input_index


def _build_atom_name_mapping(input_atom_names, target_atom_names):
    return _build_index_mapping(
        input_keys=input_atom_names,
        target_keys=target_atom_names,
        key_label="Atom-name",
    )


def _build_atom_residue_name_mapping(
    input_atom_names,
    target_atom_names,
    input_atom_residue_index,
    target_atom_residue_index,
):
    input_keys = list(zip(input_atom_residue_index.tolist(), input_atom_names))
    target_keys = list(zip(target_atom_residue_index.tolist(), target_atom_names))
    return _build_index_mapping(
        input_keys=input_keys,
        target_keys=target_keys,
        key_label="(residue_index, atom_name)",
    )


def load_external_conditioning_npz(
    path,
    target_frame_idx,
    randomize_coords=False,
    random_seed=None,
    output_dir=None,
):
    npz_path = Path(path)
    if not npz_path.exists():
        raise FileNotFoundError(f"Target conditioning NPZ not found: {npz_path}")

    rng = np.random.default_rng(random_seed) if randomize_coords else None
    with np.load(npz_path, allow_pickle=False) as npz_data:
        (
            target_atom_coords,
            coords_key,
            target_coords_frame_idx,
            target_frame_coords,
            num_coordinate_frames,
        ) = _load_target_coords(
            npz_data,
            target_frame_idx,
            randomize_coords=randomize_coords,
            rng=rng,
        )
        target_atom_names, names_key = _load_target_atom_names(npz_data)
        target_atom_residue_index, residue_key = _load_target_atom_residue_index(npz_data)
        target_dihedrals, dihedrals_key = _load_target_dihedrals(npz_data, target_frame_idx)
        target_dihedral_atom_indices, dihedral_atom_indices_key = _load_target_dihedral_atom_indices(npz_data)
        target_dihedral_mask, dihedral_mask_key = _load_target_dihedral_mask(npz_data)

    if target_atom_coords.shape[0] != target_atom_names.shape[0]:
        raise ValueError(
            "Target NPZ has inconsistent sizes between coordinates and atom names: "
            f"{target_atom_coords.shape[0]} vs {target_atom_names.shape[0]}."
        )
    if target_atom_residue_index is not None and target_atom_residue_index.shape[0] != target_atom_names.shape[0]:
        raise ValueError(
            "Target NPZ has inconsistent sizes between atom_names and atom_residue_index: "
            f"{target_atom_names.shape[0]} vs {target_atom_residue_index.shape[0]}."
        )
    dihedral_fields_present = [
        target_dihedrals is not None,
        target_dihedral_atom_indices is not None,
        target_dihedral_mask is not None,
    ]
    if any(dihedral_fields_present) and not all(dihedral_fields_present):
        raise ValueError(
            "Target NPZ must provide all dihedral fields together: "
            "`dihedrals`, `dihedral_atom_indices`, and `dihedral_mask`."
        )
    if all(dihedral_fields_present):
        if target_dihedrals.ndim == 2:
            dihedral_shape = target_dihedrals.shape
        elif target_dihedrals.ndim == 3:
            dihedral_shape = target_dihedrals.shape[1:]
        else:
            raise ValueError(
                f"Unexpected target dihedrals shape {target_dihedrals.shape}."
            )
        if dihedral_shape != target_dihedral_mask.shape:
            raise ValueError(
                "Target NPZ has inconsistent shapes between dihedrals and dihedral_mask: "
                f"{target_dihedrals.shape} vs {target_dihedral_mask.shape}."
            )
        if target_dihedral_atom_indices.shape[:2] != dihedral_shape:
            raise ValueError(
                "Target NPZ has inconsistent shapes between dihedrals and dihedral_atom_indices: "
                f"{target_dihedrals.shape} vs {target_dihedral_atom_indices.shape}."
            )
        target_dihedral_atom_indices = normalize_dihedral_atom_indices(
            target_dihedral_atom_indices,
            num_atoms=target_atom_coords.shape[0],
            dihedral_mask=target_dihedral_mask,
            context="Target NPZ",
        )

    _save_target_conditioning_reference_structures(
        output_dir=output_dir,
        npz_path=npz_path,
        target_atom_names=target_atom_names,
        target_atom_residue_index=target_atom_residue_index,
        target_frame_coords=target_frame_coords,
        target_frame_idx=target_frame_idx,
        target_coords=target_atom_coords,
        target_coords_frame_idx=target_coords_frame_idx,
    )

    print(
        "Loaded target conditioning NPZ: "
        f"path={npz_path}, coords_key={coords_key}, names_key={names_key}, "
        f"residue_key={residue_key}, dihedrals_key={dihedrals_key}, "
        f"dihedral_atom_indices_key={dihedral_atom_indices_key}, dihedral_mask_key={dihedral_mask_key}, "
        f"num_atoms={target_atom_coords.shape[0]}, target_frame_idx={target_frame_idx}, "
        f"target_coords_frame_idx={target_coords_frame_idx}, num_coordinate_frames={num_coordinate_frames}, "
        f"randomize_coords={randomize_coords}."
    )

    return {
        "target_atom_coords": target_atom_coords,
        "target_atom_names": target_atom_names,
        "target_atom_residue_index": target_atom_residue_index,
        "dihedrals": target_dihedrals,
        "dihedral_atom_indices": target_dihedral_atom_indices,
        "dihedral_mask": target_dihedral_mask,
        "path": str(npz_path),
        "coords_key": coords_key,
        "names_key": names_key,
        "residue_key": residue_key,
        "dihedrals_key": dihedrals_key,
        "dihedral_atom_indices_key": dihedral_atom_indices_key,
        "dihedral_mask_key": dihedral_mask_key,
        "target_frame_idx": target_frame_idx,
        "target_coords_frame_idx": target_coords_frame_idx,
        "num_coordinate_frames": num_coordinate_frames,
        "randomize_coords": randomize_coords,
    }


def attach_aligned_target_to_batch(batch, target_data, coord_scale):
    decoded_atom_names = _decode_batch_atom_names(batch)
    if len(decoded_atom_names) == 0:
        raise ValueError("Batch does not contain atoms.")

    target_atom_coords = target_data["target_atom_coords"]
    target_atom_names = target_data["target_atom_names"]
    target_atom_residue_index = target_data.get("target_atom_residue_index")
    target_dihedrals = target_data.get("dihedrals")
    target_dihedral_atom_indices = target_data.get("dihedral_atom_indices")
    target_dihedral_mask = target_data.get("dihedral_mask")
    decoded_atom_residue_indices = (
        _decode_batch_atom_residue_indices(batch)
        if target_atom_residue_index is not None
        else None
    )

    B, N, _ = batch["coords"].shape
    aligned_target = torch.zeros(
        (B, N, 3),
        dtype=batch["coords"].dtype,
        device=batch["coords"].device,
    )
    has_dihedral_data = (
        target_dihedrals is not None
        and target_dihedral_atom_indices is not None
        and target_dihedral_mask is not None
    )
    if has_dihedral_data:
        target_dihedrals_tensor = torch.as_tensor(
            target_dihedrals,
            dtype=batch["coords"].dtype,
            device=batch["coords"].device,
        )
        if target_dihedrals_tensor.ndim == 2:
            aligned_dihedrals = target_dihedrals_tensor.unsqueeze(0).expand(B, -1, -1).clone()
        elif target_dihedrals_tensor.ndim == 3:
            aligned_dihedrals = target_dihedrals_tensor.unsqueeze(0).expand(B, -1, -1, -1).clone()
        else:
            raise ValueError(
                f"Target dihedrals must be 2D or 3D, got shape {tuple(target_dihedrals_tensor.shape)}."
            )
        R, D, num_dihedral_atoms = target_dihedral_atom_indices.shape
        if num_dihedral_atoms != 4:
            raise ValueError(
                f"Target dihedral atom indices must have last dimension 4, got shape {tuple(target_dihedral_atom_indices.shape)}."
            )
        expected_dihedral_atom_index_shape = (R, D, num_dihedral_atoms)
        aligned_dihedral_atom_indices = torch.full(
            (B, R, D, num_dihedral_atoms),
            fill_value=-1,
            dtype=torch.long,
            device=batch["coords"].device,
        )
        target_dihedral_mask_tensor = torch.as_tensor(
            target_dihedral_mask,
            dtype=torch.bool,
            device=batch["coords"].device,
        )
        aligned_dihedral_mask = (
            target_dihedral_mask_tensor.unsqueeze(0).expand(B, -1, -1).clone()
        )

    # Align target atom coordinates to the input batch atoms across all batch elements.
    # For each batch index:
    #   - Extract the input atom names for the current batch element.
    #   - If residue indices are present, attempt to build a mapping from (atom name, residue index)
    #     pairs between the input and target. This is stricter and accounts for homonymous atoms
    #     in different residues.
    #   - If the mapping fails, a common reason is that the target residue indices are 1-based
    #     while the input indices are 0-based. In that case, shift the target indices and try again.
    #   - If no residue indices are available, build a mapping only by atom names.
    for b in range(B):
        input_atom_names = decoded_atom_names[b]
        if decoded_atom_residue_indices is not None:
            input_atom_residue_index = decoded_atom_residue_indices[b]
            try:
                mapping = _build_atom_residue_name_mapping(
                    input_atom_names=input_atom_names,
                    target_atom_names=target_atom_names,
                    input_atom_residue_index=input_atom_residue_index,
                    target_atom_residue_index=target_atom_residue_index,
                )
            except ValueError:
                # Common case: target residue indices can be 1-based.
                if np.min(target_atom_residue_index) < 1:
                    raise
                shifted_target_residue_index = target_atom_residue_index - 1
                mapping = _build_atom_residue_name_mapping(
                    input_atom_names=input_atom_names,
                    target_atom_names=target_atom_names,
                    input_atom_residue_index=input_atom_residue_index,
                    target_atom_residue_index=shifted_target_residue_index,
                )
        else:
            mapping = _build_atom_name_mapping(input_atom_names, target_atom_names)

        mapped_target_coords = target_atom_coords[mapping] / float(coord_scale)
        aligned_target[b, : len(input_atom_names)] = torch.as_tensor(
            mapped_target_coords,
            dtype=batch["coords"].dtype,
            device=batch["coords"].device,
        )
        if has_dihedral_data:
            remapped_dihedral_atom_indices = remap_dihedral_atom_indices_to_input(
                target_dihedral_atom_indices,
                target_index_for_input_index=mapping,
                dihedral_mask=target_dihedral_mask,
                context="Target NPZ",
            )
            if remapped_dihedral_atom_indices.shape != expected_dihedral_atom_index_shape:
                raise ValueError(
                    "Remapped dihedral atom indices have unexpected shape: "
                    f"expected {expected_dihedral_atom_index_shape}, got {remapped_dihedral_atom_indices.shape}."
                )
            aligned_dihedral_atom_indices[b] = torch.as_tensor(
                remapped_dihedral_atom_indices,
                dtype=torch.long,
                device=batch["coords"].device,
            )


    batch["target_atom_coords_aligned"] = aligned_target
    if has_dihedral_data:
        batch["dihedrals"] = aligned_dihedrals
        batch["dihedral_atom_indices"] = aligned_dihedral_atom_indices
        batch["dihedral_mask"] = aligned_dihedral_mask
    return batch


def get_config_path(relative_path):
    """Get the absolute path to a config file using importlib.resources."""
    try:
        # Remove 'configs/' prefix if present since we access configs directly as a subpackage
        config_subpath = relative_path.replace('configs/', '')

        # Access configs as a subpackage resource
        config_files = resources.files('simplefold.configs')
        config_path = config_files / config_subpath

        if config_path.is_file():
            return str(config_path)

    except Exception as e:
        pass

    # If importlib.resources fails, raise an informative error
    raise FileNotFoundError(
        f"Could not find config file: {relative_path}. "
        f"Expected to find it in the simplefold.configs package."
    )



def initialize_folding_model(args):
    # define folding model
    simplefold_model = args.simplefold_model

    # create checkpoint directory
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_path = os.path.join(ckpt_dir, f"{simplefold_model}.ckpt")

    # create folding model
    ckpt_path = os.path.join(ckpt_dir, f"{simplefold_model}.ckpt")
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_dir, exist_ok=True)
        os.system(f"curl -L {ckpt_url_dict[simplefold_model]} -o {ckpt_path}")
    cfg_path = get_config_path(f"configs/model/architecture/foldingdit_{simplefold_model[11:]}.yaml")

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # load model checkpoint
    if args.backend == 'torch':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_config = omegaconf.OmegaConf.load(cfg_path)
        model = hydra.utils.instantiate(model_config)
        model.load_state_dict(checkpoint, strict=True)
        model = model.to(device)
    elif args.backend == 'mlx':
        device = "cpu"
        # replace torch implementations with mlx
        with open(cfg_path, "r") as f:
            yaml_str = f.read()
        yaml_str = yaml_str.replace('torch', 'mlx')

        model_config = omegaconf.OmegaConf.create(yaml_str)
        model = hydra.utils.instantiate(model_config)
        mlx_state_dict = {k: mx.array(v) for k, v in starmap(map_torch_to_mlx, checkpoint.items()) if k is not None}
        model.update(tree_unflatten(list(mlx_state_dict.items())))
    print(f"Folding model {simplefold_model} loaded.")
    print(f"Using device: {device}.")

    model.eval()
    return model, device


def initialize_plddt_module(args, device):
    if not args.plddt:
        return None, None

    # load pLDDT module if specified
    plddt_ckpt_path = os.path.join(args.ckpt_dir, "plddt.ckpt")
    if not os.path.exists(plddt_ckpt_path):
        os.makedirs(args.ckpt_dir, exist_ok=True)
        os.system(f"curl -L {plddt_ckpt_url} -o {plddt_ckpt_path}")

    plddt_module_path = get_config_path("configs/model/architecture/plddt_module.yaml")
    plddt_checkpoint = torch.load(plddt_ckpt_path, map_location="cpu", weights_only=False)

    if args.backend == "torch":
        plddt_config = omegaconf.OmegaConf.load(plddt_module_path)
        plddt_out_module = hydra.utils.instantiate(plddt_config)
        plddt_out_module.load_state_dict(plddt_checkpoint, strict=True)
        plddt_out_module = plddt_out_module.to(device)
    elif args.backend == "mlx":
        # replace torch implementations with mlx
        with open(plddt_module_path, "r") as f:
            yaml_str = f.read()
        yaml_str = yaml_str.replace('torch', 'mlx')

        plddt_config = omegaconf.OmegaConf.create(yaml_str)
        plddt_out_module = hydra.utils.instantiate(plddt_config)

        mlx_state_dict = {k: mx.array(v) for k, v in starmap(map_plddt_torch_to_mlx, plddt_checkpoint.items()) if k is not None}
        plddt_out_module.update(tree_unflatten(list(mlx_state_dict.items())))

    plddt_out_module.eval()
    print(f"pLDDT output module loaded with {args.backend} backend.")

    plddt_latent_ckpt_path = os.path.join(args.ckpt_dir, "simplefold_1.6B.ckpt")
    if not os.path.exists(plddt_latent_ckpt_path):
        os.makedirs(args.ckpt_dir, exist_ok=True)
        os.system(f"curl -L {ckpt_url_dict['simplefold_1.6B']} -o {plddt_latent_ckpt_path}")

    plddt_latent_config_path = get_config_path("configs/model/architecture/foldingdit_1.6B.yaml")
    plddt_latent_checkpoint = torch.load(plddt_latent_ckpt_path, map_location="cpu", weights_only=False)

    if args.backend == "torch":
        plddt_latent_config = omegaconf.OmegaConf.load(plddt_latent_config_path)
        plddt_latent_module = hydra.utils.instantiate(plddt_latent_config)
        plddt_latent_module.load_state_dict(plddt_latent_checkpoint, strict=True)
        plddt_latent_module = plddt_latent_module.to(device)
    elif args.backend == "mlx":
        # replace torch implementations with mlx
        with open(plddt_latent_config_path, "r") as f:
            yaml_str = f.read()
        yaml_str = yaml_str.replace('torch', 'mlx')

        plddt_latent_config = omegaconf.OmegaConf.create(yaml_str)
        plddt_latent_module = hydra.utils.instantiate(plddt_latent_config)
        mlx_state_dict = {k: mx.array(v) for k, v in starmap(map_torch_to_mlx, plddt_latent_checkpoint.items()) if k is not None}
        plddt_latent_module.update(tree_unflatten(list(mlx_state_dict.items())))

    plddt_latent_module.eval()
    print(f"pLDDT latent module loaded with {args.backend} backend.")

    return plddt_latent_module, plddt_out_module


def initialize_esm_model(args, device):
    # load ESM2 model
    esm_model, esm_dict = esm_registry["esm2_3B"]()
    af2_to_esm = _af2_to_esm(esm_dict)

    if args.backend == 'torch':
        esm_model = esm_model.to(device)
        af2_to_esm = af2_to_esm.to(device)
    elif args.backend == 'mlx':
        esm_model_mlx = ESM2MLX(num_layers=36, embed_dim=2560, attention_heads=40)
        esm_state_dict_torch = esm_model.cpu().state_dict()

        esm_state_dict_torch = {k: mx.array(v) for k, v in starmap(map_torch_to_mlx, esm_state_dict_torch.items()) if k is not None}
        esm_model_mlx.update(tree_unflatten(list(esm_state_dict_torch.items())))
        esm_model = esm_model_mlx
    print(f"pLM ESM-3B loaded with {args.backend} backend.")

    esm_model.eval()
    return esm_model, esm_dict, af2_to_esm


def initialize_others(args, device):
    # prepare data tokenizer, featurizer, and processor
    tokenizer = BoltzTokenizer()
    featurizer = BoltzFeaturizer()
    processor = ProteinDataProcessor(
        device=device,
        scale=16.0,
        ref_scale=5.0,
        multiplicity=1,
        inference_multiplicity=args.nsample_per_protein,
        backend=args.backend,
    )

    # define flow process and sampler
    flow = LinearPath()

    if args.backend == "torch":
        sampler_cls = EMSampler
    elif args.backend == "mlx":
        sampler_cls = EMSamplerMLX

    sampler_kwargs = dict(
        num_timesteps=args.num_steps,
        t_start=1e-4,
        tau=args.tau,
        log_timesteps=True,
        w_cutoff=0.99,
    )
    if args.backend == "torch":
        sampler_kwargs["output_dir"] = Path(args.output_dir)

    sampler = sampler_cls(**sampler_kwargs)
    return tokenizer, featurizer, processor, flow, sampler


def generate_structure(
    args, batch, sampler, flow, processor,
    model, plddt_latent_module, plddt_out_module, device
):
    # run inference for target protein
    if args.backend == "torch":
        noise = torch.randn_like(batch['coords']).to(device)
    elif args.backend == "mlx":
        noise = mx.random.normal(batch['coords'].shape)
    out_dict = sampler.sample(model, flow, noise, batch)

    if args.plddt:
        if args.backend == "torch":
            t = torch.ones(batch['coords'].shape[0], device=device)
            # use unscaled coords to extract latent for pLDDT prediction
            out_feat = plddt_latent_module(
                out_dict["denoised_coords"].detach(), t, batch)
            plddt_out_dict = plddt_out_module(
                out_feat["latent"].detach(),
                batch,
            )
        elif args.backend == "mlx":
            t = mx.ones(batch['coords'].shape[0])
            # use unscaled coords to extract latent for pLDDT prediction
            out_feat = plddt_latent_module(
                out_dict["denoised_coords"], t, batch)
            plddt_out_dict = plddt_out_module(
                out_feat["latent"],
                batch,
            )
        # scale pLDDT to [0, 100]
        plddts = plddt_out_dict["plddt"] * 100.0
    else:
        plddts = None

    out_dict = processor.postprocess(out_dict, batch)
    # sampled_coord = out_dict['denoised_coords'].detach()
    if args.backend == "torch":
        sampled_coord = out_dict['denoised_coords'].detach()
    else:
        sampled_coord = out_dict['denoised_coords']

    pad_mask = batch['atom_pad_mask']
    return sampled_coord, pad_mask, plddts


def predict_structures_from_fastas(args):
    # create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir = output_dir / f"predictions_{args.simplefold_model}"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    cache = output_dir / "cache"
    cache.mkdir(parents=True, exist_ok=True)

    # set random seed for reproducibility
    pl.seed_everything(args.seed, workers=True)

    if args.backend == "mlx" and not MLX_AVAILABLE:
        args.backend = "torch"
        print("MLX not available, switch to torch backend.")

    if args.target_conditioning_npz is not None and args.backend != "torch":
        raise ValueError(
            "External target conditioning NPZ is currently supported only with "
            "the torch backend."
        )

    target_conditioning_data = None
    if args.target_conditioning_npz is not None:
        random_target_coords = bool(getattr(args, "random_target_coords", False))
        target_conditioning_data = load_external_conditioning_npz(
            args.target_conditioning_npz,
            target_frame_idx=args.target_frame_idx,
            randomize_coords=random_target_coords,
            random_seed=args.seed,
            output_dir=output_dir,
        )

    # initialize models
    model, device = initialize_folding_model(args)
    plddt_latent_module, plddt_out_module = initialize_plddt_module(args, device)
    esm_model, esm_dict, af2_to_esm = initialize_esm_model(args, device)

    # initialize other components
    tokenizer, featurizer, processor, flow, sampler = initialize_others(args, device)

    # just as in InferenceWrapper.process_input(self, aa_seq)
    # process fasta files to input format
    download_fasta_utilities(cache)
    data = check_fasta_inputs(Path(args.fasta_path))
    if not data:
        raise ValueError("No valid input files found. Please check the input directory.")
    process_fastas(
        data=data,
        out_dir=output_dir,
        ccd_path=cache / "ccd.pkl",
    )

    for struct_file in output_dir.glob("structures/*.npz"):
        record_file = output_dir / "records" / f"{struct_file.stem}.json"

        # prepare the target protein data for inference
        batch, structure, record = process_one_inference_structure(
            struct_file, record_file,
            tokenizer, featurizer, processor,
            esm_model, esm_dict, af2_to_esm,
        )

        if target_conditioning_data is not None:
            try:
                batch = attach_aligned_target_to_batch(
                    batch,
                    target_conditioning_data,
                    coord_scale=processor.scale,
                )
            except Exception as e:
                raise ValueError(
                    f"Failed to align target conditioning NPZ to structure `{record.id}`: {e}"
                ) from e

        sampled_coord, pad_mask, plddts = generate_structure(
            args, batch, sampler, flow, processor,
            model, plddt_latent_module, plddt_out_module, device
        )

        for i in range(args.nsample_per_protein):
            sampled_coord_i = sampled_coord[i]
            pad_mask_i = pad_mask[i]

            # save the generated structure
            structure_save = process_structure(
                deepcopy(structure), sampled_coord_i, pad_mask_i, record, backend=args.backend
            )
            outname = f"{record.id}_sampled_{i}"
            save_structure(
                structure_save, prediction_dir, outname,
                output_format=args.output_format,
                plddts=plddts[i] if plddts is not None else None
            )
