#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

# Started from https://github.com/jwohlwend/boltz,
# licensed under MIT License, Copyright (c) 2024 Jeremy Wohlwend, Gabriele Corso, Saro Passaro.

import argparse
import json
import multiprocessing
import os
import pickle
import traceback
from dataclasses import asdict, dataclass, replace
from functools import partial
from itertools import chain
from pathlib import Path
from typing import Any, Optional

import numpy as np
import rdkit
from redis import Redis
from tqdm import tqdm
from p_tqdm import p_umap

from boltz_data_pipeline.filter.static.filter import StaticFilter
from boltz_data_pipeline.filter.static.ligand import ExcludedLigands
from boltz_data_pipeline.filter.static.polymer import (
    ClashingChainsFilter,
    ConsecutiveCA,
    MinimumLengthFilter,
    UnknownFilter,
)
from boltz_data_pipeline.types import ChainInfo, Record, Target
from utils.mmcif_utils import parse_pdb


"""
run:
    wget https://boltz1.s3.us-east-2.amazonaws.com/ccd.rdb
    redis-server --dbfilename ccd.rdb --port 7777
before running the script to start the redis server with the CCD data.
"""


@dataclass(frozen=True, slots=True)
class PDBFile:
    """A raw PDB structure file."""

    id: str
    path: str


class Resource:
    """A shared resource for processing."""

    def __init__(self, host: str, port: int) -> None:
        """Initialize the redis database."""
        self._host = host
        self._port = port
        self._redis: Optional[Redis] = None
        self._pid: Optional[int] = None

    def _get_redis(self) -> Redis:
        """Get a redis client that is safe to use in the current process."""
        pid = os.getpid()
        if self._redis is None or self._pid != pid:
            # Avoid sharing connection pools across processes.
            self._redis = Redis(host=self._host, port=self._port)
            self._pid = pid
        return self._redis

    def __getstate__(self) -> dict[str, Any]:
        """Avoid pickling redis connection pools across processes."""
        return {
            "_host": self._host,
            "_port": self._port,
            "_redis": None,
            "_pid": None,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self._host = state["_host"]
        self._port = state["_port"]
        self._redis = None
        self._pid = None

    def get(self, key: str) -> Any:
        """Get an item from the Redis database."""
        value = self._get_redis().get(key)
        if value is not None:
            value = pickle.loads(value)
        return value

    def __getitem__(self, key: str) -> Any:
        """Get an item from the resource."""
        out = self.get(key)
        if out is None:
            raise KeyError(key)
        return out


def fetch(data_dir: Path, max_file_size: Optional[int] = None) -> list[PDBFile]:
    """Fetch the PDB files."""
    data: list[PDBFile] = []
    excluded = 0
    for file in chain(data_dir.rglob("*.pdb*"), data_dir.rglob("*.ent*")):
        # Derive a stable ID from the filename (handles `.pdb` and `.pdb.gz`).
        record_id = file.name.lower()
        for compression_suffix in (".gz", ".bz2", ".xz", ".zip"):
            if record_id.endswith(compression_suffix):
                record_id = record_id[: -len(compression_suffix)]
                break
        for pdb_suffix in (".pdb", ".ent"):
            if record_id.endswith(pdb_suffix):
                record_id = record_id[: -len(pdb_suffix)]
                break

        # Check file size and skip if too large
        if max_file_size is not None and (file.stat().st_size > max_file_size):
            excluded += 1
            continue

        data.append(PDBFile(id=record_id, path=str(file)))

    if not data:
        print(
            f"No PDB files found under {data_dir} (expected files matching '*.pdb*' or '*.ent*'). "
            "Did you place your PDB(s) there and point --data_dir to the right folder?"
        )
    print(f"Excluded {excluded} files due to size.")
    return data


def finalize(out_dir: Path) -> None:
    """Run post-processing in main thread."""
    # Group records into a manifest
    records_dir = out_dir / "records"

    failed_count = 0
    records = []
    for record in records_dir.iterdir():
        path = Path(record)
        try:
            with path.open("r") as f:
                records.append(json.load(f))
        except Exception:
            failed_count += 1
            print(f"Failed to parse {record}")
    print(f"Failed to parse {failed_count} entries")

    # Save manifest
    outpath = out_dir / "manifest.json"
    with outpath.open("w") as f:
        json.dump(records, f)


def parse(data: PDBFile, resource: Resource, clusters: dict, use_assembly: bool) -> Target:
    """Process a structure."""
    pdb_id = data.id.lower()

    parsed = parse_pdb(data.path, resource, use_assembly=use_assembly)
    structure = parsed.data
    structure_info = parsed.info

    chain_info = []
    for i, chain in enumerate(structure.chains):
        key = f"{pdb_id}_{chain['entity_id']}"
        chain_info.append(
            ChainInfo(
                chain_id=i,
                chain_name=chain["name"],
                msa_id="",
                mol_type=int(chain["mol_type"]),
                cluster_id=clusters.get(key, -1),
                num_residues=int(chain["res_num"]),
            )
        )

    record = Record(
        id=data.id,
        structure=structure_info,
        chains=chain_info,
        interfaces=[],
    )

    return Target(structure=structure, record=record)


def process_structure(
    data: PDBFile,
    resource: Resource,
    out_dir: Path,
    filters: list[StaticFilter],
    clusters: dict,
    use_assembly: bool,
) -> None:
    """Process a target."""
    struct_path = out_dir / "structures" / f"{data.id}.npz"
    record_path = out_dir / "records" / f"{data.id}.json"

    if struct_path.exists() and record_path.exists():
        return

    try:
        target: Target = parse(data, resource, clusters, use_assembly=use_assembly)
        structure = target.structure

        mask = structure.mask
        if filters is not None:
            for f in filters:
                filter_mask = f.filter(structure)
                mask = mask & filter_mask
    except Exception:
        traceback.print_exc()
        print(f"Failed to parse {data.id}")
        return

    # Replace chains and interfaces
    chains_out = []
    for i, chain in enumerate(target.record.chains):
        chains_out.append(replace(chain, valid=bool(mask[i])))

    # Replace structure and record
    structure = replace(structure, mask=mask)
    record = replace(target.record, chains=chains_out, interfaces=[])
    target = replace(target, structure=structure, record=record)

    # Dump structure
    np.savez_compressed(struct_path, **asdict(structure))

    # Dump record
    with record_path.open("w") as f:
        json.dump(asdict(record), f)


def process(args) -> None:
    """Run the data processing task."""
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Create output directories
    records_dir = args.out_dir / "records"
    records_dir.mkdir(parents=True, exist_ok=True)

    structure_dir = args.out_dir / "structures"
    structure_dir.mkdir(parents=True, exist_ok=True)

    # Load filters
    filters = [
        ExcludedLigands(),
        MinimumLengthFilter(min_len=4, max_len=5000),
        UnknownFilter(),
        ConsecutiveCA(max_dist=10.0),
        ClashingChainsFilter(freq=0.3, dist=1.7),
    ]

    # Set default pickle properties
    pickle_option = rdkit.Chem.PropertyPickleOptions.AllProps
    rdkit.Chem.SetDefaultPickleProperties(pickle_option)

    # Load shared data from redis
    resource = Resource(host=args.redis_host, port=args.redis_port)

    # Get data points
    print("Fetching data...")
    data = fetch(args.data_dir, max_file_size=args.max_file_size)
    if not data:
        raise SystemExit(1)

    # Check if we can run in parallel
    max_processes = multiprocessing.cpu_count()
    num_processes = max(1, min(args.num_processes, max_processes, len(data)))
    parallel = num_processes > 1

    # Run processing
    print("Processing data...")
    if parallel:
        fn = partial(
            process_structure,
            resource=resource,
            out_dir=args.out_dir,
            clusters={},
            filters=filters,
            use_assembly=args.use_assembly,
        )
        p_umap(fn, data, num_cpus=num_processes)
    else:
        for item in tqdm(data):
            process_structure(
                item,
                resource=resource,
                out_dir=args.out_dir,
                clusters={},
                filters=filters,
                use_assembly=args.use_assembly,
            )

    # Finalize
    finalize(args.out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process PDB data.")
    parser.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="The directory containing the PDB files.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default="data",
        help="The output directory.",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=multiprocessing.cpu_count(),
        help="The number of processes.",
    )
    parser.add_argument(
        "--redis-host",
        type=str,
        default="localhost",
        help="The Redis host.",
    )
    parser.add_argument(
        "--redis-port",
        type=int,
        default=7777,
        help="The Redis port.",
    )
    parser.add_argument(
        "--use-assembly",
        action="store_true",
        help="Whether to use assembly 1.",
    )
    parser.add_argument(
        "--max-file-size",
        type=int,
        default=None,
    )
    args = parser.parse_args()
    process(args)
