#!/usr/bin/env python3
"""Evaluate cluster-conditioned SimpleFold samples.

The implementation lives in ``simplefold.conditioned_sampling`` so the
previous monolithic workflow can be debugged one activity at a time. This file
remains as the stable command-line entry point:

    python scripts/sample_with_conditioning.py [options]

Use ``-N -1`` with a raw NPZ input to sample every trajectory frame in the
order stored in that file.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SIMPLEFOLD_ROOT = SRC_ROOT / "simplefold"
for import_path in (SRC_ROOT, SIMPLEFOLD_ROOT):
    import_path_str = str(import_path)
    if import_path_str not in sys.path:
        sys.path.insert(0, import_path_str)

from simplefold.conditioned_sampling.cli import parse_args  # noqa: E402,F401
from simplefold.conditioned_sampling.conditioning import (  # noqa: E402,F401
    cluster_labels_to_tensor,
    exact_cluster_labels,
    pad_cluster_labels,
    prepare_conditioned_batch,
    set_batch_cluster_labels,
    validate_cluster_labels_for_model,
)
from simplefold.conditioned_sampling.constants import (  # noqa: E402,F401
    CLUSTER_KEY,
    CONDITIONED_EVAL_SAMPLED_CIF_TOKEN,
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_CONDITIONED_EVAL_PDB_BASE_PATH,
    DEFAULT_DATA_PATH,
    DEFAULT_RAW_NPZ_CANDIDATES,
    PDB_ATOM_RECORDS,
    REPO_ROOT,
    SIMPLEFOLD_ROOT,
)
from simplefold.conditioned_sampling.data import (  # noqa: E402,F401
    build_structure_and_record_from_raw,
    choose_frame_position,
    find_processed_paths,
    load_conditioning_label_rows,
    load_processed_frame,
    load_raw_frame,
    match_processed_record_id,
    np_scalar_to_string,
    resolve_data_path,
    resolve_labels_npz_path,
    resolve_processed_dir,
    resolve_raw_npz_path,
)
from simplefold.conditioned_sampling.dihedrals import (  # noqa: E402,F401
    add_dihedral_validation_summary,
    circular_difference,
    compute_dihedral_angles,
    summarize_dihedrals,
)
from simplefold.conditioned_sampling.evaluation import (  # noqa: E402,F401
    evaluate_coordinate_alignment,
    evaluate_sample_dihedrals,
)
from simplefold.conditioned_sampling.geometry import kabsch_align  # noqa: E402,F401
from simplefold.conditioned_sampling.modeling import (  # noqa: E402,F401
    instantiate_and_load_model,
    load_checkpoint,
    resolve_checkpoint_path,
    resolve_device,
    strip_model_prefix,
)
from simplefold.conditioned_sampling.outputs import (  # noqa: E402,F401
    format_optional_float,
    output_stem_for_sample,
    write_atomwise_csv,
    write_dihedral_csv,
    write_dihedral_histograms,
    write_report,
)
from simplefold.conditioned_sampling.pdb_conversion import (  # noqa: E402,F401
    conditioned_eval_sampled_pdb_path,
    ensure_conditioned_eval_sampled_pdb,
    load_sampled_pdb_dihedral_coords,
    read_pdb_atom_coordinates,
    resolve_conditioned_eval_converter_base_path,
    run_conditioned_eval_cif_to_pdb_converter,
    validate_sampled_pdb_matches_coords,
)
from simplefold.conditioned_sampling.sampling import (  # noqa: E402,F401
    prepare_sampling_conditioning,
    run_model_sampler,
    sample_conditioned_structure,
)
from simplefold.conditioned_sampling.workflow import main  # noqa: E402


if __name__ == "__main__":
    main()
