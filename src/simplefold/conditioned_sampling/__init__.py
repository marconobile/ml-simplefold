"""Cluster-conditioned SimpleFold sampling helpers.

The modules in this package split the former monolithic
``scripts/sample_with_conditioning.py`` workflow by activity:

- ``cli``: command-line argument parsing
- ``data``: raw/processed frame and label loading
- ``conditioning``: batch construction and cluster-label injection
- ``modeling``: device, checkpoint, and model loading
- ``sampling``: conditioned structure generation and per-sample orchestration
- ``evaluation``: coordinate and dihedral evaluation workflows
- ``geometry``, ``pdb_conversion``, ``dihedrals``: evaluation primitives
- ``outputs``: reports, CSV files, histograms, and NPZ artifacts
- ``workflow``: end-to-end command orchestration
"""
