#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#
"""Compatibility package for legacy ``boltz_data_pipeline`` imports."""

from importlib import import_module

_real_package = import_module("simplefold.boltz_data_pipeline")

# Let imports such as ``boltz_data_pipeline.types`` resolve to the package
# vendored under ``simplefold`` while preserving the historical top-level name
# used by pickled tokenized structures.
__path__ = list(_real_package.__path__)
