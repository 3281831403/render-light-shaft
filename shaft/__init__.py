"""Deterministic volumetric light-shaft integrator (exercise package)."""

from .march import (
    dither_offsets,
    energy_check,
    phase_hg,
    sample_steps,
    shadow_occlusion,
    shaft,
    shaft_batch,
    validate_shaft,
)

__all__ = [
    "dither_offsets",
    "energy_check",
    "phase_hg",
    "sample_steps",
    "shadow_occlusion",
    "shaft",
    "shaft_batch",
    "validate_shaft",
]
