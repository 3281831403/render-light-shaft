"""Behavioral tests for the volumetric light-shaft exercise.

The file is the grading contract and must not be modified.  There are exactly
15 tests: ten are aimed at the ten hidden defects, and five are invariants
(occlusion, analytic energy, step convergence, energy cap, batch independence).
"""

from __future__ import annotations

import math

import pytest

from shaft.march import (
    dither_offsets,
    energy_check,
    phase_hg,
    sample_steps,
    shadow_occlusion,
    shaft,
    shaft_batch,
    validate_shaft,
)

ONE_OVER_FOUR_PI = 1.0 / (4.0 * math.pi)


def _flat_depth(height: int, width: int, value: float) -> list[list[float]]:
    return [[float(value) for _ in range(width)] for _ in range(height)]


def _ray(**overrides: object) -> dict[str, object]:
    ray: dict[str, object] = {
        "uv0": (0.1, 0.1),
        "uv1": (0.9, 0.9),
        "depth0": 0.1,
        "depth1": 0.1,
        "t0": 0.0,
        "t1": 1.0,
        "sigma_t": 0.0,
        "jitter": 0.0,
        "seed": 0,
    }
    ray.update(overrides)
    return ray


def _light(**overrides: object) -> dict[str, object]:
    light: dict[str, object] = {
        "intensity": 1.0,
        "cos_theta": 0.0,
        "g": 0.0,
    }
    light.update(overrides)
    return light


# ---------------------------------------------------------------------------
# One test per hidden defect.
# ---------------------------------------------------------------------------


def test_shadow_occlusion_equal_depth_is_visible() -> None:
    """Defect 1: an equal-depth surface must not self-shadow."""
    depth_map = _flat_depth(8, 8, 0.25)

    visible = shadow_occlusion(depth_map, (0.5, 0.5), 0.25, 0.0)
    occluded = shadow_occlusion(depth_map, (0.5, 0.5), 0.2505, 0.0)

    assert abs(visible - 0.0) <= 1e-9
    assert abs(occluded - 1.0) <= 1e-9


def test_shadow_occlusion_applies_bias_on_sloped_surface() -> None:
    """Defect 2: a sloped receiver needs a depth bias to avoid acne."""
    # Column c stores depth 0.20 + 0.02*c, so column 5 stores 0.30.
    depth_map = [
        [0.20 + 0.02 * col for col in range(10)]
        for _ in range(10)
    ]
    uv = (0.55, 0.5)
    receiver_depth = 0.3025

    with_bias = shadow_occlusion(depth_map, uv, receiver_depth, 0.003)
    without_bias = shadow_occlusion(depth_map, uv, receiver_depth, 0.0)

    assert abs(with_bias - 0.0) <= 1e-9
    assert abs(without_bias - 1.0) <= 1e-9


def test_sample_steps_jitter_stays_inside_each_step() -> None:
    """Defect 3: full-strength jitter must not leave the current cell."""
    t0, t1, n = 2.0, 3.0, 16
    samples = sample_steps(t0, t1, n, 1.0, 12345)
    step = (t1 - t0) / n

    assert len(samples) == n
    for i, t in enumerate(samples):
        lo = t0 + i * step
        hi = lo + step
        assert lo - 1e-12 <= t <= hi + 1e-12, (i, t, lo, hi)
    assert all(t0 - 1e-12 <= t <= t1 + 1e-12 for t in samples)


def test_shaft_attenuates_with_distance() -> None:
    """Defect 4: ignoring transmittance makes distant samples too bright."""
    depth_map = _flat_depth(8, 8, 1.0)
    ray = _ray(
        uv0=(0.5, 0.5),
        uv1=(0.5, 0.5),
        depth0=0.2,
        depth1=0.2,
        sigma_t=2.0,
        jitter=0.0,
        seed=0,
    )
    light = _light(intensity=1.0, cos_theta=0.32, g=0.0)

    value = shaft(depth_map, ray, light, 2048, 0.0)
    expected = ONE_OVER_FOUR_PI * (1.0 - math.exp(-2.0)) / 2.0

    assert value == pytest.approx(expected, abs=1e-6)
    # A no-transmittance implementation would return 1/(4*pi).
    assert value < 0.75 * ONE_OVER_FOUR_PI


def test_shaft_ignores_occluded_samples() -> None:
    """Defect 5: occluded samples must not add scattered light."""
    # Columns 0..49 are open sky, columns 50..99 are a wall.
    depth_map = [[1.0] * 50 + [0.0] * 50 for _ in range(8)]
    ray = _ray(
        uv0=(0.0, 0.5),
        uv1=(1.0, 0.5),
        depth0=0.5,
        depth1=0.5,
        sigma_t=0.5,
        jitter=0.0,
        seed=0,
    )
    light = _light(intensity=1.0, cos_theta=0.0, g=0.0)

    value = shaft(depth_map, ray, light, 1000, 0.0)
    expected = ONE_OVER_FOUR_PI * (1.0 - math.exp(-0.25)) / 0.5
    fully_visible = ONE_OVER_FOUR_PI * (1.0 - math.exp(-0.5)) / 0.5

    assert value == pytest.approx(expected, abs=1e-5)
    assert value < fully_visible - 1e-4


def test_phase_hg_clamps_g_at_one() -> None:
    """Defect 6: g == +/-1 must be clamped instead of dividing by zero."""
    p_plus = phase_hg(0.5, 1.0)
    p_minus = phase_hg(0.5, -1.0)

    assert math.isfinite(p_plus)
    assert p_plus > 0.0
    assert math.isfinite(p_minus)
    assert p_minus > 0.0
    assert phase_hg(0.0, 0.0) == pytest.approx(ONE_OVER_FOUR_PI, abs=1e-12)


def test_dither_offsets_vary_per_pixel() -> None:
    """Defect 7: using one seed for every pixel leaves banding intact."""
    offsets = dither_offsets(64, 2026)
    repeated = dither_offsets(64, 2026)
    other_seed = dither_offsets(64, 2027)

    assert len(offsets) == 64
    assert all(0.0 <= x < 1.0 for x in offsets)
    assert len({round(x, 12) for x in offsets}) > 1
    assert all(abs(a - b) <= 1e-12 for a, b in zip(offsets, repeated))
    assert any(abs(a - b) > 1e-9 for a, b in zip(offsets, other_seed))


def test_energy_check_normalizes_transmittance() -> None:
    """Defect 8: RGB-like sums are not normalized transmittance."""
    value = energy_check([0.2, 0.3, 0.5], [0.1, 0.5, 0.9])

    assert value == pytest.approx(0.62, abs=1e-12)
    assert 0.0 <= value <= 1.0


def test_shaft_batch_does_not_share_samples() -> None:
    """Defect 9: each ray owns its sample positions and seeds."""
    depth_map = _flat_depth(16, 16, 1.0)
    light = _light(intensity=1.3, cos_theta=0.7, g=0.2)
    rays = [
        _ray(
            uv0=(0.05, 0.1),
            uv1=(0.25, 0.2),
            depth0=0.1,
            depth1=0.1,
            t0=0.0,
            t1=1.0,
            sigma_t=0.2,
            jitter=0.75,
            seed=11,
        ),
        _ray(
            uv0=(0.8, 0.9),
            uv1=(0.6, 0.7),
            depth0=0.3,
            depth1=0.3,
            t0=2.0,
            t1=4.0,
            sigma_t=1.4,
            jitter=0.75,
            seed=99,
        ),
    ]
    config = {
        "depth_map": depth_map,
        "light": light,
        "n": 257,
        "bias": 0.0,
    }

    batch = shaft_batch(rays, config)

    assert batch["count"] == 2
    assert len(batch["values"]) == 2
    for got, ray in zip(batch["values"], rays):
        expected = shaft(depth_map, ray, light, 257, 0.0)
        assert got == pytest.approx(expected, abs=1e-9)


def test_validate_shaft_rejects_out_of_range_uv() -> None:
    """Defect 10: invalid UVs, depths, and NaNs must be reported."""
    depth_map = _flat_depth(8, 8, 0.5)
    valid_ray = _ray(
        uv0=(0.0, 0.0),
        uv1=(1.0, 1.0),
        depth0=0.2,
        depth1=0.4,
    )

    assert validate_shaft(depth_map, valid_ray) == []

    uv_errors = validate_shaft(depth_map, _ray(uv1=(1.25, 0.5)))
    assert any(message.startswith("uv") for message in uv_errors)

    depth_errors = validate_shaft(depth_map, _ray(depth0=-0.1))
    assert any(message.startswith("depth") for message in depth_errors)

    nan_errors = validate_shaft(depth_map, _ray(depth1=math.nan))
    assert any(message.startswith("nan") for message in nan_errors)


# ---------------------------------------------------------------------------
# Five invariants independent of any one defect.
# ---------------------------------------------------------------------------


def test_fully_occluded_shaft_is_zero() -> None:
    """A ray behind an opaque wall contributes no light."""
    depth_map = _flat_depth(16, 16, 0.0)
    ray = _ray(
        uv0=(0.2, 0.2),
        uv1=(0.8, 0.8),
        depth0=0.9,
        depth1=0.9,
        t0=0.0,
        t1=2.0,
        sigma_t=0.3,
        jitter=0.8,
        seed=4,
    )
    light = _light(intensity=3.0, cos_theta=0.1, g=0.4)

    value = shaft(depth_map, ray, light, 257, 0.0)

    assert abs(value) <= 1e-9


def test_fully_visible_shaft_matches_analytic_integral() -> None:
    """With full visibility the midpoint sum matches the exponential integral."""
    depth_map = _flat_depth(8, 8, 1.0)
    ray = _ray(
        uv0=(0.4, 0.4),
        uv1=(0.6, 0.6),
        depth0=0.3,
        depth1=0.3,
        t0=0.0,
        t1=1.0,
        sigma_t=0.8,
        jitter=0.0,
        seed=0,
    )
    light = _light(intensity=2.0, cos_theta=0.2, g=0.0)

    value = shaft(depth_map, ray, light, 4096, 0.0)
    expected = 2.0 * ONE_OVER_FOUR_PI * (1.0 - math.exp(-0.8)) / 0.8

    assert value == pytest.approx(expected, abs=1e-6)


def test_shaft_converges_when_steps_double() -> None:
    """Doubling the step count changes the result by less than 1e-3."""
    depth_map = _flat_depth(8, 8, 1.0)
    ray = _ray(
        uv0=(0.3, 0.3),
        uv1=(0.7, 0.7),
        depth0=0.2,
        depth1=0.2,
        t0=0.0,
        t1=1.5,
        sigma_t=0.45,
        jitter=0.0,
        seed=0,
    )
    light = _light(intensity=1.3, cos_theta=0.6, g=0.1)

    coarse = shaft(depth_map, ray, light, 64, 0.0)
    fine = shaft(depth_map, ray, light, 128, 0.0)

    assert abs(coarse - fine) < 1e-3


def test_energy_never_exceeds_one() -> None:
    """Normalized transmitted energy is always capped at one."""
    cases = [
        ([0.1, 0.2, 0.3], [0.2, 0.5, 0.9]),
        ([1.0, 2.0, 3.0], [1.0, 1.0, 1.0]),
        ([0.5, 0.5], [0.0, 0.0]),
        ([], []),
    ]

    for samples, transmittance in cases:
        value = energy_check(samples, transmittance)
        assert 0.0 <= value <= 1.0

    assert energy_check([0.1, 0.2, 0.3], [1.0, 1.0, 1.0]) == pytest.approx(1.0)
    assert energy_check([0.5, 0.5], [0.0, 0.0]) == pytest.approx(0.0)


def test_batch_matches_individual_rays() -> None:
    """Batch integration must equal two independent single-ray integrations."""
    depth_map = _flat_depth(8, 8, 1.0)
    light = _light(intensity=0.9, cos_theta=0.35, g=0.15)
    rays = [
        _ray(
            uv0=(0.2, 0.2),
            uv1=(0.4, 0.4),
            depth0=0.15,
            depth1=0.15,
            t0=0.0,
            t1=1.0,
            sigma_t=0.6,
            jitter=0.0,
            seed=3,
        ),
        _ray(
            uv0=(0.7, 0.7),
            uv1=(0.5, 0.5),
            depth0=0.35,
            depth1=0.35,
            t0=1.0,
            t1=2.5,
            sigma_t=1.1,
            jitter=0.4,
            seed=8,
        ),
    ]
    config = {
        "depth_map": depth_map,
        "light": light,
        "n": 96,
        "bias": 1e-4,
    }

    batch = shaft_batch(rays, config)
    individual = [shaft(depth_map, ray, light, 96, 1e-4) for ray in rays]

    assert batch["values"] == pytest.approx(individual, abs=1e-9)
