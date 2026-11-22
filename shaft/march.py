"""Deterministic CPU volumetric light-shaft integrator.

The module deliberately ships as eight unimplemented contracts.  The exercise
is to make the contracts in this file true without changing the tests.  Only
``math`` (and the standard library) is needed; no random source is allowed.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

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

_DEPTH_TOLERANCE = 1e-9
_G_EPSILON = 1e-12


def _hash01(seed: int, index: int) -> float:
    """Deterministic pseudo-random value in ``[0, 1)`` from ``seed``/``index``.

    A sine hash built on ``math`` only; the fractional part keeps the result
    inside the half-open unit interval for any sign of the intermediate value.
    """

    x = math.sin(seed * 12.9898 + (index + 1) * 78.233) * 43758.5453
    return x - math.floor(x)


def shadow_occlusion(
    depth_map: Sequence[Sequence[float]],
    uv: tuple[float, float],
    depth: float,
    bias: float,
) -> float:
    """Return shadow-map occlusion for one receiver sample.

    ``depth_map`` is a non-empty rectangular grid of stored light-space
    depths in ``[0, 1]``; rows grow with ``v`` and columns grow with ``u``.
    ``uv`` is a normalized texture coordinate.  Sampling is nearest-texel:
    ``col = floor(u * width)`` and ``row = floor(v * height)``, with the
    indices clamped to the valid edge texel.  ``depth`` is the receiver depth
    in the same units and ``bias`` is a non-negative constant depth bias.

    The receiver is occluded exactly when ``depth > stored + bias``.  A stored
    depth equal to the receiver depth is visible, even when ``bias == 0``.
    Numerical comparisons use an absolute tolerance of ``1e-9`` so that an
    equal surface never self-shadows through floating-point noise.

    Returns ``1.0`` when occluded and ``0.0`` when visible.  Fractional
    values are not part of the contract.
    """

    height = len(depth_map)
    width = len(depth_map[0])
    u, v = uv

    col = min(max(int(math.floor(u * width)), 0), width - 1)
    row = min(max(int(math.floor(v * height)), 0), height - 1)
    stored = depth_map[row][col]

    if depth > stored + bias + _DEPTH_TOLERANCE:
        return 1.0
    return 0.0


def sample_steps(
    t0: float,
    t1: float,
    n: int,
    jitter: float,
    seed: int,
) -> list[float]:
    """Return ``n`` deterministic jittered sample positions in ``[t0, t1]``.

    The interval is split into ``n`` equal cells of width
    ``step = (t1 - t0) / n``.  Cell ``i`` has midpoint
    ``t0 + (i + 0.5) * step``.  ``jitter`` must be in ``[0, 1]`` and the
    returned offset is ``(u_i - 0.5) * jitter * step``, where ``u_i`` is a
    deterministic value in ``[0, 1]`` derived from ``seed`` and ``i``.  Every
    result must therefore stay inside its own cell, including when
    ``jitter == 1``.

    The function must be pure and deterministic for identical arguments: the
    same ``seed`` and ``n`` produce the same list.  Different seeds must vary
    the sequence.  ``n == 0`` returns ``[]``.  Do not use the ``random``
    module; derive the per-cell value with ``math`` only.
    """

    if n <= 0:
        return []

    jitter = min(max(jitter, 0.0), 1.0)
    step = (t1 - t0) / n
    samples = []
    for i in range(n):
        midpoint = t0 + (i + 0.5) * step
        offset = (_hash01(seed, i) - 0.5) * jitter * step
        samples.append(midpoint + offset)
    return samples


def phase_hg(cos_theta: float, g: float) -> float:
    """Return the Henyey-Greenstein phase function.

    The normalized phase function is

    ``(1 - g**2) / (4 * pi * (1 + g**2 - 2 * g * cos_theta)**1.5)``.

    ``cos_theta`` is clamped to ``[-1, 1]``.  ``g`` is the anisotropy
    parameter and must be clamped to the open interval ``(-1, 1)`` (use a
    small epsilon such as ``1e-12``) before evaluating the formula; in
    particular ``phase_hg(0.5, 1.0)`` and ``phase_hg(0.5, -1.0)`` must return
    finite positive floats instead of dividing by zero.

    For ``g == 0`` the result is exactly the isotropic value
    ``1 / (4 * pi)`` within ``1e-9``.
    """

    cos_theta = min(max(cos_theta, -1.0), 1.0)
    g = min(max(g, -1.0 + _G_EPSILON), 1.0 - _G_EPSILON)

    g2 = g * g
    denom = 1.0 + g2 - 2.0 * g * cos_theta
    return (1.0 - g2) / (4.0 * math.pi * denom ** 1.5)


def shaft(
    depth_map: Sequence[Sequence[float]],
    ray: Mapping[str, Any],
    light: Mapping[str, Any],
    n: int,
    bias: float,
) -> float:
    """Integrate the single-scattering contribution of one ray segment.

    ``ray`` describes a segment in shadow-map space and has these keys:

    ``uv0``, ``uv1``
        Endpoint texture coordinates as ``(u, v)`` pairs.
    ``depth0``, ``depth1``
        Endpoint light-space depths in ``[0, 1]``.
    ``t0``, ``t1``
        Integration interval with ``t1 > t0``.
    ``sigma_t``
        Non-negative extinction coefficient in inverse ``t`` units.
    ``jitter``, ``seed``
        Optional sampling controls; defaults are ``0.0`` and ``0``.

    ``light`` has ``intensity >= 0``, ``cos_theta`` in ``[-1, 1]``, and the
    Henyey-Greenstein ``g`` value.

    Use :func:`sample_steps` to obtain the ``n`` positions.  For a position
    ``t`` let ``s = (t - t0) / (t1 - t0)`` and interpolate both the texture
    coordinate and depth linearly from the segment endpoints.  The visibility
    is ``1 - shadow_occlusion(depth_map, uv, depth, bias)``.  The transmittance
    from the segment start is ``exp(-sigma_t * (t - t0))``.  The returned
    value is the midpoint quadrature

    ``step * sum(visibility * transmittance * intensity * phase_hg(cos_theta, g))``

    with ``step = (t1 - t0) / n``.  Occluded samples contribute exactly zero,
    and no sample may contribute when its transmittance or visibility is zero.
    The phase function is not optional: energy must fall off with distance.
    """

    if n <= 0:
        return 0.0

    uv0 = ray["uv0"]
    uv1 = ray["uv1"]
    depth0 = float(ray["depth0"])
    depth1 = float(ray["depth1"])
    t0 = float(ray["t0"])
    t1 = float(ray["t1"])
    sigma_t = float(ray["sigma_t"])
    jitter = float(ray.get("jitter", 0.0))
    seed = int(ray.get("seed", 0))

    intensity = float(light["intensity"])
    phase = phase_hg(float(light["cos_theta"]), float(light["g"]))

    step = (t1 - t0) / n
    total = 0.0
    for t in sample_steps(t0, t1, n, jitter, seed):
        s = (t - t0) / (t1 - t0)
        uv = (uv0[0] + (uv1[0] - uv0[0]) * s, uv0[1] + (uv1[1] - uv0[1]) * s)
        depth = depth0 + (depth1 - depth0) * s
        visibility = 1.0 - shadow_occlusion(depth_map, uv, depth, bias)
        if visibility == 0.0:
            continue
        transmittance = math.exp(-sigma_t * (t - t0))
        if transmittance == 0.0:
            continue
        total += visibility * transmittance
    return step * total * intensity * phase


def dither_offsets(n: int, seed: int) -> list[float]:
    """Return one deterministic dither offset per pixel.

    The result has length ``n`` and every value is in ``[0, 1)``.  Offset
    ``i`` must be derived from both ``seed`` and ``i`` so that different
    pixels do not receive the same offset.  For ``n > 1`` the sequence must not
    be constant; otherwise the dither does not break up banding.

    Repeated calls with the same ``n`` and ``seed`` must return identical
    lists.  Different seeds must produce different sequences.  ``n == 0``
    returns ``[]``.  Do not use the ``random`` module; use ``math`` only.
    """

    if n <= 0:
        return []
    return [_hash01(seed, i) for i in range(n)]


def energy_check(
    samples: Sequence[float],
    transmittance: float | Sequence[float],
) -> float:
    """Return the normalized transmitted energy in ``[0, 1]``.

    ``samples`` are non-negative scalar sample weights (not RGB channels).
    ``transmittance`` is either one scalar in ``[0, 1]`` or a sequence of
    values in ``[0, 1]`` parallel to ``samples``.  The result is the
    transmittance weighted by the samples:

    ``sum(samples[i] * transmittance[i]) / sum(samples[i])``.

    An empty input or a zero total weight returns ``0.0``.  The result must
    be clamped to ``[0, 1]``.  Summing RGB components directly is not this
    contract; the normalization is what keeps the result from exceeding one.
    """

    if not samples:
        return 0.0

    if isinstance(transmittance, (int, float)):
        weights = [float(transmittance)] * len(samples)
    else:
        weights = list(transmittance)

    total_weight = 0.0
    weighted = 0.0
    for sample, transmit in zip(samples, weights):
        total_weight += sample
        weighted += sample * transmit

    if total_weight == 0.0:
        return 0.0
    return min(max(weighted / total_weight, 0.0), 1.0)


def validate_shaft(
    depth_map: Sequence[Sequence[float]],
    ray: Mapping[str, Any],
) -> list[str]:
    """Return a list of validation messages for one shaft ray.

    An empty list means the input is valid.  Otherwise every detected problem
    is reported, using stable lower-case prefixes:

    ``"uv"``
        ``uv0`` or ``uv1`` is malformed or outside ``[0, 1]``.
    ``"depth"``
        ``depth0`` or ``depth1`` is outside ``[0, 1]``.
    ``"depth_map"``
        The map is empty, ragged, or contains a value outside ``[0, 1]``.
    ``"nan"``
        Any relevant scalar (UV, depth, map entry, or segment bound) is not
        finite.

    In particular, sampling coordinates outside ``[0, 1]`` must never be
    accepted silently.  Non-finite input is reported as ``"nan"`` rather than
    as a range error.  The function only validates; it does not integrate.
    """

    messages: list[str] = []

    def _is_finite_number(value: Any) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)

    # --- UV endpoints -----------------------------------------------------
    for key in ("uv0", "uv1"):
        value = ray.get(key)
        if (
            not isinstance(value, (tuple, list))
            or len(value) != 2
            or not all(isinstance(c, (int, float)) for c in value)
        ):
            messages.append(f"uv: {key} is malformed")
            continue
        if not all(math.isfinite(c) for c in value):
            messages.append(f"nan: {key} is not finite")
            continue
        if not all(0.0 <= c <= 1.0 for c in value):
            messages.append(f"uv: {key} is outside [0, 1]")

    # --- Depth endpoints --------------------------------------------------
    for key in ("depth0", "depth1"):
        value = ray.get(key)
        if not isinstance(value, (int, float)):
            messages.append(f"depth: {key} is malformed")
            continue
        if not math.isfinite(value):
            messages.append(f"nan: {key} is not finite")
            continue
        if not 0.0 <= value <= 1.0:
            messages.append(f"depth: {key} is outside [0, 1]")

    # --- Segment bounds ---------------------------------------------------
    for key in ("t0", "t1"):
        value = ray.get(key)
        if not _is_finite_number(value):
            messages.append(f"nan: {key} is not finite")

    # --- Depth map --------------------------------------------------------
    if not depth_map or not all(len(row) > 0 for row in depth_map):
        messages.append("depth_map: map is empty")
    else:
        width = len(depth_map[0])
        if any(len(row) != width for row in depth_map):
            messages.append("depth_map: map is ragged")
        saw_nan = False
        saw_range = False
        for row in depth_map:
            for entry in row:
                if not isinstance(entry, (int, float)) or not math.isfinite(entry):
                    saw_nan = True
                elif not 0.0 <= entry <= 1.0:
                    saw_range = True
        if saw_nan:
            messages.append("nan: depth_map contains a non-finite value")
        if saw_range:
            messages.append("depth_map: value outside [0, 1]")

    return messages


def shaft_batch(
    rays: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Integrate a batch of rays independently.

    ``config`` contains ``depth_map``, ``light``, ``n``, and ``bias``.  The
    ``light`` value is passed through to :func:`shaft` unchanged.  The return value is

    ``{"values": [float, ...], "count": len(rays)}``

    with one value per input ray in the same order.  Each ray must be
    integrated with its own ``t0``, ``t1``, ``uv`` endpoints, depth endpoints,
    extinction, jitter, and seed.  Sampling positions, transmittance, and
    visibility must never be reused between rays; the result for a ray must
    equal calling :func:`shaft` for that ray alone.  An empty ``rays``
    sequence returns ``{"values": [], "count": 0}``.
    """

    depth_map = config["depth_map"]
    light = config["light"]
    n = config["n"]
    bias = config["bias"]

    values = [shaft(depth_map, ray, light, n, bias) for ray in rays]
    return {"values": values, "count": len(rays)}
