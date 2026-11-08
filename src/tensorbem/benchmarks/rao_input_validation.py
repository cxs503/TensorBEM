"""Probe the RAO entry point with non-physical inputs: rejected vs raw algebra.

The impedance assembly ``Z xi = F`` is a linear solve, so every finite input
produces an output.  This driver runs the same list of malformed inputs twice,
on a coarse hemisphere:

* through ``compute_rao_at_frequency`` (the validating entry point), which must
  raise ``ValueError`` for each of them;
* through ``_assemble_rao_at_frequency`` (the raw algebra), which shows *what
  the code used to return* — a finite, plausible-looking, wrong RAO vector.

Keeping the raw assembly callable makes the "before" column of
``docs/rao_input_validation.md`` reproducible after the fix, without checking
out an old revision.  A well-posed case is run through both to show the
validating path is bit-identical to the raw one.

Usage::

    PYTHONPATH=src python3 -m tensorbem.benchmarks.rao_input_validation \
        --json docs/rao_input_validation/scan.json
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
from typing import Any

import numpy as np

from ..geometry3d import hemisphere_hull_mesh
from ..rao import (
    _assemble_rao_at_frequency,
    compute_rao_at_frequency,
    compute_rao_spectrum,
)

OMEGA = 1.0
MASS = 1000.0
CG = np.array([0.0, 0.0, -0.1])
DENSITY = 1025.0
GRAVITY = 9.80665


def _raw(hull, **overrides: Any):
    kwargs = dict(
        hull=hull,
        omega=OMEGA,
        wave_angle=0.0,
        mass=MASS,
        center_of_gravity=CG.copy(),
        hydrostatic_props=None,
        amplitude=1.0,
        density=DENSITY,
        gravity=GRAVITY,
        inertia=None,
        extra_damping=None,
        chief_points="auto",
    )
    kwargs.update(overrides)
    return _assemble_rao_at_frequency(**kwargs)


def _public(hull, **overrides: Any):
    kwargs = dict(
        hull=hull,
        omega=OMEGA,
        wave_angle=0.0,
        mass=MASS,
        center_of_gravity=CG.copy(),
    )
    kwargs.update(overrides)
    return compute_rao_at_frequency(**kwargs)


CASES: list[tuple[str, dict[str, Any], str]] = [
    ("baseline (valid)", {}, "positive control"),
    ("mass = -1000 kg", {"mass": -1000.0}, "negative body mass"),
    ("mass = 0 kg", {"mass": 0.0}, "massless body"),
    ("mass = NaN", {"mass": float("nan")}, "NaN poisoning"),
    ("omega = -1 rad/s", {"omega": -1.0}, "omega**2 hides the sign"),
    ("omega = NaN", {"omega": float("nan")}, "NaN poisoning"),
    ("cg = (0, 0, NaN)", {"center_of_gravity": np.array([0.0, 0.0, float("nan")])}, "NaN poisoning"),
    ("cg shape (2,)", {"center_of_gravity": np.array([0.0, 0.0])}, "bare IndexError"),
    ("inertia 2x2", {"inertia": np.eye(2)}, "uncontrolled broadcast error"),
    (
        "inertia asymmetric",
        {"inertia": np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])},
        "asymmetric J accepted",
    ),
    ("inertia NaN", {"inertia": np.full((3, 3), np.nan)}, "NaN poisoning"),
    ("extra_damping shape (6,)", {"extra_damping": np.zeros(6)}, "vector broadcast over matrix"),
    ("extra_damping NaN", {"extra_damping": np.full((6, 6), np.nan)}, "NaN poisoning"),
    ("extra_damping = -1e6 I", {"extra_damping": -1.0e6 * np.eye(6)}, "negative damping = energy source"),
]


def _summarise(result: Any) -> dict[str, Any]:
    amplitude = np.asarray(result.rao_amplitude, dtype=float)
    return {
        "returned": True,
        "heave_amplitude": None if not np.all(np.isfinite(amplitude)) else round(float(amplitude[2]), 6),
        "max_amplitude": None if not np.all(np.isfinite(amplitude)) else round(float(np.max(amplitude)), 6),
        "non_finite_entries": int(np.sum(~np.isfinite(amplitude))),
        "impedance_residual": None
        if not math.isfinite(float(result.impedance_residual))
        else float(result.impedance_residual),
    }


def _attempt(fn, **kwargs) -> dict[str, Any]:
    try:
        return _summarise(fn(**kwargs))
    except ValueError as exc:
        return {"returned": False, "error_type": "ValueError", "error": str(exc)}
    except Exception as exc:  # noqa: BLE001 - we are documenting the failure modes
        return {"returned": False, "error_type": type(exc).__name__, "error": str(exc)}


def single_frequency_scan(hull) -> list[dict[str, Any]]:
    rows = []
    for name, overrides, note in CASES:
        rows.append(
            {
                "case": name,
                "note": note,
                "public_api": _attempt(_public, hull=hull, **overrides),
                "raw_assembly": _attempt(_raw, hull=hull, **overrides),
            }
        )
    return rows


def sweep_scan(hull) -> list[dict[str, Any]]:
    rows = []
    for name, omegas in [
        ("omegas = [-1, -2]", [-1.0, -2.0]),
        ("omegas contains 0", [1.0, 0.0]),
        ("omegas contains NaN", [1.0, float("nan")]),
        ("omegas shape (1, 2)", [[1.0, 2.0]]),
        ("omegas = [0.6, 1.2] (valid)", [0.6, 1.2]),
    ]:
        try:
            spectrum = compute_rao_spectrum(
                hull=hull,
                omegas=omegas,
                wave_angle=0.0,
                mass=MASS,
                center_of_gravity=CG.copy(),
            )
            rows.append(
                {
                    "case": name,
                    "returned": True,
                    "heave_amplitudes": [round(float(v), 6) for v in spectrum.rao_amplitudes[:, 2]],
                }
            )
        except ValueError as exc:
            rows.append({"case": name, "returned": False, "error": str(exc)})
    return rows


def run() -> dict[str, Any]:
    hull = hemisphere_hull_mesh(subdivisions=1)
    return {
        "mesh": {"family": "hemisphere", "subdivisions": 1, "panels": int(hull.centroids.shape[0])},
        "state": {
            "omega": OMEGA,
            "mass": MASS,
            "center_of_gravity": CG.tolist(),
            "density": DENSITY,
            "gravity": GRAVITY,
        },
        "single_frequency": single_frequency_scan(hull),
        "frequency_sweep": sweep_scan(hull),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=pathlib.Path, default=None)
    args = parser.parse_args(argv)

    payload = run()
    text = json.dumps(payload, indent=2, sort_keys=False)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n")
        print(f"wrote {args.json}")
    else:
        print(text)

    rejected = sum(1 for row in payload["single_frequency"] if not row["public_api"]["returned"])
    print(
        f"{rejected}/{len(payload['single_frequency'])} single-frequency cases rejected; "
        f"{sum(1 for row in payload['frequency_sweep'] if not row['returned'])}/"
        f"{len(payload['frequency_sweep'])} sweep cases rejected"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())