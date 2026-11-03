"""Impact of the rigid-body mass-matrix correction on the RAO chain.

Run::

    PYTHONPATH=src python examples/mass_matrix_coupling_impact.py \
        --output artifacts/mass_matrix_audit.json

The audited defects in :func:`tensorbem.hydrostatic.compute_mass_matrix` were

1. the translation/rotation coupling blocks were transposed (``-m*S(r_g)^T`` in
   the ``(v, omega)`` block instead of ``-m*S(r_g)``), and
2. the default-inertia branch built the inertia about the reference point and
   then applied the parallel-axis transfer again.

``rao.py`` calls ``compute_mass_matrix`` inside the impedance matrix
``Z = -omega^2*(M + A) + i*omega*B + C``, so both defects propagate into the
RAO whenever the centre of gravity is off the reference point. This script
quantifies that propagation: it solves the *same* radiation/diffraction problem
twice, once with the corrected mass matrix and once with the legacy (audited)
one, and reports the relative change of the RAO per degree of freedom.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np

from tensorbem import rao as rao_module
from tensorbem.geometry3d import hemisphere_hull_mesh
from tensorbem.hydrostatic import (
    compute_displaced_volume_and_buoyancy,
    compute_mass_matrix,
)
from tensorbem.rao import compute_rao_at_frequency

DENSITY = 1025.0
GRAVITY = 9.80665
DOF_NAMES = ("surge", "sway", "heave", "roll", "pitch", "yaw")


def legacy_mass_matrix(mass: float, center_of_gravity, inertia=None) -> np.ndarray:
    """The audited pre-fix implementation, kept here as the control arm."""
    cg = np.asarray(center_of_gravity, dtype=float)
    x, y, z = cg
    s = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=float)
    m11 = mass * np.eye(3)
    m12 = -mass * s.T
    m21 = -mass * s
    if inertia is not None:
        j = np.asarray(inertia, dtype=float)
    else:
        r = np.linalg.norm(cg)
        if r > 1e-10:
            k = 0.4 * r
            j_cg = mass * k**2 * np.eye(3)
            j = j_cg + mass * (r**2 * np.eye(3) - np.outer(cg, cg))
        else:
            j = mass * 0.4**2 * np.eye(3)
    m22 = j + mass * (s @ s.T)
    m = np.zeros((6, 6))
    m[0:3, 0:3] = m11
    m[0:3, 3:] = m12
    m[3:, 0:3] = m21
    m[3:, 3:] = m22
    return m


def _rao_case(subdivisions: int, radius: float, ka: float, cg_z_ratio: float):
    hull = hemisphere_hull_mesh(subdivisions=subdivisions, radius=radius)
    volume, _cb = compute_displaced_volume_and_buoyancy(hull)
    mass = DENSITY * volume
    cg = np.array([0.0, 0.0, -cg_z_ratio * radius])
    inertia = mass * (0.4 * radius) ** 2 * np.eye(3)  # about the CG
    omega = float(np.sqrt(GRAVITY * ka / radius))  # deep-water dispersion, ka = omega^2 a / g
    return hull, mass, cg, inertia, omega


def _solve(hull, mass, cg, inertia, omega, mass_matrix_fn):
    original = rao_module.compute_mass_matrix
    rao_module.compute_mass_matrix = mass_matrix_fn
    try:
        return compute_rao_at_frequency(
            hull, omega, 0.0, mass, cg, inertia=inertia,
            density=DENSITY, gravity=GRAVITY,
        )
    finally:
        rao_module.compute_mass_matrix = original


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/mass_matrix_audit.json"))
    parser.add_argument("--subdivisions", type=int, default=2)
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--ka", type=float, nargs="+", default=(0.5, 1.0))
    parser.add_argument("--cg-z-ratio", type=float, default=0.5)
    args = parser.parse_args()

    report: dict = {
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "case": {
            "hull": "floating hemisphere (hemisphere_hull_mesh)",
            "subdivisions": args.subdivisions,
            "radius_m": args.radius,
            "density": DENSITY,
            "gravity": GRAVITY,
            "ka_values": list(args.ka),
            "cg": "z = -cg_z_ratio*radius, x = y = 0",
            "cg_z_ratio": args.cg_z_ratio,
            "inertia": "J_cg = m*(0.4*radius)^2*I (explicit, about the CG)",
        },
        "frequencies": [],
    }

    for ka in args.ka:
        hull, mass, cg, inertia, omega = _rao_case(
            args.subdivisions, args.radius, ka, args.cg_z_ratio
        )
        corrected = _solve(hull, mass, cg, inertia, omega, compute_mass_matrix)
        legacy = _solve(hull, mass, cg, inertia, omega, legacy_mass_matrix)
        m_new = compute_mass_matrix(mass, cg, inertia)
        m_old = legacy_mass_matrix(mass, cg, inertia)
        a_new = np.abs(corrected.rao)
        a_old = np.abs(legacy.rao)
        rel = np.where(a_new > 0.0, np.abs(a_old - a_new) / np.maximum(a_new, 1e-300), 0.0)
        report["frequencies"].append(
            {
                "ka": ka,
                "omega": omega,
                "rao_amplitude_corrected": a_new.tolist(),
                "rao_amplitude_legacy": a_old.tolist(),
                "rao_relative_change_per_dof": dict(zip(DOF_NAMES, rel.tolist())),
                "max_rao_relative_change": float(np.max(rel)),
                "mass_matrix_max_relative_change": float(
                    np.max(np.abs(m_new - m_old)) / np.max(np.abs(m_new))
                ),
                "impedance_residual_corrected": corrected.impedance_residual,
                "impedance_residual_legacy": legacy.impedance_residual,
            }
        )

    # Default-inertia arm: the double parallel-axis transfer seen without an
    # explicit tensor. Not a quantitative RAO reference (the heuristic itself is
    # documented as WITHHELD), but it shows the size of the bookkeeping error.
    mass_d = 1000.0
    cg_d = np.array([0.3, -0.4, -0.6])
    m_d_new = compute_mass_matrix(mass_d, cg_d)
    m_d_old = legacy_mass_matrix(mass_d, cg_d)
    report["default_inertia_arm"] = {
        "mass": mass_d,
        "cg": cg_d.tolist(),
        "rotational_block_new_diagonal": np.diag(m_d_new[3:, 3:]).tolist(),
        "rotational_block_legacy_diagonal": np.diag(m_d_old[3:, 3:]).tolist(),
        "legacy_over_corrected_diagonal": (
            np.diag(m_d_old[3:, 3:]) / np.diag(m_d_new[3:, 3:])
        ).tolist(),
        "mass_matrix_max_relative_change": float(
            np.max(np.abs(m_d_new - m_d_old)) / np.max(np.abs(m_d_new))
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())