"""Heading x frequency RAO field (seakeeping response surface) with verification.

Why this module exists
----------------------
``tensorbem.rao`` returns a 6-DOF RAO at a single (omega, beta).  Seakeeping
post-processing needs the whole *response surface* |RAO_i(ka, beta)| as a field
quantity: it is what a polar/contour plot of the response shows and what a
spectral calculation integrates over.  This module computes that field and -
more importantly - verifies it against references independent of the panel
operators.

Reference hierarchy (no self-certification)
-------------------------------------------
1. **Hulme (1982) floating hemisphere** - published multi-pole series
   (``tensorbem.analytic.hulme_hemisphere_radiation``) gives ``A33, B33, F3``
   and ``A11, B11, F1``.  Heave is a genuine single-DOF system for this body
   (no heave-pitch coupling), so

       xi3_ref / A = F3 / (-omega^2 (m + A33) + i omega B33 + C33)

   is a closed-form, operator-independent reference for |RAO_3|.
2. **Axisymmetry (Haskind/Kochin)** - for a body of revolution about ``z`` the
   horizontal exciting force vector keeps a constant magnitude and follows the
   wave heading exactly: ``F1(beta) = F_h cos(beta)``, ``F2(beta) = F_h sin(beta)``,
   while ``F3`` is heading independent.  Hence |RAO_3| is heading independent.
   These are exact identities, checked over the whole (ka, beta) grid.
3. **Long-wave heave limit** - |RAO_3| -> 1 as ``ka -> 0``.
4. **Committed baseline** - ``docs/rao_verification_metrics.json`` (64-panel
   hemisphere, ka = 0.2/0.5/1.0) must be reproduced by the coarse grid.
5. **Grid convergence** - 256-panel field vs 64-panel field.

The response surface is scale invariant: with radius ``a``, ``ka = nu a`` and the
figure axes are dimensionless, so the same cloud applies to any geometrically
similar body.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

from ..analytic import hulme_hemisphere_radiation
from ..free_surface_solver3d import (
    assemble_free_surface_operators_3d,
    solve_diffraction_3d,
    solve_radiation_3d,
)
from ..geometry3d import hemisphere_hull_mesh
from ..hydrostatic import (
    compute_displaced_volume_and_buoyancy,
    compute_hydrostatic_restoring_matrix,
    compute_mass_matrix,
)
from ..rao import compute_rao_at_frequency

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

_DEFAULT_DENSITY = 1025.0
_DEFAULT_GRAVITY = 9.80665
RADIUS = 1.0
CG_DEPTH_RATIO = 0.5  # cg at -0.5 a, same as the committed RAO baseline
#: Radius of gyration squared used for the (documented) test-device inertia
#: tensor ``J = m * RG2 * I``; identical to the committed RAO baseline
#: (``hydrostatic_rao_verification``), so pitch/surge coupling is comparable.
RADIUS_OF_GYRATION_SQ = 0.4
DOF_NAMES = ("surge", "sway", "heave", "roll", "pitch", "yaw")
DOF_UNITS = ("m/m", "m/m", "m/m", "rad/m", "rad/m", "rad/m")

#: ka sweep of the delivered cloud (dimensionless wavenumber, a = 1 m).
DEFAULT_KAS: tuple[float, ...] = (
    0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 1.0, 1.2, 1.4, 1.7, 2.0,
)
#: heading sweep; 0 deg = waves travelling towards +x.
DEFAULT_HEADINGS_DEG: tuple[float, ...] = tuple(float(d) for d in range(0, 181, 18))
FINE_SUBDIVISIONS = 3  # 4 * 4**3 = 256 panels
COARSE_SUBDIVISIONS = 2  # 4 * 4**2 = 64 panels (committed baseline resolution)


@dataclass(frozen=True)
class Body:
    """Floating hemisphere with its mass and hydrostatic restoring matrix."""

    hull: Any
    radius: float
    mass: float
    center_of_gravity: FloatArray
    restoring: FloatArray
    mass_matrix: FloatArray
    inertia: FloatArray
    displaced_volume: float

    @property
    def panel_count(self) -> int:
        return int(len(self.hull.faces))


def build_body(
    subdivisions: int,
    *,
    radius: float = RADIUS,
    density: float = _DEFAULT_DENSITY,
    gravity: float = _DEFAULT_GRAVITY,
    cg_depth_ratio: float = CG_DEPTH_RATIO,
    radius_of_gyration_sq: float = RADIUS_OF_GYRATION_SQ,
) -> Body:
    """Build the hemisphere test body used by the committed RAO baseline."""
    hull = hemisphere_hull_mesh(subdivisions, radius=radius)
    volume, _ = compute_displaced_volume_and_buoyancy(hull)
    mass = density * volume
    cg = np.array([0.0, 0.0, -cg_depth_ratio * radius], dtype=float)
    # Explicit test-device inertia about the origin (documented, not a claim
    # about a real hemisphere): avoids the heuristic diagonal default, which
    # otherwise sets the pitch response and feeds back into surge via the
    # cg offset.
    inertia = mass * radius_of_gyration_sq * radius**2 * np.eye(3)
    props = compute_hydrostatic_restoring_matrix(hull, mass, cg, density, gravity)
    return Body(
        hull=hull,
        radius=radius,
        mass=mass,
        center_of_gravity=cg,
        restoring=np.asarray(props.restoring_matrix, dtype=float),
        mass_matrix=compute_mass_matrix(mass, cg, inertia),
        inertia=inertia,
        displaced_volume=float(volume),
    )


def omega_from_ka(ka: float, radius: float = RADIUS, gravity: float = _DEFAULT_GRAVITY) -> float:
    """Deep-water dispersion: nu = ka / a, omega = sqrt(g nu)."""
    return math.sqrt(gravity * ka / radius)


def compute_field(
    body: Body,
    kas: Sequence[float],
    headings_deg: Sequence[float],
    *,
    density: float = _DEFAULT_DENSITY,
    gravity: float = _DEFAULT_GRAVITY,
    chief_points: int | str | None = "auto",
) -> dict[str, Any]:
    """Compute the RAO response surface for one body.

    Free-surface operators depend only on (mesh, nu) and the radiation problem
    does not depend on the heading, so both are assembled/solved **once per
    frequency** and reused for every heading.  The result is bit-identical to
    repeated calls of :func:`tensorbem.rao.compute_rao_at_frequency`
    (asserted by :func:`check_chain_integrity`).
    """
    kas_arr = np.asarray(kas, dtype=float)
    headings = np.deg2rad(np.asarray(headings_deg, dtype=float))
    n_f, n_b = kas_arr.size, headings.size
    rao = np.zeros((n_f, n_b, 6), dtype=complex)
    force = np.zeros((n_f, n_b, 6), dtype=complex)
    added_mass = np.zeros((n_f, 6, 6), dtype=float)
    damping = np.zeros((n_f, 6, 6), dtype=float)
    residual = np.zeros((n_f, n_b), dtype=float)

    for i, ka in enumerate(kas_arr):
        nu = float(ka) / body.radius
        ops = assemble_free_surface_operators_3d(body.hull, nu, chief_points=chief_points)
        rad = solve_radiation_3d(
            body.hull, nu, density=density, gravity=gravity, operators=ops
        )
        added_mass[i] = rad.added_mass
        damping[i] = rad.damping
        omega = math.sqrt(gravity * nu)
        z = (
            -(omega * omega) * (body.mass_matrix + rad.added_mass)
            + 1j * omega * rad.damping
            + body.restoring
        )
        for j, beta in enumerate(headings):
            diff = solve_diffraction_3d(
                body.hull,
                nu,
                wave_angle=float(beta),
                density=density,
                gravity=gravity,
                operators=ops,
            )
            f_exc = diff.excitation_force
            xi = np.linalg.solve(z, f_exc)
            rao[i, j] = xi
            force[i, j] = f_exc
            residual[i, j] = float(np.linalg.norm(z @ xi - f_exc))

    return {
        "kas": kas_arr,
        "headings_deg": np.asarray(headings_deg, dtype=float),
        "omegas": np.array([omega_from_ka(k, body.radius, gravity) for k in kas_arr]),
        "rao": rao,
        "rao_amplitude": np.abs(rao),
        "excitation_force": force,
        "added_mass": added_mass,
        "damping": damping,
        "impedance_residual": residual,
    }


def _field_worker(task: tuple[int, tuple[float, ...], tuple[float, ...]]) -> dict[str, Any]:
    """Picklable per-frequency worker used by the multiprocessing path."""
    subdivisions, kas, headings_deg = task
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    body = build_body(subdivisions)
    return {
        "subdivisions": subdivisions,
        "panel_count": body.panel_count,
        "mass": body.mass,
        "displaced_volume": body.displaced_volume,
        "restoring": body.restoring.tolist(),
        "field": compute_field(body, kas, headings_deg),
    }


def compute_field_parallel(
    subdivisions: int,
    kas: Sequence[float],
    headings_deg: Sequence[float],
    *,
    workers: int = 4,
) -> dict[str, Any]:
    """Compute the field, splitting the frequency sweep over ``workers`` processes."""
    kas = tuple(float(k) for k in kas)
    headings = tuple(float(h) for h in headings_deg)
    if workers <= 1 or len(kas) == 1:
        return _field_worker((subdivisions, kas, headings))
    import multiprocessing as mp

    tasks = [(subdivisions, (ka,), headings) for ka in kas]
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=min(workers, len(tasks))) as pool:
        chunks = pool.map(_field_worker, tasks)
    merged = {k: v for k, v in chunks[0].items() if k != "field"}
    merged["field"] = {
        key: np.concatenate([c["field"][key] for c in chunks], axis=0)
        for key in chunks[0]["field"]
    }
    for c in chunks[1:]:
        if c["panel_count"] != merged["panel_count"]:
            raise RuntimeError("panel count differs between workers")
    return merged


# ---------------------------------------------------------------------------
# references and verification
# ---------------------------------------------------------------------------


def hulme_heave_reference(
    ka: float,
    mass: float,
    c33: float,
    *,
    radius: float = RADIUS,
    density: float = _DEFAULT_DENSITY,
    gravity: float = _DEFAULT_GRAVITY,
) -> dict[str, float]:
    """Closed-form heave RAO reference from Hulme (1982) coefficients.

    Heave is decoupled for this body (``C35 = 0`` by fore/aft symmetry and no
    heave coupling in the mass matrix), so the single-DOF transfer function is

        xi3 / A = F3 / (-omega^2 (m + A33) + i omega B33 + C33).
    """
    ref = hulme_hemisphere_radiation(ka, radius=radius, density=density, gravity=gravity)
    omega = math.sqrt(gravity * ka / radius)
    denom = (
        -omega * omega * (mass + ref.heave_added_mass)
        + 1j * omega * ref.heave_damping
        + c33
    )
    value = complex(ref.excitation_force_z) / denom
    return {
        "ka": float(ka),
        "omega": omega,
        "heave_rao_amplitude": float(abs(value)),
        "heave_added_mass_reference": float(ref.heave_added_mass),
        "heave_damping_reference": float(ref.heave_damping),
        "excitation_heave_reference": float(abs(ref.excitation_force_z)),
    }


def _rel(value: float, reference: float) -> float:
    scale = abs(reference)
    return abs(value) / scale if scale == 0.0 else abs(value - reference) / scale


def _moment_heading_law(force: ComplexArray, headings: FloatArray) -> dict[str, float]:
    """Rotational excitation for an axisymmetric body: |F5| ~ |cos b|, |F4| ~ |sin b|.

    Same rotating-load argument as for the forces: the load vector of a body of
    revolution is attached to the incoming wave direction, so the pitch moment
    follows ``|cos beta|`` and the roll moment follows ``|sin beta|``.  The
    normalising values are taken at ``beta = 0`` and ``beta = 90 deg``.
    """
    headings = np.asarray(headings, dtype=float)
    beta = np.deg2rad(headings)
    abs_f = np.abs(force)
    j0 = int(np.argmin(np.abs(headings)))
    j90 = int(np.argmin(np.abs(headings - 90.0)))
    pitch = abs_f[:, j0, 4]
    roll = abs_f[:, j90, 3]
    pitch_err = np.abs(abs_f[:, :, 4] - np.outer(pitch, np.abs(np.cos(beta)))) / np.maximum(
        pitch[:, None], 1e-300
    )
    roll_err = np.abs(abs_f[:, :, 3] - np.outer(roll, np.abs(np.sin(beta)))) / np.maximum(
        roll[:, None], 1e-300
    )
    return {
        "statement": "|F5(beta)| = |F5(0)| |cos beta| and |F4(beta)| = |F4(90deg)| |sin beta|",
        "max_abs_rel_error_pitch_moment": float(np.max(pitch_err)),
        "max_abs_rel_error_roll_moment": float(np.max(roll_err)),
        "pitch_moment_at_beta0": pitch.tolist(),
        "roll_moment_at_beta90": roll.tolist(),
    }


def verify(
    field: dict[str, Any],
    *,
    coarse: dict[str, Any] | None = None,
    committed_path: Path | None = None,
    density: float = _DEFAULT_DENSITY,
    gravity: float = _DEFAULT_GRAVITY,
) -> dict[str, Any]:
    """Run every independent check and return a JSON-ready report."""
    data = field["field"]
    kas = np.asarray(data["kas"], dtype=float)
    headings = np.asarray(data["headings_deg"], dtype=float)
    rao_amp = np.asarray(data["rao_amplitude"], dtype=float)
    force = np.asarray(data["excitation_force"], dtype=complex)
    restoring = np.asarray(field["restoring"], dtype=float)
    c33 = float(restoring[2, 2])
    mass = float(field["mass"])
    j0 = int(np.argmin(np.abs(headings)))  # beta = 0

    # V1: heave RAO / excitation / A33 / B33 against the Hulme series.
    heave_rows = []
    for i, ka in enumerate(kas):
        ref = hulme_heave_reference(float(ka), mass, c33, density=density, gravity=gravity)
        heave_rows.append(
            {
                **ref,
                "heave_rao_reference": ref["heave_rao_amplitude"],
                "heave_rao_bem": float(rao_amp[i, j0, 2]),
                "heave_rao_rel_error": _rel(rao_amp[i, j0, 2], ref["heave_rao_amplitude"]),
                "excitation_heave_bem": float(abs(force[i, j0, 2])),
                "excitation_heave_rel_error": _rel(
                    abs(force[i, j0, 2]), ref["excitation_heave_reference"]
                ),
                "heave_added_mass_bem": float(data["added_mass"][i, 2, 2]),
                "heave_added_mass_rel_error": _rel(
                    data["added_mass"][i, 2, 2], ref["heave_added_mass_reference"]
                ),
                "heave_damping_bem": float(data["damping"][i, 2, 2]),
                "heave_damping_rel_error": _rel(
                    data["damping"][i, 2, 2], ref["heave_damping_reference"]
                ),
            }
        )

    # V2: axisymmetry - |F1(beta)| = |F_h| |cos(beta)|, |F2(beta)| = |F_h| |sin(beta)|
    # for a body of revolution; |RAO_3| is heading independent.
    beta_rad = np.deg2rad(headings)
    f_h = np.abs(force[:, j0, 0])
    cos_err = np.abs(np.abs(force[:, :, 0]) - np.outer(f_h, np.abs(np.cos(beta_rad)))) / np.maximum(
        np.abs(f_h)[:, None], 1e-300
    )
    sin_err = np.abs(np.abs(force[:, :, 1]) - np.outer(f_h, np.abs(np.sin(beta_rad)))) / np.maximum(
        np.abs(f_h)[:, None], 1e-300
    )
    heave_col = rao_amp[:, j0, 2]
    heave_spread = float(
        np.max(np.abs(rao_amp[:, :, 2] - heave_col[:, None]) / heave_col[:, None])
    )

    # V3: long-wave heave limit.
    i_min = int(np.argmin(kas))
    long_wave = {
        "ka": float(kas[i_min]),
        "heave_rao": float(rao_amp[i_min, j0, 2]),
        "reference": 1.0,
        "relative_error": float(_rel(rao_amp[i_min, j0, 2], 1.0)),
    }

    # V4: reproduce the committed baseline artefact.
    committed = None
    if committed_path is not None and Path(committed_path).exists():
        blob = json.loads(Path(committed_path).read_text())
        ref_rows = blob["hulme_crosscheck"]["frequencies"]
        source = (coarse or field)["field"]
        c_kas = np.asarray(source["kas"], dtype=float)
        c_amp = np.asarray(source["rao_amplitude"], dtype=float)
        rows = []
        for ref in ref_rows:
            idx = int(np.argmin(np.abs(c_kas - ref["ka"])))
            if abs(c_kas[idx] - ref["ka"]) > 1e-12:
                continue
            rows.append(
                {
                    "ka": float(ref["ka"]),
                    "heave_rao_committed": float(ref["heave_rao_amplitude"]),
                    "heave_rao_recomputed": float(c_amp[idx, j0, 2]),
                    "heave_rao_rel_error": _rel(
                        c_amp[idx, j0, 2], ref["heave_rao_amplitude"]
                    ),
                    "surge_rao_committed": float(ref.get("surge_rao_amplitude", float("nan"))),
                    "surge_rao_recomputed": float(c_amp[idx, j0, 0]),
                }
            )
        committed = {
            "artifact": str(committed_path),
            "panel_count_committed": int(blob["hulme_crosscheck"]["panel_count"]),
            "panel_count_recomputed": int((coarse or field)["panel_count"]),
            "subdivisions_recomputed": int((coarse or field)["subdivisions"]),
            "rows": rows,
            "max_heave_rao_rel_error": max(
                (r["heave_rao_rel_error"] for r in rows), default=float("nan")
            ),
        }

    # V5: discretisation - 256 vs 64 panels.
    convergence = None
    if coarse is not None:
        c_amp = np.asarray(coarse["field"]["rao_amplitude"], dtype=float)
        rel = np.abs(rao_amp - c_amp) / np.maximum(np.abs(rao_amp), 1e-12)
        convergence = {
            "fine_panels": int(field["panel_count"]),
            "coarse_panels": int(coarse["panel_count"]),
            "per_dof_max_abs_rel_difference": {
                DOF_NAMES[d]: float(np.max(rel[:, :, d])) for d in range(6)
            },
            "heave_max_abs_rel_difference": float(np.max(rel[:, :, 2])),
            "heave_rel_difference_beta0": [float(rel[i, j0, 2]) for i in range(len(kas))],
        }

    return {
        "heave_reference_vs_hulme": {
            "reference": (
                "Hulme (1982) multi-pole series via "
                "tensorbem.analytic.hulme_hemisphere_radiation"
            ),
            "c33": c33,
            "mass": mass,
            "rows": heave_rows,
            "max_heave_rao_rel_error": max(r["heave_rao_rel_error"] for r in heave_rows),
            "max_excitation_heave_rel_error": max(
                r["excitation_heave_rel_error"] for r in heave_rows
            ),
            "max_heave_added_mass_rel_error": max(
                r["heave_added_mass_rel_error"] for r in heave_rows
            ),
            "max_heave_damping_rel_error": max(
                r["heave_damping_rel_error"] for r in heave_rows
            ),
        },
        "axisymmetry": {
            "statement": (
                "F1(beta) = F_h cos(beta) and F2(beta) = F_h sin(beta) exactly for a "
                "body of revolution (Haskind); |RAO_3| is heading independent"
            ),
            "max_abs_rel_error_excitation_x": float(np.max(cos_err)),
            "max_abs_rel_error_excitation_y": float(np.max(sin_err)),
            "max_abs_rel_spread_heave_rao": heave_spread,
        },
        "moment_excitation_heading_law": _moment_heading_law(force, headings),
        "long_wave_heave_limit": long_wave,
        "impedance_residual_max": float(np.max(data["impedance_residual"])),
        "committed_baseline": committed,
        "discretisation": convergence,
    }


def check_chain_integrity(
    body: Body,
    ka: float,
    headings_deg: Iterable[float],
    *,
    chief_points: int | str | None = "auto",
) -> dict[str, float]:
    """The field driver must be bit-identical to the public RAO entry point.

    Guards the operator-reuse optimisation: reusing one operator assembly for
    the radiation and all diffraction solves is only legitimate if it changes
    nothing.
    """
    headings = tuple(float(h) for h in headings_deg)
    field = compute_field(body, (ka,), headings, chief_points=chief_points)
    amo = np.asarray(field["rao_amplitude"], dtype=float)
    force = np.asarray(field["excitation_force"], dtype=complex)
    worst_rao = 0.0
    worst_force = 0.0
    for j, beta in enumerate(headings):
        res = compute_rao_at_frequency(
            body.hull,
            omega_from_ka(ka, body.radius),
            math.radians(beta),
            body.mass,
            body.center_of_gravity,
            inertia=body.inertia,
            chief_points=chief_points,
        )
        worst_rao = max(worst_rao, float(np.max(np.abs(np.abs(res.rao) - amo[0, j]))))
        worst_force = max(
            worst_force, float(np.max(np.abs(res.excitation_force - force[0, j])))
        )
    return {
        "max_abs_rao_difference": worst_rao,
        "max_abs_excitation_difference": worst_force,
    }


# ---------------------------------------------------------------------------
# serialisation
# ---------------------------------------------------------------------------


def field_to_json(field: dict[str, Any]) -> dict[str, Any]:
    """Convert the field arrays into JSON-safe nested lists."""
    data = field["field"]
    return {
        "subdivisions": int(field["subdivisions"]),
        "panel_count": int(field["panel_count"]),
        "mass": float(field["mass"]),
        "displaced_volume": float(field["displaced_volume"]),
        "restoring": np.asarray(field["restoring"], dtype=float).tolist(),
        "kas": np.asarray(data["kas"], dtype=float).tolist(),
        "headings_deg": np.asarray(data["headings_deg"], dtype=float).tolist(),
        "omegas": np.asarray(data["omegas"], dtype=float).tolist(),
        "rao_amplitudes": np.asarray(data["rao_amplitude"], dtype=float).tolist(),
        "excitation_force_abs": np.abs(np.asarray(data["excitation_force"], dtype=complex)).tolist(),
        "added_mass_heave": np.asarray(data["added_mass"], dtype=float)[:, 2, 2].tolist(),
        "damping_heave": np.asarray(data["damping"], dtype=float)[:, 2, 2].tolist(),
        "impedance_residual_max": float(np.max(data["impedance_residual"])),
    }


def field_from_json(blob: dict[str, Any]) -> dict[str, Any]:
    """Inverse of :func:`field_to_json` for the figure stage."""
    return {
        "subdivisions": blob["subdivisions"],
        "panel_count": blob["panel_count"],
        "mass": blob["mass"],
        "displaced_volume": blob["displaced_volume"],
        "restoring": np.asarray(blob["restoring"], dtype=float),
        "field": {
            "kas": np.asarray(blob["kas"], dtype=float),
            "headings_deg": np.asarray(blob["headings_deg"], dtype=float),
            "omegas": np.asarray(blob["omegas"], dtype=float),
            "rao_amplitude": np.asarray(blob["rao_amplitudes"], dtype=float),
            "excitation_force_abs": np.asarray(blob["excitation_force_abs"], dtype=float),
            "added_mass_heave": np.asarray(blob["added_mass_heave"], dtype=float),
            "damping_heave": np.asarray(blob["damping_heave"], dtype=float),
            "impedance_residual_max": blob["impedance_residual_max"],
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--kas", type=float, nargs="+", default=list(DEFAULT_KAS))
    parser.add_argument(
        "--headings", type=float, nargs="+", default=list(DEFAULT_HEADINGS_DEG)
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=str, default="data/rao_heading_field")
    parser.add_argument("--figures", type=str, default="figures")
    parser.add_argument("--skip-coarse", action="store_true")
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="suffix for the written artefacts, e.g. --tag _part0 for a ka chunk",
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    fig_dir = Path(args.figures)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    fine = compute_field_parallel(
        FINE_SUBDIVISIONS, args.kas, args.headings, workers=args.workers
    )
    coarse = None
    if not args.skip_coarse:
        coarse = compute_field_parallel(
            COARSE_SUBDIVISIONS, args.kas, args.headings, workers=args.workers
        )

    body = build_body(FINE_SUBDIVISIONS)
    chain = check_chain_integrity(body, float(args.kas[-1]), (0.0, 90.0))
    report = verify(
        fine,
        coarse=coarse,
        committed_path=Path("docs/rao_verification_metrics.json"),
    )
    report["chain_integrity"] = chain
    report["fine"] = {k: v for k, v in field_to_json(fine).items() if k != "rao_amplitudes"}
    if coarse is not None:
        report["coarse"] = {
            k: v for k, v in field_to_json(coarse).items() if k != "rao_amplitudes"
        }
    (out_dir / f"rao_field{args.tag}.json").write_text(
        json.dumps(field_to_json(fine), indent=1) + "\n"
    )
    if coarse is not None:
        (out_dir / f"rao_field_coarse{args.tag}.json").write_text(
            json.dumps(field_to_json(coarse), indent=1) + "\n"
        )
    (out_dir / f"verification{args.tag}.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(
        {
            "panels": fine["panel_count"],
            "grid": [len(args.kas), len(args.headings)],
            "chain_integrity": chain,
            "max_heave_rao_rel_error_vs_hulme": report["heave_reference_vs_hulme"][
                "max_heave_rao_rel_error"
            ],
            "axisymmetry_x": report["axisymmetry"]["max_abs_rel_error_excitation_x"],
            "committed_baseline": report["committed_baseline"],
            "discretisation": report["discretisation"],
        },
        indent=1,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())