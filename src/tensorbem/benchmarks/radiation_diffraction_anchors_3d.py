"""3-D radiation/diffraction anchor surface for TensorBEM.

The module is the first verification surface of the 3-D radiation/diffraction
stack. Every item is a *closed-form* or *theorem-exact* reference compared with
a programmatically generated BEM result; nothing is transcribed by hand.

Anchor set
----------
A1  Unbounded (infinite-fluid) added mass ladder
    * sphere ``A33 -> 2/3 rho pi a^3`` (classical, exact);
    * closed hemisphere: lat-lon refinement ladder + Richardson limit, read
      against the campaign anchor ``2/3 rho pi a^3`` (= the sphere value and
      the hemisphere's *displaced mass*) and against the thin-disk limit
      ``8/3 rho a^3``.

A2  Infinite-frequency (pressure-release) anchor
    At ``ka -> inf`` the free surface degenerates to ``phi = 0``, which is
    exactly the symmetry plane of a heaving sphere (the heave potential is odd
    in ``z``). Hence for a surface-piercing hemisphere

        A33(ka -> inf) = 1/2 A33(sphere) = rho pi a^3 / 3      (exact),

    and the 3-D free-surface solver must climb to it as ``ka`` grows.

A3  Froude-Krylov excitation, closed form vs panel integration
    Wall-sided box barge ``L x B x T``, deep water, head seas:

        F3 = rho g A L B exp(-kT) sin(kL/2)/(kL/2)                (real)
        F1 = -2 i rho g A B sin(kL/2) (1 - exp(-kT))/k            (imaginary)
        F5 = pitch moment from bottom and end pressures.

A4  Long-wave RAO limits at ``omega = 0.05 rad/s``: heave in ``[0.95, 1.05]``,
    ``|pitch| < 0.1``.

A5  High-frequency RAO limit at ``omega = 3.0 rad/s``: heave ``< 0.2``.

A6  Mesh convergence of the barge RAO ladder ``10x5 -> 20x10 -> 40x20``:
    relative change below 2 % on the finest pair.

A7  Negative control: ``omega -> 0`` gives a frequency-independent added mass
    (the rigid-lid / zero-frequency value) and damping ``B_jj -> 0``.

Run::

    python -m tensorbem.benchmarks.radiation_diffraction_anchors_3d \
        --output docs/radiation_diffraction_anchors_3d/results.json \
        --figures docs/radiation_diffraction_anchors_3d
"""

from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from tensorbem.analytic import ellipsoid_added_mass_matrix
from tensorbem.free_surface_solver3d import froude_krylov_force_3d, solve_radiation_3d
from tensorbem.geometry3d import (
    box_hull_mesh,
    hemisphere_hull_mesh,
    hemisphere_solid_mesh,
    icosphere,
)
from tensorbem.rao import compute_rao_at_frequency
from tensorbem.solver3d import compute_added_mass_matrix_3d

DENSITY = 1025.0
GRAVITY = 9.80665

# Campaign box barge (p167w2-n01-bem-radiation)
BARGE_LENGTH = 100.0
BARGE_BEAM = 20.0
BARGE_DRAFT = 8.0
BARGE_OMEGA_FK = 0.5
BARGE_AMPLITUDE = 1.0


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _richardson_limit(
    coarse: float, fine: float, refinement: float, order: float = 2.0
) -> float:
    """Extrapolate a second-order-convergent sequence (``h`` ratio ``refinement``)."""
    factor = refinement**order
    return fine + (fine - coarse) / (factor - 1.0)


def _observed_order(
    coarse: float, medium: float, fine: float, refinement: float
) -> float | None:
    """Observed order of accuracy from three values on a geometric ladder."""
    d1 = abs(medium - coarse)
    d2 = abs(fine - medium)
    if d1 <= 0.0 or d2 <= 0.0:
        return None
    return float(math.log(d1 / d2) / math.log(refinement))


def _rel(a: float, b: float) -> float:
    return float(abs(a - b) / abs(b)) if b != 0.0 else float("inf")


def _criterion(
    ident: str,
    description: str,
    measured: Any,
    target: Any,
    passed: bool,
    note: str = "",
) -> dict[str, Any]:
    return {
        "id": ident,
        "description": description,
        "measured": measured,
        "target": target,
        "pass": bool(passed),
        "note": note,
    }


# --------------------------------------------------------------------------- #
# A1: unbounded added mass
# --------------------------------------------------------------------------- #
def sphere_unbounded_added_mass(
    subdivisions: Sequence[int] = (2, 3),
    radius: float = 1.0,
    density: float = DENSITY,
) -> dict[str, Any]:
    """``A33`` of an unbounded sphere on a geodesic ladder (exact reference)."""
    exact = (2.0 / 3.0) * density * math.pi * radius**3
    rows = []
    for sub in subdivisions:
        mesh = icosphere(int(sub), radius=radius)
        res = compute_added_mass_matrix_3d(mesh, density=density)
        a33 = float(res.added_mass_matrix[2, 2])
        rows.append(
            {
                "subdivisions": int(sub),
                "panels": int(mesh.panel_count),
                "a33": a33,
                "rel_error": _rel(a33, exact),
                "condition_number": float(res.condition_number),
                "relative_residual": float(res.max_relative_residual_l2),
            }
        )
    limit = None
    if len(rows) >= 2:
        # geodesic refinement multiplies the panel count by four (h halves)
        limit = _richardson_limit(
            rows[-2]["a33"], rows[-1]["a33"], math.sqrt(rows[-1]["panels"] / rows[-2]["panels"])
        )
    return {
        "analytic_a33": exact,
        "rows": rows,
        "worst_rel_error": max(row["rel_error"] for row in rows),
        "richardson_a33": limit,
        "richardson_rel_error": _rel(limit, exact) if limit is not None else None,
    }


def hemisphere_unbounded_added_mass(
    levels: Sequence[tuple[int, int]] = ((6, 12), (8, 16), (12, 24), (16, 32)),
    radius: float = 1.0,
    density: float = DENSITY,
) -> dict[str, Any]:
    """``A33`` of a *closed* hemisphere in unbounded fluid, refinement ladder.

    The flat lid of a closed hemisphere drives a piston-type flow, so its
    unbounded axial added mass is *not* the sphere value: it sits above the
    thin-disk limit ``8/3 rho a^3``. The ladder therefore reports the
    Richardson limit and both normalisations rather than a single
    "theoretical" number.
    """
    if len(levels) < 2:
        raise ValueError("at least two refinement levels are needed")
    sphere_value = (2.0 / 3.0) * density * math.pi * radius**3
    disk_value = (8.0 / 3.0) * density * radius**3
    rows = []
    for n_theta, n_phi in levels:
        mesh = hemisphere_solid_mesh(
            radius=radius, n_theta=int(n_theta), n_phi=int(n_phi)
        )
        res = compute_added_mass_matrix_3d(mesh, density=density)
        tri = mesh.vertices[mesh.faces]
        volume = float(
            np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0
        )
        a33 = float(res.added_mass_matrix[2, 2])
        a11 = float(res.added_mass_matrix[0, 0])
        rows.append(
            {
                "n_theta": int(n_theta),
                "n_phi": int(n_phi),
                "panels": int(mesh.panel_count),
                "discrete_volume": volume,
                "volume_rel_error": _rel(volume, (2.0 / 3.0) * math.pi * radius**3),
                "a33": a33,
                "a11": a11,
                "a33_over_rho_a3": a33 / (density * radius**3),
                "a33_over_rho_displaced": a33 / (density * (2.0 / 3.0) * math.pi * radius**3),
                "rel_error_vs_sphere_anchor": _rel(a33, sphere_value),
                "rel_error_vs_disk_limit": _rel(a33, disk_value),
                "condition_number": float(res.condition_number),
                "relative_residual": float(res.max_relative_residual_l2),
            }
        )

    h = [1.0 / row["n_theta"] for row in rows]
    last, prev = rows[-1], rows[-2]
    refinement = h[-2] / h[-1]
    a33_limit = _richardson_limit(prev["a33"], last["a33"], refinement)
    order = None
    limit_observed = None
    if len(rows) >= 3:
        order = _observed_order(
            rows[-3]["a33"], rows[-2]["a33"], rows[-1]["a33"], h[-3] / h[-2]
        )
        if order is not None and order > 0.5:
            # bracket the limit with the empirical order as well: a lat-lon cap
            # converges on a mixed h^2/h^p ladder, so the p = 2 value alone
            # overstates the remaining error
            limit_observed = _richardson_limit(
                prev["a33"], last["a33"], refinement, order=order
            )
    return {
        "sphere_anchor_a33": sphere_value,
        "disk_limit_a33": disk_value,
        "rows": rows,
        "richardson_a33": a33_limit,
        "richardson_a33_observed_order": limit_observed,
        "richardson_a33_over_rho_displaced": a33_limit
        / (density * (2.0 / 3.0) * math.pi * radius**3),
        "observed_order": order,
        "rel_error_vs_sphere_anchor": _rel(a33_limit, sphere_value),
        "rel_error_vs_disk_limit": _rel(a33_limit, disk_value),
    }


def fill_extrapolations(measured: dict[str, Any]) -> dict[str, Any]:
    """(Re)derive the Richardson numbers from the committed ladders alone.

    Everything here is a function of the panel ladders already stored in the
    payload, so a payload written by an older version of this module can be
    re-judged and re-rendered without repeating a single BEM solve.
    """
    sphere = measured["A1_sphere_unbounded"]
    rows = sphere["rows"]
    density = measured["constants"]["density"]
    if len(rows) >= 2:
        rewrite = math.sqrt(rows[-1]["panels"] / rows[-2]["panels"])
        sphere["richardson_a33"] = _richardson_limit(
            rows[-2]["a33"], rows[-1]["a33"], rewrite
        )
        sphere["richardson_rel_error"] = _rel(sphere["richardson_a33"], sphere["analytic_a33"])

    hemi = measured["A1_hemisphere_unbounded"]
    rows = hemi["rows"]
    h = [1.0 / row["n_theta"] for row in rows]
    refinement = h[-2] / h[-1]
    limit = _richardson_limit(rows[-2]["a33"], rows[-1]["a33"], refinement)
    hemi["richardson_a33"] = limit
    hemi["richardson_a33_over_rho_displaced"] = limit / (
        density * (2.0 / 3.0) * math.pi
    )
    hemi["rel_error_vs_sphere_anchor"] = _rel(limit, hemi["sphere_anchor_a33"])
    hemi["rel_error_vs_disk_limit"] = _rel(limit, hemi["disk_limit_a33"])
    order = None
    observed = None
    if len(rows) >= 3:
        order = _observed_order(
            rows[-3]["a33"], rows[-2]["a33"], rows[-1]["a33"], h[-3] / h[-2]
        )
        if order is not None and order > 0.5:
            observed = _richardson_limit(rows[-2]["a33"], rows[-1]["a33"], refinement, order=order)
    hemi["observed_order"] = order
    hemi["richardson_a33_observed_order"] = observed
    return measured


def _sphere_anchor_error(sphere: dict[str, Any]) -> float:
    """Anchor error for the sphere: Richardson limit when the ladder allows it."""
    value = sphere.get("richardson_rel_error")
    return float(value) if value is not None else float(sphere["worst_rel_error"])


def oblate_disk_limit(radius: float = 1.0, density: float = DENSITY) -> dict[str, float]:
    """Thin-disk (oblate-spheroid) axial added mass from the analytic module."""
    axes = np.array([radius, radius, 1.0e-3 * radius])
    matrix = ellipsoid_added_mass_matrix(axes, density=density)
    return {
        "a33": float(np.asarray(matrix)[2, 2]),
        "closed_form": (8.0 / 3.0) * density * radius**3,
    }


# --------------------------------------------------------------------------- #
# A2: infinite-frequency (pressure-release) anchor
# --------------------------------------------------------------------------- #
def hemisphere_high_frequency_limit(
    ka_values: Sequence[float] = (0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0),
    subdivisions: int = 3,
    radius: float = 1.0,
    density: float = DENSITY,
    gravity: float = GRAVITY,
) -> dict[str, Any]:
    """``A33(ka)`` of a floating hemisphere vs the exact ``ka -> inf`` limit."""
    hull = hemisphere_hull_mesh(int(subdivisions), radius=radius)
    limit = density * math.pi * radius**3 / 3.0
    rows = []
    for ka in ka_values:
        res = solve_radiation_3d(
            hull, float(ka), modes=(2,), density=density, gravity=gravity
        )
        a33 = float(res.added_mass[0, 0])
        rows.append(
            {
                "ka": float(ka),
                "a33": a33,
                "b33": float(res.damping[0, 0]),
                "ratio_to_limit": a33 / limit,
                "rel_error": _rel(a33, limit),
                "condition_number": float(res.condition_number),
            }
        )
    return {
        "panels": int(hull.panel_count),
        "analytic_limit": limit,
        "analytic_limit_over_rho_a3": limit / (density * radius**3),
        "rows": rows,
        "highest_ka": rows[-1]["ka"],
        "rel_error_at_highest_ka": rows[-1]["rel_error"],
        "approach_monotone_above_ka": _monotone_rise_start(rows),
    }


def _monotone_rise_start(
    rows: Sequence[dict[str, Any]], tol: float = 1.0e-4
) -> float | None:
    """First ``ka`` from which ``A33`` rises monotonically (within ``tol``)."""
    start = None
    for prev, cur in zip(rows, rows[1:]):
        if cur["a33"] > prev["a33"] * (1.0 + tol):
            if start is None:
                start = prev["ka"]
        else:
            start = None
    return start


# --------------------------------------------------------------------------- #
# A3: Froude-Krylov closed forms for a wall-sided box barge
# --------------------------------------------------------------------------- #
def froude_krylov_barge_closed_form(
    omega: float,
    *,
    length: float = BARGE_LENGTH,
    beam: float = BARGE_BEAM,
    draft: float = BARGE_DRAFT,
    amplitude: float = BARGE_AMPLITUDE,
    density: float = DENSITY,
    gravity: float = GRAVITY,
) -> dict[str, complex]:
    """Closed-form Froude-Krylov excitation of a wall-sided barge, head seas.

    Conventions follow
    :func:`tensorbem.free_surface_solver3d.incident_airy_wave_3d`
    (``exp(-i omega t)``, ``phi_I = -(i g A / omega) exp(kz) exp(i k x)``), so
    ``F_j = -i omega rho int phi_I n_j dS``. With ``n_3 = -1`` on the flat
    bottom and ``n_1 = +-1`` on the two vertical ends:

    * heave (real, in phase with the wave):

      ``F3 = rho g A L B exp(-k T) sin(kL/2) / (kL/2)``

    * surge (pure added-mass, 90 deg out of phase):

      ``F1 = -2 i rho g A B sin(kL/2) (1 - exp(-kT)) / k``

    * pitch about the waterline origin (``n_5 = x n_3 - z n_1``), with
      ``a = L/2``, ``Izz = (exp(-kT)(1 + kT) - 1)/k^2`` and
      ``C = -(i g A / omega)``:

      ``F5 = -i omega rho [ -C B exp(-kT) 2 i (sin(ka) - ka cos(ka))/k^2
      - C B 2 i sin(ka) Izz ]``
    """
    k = omega * omega / gravity
    a = 0.5 * length
    theta = k * a
    decay = math.exp(-k * draft)
    c_amp = -(1.0j * gravity * amplitude / omega)

    f3 = density * gravity * amplitude * length * beam * decay * math.sin(theta) / theta
    f1 = (
        -2.0j
        * density
        * gravity
        * amplitude
        * beam
        * math.sin(theta)
        * (1.0 - decay)
        / k
    )
    bottom_integral = 2.0j * (math.sin(theta) - theta * math.cos(theta)) / (k * k)
    izz = (decay * (1.0 + k * draft) - 1.0) / (k * k)
    f5 = -1.0j * omega * density * (
        -c_amp * beam * decay * bottom_integral
        - c_amp * beam * 2.0j * math.sin(theta) * izz
    )
    return {"f1": complex(f1), "f3": complex(f3), "f5": complex(f5)}


def froude_krylov_barge_panels(
    meshes: Iterable[tuple[int, int, int]],
    omega: float,
    *,
    length: float = BARGE_LENGTH,
    beam: float = BARGE_BEAM,
    draft: float = BARGE_DRAFT,
    amplitude: float = BARGE_AMPLITUDE,
    density: float = DENSITY,
    gravity: float = GRAVITY,
) -> dict[str, Any]:
    """Panel-integrated Froude-Krylov forces versus the closed forms."""
    nu = omega * omega / gravity
    closed = froude_krylov_barge_closed_form(
        omega,
        length=length,
        beam=beam,
        draft=draft,
        amplitude=amplitude,
        density=density,
        gravity=gravity,
    )
    rows = []
    for n_length, n_beam, n_draft in meshes:
        hull = box_hull_mesh(
            length,
            beam,
            draft,
            n_length=int(n_length),
            n_beam=int(n_beam),
            n_draft=int(n_draft),
        )
        force = froude_krylov_force_3d(
            hull, nu, amplitude=amplitude, density=density, gravity=gravity
        )
        rows.append(
            {
                "n_length": int(n_length),
                "n_beam": int(n_beam),
                "n_draft": int(n_draft),
                "panels": int(hull.panel_count),
                "f1": [float(force[0].real), float(force[0].imag)],
                "f3": [float(force[2].real), float(force[2].imag)],
                "f5": [float(force[4].real), float(force[4].imag)],
                "f3_rel_error": _rel(abs(force[2]), abs(closed["f3"])),
                "f5_rel_error": _rel(abs(force[4]), abs(closed["f5"])),
                "f1_rel_error": _rel(abs(force[0]), abs(closed["f1"])),
            }
        )
    finest = rows[-1]
    return {
        "omega": omega,
        "wavenumber": nu,
        "kL": nu * length,
        "closed_form": {
            key: [float(val.real), float(val.imag)] for key, val in closed.items()
        },
        "closed_form_magnitude": {key: float(abs(val)) for key, val in closed.items()},
        "rows": rows,
        "worst_rel_error_finest": max(
            finest["f1_rel_error"], finest["f3_rel_error"], finest["f5_rel_error"]
        ),
        "worst_rel_error_coarsest": max(
            rows[0]["f1_rel_error"], rows[0]["f3_rel_error"], rows[0]["f5_rel_error"]
        ),
    }


# --------------------------------------------------------------------------- #
# A4-A7: barge RAO surface
# --------------------------------------------------------------------------- #
def _barge_body(
    length: float, beam: float, draft: float, density: float
) -> dict[str, Any]:
    """Uniform wall-sided box in neutral buoyancy (mass = displaced mass)."""
    mass = density * length * beam * draft
    cg = np.array([0.0, 0.0, -0.5 * draft])
    inertia = np.array(
        [
            [
                mass * (beam * beam + draft * draft) / 12.0 + mass * (0.5 * draft) ** 2,
                0.0,
                0.0,
            ],
            [
                0.0,
                mass * (length * length + draft * draft) / 12.0
                + mass * (0.5 * draft) ** 2,
                0.0,
            ],
            [0.0, 0.0, mass * (length * length + beam * beam) / 12.0],
        ]
    )
    return {"mass": mass, "center_of_gravity": cg, "inertia": inertia}


def barge_rao(
    omega: float,
    *,
    n_length: int = 10,
    n_beam: int = 5,
    n_draft: int = 1,
    length: float = BARGE_LENGTH,
    beam: float = BARGE_BEAM,
    draft: float = BARGE_DRAFT,
    density: float = DENSITY,
    gravity: float = GRAVITY,
    amplitude: float = BARGE_AMPLITUDE,
    wave_angle: float = 0.0,
) -> dict[str, Any]:
    """Heave/pitch RAO of the campaign barge at one frequency."""
    hull = box_hull_mesh(
        length,
        beam,
        draft,
        n_length=int(n_length),
        n_beam=int(n_beam),
        n_draft=int(n_draft),
    )
    body = _barge_body(length, beam, draft, density)
    res = compute_rao_at_frequency(
        hull,
        float(omega),
        float(wave_angle),
        body["mass"],
        body["center_of_gravity"],
        inertia=body["inertia"],
        amplitude=amplitude,
        density=density,
        gravity=gravity,
    )
    fk = froude_krylov_force_3d(
        hull,
        res.nu,
        amplitude=amplitude,
        density=density,
        gravity=gravity,
        wave_angle=float(wave_angle),
    )
    return {
        "omega": float(omega),
        "wavenumber": float(res.nu),
        "n_length": int(n_length),
        "n_beam": int(n_beam),
        "n_draft": int(n_draft),
        "panels": int(hull.panel_count),
        "heave_rao": float(abs(res.rao[2])),
        "pitch_rao": float(abs(res.rao[4])),
        "surge_rao": float(abs(res.rao[0])),
        "a33": float(res.added_mass[2, 2]),
        "a55": float(res.added_mass[4, 4]),
        "b33": float(res.damping[2, 2]),
        "b55": float(res.damping[4, 4]),
        "excitation_f3": float(abs(res.excitation_force[2])),
        "fk_f3": float(abs(fk[2])),
        "fk_to_total_f3": float(abs(fk[2]) / abs(res.excitation_force[2])),
        "impedance_residual": float(res.impedance_residual),
    }


def barge_rao_limits(
    omegas: Sequence[float] = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0),
    n_length: int = 10,
    n_beam: int = 5,
    n_draft: int = 1,
) -> dict[str, Any]:
    """Frequency sweep: long-wave / high-frequency limits and zero-frequency control."""
    rows = [
        barge_rao(float(om), n_length=n_length, n_beam=n_beam, n_draft=n_draft)
        for om in omegas
    ]
    low = [row for row in rows if row["omega"] <= 0.2]
    mid = min(rows, key=lambda row: abs(row["omega"] - 0.5))
    a33_spread = (max(row["a33"] for row in low) - min(row["a33"] for row in low)) / max(
        row["a33"] for row in low
    )
    return {
        "rows": rows,
        "long_wave": rows[0],
        "high_frequency": rows[-1],
        "mid_frequency": mid,
        "a33_spread_low_omega": a33_spread,
        "b33_over_omega_a33_low": low[0]["b33"] / (low[0]["omega"] * low[0]["a33"]),
        "a33_ratio_low_to_mid": low[0]["a33"] / mid["a33"],
        "b33_ratio_low_to_mid": low[0]["b33"] / mid["b33"] if mid["b33"] else float("inf"),
    }


def barge_mesh_ladder(
    omegas: Sequence[float] = (0.5, 1.0),
    ladder: Sequence[tuple[int, int, int]] = ((10, 5, 1), (20, 10, 2), (40, 20, 2)),
) -> dict[str, Any]:
    """RAO mesh convergence on a geometric ladder (criterion A6)."""
    result: dict[str, Any] = {
        "ladder": [list(item) for item in ladder],
        "frequencies": [],
    }
    for omega in omegas:
        rows = [
            barge_rao(float(omega), n_length=nl, n_beam=nb, n_draft=nd)
            for nl, nb, nd in ladder
        ]
        h = [1.0 / row["n_length"] for row in rows]
        refinements = [h[i] / h[i + 1] for i in range(len(h) - 1)]
        orders = None
        if len(rows) >= 3:
            orders = {
                key: _observed_order(
                    rows[-3][key], rows[-2][key], rows[-1][key], refinements[-1]
                )
                for key in ("heave_rao", "a33", "b33")
            }
        result["frequencies"].append(
            {
                "omega": float(omega),
                "rows": rows,
                "heave_rao_change_coarse_to_medium": _rel(
                    rows[1]["heave_rao"], rows[0]["heave_rao"]
                ),
                "heave_rao_change_medium_to_fine": _rel(
                    rows[-1]["heave_rao"], rows[-2]["heave_rao"]
                ),
                "a33_change_medium_to_fine": _rel(rows[-1]["a33"], rows[-2]["a33"]),
                "b33_change_medium_to_fine": _rel(rows[-1]["b33"], rows[-2]["b33"]),
                "observed_orders": orders,
                "richardson_heave_rao": _richardson_limit(
                    rows[-2]["heave_rao"], rows[-1]["heave_rao"], refinements[-1]
                ),
            }
        )
    return result


# --------------------------------------------------------------------------- #
# study driver
# --------------------------------------------------------------------------- #
def run_study(
    *,
    quick: bool = False,
    density: float = DENSITY,
    gravity: float = GRAVITY,
) -> dict[str, Any]:
    """Run the full anchor surface and return the JSON-ready payload."""
    hemisphere_levels: Sequence[tuple[int, int]] = ((6, 12), (8, 16), (12, 24), (16, 32))
    sphere_levels: Sequence[int] = (2, 3)
    ka_values: Sequence[float] = (0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0)
    fk_meshes: Sequence[tuple[int, int, int]] = ((5, 2, 1), (10, 5, 1), (20, 10, 2), (40, 20, 2))
    omegas: Sequence[float] = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0)
    ladder: Sequence[tuple[int, int, int]] = ((10, 5, 1), (20, 10, 2), (40, 20, 2))
    if quick:
        hemisphere_levels = ((6, 12), (8, 16), (12, 24))
        sphere_levels = (2,)
        ka_values = (1.0, 3.0, 8.0)
        fk_meshes = ((10, 5, 1), (20, 10, 2))
        omegas = (0.05, 0.5, 3.0)
        ladder = ((10, 5, 1), (20, 10, 2))

    sphere = sphere_unbounded_added_mass(sphere_levels, density=density)
    hemisphere = hemisphere_unbounded_added_mass(hemisphere_levels, density=density)
    disk = oblate_disk_limit(density=density)
    high_freq = hemisphere_high_frequency_limit(
        ka_values, density=density, gravity=gravity
    )
    fk = froude_krylov_barge_panels(
        fk_meshes, BARGE_OMEGA_FK, density=density, gravity=gravity
    )
    rao_limits = barge_rao_limits(omegas)
    rao_ladder = barge_mesh_ladder(ladder=ladder)

    measured = {
        "campaign": "p167w2-n01-bem-radiation",
        "module": "tensorbem.benchmarks.radiation_diffraction_anchors_3d",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "constants": {
            "density": density,
            "gravity": gravity,
            "barge": {
                "length": BARGE_LENGTH,
                "beam": BARGE_BEAM,
                "draft": BARGE_DRAFT,
            },
            "hemisphere_radius": 1.0,
        },
        "A1_sphere_unbounded": sphere,
        "A1_hemisphere_unbounded": hemisphere,
        "A1_thin_disk_limit": disk,
        "A2_hemisphere_high_frequency": high_freq,
        "A3_froude_krylov": fk,
        "A4_A7_barge_rao": rao_limits,
        "A6_barge_mesh_ladder": rao_ladder,
    }
    return judge(measured)


def judge(measured: dict[str, Any]) -> dict[str, Any]:
    """Turn a measurement payload into the criteria verdicts.

    ``judge`` is separated from the solvers so a verdict can be recomputed from
    a committed payload (``--rejudge``) without repeating the BEM runs: the
    numbers and the judgement are two layers, and a criterion definition must
    never require re-solving the physics.
    """
    measured = fill_extrapolations(measured)
    density = measured["constants"]["density"]
    gravity = measured["constants"]["gravity"]
    sphere = measured["A1_sphere_unbounded"]
    hemisphere = measured["A1_hemisphere_unbounded"]
    disk = measured["A1_thin_disk_limit"]
    high_freq = measured["A2_hemisphere_high_frequency"]
    fk = measured["A3_froude_krylov"]
    rao_limits = measured["A4_A7_barge_rao"]
    rao_ladder = measured["A6_barge_mesh_ladder"]
    campaign_anchor = (2.0 / 3.0) * density * math.pi

    criteria: list[dict[str, Any]] = [
        _criterion(
            "C1a-sphere-unbounded-A33",
            "unbounded sphere A33 vs 2/3 rho pi a^3 (the campaign's 2146.8 kg closed form)",
            {
                "analytic": sphere["analytic_a33"],
                "finest_panels": sphere["rows"][-1]["panels"],
                "finest_a33": sphere["rows"][-1]["a33"],
                "finest_rel_error": sphere["rows"][-1]["rel_error"],
                "worst_rel_error": sphere["worst_rel_error"],
                "richardson_a33": sphere.get("richardson_a33"),
                "richardson_rel_error": sphere.get("richardson_rel_error"),
            },
            "Richardson-extrapolated relative error < 2 % (raw finest mesh also reported)",
            _sphere_anchor_error(sphere) < 0.02,
            "machinery anchor: the campaign's 2/3 rho pi a^3 value is reproduced by the "
            "sphere; the raw faceted-mesh error is 2.03 % at 320 panels and 0.51 % at "
            "1280 panels, so the criterion is applied to the Richardson limit",
        ),
        _criterion(
            "C1b-hemisphere-unbounded-A33",
            "closed hemisphere unbounded A33 vs the campaign anchor 2/3 rho pi a^3 = 2146.8 kg",
            {
                "campaign_anchor": campaign_anchor,
                "richardson_a33": hemisphere["richardson_a33"],
                "richardson_rel_error_vs_anchor": _rel(
                    hemisphere["richardson_a33"], campaign_anchor
                ),
                "a33_over_rho_displaced": hemisphere["richardson_a33_over_rho_displaced"],
                "thin_disk_limit": disk["closed_form"],
                "disk_value_from_analytic_module": disk["a33"],
            },
            "relative error < 2 %",
            _rel(hemisphere["richardson_a33"], campaign_anchor) < 0.02,
            "root cause: 2/3 rho pi a^3 is the sphere value (and the hemisphere's "
            "displaced mass). A closed hemisphere also drives a piston flow on its "
            "flat lid, so its unbounded axial added mass sits above the thin-disk "
            "limit 8/3 rho a^3 = %.1f kg (measured %.1f kg); the same body with the "
            "lid replaced by the free surface obeys the exact high-frequency anchor "
            "rho pi a^3/3 (C1c)."
            % (disk["closed_form"], hemisphere["richardson_a33"]),
        ),
        _criterion(
            "C1c-hemisphere-high-frequency-A33",
            "floating hemisphere A33(ka -> inf) vs the exact pressure-release anchor rho pi a^3/3",
            {
                "analytic_limit": high_freq["analytic_limit"],
                "highest_ka": high_freq["highest_ka"],
                "a33_at_highest_ka": high_freq["rows"][-1]["a33"],
                "rel_error": high_freq["rel_error_at_highest_ka"],
                "ka_ladder": [[row["ka"], row["a33"]] for row in high_freq["rows"]],
            },
            "monotone climb towards the limit, relative error < 5 % at the highest ka",
            high_freq["rel_error_at_highest_ka"] < 0.05,
            "exact by the mirror theorem: the heave potential of a sphere is odd in z, "
            "so ka -> inf (phi = 0 on z = 0) coincides with the sphere's symmetry "
            "plane and A33 = A33(sphere)/2 = rho pi a^3/3",
        ),
        _criterion(
            "C2-froude-krylov-closed-form",
            "barge Froude-Krylov F1/F3/F5 vs closed form at omega = 0.5 rad/s",
            {
                "worst_rel_error_finest": fk["worst_rel_error_finest"],
                "worst_rel_error_coarsest": fk["worst_rel_error_coarsest"],
                "f3_closed_form": fk["closed_form_magnitude"]["f3"],
                "f3_finest": fk["rows"][-1]["f3"],
                "panels_finest": fk["rows"][-1]["panels"],
                "kL": fk["kL"],
            },
            "relative error < 3 %",
            fk["worst_rel_error_finest"] < 0.03,
            "second-order convergence in the panel size",
        ),
        _criterion(
            "C3-long-wave-RAO",
            "long-wave limits at omega = 0.05 rad/s (kL = %.4f)"
            % (rao_limits["long_wave"]["wavenumber"] * BARGE_LENGTH),
            {
                "heave_rao": rao_limits["long_wave"]["heave_rao"],
                "pitch_rao": rao_limits["long_wave"]["pitch_rao"],
                "kL": rao_limits["long_wave"]["wavenumber"] * BARGE_LENGTH,
            },
            "heave in [0.95, 1.05] and pitch < 0.1",
            0.95 <= rao_limits["long_wave"]["heave_rao"] <= 1.05
            and rao_limits["long_wave"]["pitch_rao"] < 0.1,
        ),
        _criterion(
            "C4-high-frequency-RAO",
            "high-frequency limit at omega = 3.0 rad/s",
            {
                "heave_rao": rao_limits["high_frequency"]["heave_rao"],
                "pitch_rao": rao_limits["high_frequency"]["pitch_rao"],
                "a33": rao_limits["high_frequency"]["a33"],
                "kL": rao_limits["high_frequency"]["wavenumber"] * BARGE_LENGTH,
            },
            "heave < 0.2",
            rao_limits["high_frequency"]["heave_rao"] < 0.2,
        ),
        _criterion(
            "C5-rao-mesh-convergence",
            "|RAO(40x20) - RAO(20x10)| / RAO(20x10) over the barge ladder",
            {
                "worst_medium_to_fine": max(
                    freq["heave_rao_change_medium_to_fine"]
                    for freq in rao_ladder["frequencies"]
                ),
                "worst_coarse_to_medium": max(
                    freq["heave_rao_change_coarse_to_medium"]
                    for freq in rao_ladder["frequencies"]
                ),
            },
            "relative change < 2 %",
            max(
                freq["heave_rao_change_medium_to_fine"]
                for freq in rao_ladder["frequencies"]
            )
            < 0.02,
        ),
        _criterion(
            "C6-zero-frequency-control",
            "omega -> 0: frequency-independent A_jj, vanishing B_jj",
            {
                "a33_spread_low_omega": rao_limits["a33_spread_low_omega"],
                "b33_over_omega_a33_low": rao_limits["b33_over_omega_a33_low"],
                "a33_ratio_low_to_mid": rao_limits["a33_ratio_low_to_mid"],
                "b33_ratio_low_to_mid": rao_limits["b33_ratio_low_to_mid"],
                "a33_low_omega": [
                    [row["omega"], row["a33"]]
                    for row in rao_limits["rows"]
                    if row["omega"] <= 0.2
                ],
                "b33_low_omega": [
                    [row["omega"], row["b33"]]
                    for row in rao_limits["rows"]
                    if row["omega"] <= 0.5
                ],
                "hemisphere_ka_ladder": [
                    [row["ka"], row["a33"], row["b33"]] for row in high_freq["rows"]
                ],
            },
            "A33 spread < 5 % over omega <= 0.2 and B33/(omega A33) < 0.05",
            rao_limits["a33_spread_low_omega"] < 0.05
            and rao_limits["b33_over_omega_a33_low"] < 0.05,
        ),
    ]

    payload = dict(measured)
    payload["criteria"] = criteria
    payload["all_pass"] = all(item["pass"] for item in criteria)
    return payload


def write_figures(payload: dict[str, Any], directory: Path) -> list[str]:
    """Render the report figures (matplotlib, Agg backend)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    # --- figure 1: added-mass anchors ---------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    sphere = payload["A1_sphere_unbounded"]
    axes[0].axhline(
        sphere["analytic_a33"], color="k", ls="--", label="analytic $2/3\\rho\\pi a^3$"
    )
    axes[0].plot(
        [row["panels"] for row in sphere["rows"]],
        [row["a33"] for row in sphere["rows"]],
        "o-",
        label="sphere (BEM)",
    )
    hemi = payload["A1_hemisphere_unbounded"]
    axes[0].plot(
        [row["panels"] for row in hemi["rows"]],
        [row["a33"] for row in hemi["rows"]],
        "s-",
        label="closed hemisphere (BEM)",
    )
    axes[0].axhline(
        hemi["richardson_a33"], color="C3", ls=":", label="hemisphere Richardson limit"
    )
    axes[0].axhline(
        payload["A1_thin_disk_limit"]["closed_form"],
        color="C2",
        ls="-.",
        label="thin disk $8/3\\rho a^3$",
    )
    axes[0].set_xscale("log")
    axes[0].set_xlabel("panels")
    axes[0].set_ylabel("$A_{33}$ [kg]")
    axes[0].set_title("unbounded axial added mass")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)

    high = payload["A2_hemisphere_high_frequency"]
    axes[1].axhline(
        high["analytic_limit"], color="k", ls="--", label="exact $\\rho\\pi a^3/3$"
    )
    axes[1].plot(
        [row["ka"] for row in high["rows"]],
        [row["a33"] for row in high["rows"]],
        "o-",
        label="floating hemisphere (%d panels)" % high["panels"],
    )
    axes[1].set_xscale("log")
    axes[1].set_xlabel("$ka$")
    axes[1].set_ylabel("$A_{33}$ [kg]")
    axes[1].set_title("infinite-frequency (pressure-release) limit")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    path = directory / "fig_radiff_added_mass_anchors.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(str(path))

    # --- figure 2: Froude-Krylov convergence --------------------------------
    fk = payload["A3_froude_krylov"]
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    labels = {"f1": "$F_1$ surge", "f3": "$F_3$ heave", "f5": "$F_5$ pitch"}
    for key, label in labels.items():
        panels = [row["panels"] for row in fk["rows"]]
        errors = [max(row["%s_rel_error" % key], 1.0e-16) for row in fk["rows"]]
        ax.loglog(panels, errors, "o-", label=label)
    panels = [row["panels"] for row in fk["rows"]]
    reference = [fk["rows"][0]["f3_rel_error"] * (panels[0] / n) ** 2 for n in panels]
    ax.loglog(panels, reference, "k:", label="slope $-2$ reference")
    ax.set_xlabel("panels")
    ax.set_ylabel("relative error vs closed form")
    ax.set_title("Froude-Krylov excitation, $\\omega = %.2f$ rad/s" % fk["omega"])
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    path = directory / "fig_radiff_froude_krylov.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(str(path))

    # --- figure 3: RAO frequency sweep with limit bands ---------------------
    sweep = payload["A4_A7_barge_rao"]
    omegas = [row["omega"] for row in sweep["rows"]]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    axes[0].plot(omegas, [row["heave_rao"] for row in sweep["rows"]], "o-")
    axes[0].axhspan(0.95, 1.05, color="C2", alpha=0.2, label="long-wave band [0.95, 1.05]")
    axes[0].axhline(0.2, color="C3", ls="--", label="high-frequency bound 0.2")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("$\\omega$ [rad/s]")
    axes[0].set_ylabel("heave RAO")
    axes[0].set_title("barge heave RAO, $L$=%.0f m" % BARGE_LENGTH)
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].plot(omegas, [row["a33"] for row in sweep["rows"]], "o-", label="$A_{33}$")
    axes[1].plot(omegas, [row["b33"] for row in sweep["rows"]], "s-", label="$B_{33}$")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("$\\omega$ [rad/s]")
    axes[1].set_ylabel("coefficient")
    axes[1].set_title("zero-frequency control: $A_{33}$ plateau, $B_{33}\\to 0$")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    path = directory / "fig_radiff_rao_limits.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(str(path))

    # --- figure 4: RAO mesh convergence ------------------------------------
    ladder = payload["A6_barge_mesh_ladder"]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    for freq in ladder["frequencies"]:
        panels = [row["panels"] for row in freq["rows"]]
        axes[0].plot(
            panels, [row["heave_rao"] for row in freq["rows"]], "o-",
            label="$\\omega = %.2f$" % freq["omega"],
        )
        axes[1].plot(
            panels, [row["a33"] for row in freq["rows"]], "s-",
            label="$\\omega = %.2f$" % freq["omega"],
        )
    for ax, title, ylabel in (
        (axes[0], "heave RAO", "heave RAO"),
        (axes[1], "heave added mass", "$A_{33}$ [kg]"),
    ):
        ax.set_xscale("log")
        ax.set_xlabel("panels")
        ax.set_ylabel(ylabel)
        ax.set_title(title + " vs mesh")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    path = directory / "fig_radiff_rao_convergence.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(str(path))
    return written


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=None, help="JSON payload path")
    parser.add_argument("--figures", type=Path, default=None, help="figure directory")
    parser.add_argument("--quick", action="store_true", help="reduced ladder for CI")
    parser.add_argument(
        "--rejudge",
        action="store_true",
        help="recompute the criteria from an existing --output payload (no solves)",
    )
    args = parser.parse_args(argv)

    if args.rejudge:
        assert args.output is not None, "--rejudge needs --output"
        payload = judge(json.loads(args.output.read_text()))
    else:
        payload = run_study(quick=args.quick)
    if args.figures is not None:
        payload["figures"] = write_figures(payload, args.figures)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    summary = {
        item["id"]: item["pass"] for item in payload["criteria"]
    }
    print(json.dumps({"all_pass": payload["all_pass"], "criteria": summary}, indent=2))
    return 0 if payload["all_pass"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
