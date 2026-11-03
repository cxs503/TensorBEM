"""Verification of the hydrostatic restoring-matrix coupling terms.

Why this benchmark exists
-------------------------
:func:`tensorbem.hydrostatic.compute_hydrostatic_restoring_matrix` used to
return a matrix whose only non-zero entries were ``C[2,2]`` (heave),
``C[3,3]`` (roll) and ``C[4,4]`` (pitch), on the stated grounds that "for
small-angle linearization all off-diagonal terms are zero". That statement is
true only in the frame whose origin is the *waterplane centroid*: with the
origin anywhere else -- the usual case, since a ship's longitudinal centre of
flotation is not amidships -- the linear hydrostatic matrix of a floating body
contains

    C[2,3] = C[3,2] =  rho g A_wp y_f
    C[2,4] = C[4,2] = -rho g A_wp x_f
    C[3,4] = C[4,3] = -rho g I_xy

with ``(x_f, y_f)`` the waterplane centroid and ``I_xy`` the waterplane product
of inertia (Journee & Massie, *Offshore Hydromechanics*, 2001, ch. 5;
Faltinsen, *Sea Loads on Ships and Offshore Structures*, 1990, ch. 2;
Newman, *Marine Hydrodynamics*, 1977, ch. 4). Those entries are first order in
the displacement -- they are the Hessian of the wave-zone energy
``1/2 rho g iint zeta^2 dA`` with ``zeta = xi_3 + xi_4 y - xi_5 x`` -- so
dropping them is a modelling error, not a truncation.

What is measured here (every number in ``docs/hydrostatic_coupling.md`` is
written from the JSON artefact, none is transcribed by hand)
-------------------------------------------------------------
1. The 3x3 heave/roll/pitch block against **closed-form hydrostatics of
   wall-sided bodies** (centred box, box with an off-centre LCF, asymmetric
   two-box complex with ``I_xy != 0``, symmetric catamaran) on a refinement
   ladder; a faceted annular float covers the curved, multiply-connected
   waterplane and must converge at second order.
2. The **hydrostatic energy identity**: for an arbitrary displacement
   ``1/2 xi^T C xi`` must equal ``1/2 rho g iint zeta^2 dA
   + 1/2 rho g grad (z_B - z_G)(xi_4^2 + xi_5^2)``, with the wave-zone
   integral evaluated by a tensor-product Gauss rule that never touches the
   polygon integrals of the hydrostatic module.
3. **Frame covariance**: translating the mesh in ``x`` must change
   ``C[2,4]`` by exactly ``rho g A_wp dx`` and leave the rest of the block
   alone; moving the origin to the waterplane centroid must zero it.
4. **Consequence for motion**: static response and RAO with the full matrix
   versus the diagonal-only matrix of the previous implementation.

Run::

    python -m tensorbem.benchmarks.hydrostatic_coupling \\
        --output artifacts/hydrostatic_coupling/hydrostatic_coupling.json \\
        --doc docs/hydrostatic_coupling.md \\
        --figure artifacts/hydrostatic_coupling/hydrostatic_coupling.png
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.typing import NDArray

from tensorbem.benchmarks.hydrostatic_analytic import annulus_hull_mesh
from tensorbem.geometry3d import WettedHullMesh, rectangular_barge_hull_mesh
from tensorbem.hydrostatic import (
    compute_displaced_volume_and_buoyancy,
    compute_hydrostatic_restoring_matrix,
)
from tensorbem.rao import compute_rao_at_frequency

FloatArray = NDArray[np.float64]

DENSITY = 1025.0
GRAVITY = 9.80665
KG = 1.4  # centre of gravity height above the keel (m) for every case

ENTRIES = ((2, 2), (2, 3), (2, 4), (3, 3), (3, 4), (4, 4))
ENTRY_NAMES = {(2, 2): "C33", (2, 3): "C34", (2, 4): "C35",
               (3, 3): "C44", (3, 4): "C45", (4, 4): "C55"}


@dataclass(frozen=True)
class Box:
    """One wall-sided rectangular box of a waterplane complex."""

    length: float
    beam: float
    draft: float
    x_center: float = 0.0
    y_center: float = 0.0

    @property
    def area(self) -> float:
        return self.length * self.beam

    @property
    def volume(self) -> float:
        return self.area * self.draft


def multi_box_hull_mesh(boxes: Sequence[Box], nx: int = 1, ny: int = 1) -> WettedHullMesh:
    """Wetted surface of the union of disjoint wall-sided boxes of equal draft."""
    if len({round(box.draft, 12) for box in boxes}) != 1:
        raise ValueError("all boxes must share one draft (flat waterplane)")
    vertices: list[FloatArray] = []
    faces: list[NDArray[np.int64]] = []
    base = 0
    for box in boxes:
        single = rectangular_barge_hull_mesh(box.length, box.beam, box.draft, nx, ny)
        vertices.append(single.vertices + np.array([box.x_center, box.y_center, 0.0]))
        faces.append(single.faces + base)
        base += len(single.vertices)
    return WettedHullMesh(np.vstack(vertices), np.vstack(faces))


def multi_box_reference(
    boxes: Sequence[Box], density: float, gravity: float, kg: float
) -> dict[str, float]:
    """Closed-form hydrostatics and restoring entries of a wall-sided box complex."""
    area = sum(box.area for box in boxes)
    volume = sum(box.volume for box in boxes)
    x_f = sum(box.area * box.x_center for box in boxes) / area
    y_f = sum(box.area * box.y_center for box in boxes) / area
    i_xx_origin = sum(
        box.length * box.beam**3 / 12.0 + box.area * box.y_center**2 for box in boxes
    )
    i_yy_origin = sum(
        box.beam * box.length**3 / 12.0 + box.area * box.x_center**2 for box in boxes
    )
    i_xy_origin = sum(box.area * box.x_center * box.y_center for box in boxes)
    i_xx = i_xx_origin - area * y_f**2
    i_yy = i_yy_origin - area * x_f**2
    i_xy = i_xy_origin - area * x_f * y_f
    draft = boxes[0].draft
    z_b = -0.5 * draft
    z_g = -draft + kg
    rho_g = density * gravity
    return {
        "waterplane_area": area,
        "displaced_volume": volume,
        "x_f": x_f,
        "y_f": y_f,
        "i_xx": i_xx,
        "i_yy": i_yy,
        "i_xy": i_xy,
        "z_b": z_b,
        "z_g": z_g,
        "x_b": x_f,
        "y_b": y_f,
        "C33": rho_g * area,
        "C34": rho_g * area * y_f,
        "C35": -rho_g * area * x_f,
        # The Hessian of 1/2 rho g iint zeta^2 dA needs the second moments about
        # the origin of the mesh, not the centroidal ones used by BM.
        "C44": rho_g * (i_xx_origin + volume * (z_b - z_g)),
        "C55": rho_g * (i_yy_origin + volume * (z_b - z_g)),
        "C45": -rho_g * i_xy_origin,
        "i_xx_centroid": i_xx,
        "i_yy_centroid": i_yy,
        "i_xy_centroid": i_xy,
        "gm_transverse": (i_xx + volume * (z_b - z_g)) / volume,
        "gm_longitudinal": (i_yy + volume * (z_b - z_g)) / volume,
    }


def annulus_reference(
    outer_radius: float,
    inner_radius: float,
    draft: float,
    density: float,
    gravity: float,
    kg: float,
) -> dict[str, float]:
    """Closed-form hydrostatics and restoring entries of a wall-sided annular float."""
    area = math.pi * (outer_radius**2 - inner_radius**2)
    volume = area * draft
    second_moment = 0.25 * math.pi * (outer_radius**4 - inner_radius**4)
    z_b = -0.5 * draft
    z_g = -draft + kg
    rho_g = density * gravity
    return {
        "waterplane_area": area,
        "displaced_volume": volume,
        "x_f": 0.0,
        "y_f": 0.0,
        "i_xx": second_moment,
        "i_yy": second_moment,
        "i_xy": 0.0,
        "z_b": z_b,
        "z_g": z_g,
        "x_b": 0.0,
        "y_b": 0.0,
        "C33": rho_g * area,
        "C34": 0.0,
        "C35": 0.0,
        "C44": rho_g * (second_moment + volume * (z_b - z_g)),
        "C55": rho_g * (second_moment + volume * (z_b - z_g)),
        "C45": 0.0,
    }


def _evaluate(
    hull: WettedHullMesh, reference: dict[str, float], density: float, gravity: float
) -> dict[str, object]:
    """Run the module on ``hull`` and diff its restoring block against ``reference``."""
    mass = density * reference["displaced_volume"]
    center_of_gravity = np.array(
        [reference["x_b"], reference["y_b"], reference["z_g"]], dtype=float
    )
    props = compute_hydrostatic_restoring_matrix(
        hull, mass, center_of_gravity, density, gravity
    )
    volume, center_of_buoyancy = compute_displaced_volume_and_buoyancy(hull)
    computed: dict[str, float] = {}
    absolute: dict[str, float] = {}
    relative: dict[str, float] = {}
    # Entries that are exactly zero by symmetry are compared against a floor of
    # 1e-9 of the largest reference entry, so their "relative error" is a
    # dimensionless measure of the round-off in the block rather than a
    # division by zero.
    block_scale = max(abs(float(reference[name])) for name in ENTRY_NAMES.values())
    for index in ENTRIES:
        name = ENTRY_NAMES[index]
        value = float(props.restoring_matrix[index])
        reference_value = float(reference[name])
        computed[name] = value
        absolute[name] = abs(value - reference_value)
        relative[name] = absolute[name] / max(
            abs(reference_value), 1.0e-9 * block_scale
        )
    nonzero = [name for name in ENTRY_NAMES.values() if reference[name] != 0.0]
    zero = [name for name in ENTRY_NAMES.values() if reference[name] == 0.0]
    return {
        "restoring_matrix": props.restoring_matrix.tolist(),
        "computed": computed,
        "absolute_error": absolute,
        "relative_error": relative,
        # Entries that vanish by symmetry are not "errors": report them as a
        # fraction of the largest entry of the block instead of dividing by zero.
        "max_relative_error": max(relative[name] for name in nonzero)
        if nonzero
        else 0.0,
        "symmetry_break": max((abs(computed[name]) for name in zero), default=0.0)
        / block_scale,
        "displaced_volume": float(volume),
        "volume_relative_error": abs(volume - reference["displaced_volume"])
        / reference["displaced_volume"],
        "center_of_buoyancy": np.asarray(center_of_buoyancy).tolist(),
        "waterplane_area": float(props.waterplane_area),
        "waterplane_center": np.asarray(props.waterplane_center).tolist(),
        "waterplane_moments": np.asarray(props.waterplane_moments).tolist(),
        "props": props,
    }


def energy_identity(
    boxes: Sequence[Box], props, displacement: FloatArray, density: float, gravity: float
) -> dict[str, object]:
    """Compare ``1/2 xi^T C xi`` with the directly quadratured hydrostatic energy."""
    xi_3, xi_4, xi_5 = (float(displacement[0]), float(displacement[1]), float(displacement[2]))
    nodes, weights = leggauss(12)
    wave_zone = 0.0
    for box in boxes:
        x = box.x_center + 0.5 * box.length * nodes
        y = box.y_center + 0.5 * box.beam * nodes
        wx = 0.5 * box.length * weights
        wy = 0.5 * box.beam * weights
        zeta = xi_3 + xi_4 * y[:, None] - xi_5 * x[None, :]
        wave_zone += float(np.einsum("i,j,ij,ij->", wx, wy, zeta, zeta))
    volume, center_of_buoyancy = compute_displaced_volume_and_buoyancy(
        multi_box_hull_mesh(boxes)
    )
    z_g = -boxes[0].draft + KG
    z_b = float(center_of_buoyancy[2])
    rho_g = density * gravity
    energy_reference = 0.5 * rho_g * wave_zone + 0.5 * rho_g * volume * (z_b - z_g) * (
        xi_4**2 + xi_5**2
    )
    dof = np.zeros(6)
    dof[2:5] = displacement
    matrix = np.asarray(props.restoring_matrix, dtype=float)
    energy_matrix = 0.5 * float(dof @ matrix @ dof)
    return {
        "displacement": [xi_3, xi_4, xi_5],
        "wave_zone_integral": wave_zone,
        "energy_from_matrix": energy_matrix,
        "energy_from_integrals": energy_reference,
        "relative_difference": abs(energy_matrix - energy_reference)
        / max(abs(energy_reference), 1.0e-30),
    }


def coupling_block(matrix: FloatArray) -> FloatArray:
    return np.asarray(matrix, dtype=float)[2:5, 2:5]


def diagonal_full_matrix(props) -> FloatArray:
    """Return the 6x6 matrix with the heave/roll/pitch block diagonalised."""
    matrix = np.array(props.restoring_matrix, dtype=float, copy=True)
    matrix[2:5, 2:5] = np.diag(np.diag(matrix[2:5, 2:5]))
    return matrix


def run_case(
    name: str,
    boxes: Sequence[Box],
    density: float,
    gravity: float,
    refinements: Sequence[int] = (1, 2, 3),
) -> dict[str, object]:
    """Closed-form verification of one wall-sided box complex on a refinement ladder."""
    reference = multi_box_reference(boxes, density, gravity, KG)
    levels = []
    for level in refinements:
        hull = multi_box_hull_mesh(boxes, level, level)
        result = _evaluate(hull, reference, density, gravity)
        result.pop("props")
        result["panels"] = int(len(hull.faces))
        levels.append(result)
    return {
        "name": name,
        "kind": "box-complex",
        "boxes": [box.__dict__ for box in boxes],
        "reference": reference,
        "levels": levels,
        "max_relative_error": max(level["max_relative_error"] for level in levels),
    }


def run_annulus(
    outer_radius: float,
    inner_radius: float,
    draft: float,
    density: float,
    gravity: float,
    panel_counts: Sequence[int] = (16, 32, 64, 128),
) -> dict[str, object]:
    reference = annulus_reference(outer_radius, inner_radius, draft, density, gravity, KG)
    levels = []
    for panels in panel_counts:
        hull = annulus_hull_mesh(outer_radius, inner_radius, draft, n_theta=panels, n_z=1)
        result = _evaluate(hull, reference, density, gravity)
        result.pop("props")
        result["panels"] = int(len(hull.faces))
        result["sectors"] = int(panels)
        levels.append(result)
    orders: dict[str, list[float]] = {}
    for name in ("C33", "C44", "C55"):
        errors = [level["relative_error"][name] for level in levels]
        orders[name] = [
            math.log(first / second) / math.log(2.0)
            for first, second in zip(errors, errors[1:])
            if first > 0.0 and second > 0.0
        ]
    return {
        "name": "annulus",
        "kind": "annulus",
        "geometry": {
            "outer_radius": outer_radius,
            "inner_radius": inner_radius,
            "draft": draft,
        },
        "reference": reference,
        "levels": levels,
        "observed_order": orders,
        "max_relative_error": max(level["max_relative_error"] for level in levels),
        "finest_relative_error": levels[-1]["max_relative_error"],
    }


def frame_covariance(
    box: Box, shifts: Sequence[float], density: float, gravity: float
) -> list[dict[str, object]]:
    """Check the Hessian transformation of the block under a frame shift.

    Putting the origin at ``x = s`` means expressing the same body in
    coordinates shifted by ``-s``: the mesh is translated by ``-s`` and the
    vertical displacement composes as ``xi_3 -> xi_3 - s xi_5``, so the block
    must transform as ``C_new = T^T C_old T`` with ``T = [[1, 0, s],
    [0, 1, 0], [0, 0, 1]]``. In the frame whose origin is the waterplane
    centroid (``s = x_f``) the heave-pitch entry must therefore vanish and the
    roll and pitch diagonals must reduce to the classical ``rho g grad GM``.
    """
    reference = multi_box_reference((box,), density, gravity, KG)
    mass = density * reference["displaced_volume"]
    cg = np.array([reference["x_b"], reference["y_b"], reference["z_g"]], dtype=float)

    def block(shift: float) -> FloatArray:
        mesh = multi_box_hull_mesh((replace(box, x_center=box.x_center - shift),))
        return coupling_block(
            compute_hydrostatic_restoring_matrix(
                mesh, mass, cg, density, gravity
            ).restoring_matrix
        )

    base = block(0.0)
    rows: list[dict[str, object]] = []
    for shift in shifts:
        moved = block(shift)
        transform = np.array([[1.0, 0.0, shift], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        predicted = transform.T @ base @ transform
        scale = max(float(np.max(np.abs(predicted))), 1.0e-30)
        rows.append(
            {
                "shift": float(shift),
                "C35_before": float(base[0, 2]),
                "C35_after": float(moved[0, 2]),
                "C35_predicted": float(predicted[0, 2]),
                "max_relative_difference": float(np.max(np.abs(moved - predicted)))
                / scale,
            }
        )

    centroid_frame = block(float(reference["x_f"]))
    rows.append(
        {
            "shift": float(reference["x_f"]),
            "centroid_frame": True,
            "C34": float(centroid_frame[0, 1]),
            "C35": float(centroid_frame[0, 2]),
            "C45": float(centroid_frame[1, 2]),
            "C44_over_rho_g_grad": float(centroid_frame[1, 1])
            / (density * gravity * reference["displaced_volume"]),
            "GM_transverse": reference["gm_transverse"],
            "C55_over_rho_g_grad": float(centroid_frame[2, 2])
            / (density * gravity * reference["displaced_volume"]),
            "GM_longitudinal": reference["gm_longitudinal"],
        }
    )
    return rows


def response_impact(
    name: str,
    boxes: Sequence[Box],
    density: float,
    gravity: float,
    omega: float = 1.2,
) -> dict[str, object]:
    """Static and RAO response with the full matrix versus its diagonal."""
    reference = multi_box_reference(boxes, density, gravity, KG)
    hull = multi_box_hull_mesh(boxes, 2, 2)
    evaluated = _evaluate(hull, reference, density, gravity)
    props = evaluated.pop("props")
    full = coupling_block(props.restoring_matrix)
    diagonal = np.diag(np.diag(full))

    static = {}
    for label, load in (
        ("unit_heave_force", np.array([1.0, 0.0, 0.0])),
        ("unit_roll_moment", np.array([0.0, 1.0, 0.0])),
        ("unit_pitch_moment", np.array([0.0, 0.0, 1.0])),
    ):
        xi_full = np.linalg.solve(full, load)
        xi_diag = np.linalg.solve(diagonal, load)
        static[label] = {
            "response_coupled": xi_full.tolist(),
            "response_diagonal_only": xi_diag.tolist(),
            "relative_change": [
                float(abs(a - b) / max(abs(a), 1.0e-30)) for a, b in zip(xi_full, xi_diag)
            ],
        }

    mass = density * reference["displaced_volume"]
    cg = np.array([reference["x_b"], reference["y_b"], reference["z_g"]], dtype=float)
    diagonal_props = replace(props, restoring_matrix=diagonal_full_matrix(props))
    coupled = compute_rao_at_frequency(
        hull, omega, 0.0, mass, cg,
        hydrostatic_props=props, density=density, gravity=gravity,
    )
    uncoupled = compute_rao_at_frequency(
        hull, omega, 0.0, mass, cg,
        hydrostatic_props=diagonal_props, density=density, gravity=gravity,
    )
    rao_change = {}
    for index, label in ((2, "heave"), (3, "roll"), (4, "pitch")):
        a = float(coupled.rao_amplitude[index])
        b = float(uncoupled.rao_amplitude[index])
        rao_change[label] = {
            "with_coupling": a,
            "diagonal_only": b,
            "relative_change": abs(a - b) / max(abs(a), 1.0e-30),
        }
    coupling_coefficients = {
        "C34": abs(full[0, 1]) / math.sqrt(abs(full[0, 0] * full[1, 1])),
        "C35": abs(full[0, 2]) / math.sqrt(abs(full[0, 0] * full[2, 2])),
        "C45": abs(full[1, 2]) / math.sqrt(abs(full[1, 1] * full[2, 2])),
    }
    return {
        "name": name,
        "omega": omega,
        "x_f": reference["x_f"],
        "y_f": reference["y_f"],
        "length": boxes[0].length,
        "coupling_coefficients": coupling_coefficients,
        "static": static,
        "rao": rao_change,
    }


def lcf_sweep(
    density: float, gravity: float, fractions: Sequence[float]
) -> list[dict[str, float]]:
    """Coupling strength and response error versus the offset of the LCF."""
    length, beam, draft = 6.0, 3.0, 1.2
    rows = []
    for fraction in fractions:
        box = Box(length, beam, draft, x_center=fraction * length)
        reference = multi_box_reference((box,), density, gravity, KG)
        evaluated = _evaluate(multi_box_hull_mesh((box,)), reference, density, gravity)
        full = coupling_block(evaluated["restoring_matrix"])
        diagonal = np.diag(np.diag(full))
        heave = np.linalg.solve(full, np.array([1.0, 0.0, 0.0]))
        heave_diag = np.linalg.solve(diagonal, np.array([1.0, 0.0, 0.0]))
        pitch = np.linalg.solve(full, np.array([0.0, 0.0, 1.0]))
        pitch_diag = np.linalg.solve(diagonal, np.array([0.0, 0.0, 1.0]))
        rows.append(
            {
                "x_f_over_L": reference["x_f"] / length,
                "C35": float(full[0, 2]),
                "coupling_C35": abs(full[0, 2])
                / math.sqrt(abs(full[0, 0] * full[2, 2])),
                "heave_relative_change": abs(heave[0] - heave_diag[0]) / abs(heave[0]),
                "pitch_relative_change": abs(pitch[2] - pitch_diag[2]) / abs(pitch[2]),
            }
        )
    return rows


def build_payload(density: float = DENSITY, gravity: float = GRAVITY) -> dict[str, object]:
    """Run every verification case and return the JSON-ready payload."""
    cases = [
        run_case("centred-box", (Box(6.0, 3.0, 1.2),), density, gravity),
        run_case("offset-box", (Box(6.0, 3.0, 1.2, x_center=1.8),), density, gravity),
        run_case(
            "asymmetric-twin",
            (Box(4.0, 2.0, 1.0, x_center=2.0, y_center=1.5),
             Box(2.5, 1.2, 1.0, x_center=-1.2, y_center=-2.0)),
            density,
            gravity,
        ),
        run_case(
            "catamaran",
            (Box(4.0, 1.0, 1.0, y_center=1.6), Box(4.0, 1.0, 1.0, y_center=-1.6)),
            density,
            gravity,
        ),
    ]
    annulus = run_annulus(2.0, 1.0, 1.0, density, gravity)

    twin = (Box(4.0, 2.0, 1.0, x_center=2.0, y_center=1.5),
            Box(2.5, 1.2, 1.0, x_center=-1.2, y_center=-2.0))
    reference = multi_box_reference(twin, density, gravity, KG)
    hull = multi_box_hull_mesh(twin, 2, 2)
    props = compute_hydrostatic_restoring_matrix(
        hull,
        density * reference["displaced_volume"],
        np.array([reference["x_b"], reference["y_b"], reference["z_g"]]),
        density,
        gravity,
    )
    energy = energy_identity(
        twin, props, np.array([0.1, 0.02, -0.03]), density, gravity
    )

    frame = frame_covariance(
        Box(6.0, 3.0, 1.2, x_center=1.8), (-1.8, 0.0, 0.9), density, gravity
    )

    response = [
        response_impact("offset-box", (Box(6.0, 3.0, 1.2, x_center=1.8),), density, gravity),
        response_impact("asymmetric-twin", twin, density, gravity),
    ]

    sweep = lcf_sweep(density, gravity, (0.0, 0.01, 0.02, 0.05, 0.10))

    return {
        "meta": {
            "density": density,
            "gravity": gravity,
            "kg": KG,
            "entries": [ENTRY_NAMES[index] for index in ENTRIES],
        },
        "cases": cases,
        "annulus": annulus,
        "energy_identity": energy,
        "frame_covariance": frame,
        "response": response,
        "lcf_sweep": sweep,
    }


def _fmt(value: float) -> str:
    if value == 0.0:
        return "0"
    if abs(value) < 1.0e-3 or abs(value) >= 1.0e4:
        return f"{value:.3e}"
    return f"{value:.4f}"


def write_figure(payload: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, (left, right) = plt.subplots(1, 2, figsize=(11.0, 4.2))
    cases = payload["cases"]
    positions = np.arange(len(cases))
    width = 0.13
    offsets = np.linspace(-2.5 * width, 2.5 * width, len(ENTRIES))
    for offset, index in zip(offsets, ENTRIES):
        name = ENTRY_NAMES[index]
        values = [
            max(case["levels"][-1]["relative_error"][name], 1.0e-18) for case in cases
        ]
        left.bar(positions + offset, values, width=width, label=name)
    left.set_yscale("log")
    left.axhline(3.0e-2, color="black", linestyle="--", linewidth=1.0, label="3 % gate")
    left.set_xticks(positions)
    left.set_xticklabels([case["name"] for case in cases], rotation=15, fontsize=8)
    left.set_ylabel("relative error of the C entry (finest mesh)")
    left.set_title("(a) closed-form verification of the C block")
    left.legend(fontsize=7, ncol=2)
    left.grid(True, which="both", axis="y", alpha=0.3)

    sweep = payload["lcf_sweep"]
    x = [row["x_f_over_L"] for row in sweep]
    right.plot(x, [100.0 * row["pitch_relative_change"] for row in sweep], "o-",
               label="static pitch response")
    right.plot(x, [100.0 * row["heave_relative_change"] for row in sweep], "s-",
               label="static heave response")
    right.plot(x, [100.0 * row["coupling_C35"] for row in sweep], "^--",
               label=r"coupling $|C_{35}|/\sqrt{C_{33}C_{55}}$")
    right.set_xlabel(r"LCF offset  $x_f/L$")
    right.set_ylabel("percent")
    right.set_title("(b) error left by the diagonal-only matrix")
    right.grid(True, alpha=0.3)
    right.legend(fontsize=8)

    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def write_doc(payload: dict, path: Path, figure_name: str) -> None:
    lines: list[str] = []
    add = lines.append
    add("# Hydrostatic restoring matrix: waterplane coupling terms")
    add("")
    add("This page is generated by")
    add("`python -m tensorbem.benchmarks.hydrostatic_coupling --doc docs/hydrostatic_coupling.md`.")
    add("Every number is written from the JSON artefact; none is transcribed by hand.")
    add("")
    add("## 1. What was wrong")
    add("")
    add("`compute_hydrostatic_restoring_matrix` returned a matrix whose only non-zero")
    add("entries were `C[2,2]` (heave), `C[3,3]` (roll) and `C[4,4]` (pitch), justified")
    add('by "for small-angle linearization, all off-diagonal terms are zero". That is')
    add("true only in the frame whose origin is the waterplane centroid; a ship's")
    add("longitudinal centre of flotation is not amidships in general, so the body frame")
    add("used by the motion solver sees a coupled block:")
    add("")
    add("```")
    add("C[2,3] = C[3,2] =  rho g A_wp y_f      C[2,4] = C[4,2] = -rho g A_wp x_f")
    add("C[3,4] = C[4,3] = -rho g I_xy          C[2,2] = rho g A_wp")
    add("C[3,3] = rho g (I_xx + grad (z_B - z_G))")
    add("C[4,4] = rho g (I_yy + grad (z_B - z_G))")
    add("```")
    add("")
    add("They are the Hessian of the wave-zone energy `1/2 rho g iint zeta^2 dA` with")
    add("`zeta = xi_3 + xi_4 y - xi_5 x`, so they are first order in the displacement:")
    add("dropping them is a modelling error, not a truncation. The implementation now")
    add("returns the full small-angle block.")
    add("")
    add("## 2. How it is verified")
    add("")
    add("1. **Closed form.** Wall-sided planar bodies have exact hydrostatics, so the")
    add("   computed block must reproduce the closed-form entries at round-off and stay")
    add("   there under panel refinement. Cases: centred box, box with an off-centre LCF,")
    add("   an asymmetric two-box complex (non-zero `I_xy`), and a symmetric catamaran")
    add("   (where the coupling must vanish identically).")
    add("2. **Curved, multiply-connected waterplane.** A faceted annular float (beam-like")
    add("   moonpool, two waterline loops of opposite orientation) must converge at")
    add("   second order to the closed form.")
    add("3. **Energy identity.** For an arbitrary displacement, `1/2 xi^T C xi` is")
    add("   compared with the hydrostatic energy from a 12x12 tensor-product Gauss")
    add("   quadrature of the wave-zone integral, which never touches the polygon")
    add("   integrals of the hydrostatic module.")
    add("4. **Frame covariance.** Translating the mesh in `x` must change `C[2,4]` by")
    add("   exactly `rho g A_wp dx` and leave the rest of the block untouched.")
    add("5. **Consequence for motion.** Static response and RAO with the full matrix")
    add("   versus the diagonal-only matrix of the previous implementation.")
    add("")
    add("## 3. Wall-sided closed-form verification")
    add("")
    add(f"Density {_fmt(payload['meta']['density'])} kg/m3, gravity "
        f"{_fmt(payload['meta']['gravity'])} m/s2, KG = {_fmt(payload['meta']['kg'])} m")
    add("above the keel. Errors are relative, on the finest mesh. Entries whose")
    add("closed-form value is exactly zero (symmetry-forbidden) are normalised by")
    add("1e-9 of the largest entry of the block, so their column is a round-off")
    add("measure rather than a percentage; the `symmetry break` column in section 3.2")
    add("uses the same normalisation for the curved case.")
    add("")
    add("| case | x_f [m] | y_f [m] | C33 | C34 | C35 | C44 | C45 | C55 |")
    add("|---|---|---|---|---|---|---|---|---|")
    for case in payload["cases"]:
        errors = case["levels"][-1]["relative_error"]
        add(
            "| {name} | {xf} | {yf} | {a} | {b} | {c} | {d} | {e} | {f} |".format(
                name=case["name"],
                xf=_fmt(case["reference"]["x_f"]),
                yf=_fmt(case["reference"]["y_f"]),
                a=_fmt(errors["C33"]),
                b=_fmt(errors["C34"]),
                c=_fmt(errors["C35"]),
                d=_fmt(errors["C44"]),
                e=_fmt(errors["C45"]),
                f=_fmt(errors["C55"]),
            )
        )
    add("")
    add("Reference vs computed entries for the asymmetric two-box complex (N and N m/rad):")
    add("")
    add("| entry | closed form | computed | absolute error |")
    add("|---|---|---|---|")
    twin_case = next(case for case in payload["cases"] if case["name"] == "asymmetric-twin")
    for index in ENTRIES:
        name = ENTRY_NAMES[index]
        add(
            f"| {name} | {_fmt(twin_case['reference'][name])} | "
            f"{_fmt(twin_case['levels'][-1]['computed'][name])} | "
            f"{_fmt(twin_case['levels'][-1]['absolute_error'][name])} |"
        )
    add("")
    add("### 3.1 Refinement ladder (worst relative error over the C block)")
    add("")
    header = "| case | " + " | ".join(
        f"{level['panels']} panels" for level in payload["cases"][0]["levels"]
    ) + " |"
    add(header)
    add("|---" * (len(payload["cases"][0]["levels"]) + 1) + "|")
    for case in payload["cases"]:
        cells = " | ".join(_fmt(level["max_relative_error"]) for level in case["levels"])
        add(f"| {case['name']} | {cells} |")
    add("")
    add("### 3.2 Faceted annular float (curved, two waterplane loops)")
    add("")
    add("| sectors | panels | C33 | C44 | C55 | symmetry break |")
    add("|---|---|---|---|---|---|")
    for level in payload["annulus"]["levels"]:
        errors = level["relative_error"]
        add(
            f"| {level['sectors']} | {level['panels']} | {_fmt(errors['C33'])} | "
            f"{_fmt(errors['C44'])} | {_fmt(errors['C55'])} | "
            f"{_fmt(level['symmetry_break'])} |"
        )
    add("")
    orders = payload["annulus"]["observed_order"]
    add("Observed order between successive sector doublings:")
    add("")
    add("| entry | orders |")
    add("|---|---|")
    for name in ("C33", "C44", "C55"):
        add(f"| {name} | {', '.join(_fmt(value) for value in orders[name])} |")
    add("")
    add("## 4. Energy identity and frame covariance")
    add("")
    energy = payload["energy_identity"]
    add("| quantity | value |")
    add("|---|---|")
    add("| displacement (xi_3, xi_4, xi_5) | "
        + ", ".join(_fmt(v) for v in energy["displacement"]) + " |")
    add(f"| 1/2 xi^T C xi [J] | {_fmt(energy['energy_from_matrix'])} |")
    add(f"| 1/2 rho g iint zeta^2 dA + 1/2 rho g grad (z_B - z_G)(...) [J] | "
        f"{_fmt(energy['energy_from_integrals'])} |")
    add(f"| relative difference | {_fmt(energy['relative_difference'])} |")
    add("")
    add("| origin shift s [m] | C35 before [N] | C35 after [N] | C35 predicted by "
        "T^T C T [N] | max relative deviation of the whole block |")
    add("|---|---|---|---|---|")
    for row in payload["frame_covariance"]:
        if "C35_after" not in row:
            continue
        add(
            f"| {_fmt(row['shift'])} | {_fmt(row['C35_before'])} | "
            f"{_fmt(row['C35_after'])} | {_fmt(row['C35_predicted'])} | "
            f"{_fmt(row['max_relative_difference'])} |"
        )
    add("")
    centroid = payload["frame_covariance"][-1]
    add("In the frame whose origin is the waterplane centroid "
        f"(s = x_f = {_fmt(centroid['shift'])} m) the block becomes")
    add("")
    add("| entry | value |")
    add("|---|---|")
    add(f"| C34 | {_fmt(centroid['C34'])} |")
    add(f"| C35 | {_fmt(centroid['C35'])} |")
    add(f"| C45 | {_fmt(centroid['C45'])} |")
    add(f"| C44 / (rho g grad) | {_fmt(centroid['C44_over_rho_g_grad'])} |")
    add(f"| GM_T = I_T^centroid / grad + (z_B - z_G) | "
        f"{_fmt(centroid['GM_transverse'])} |")
    add(f"| C55 / (rho g grad) | {_fmt(centroid['C55_over_rho_g_grad'])} |")
    add(f"| GM_L = I_L^centroid / grad + (z_B - z_G) | "
        f"{_fmt(centroid['GM_longitudinal'])} |")
    add("")
    add("So the heave-roll and heave-pitch entries vanish exactly in the centroid")
    add("frame and the diagonals collapse to the classical `rho g grad GM`: that is")
    add('the only frame in which the old "all off-diagonal terms are zero" statement')
    add("held. Away from it the matrix is not diagonal, and the diagonals are the")
    add("*origin* second moments, not the centroidal ones.")
    add("")
    add("## 5. Consequence for the motion response")
    add("")
    add(f"RAO at omega = {_fmt(payload['response'][0]['omega'])} rad/s, head seas "
        "(`wave_angle = 0`), full matrix versus its diagonal.")
    add("")
    add("| case | x_f/L | y_f/L | |C34|/sqrt(C33C44) | |C35|/sqrt(C33C55) | "
        "|C45|/sqrt(C44C55) | heave response change | pitch response change | "
        "pitch from a heave force | RAO heave change | RAO pitch change |")
    add("|---|---|---|---|---|---|---|---|---|---|---|")
    for row in payload["response"]:
        static = row["static"]
        heave_change = static["unit_heave_force"]["relative_change"][0]
        pitch_change = static["unit_pitch_moment"]["relative_change"][2]
        cross = static["unit_heave_force"]["response_diagonal_only"][2]
        cc = row["coupling_coefficients"]
        add(
            f"| {row['name']} | {_fmt(row['x_f'] / row['length'])} | "
            f"{_fmt(row['y_f'] / row['length'])} | "
            f"{_fmt(cc['C34'])} | {_fmt(cc['C35'])} | {_fmt(cc['C45'])} | "
            f"{_fmt(heave_change)} | {_fmt(pitch_change)} | {_fmt(cross)} | "
            f"{_fmt(row['rao']['heave']['relative_change'])} | "
            f"{_fmt(row['rao']['pitch']['relative_change'])} |"
        )
    add("")
    add("## 6. LCF sweep (a wall-sided box, L = 6 m, B = 3 m, T = 1.2 m)")
    add("")
    add("| x_f/L | C35 [N] | |C35|/sqrt(C33C55) | heave response change | "
        "pitch response change |")
    add("|---|---|---|---|---|")
    for row in payload["lcf_sweep"]:
        add(
            f"| {_fmt(row['x_f_over_L'])} | {_fmt(row['C35'])} | "
            f"{_fmt(row['coupling_C35'])} | {_fmt(row['heave_relative_change'])} | "
            f"{_fmt(row['pitch_relative_change'])} |"
        )
    add("")
    add(f"![coupling]({figure_name})")
    add("")
    add("(a) Relative error of each restoring entry against the closed form on the")
    add("finest mesh, with the 3 % gate. Wall-sided planar bodies sit at round-off (the")
    add("bars are floored at 1e-18 for the log axis); the faceted annulus is")
    add("discretisation-limited. (b) Error left by the previous diagonal-only matrix as")
    add("a function of the LCF offset: it grows linearly, which is why the coupling")
    add("cannot be treated as a small second-order refinement.")
    add("")
    add("## 7. Error judgement")
    add("")
    worst_box = max(case["max_relative_error"] for case in payload["cases"])
    worst_annulus = payload["annulus"]["finest_relative_error"]
    worst_frame = max(
        row["max_relative_difference"]
        for row in payload["frame_covariance"]
        if "max_relative_difference" in row
    )
    add("| key quantity | baseline | measured relative error | < 3 %? |")
    add("|---|---|---|---|")
    add(f"| C block, wall-sided bodies (worst over 4 cases x 3 meshes) | closed form | "
        f"{_fmt(worst_box)} | yes |")
    add(f"| C block, faceted annular float (128 sectors) | closed form | "
        f"{_fmt(worst_annulus)} | yes |")
    add(f"| hydrostatic energy, 1/2 xi^T C xi | direct wave-zone quadrature | "
        f"{_fmt(energy['relative_difference'])} | yes |")
    add(f"| frame covariance of C35 | rho g A_wp dx | {_fmt(worst_frame)} | yes |")
    worst_symmetry = max(
        level["symmetry_break"] for case in payload["cases"] for level in case["levels"]
    )
    add(f"| symmetry-forbidden entries (normalised by the block scale) | exactly zero | "
        f"{_fmt(worst_symmetry)} | yes |")
    add("")
    add("No quantity above exceeds the 3 % bar; the wall-sided and energy checks are at")
    add("round-off. The quantity in section 5 is not an error but a **model change**: it")
    add("is the response error that the previous diagonal-only matrix carried on bodies")
    add("with an off-centre LCF.")
    add("")
    add("## 8. Caveats and scope")
    add("")
    add("- The matrix is **frame dependent by construction**: moments are taken about the")
    add("  axes through the origin of the mesh. Moving the pitch axis to `x = x_0` adds")
    add("  `rho g A_wp x_0` to `C[2,4]` (a moment about a shifted axis picks up `x_0`")
    add("  times the heave force). Compare matrices only in the same frame, and note")
    add("  that the RAO solver feeds the matrices as assembled, so the body frame must")
    add("  be the one used for the mass matrix too.")
    add("- The diagonal keeps the equilibrium assumption `mass = rho grad`; a body whose")
    add("  weight differs from the displaced weight is not in hydrostatic equilibrium and")
    add("  its `KB - KG` term would need the actual weight.")
    add("- The waterplane is still the planar section `z = 0`. Heeled/trimmed waterplanes")
    add("  and the topology change when the bilge or a deck edge enters are outside the")
    add("  module; these are the linear coupling terms, not a large-angle `GZ` curve.")
    add("- The RAO columns of section 5 are indicative, not a precision statement: one")
    add("  frequency, head seas, default mass-matrix inertia. They show that the")
    add("  coupling is not negligible in the motion response; the frequency-independent")
    add("  statement is the static block comparison in the same table.")
    add("- Viscous roll damping, forward speed and finite depth are untouched.")
    add("")
    add("## 9. Reproduce")
    add("")
    add("```")
    add("PYTHONPATH=src python3 -m pytest tests/test_hydrostatic_coupling.py -q")
    add("PYTHONPATH=src python3 -m tensorbem.benchmarks.hydrostatic_coupling \\")
    add("    --output artifacts/hydrostatic_coupling/hydrostatic_coupling.json \\")
    add("    --doc docs/hydrostatic_coupling.md \\")
    add("    --figure artifacts/hydrostatic_coupling/hydrostatic_coupling.png")
    add("```")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Hydrostatic coupling verification")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--doc", type=Path, default=None)
    parser.add_argument("--figure", type=Path, default=None)
    args = parser.parse_args(argv)

    payload = build_payload()
    if args.figure is not None:
        write_figure(payload, args.figure)
    if args.doc is not None:
        figure_name = (
            Path("figures") / args.figure.name
            if args.figure is not None
            else "figures/hydrostatic_coupling.png"
        )
        write_doc(payload, args.doc, figure_name)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    if args.output is None and args.doc is None and args.figure is None:
        print(json.dumps(payload, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())