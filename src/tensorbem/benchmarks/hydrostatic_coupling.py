"""Verification study for the hydrostatic restoring coupling terms.

The historical ``compute_hydrostatic_restoring_matrix`` returns a diagonal matrix.
That form is exact only inside a narrow domain (waterplane centroid on the
reference vertical axis, zero waterplane product of inertia about it, body in
flotation equilibrium).  This module

* builds exact prismatic hulls with prescribed (possibly eccentric and
  asymmetric) waterplanes,
* validates the complete 6x6 matrix against closed-form expressions and against
  an independent finite-difference oracle that integrates the wetted polyhedron
  after clipping it at the calm waterplane,
* quantifies the error of the diagonal-only matrix outside its domain, and
* measures how the coupling terms propagate into the RAO chain.

Run ``python -m tensorbem.benchmarks.hydrostatic_coupling --output <dir>`` to
write ``hydrostatic_coupling_study.json`` plus the figures used by
``dev_report_tensorbem-restoring-matrix-audit.md``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from ..geometry3d import WettedHullMesh
from ..hydrostatic import (
    compute_hydrostatic_restoring_matrix,
    compute_hydrostatic_restoring_matrix_full,
)
from ..rao import compute_rao_at_frequency

RHO = 1025.0
G = 9.80665
DEFAULT_FREEBOARD = 0.5

#: Prism waterplane used as the asymmetric (no mirror symmetry) case.
TRAPEZOID = np.array(
    [(-1.00, -0.60), (1.20, -0.80), (1.50, 1.00), (-0.90, 0.70)], dtype=float
)

MODE_NAMES = ("surge", "sway", "heave", "roll", "pitch", "yaw")


# --------------------------------------------------------------------------- #
# geometry helpers
# --------------------------------------------------------------------------- #
def rectangle(x0: float, y0: float, length: float, beam: float) -> np.ndarray:
    """Counter-clockwise rectangle waterplane with its lower corner at (x0, y0)."""
    return np.array(
        [(x0, y0), (x0 + length, y0), (x0 + length, y0 + beam), (x0, y0 + beam)],
        dtype=float,
    )


def polygon_area(polygon: np.ndarray) -> float:
    """Area of a simple polygon (shoelace formula)."""
    x = np.asarray(polygon, dtype=float)[:, 0]
    y = np.asarray(polygon, dtype=float)[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def prism_hull(polygon: np.ndarray, draft: float) -> WettedHullMesh:
    """Wetted surface of a vertical prism with a convex waterplane polygon.

    Every face is planar, so the waterplane polygon and the enclosed volume are
    represented exactly by the ten-or-so triangles.
    """
    polygon = np.asarray(polygon, dtype=float)
    n = len(polygon)
    vertices = [(x, y, -draft) for x, y in polygon] + [(x, y, 0.0) for x, y in polygon]
    reversed_index = list(reversed(range(n)))
    faces = []
    for k in range(1, n - 1):
        faces.append((reversed_index[0], reversed_index[k], reversed_index[k + 1]))
    for i in range(n):
        j = (i + 1) % n
        faces.append((i, j, j + n))
        faces.append((i, j + n, i + n))
    return WettedHullMesh(np.array(vertices, dtype=float), np.array(faces, dtype=np.int64))


# --------------------------------------------------------------------------- #
# independent finite-difference oracle
# --------------------------------------------------------------------------- #
def _rotation(xi: np.ndarray) -> np.ndarray:
    phi, theta, psi = xi[3], xi[4], xi[5]
    cx, sx = math.cos(phi), math.sin(phi)
    cy, sy = math.cos(theta), math.sin(theta)
    cz, sz = math.cos(psi), math.sin(psi)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def _prism_polyhedron(
    polygon: np.ndarray, draft: float, freeboard: float = DEFAULT_FREEBOARD
) -> list[np.ndarray]:
    """Closed convex polyhedron (bottom, sides, deck) with outward face normals."""
    polygon = np.asarray(polygon, dtype=float)
    n = len(polygon)
    bottom = np.column_stack([polygon, np.full(n, -draft)])
    deck = np.column_stack([polygon, np.full(n, freeboard)])
    faces = [bottom[::-1].copy(), deck.copy()]
    for i in range(n):
        j = (i + 1) % n
        faces.append(np.array([bottom[i], bottom[j], deck[j], deck[i]]))
    return faces


def clip_below_waterplane(faces: list[np.ndarray], tol: float = 1.0e-12) -> list[np.ndarray]:
    """Clip a convex polyhedron by the half space ``z <= 0`` (exact)."""
    clipped: list[np.ndarray] = []
    on_plane: list[np.ndarray] = []
    for face in faces:
        kept: list[np.ndarray] = []
        count = len(face)
        for k in range(count):
            p = face[k]
            q = face[(k + 1) % count]
            inside_p = p[2] <= tol
            inside_q = q[2] <= tol
            if inside_p:
                kept.append(p)
            if inside_p != inside_q:
                s = p[2] / (p[2] - q[2])
                point = p + s * (q - p)
                point = np.array([point[0], point[1], 0.0])
                kept.append(point)
                on_plane.append(point)
        if len(kept) >= 3:
            clipped.append(np.array(kept))
    if len(on_plane) >= 3:
        points = np.array(on_plane)
        unique: list[np.ndarray] = []
        for point in points:
            if not any(np.linalg.norm(point - other) <= 1.0e-9 for other in unique):
                unique.append(point)
        points = np.array(unique)
        if len(points) >= 3:
            centre = points.mean(axis=0)
            angles = np.arctan2(points[:, 1] - centre[1], points[:, 0] - centre[0])
            clipped.append(points[np.argsort(angles)])
    return clipped


def volume_and_centroid(faces: list[np.ndarray]) -> tuple[float, np.ndarray]:
    """Volume and centroid of a closed polyhedron with outward face normals."""
    volume = 0.0
    moment = np.zeros(3)
    for face in faces:
        a = face[0]
        for k in range(1, len(face) - 1):
            b, c = face[k], face[k + 1]
            signed = float(np.dot(a, np.cross(b, c))) / 6.0
            volume += signed
            moment += signed * (a + b + c) / 4.0
    return volume, moment / volume


def generalized_hydrostatic_force(
    polygon: np.ndarray,
    draft: float,
    mass: float,
    center_of_gravity: np.ndarray,
    xi: np.ndarray,
    density: float = RHO,
    gravity: float = G,
    freeboard: float = DEFAULT_FREEBOARD,
) -> np.ndarray:
    """Buoyancy + weight force/moment (about the body origin) for the motion ``xi``.

    ``xi = (surge, sway, heave, roll, pitch, yaw)``; the rigid motion is
    ``X = R xi r + (xi1, xi2, xi3)``.  This is the oracle: it never uses the
    linearised expressions, it integrates the wetted polyhedron directly.
    """
    rotation = _rotation(xi)
    translated = [
        (rotation @ face.T).T + xi[0:3]
        for face in _prism_polyhedron(polygon, draft, freeboard)
    ]
    wetted = clip_below_waterplane(translated)
    volume, center_of_buoyancy = volume_and_centroid(wetted)

    buoyancy = np.array([0.0, 0.0, density * gravity * volume])
    weight = np.array([0.0, 0.0, -mass * gravity])
    center_of_gravity_space = rotation @ np.asarray(center_of_gravity, float) + xi[0:3]

    force = buoyancy + weight
    moment = np.cross(center_of_buoyancy, buoyancy) + np.cross(
        center_of_gravity_space, weight
    )
    return np.concatenate([force, moment])


def finite_difference_restoring_matrix(
    polygon: np.ndarray,
    draft: float,
    mass: float,
    center_of_gravity: np.ndarray,
    step: float = 2.0e-5,
    freeboard: float = DEFAULT_FREEBOARD,
) -> np.ndarray:
    """``C[i, j] = -d F_i / d xi_j`` from central differences of the exact force."""
    matrix = np.zeros((6, 6))
    for j in range(6):
        delta = np.zeros(6)
        delta[j] = step
        plus = generalized_hydrostatic_force(
            polygon, draft, mass, center_of_gravity, delta, freeboard=freeboard
        )
        minus = generalized_hydrostatic_force(
            polygon, draft, mass, center_of_gravity, -delta, freeboard=freeboard
        )
        matrix[:, j] = -(plus - minus) / (2.0 * step)
    return matrix


def equilibrium_center_of_gravity(
    polygon: np.ndarray, draft: float, z_g: float, freeboard: float = DEFAULT_FREEBOARD
) -> np.ndarray:
    """Centre of gravity on the vertical through the centre of buoyancy.

    The restoring matrix is defined about an equilibrium configuration; a
    horizontal CG offset leaves a constant moment imbalance that rotates with the
    yaw angle and contaminates the coupling entries.
    """
    wetted = clip_below_waterplane(_prism_polyhedron(polygon, draft, freeboard))
    _, center_of_buoyancy = volume_and_centroid(wetted)
    return np.array([center_of_buoyancy[0], center_of_buoyancy[1], z_g])


def relative_error(value: float, reference: float) -> float:
    scale = abs(reference)
    if scale == 0.0:
        return abs(value)
    return abs(value - reference) / scale


# --------------------------------------------------------------------------- #
# study cases
# --------------------------------------------------------------------------- #
CASES = {
    "symmetric_barge": dict(
        polygon=rectangle(-2.0, -1.0, 4.0, 2.0), draft=1.0, z_g=-0.35
    ),
    "eccentric_barge": dict(
        polygon=rectangle(1.0, 0.8, 4.0, 2.0), draft=1.0, z_g=-0.35
    ),
    "trapezoid_waterplane": dict(polygon=TRAPEZOID, draft=0.8, z_g=-0.45),
    "triangle_waterplane": dict(
        polygon=np.array([(0.0, 0.0), (2.5, 0.2), (1.6, 1.7)]), draft=0.9, z_g=-0.50
    ),
}

CASE_TITLES = {
    "symmetric_barge": "symmetric barge, origin at the waterplane centroid",
    "eccentric_barge": "rectangular barge offset 1.0 m / 0.8 m from the origin",
    "trapezoid_waterplane": "trapezoidal waterplane (no mirror symmetry)",
    "triangle_waterplane": "triangular waterplane (no mirror symmetry)",
}


def run_case(name: str, case: dict) -> dict:
    """Compare legacy, complete, closed-form and oracle matrices for one case."""
    polygon = case["polygon"]
    draft = case["draft"]
    area = polygon_area(polygon)
    volume = area * draft
    mass = RHO * volume
    hull = prism_hull(polygon, draft)
    cg = equilibrium_center_of_gravity(polygon, draft, case["z_g"])

    legacy = compute_hydrostatic_restoring_matrix(hull, mass, cg)
    full = compute_hydrostatic_restoring_matrix_full(hull, mass, cg)
    oracle = finite_difference_restoring_matrix(polygon, draft, mass, cg)

    x_c, y_c = full.waterplane_center
    i_xx_o, i_yy_o, i_xy_o = full.waterplane_moments_about_origin
    z_b = full.center_of_buoyancy[2]

    closed = np.zeros((6, 6))
    closed[2, 2] = RHO * G * area
    closed[2, 3] = closed[3, 2] = RHO * G * area * y_c
    closed[2, 4] = closed[4, 2] = -RHO * G * area * x_c
    closed[3, 3] = RHO * G * (i_xx_o + volume * z_b) - mass * G * cg[2]
    closed[4, 4] = RHO * G * (i_yy_o + volume * z_b) - mass * G * cg[2]
    closed[3, 4] = closed[4, 3] = -RHO * G * i_xy_o

    entries = []
    diagonal_stiffness = max(
        abs(full.restoring_matrix[2, 2]),
        abs(full.restoring_matrix[3, 3]),
        abs(full.restoring_matrix[4, 4]),
    )
    for i in range(6):
        for j in range(6):
            value = float(full.restoring_matrix[i, j])
            reference = float(closed[i, j])
            oracle_value = float(oracle[i, j])
            if value == 0.0 and reference == 0.0 and abs(oracle_value) < 1.0e-6 * diagonal_stiffness:
                continue
            entries.append(
                {
                    "row": MODE_NAMES[i],
                    "column": MODE_NAMES[j],
                    "complete": value,
                    "legacy": float(legacy.restoring_matrix[i, j]),
                    "closed_form": reference,
                    "oracle": oracle_value,
                    "error_vs_closed_form": relative_error(value, reference),
                    "error_vs_oracle": relative_error(oracle_value, value),
                    "legacy_relative_error": (
                        relative_error(float(legacy.restoring_matrix[i, j]), value)
                        if value != 0.0
                        else abs(float(legacy.restoring_matrix[i, j]))
                    ),
                }
            )

    return {
        "title": CASE_TITLES[name],
        "waterplane_area": area,
        "displaced_volume": volume,
        "mass": mass,
        "waterplane_center": [float(x_c), float(y_c)],
        "waterplane_moments_about_origin": [float(v) for v in (i_xx_o, i_yy_o, i_xy_o)],
        "center_of_buoyancy": [float(v) for v in full.center_of_buoyancy],
        "center_of_gravity": [float(v) for v in cg],
        "equilibrium_relative_mismatch": float(full.equilibrium_relative_mismatch),
        "diagonal_adequate": bool(full.diagonal_adequate),
        "notes": list(full.notes),
        "legacy_matrix": legacy.restoring_matrix.tolist(),
        "complete_matrix": full.restoring_matrix.tolist(),
        "coupling_terms": full.coupling_terms.tolist(),
        "oracle_matrix": oracle.tolist(),
        "entries": entries,
        "max_error_vs_closed_form": max(
            (entry["error_vs_closed_form"] for entry in entries), default=0.0
        ),
        "max_error_vs_oracle": max(
            (entry["error_vs_oracle"] for entry in entries), default=0.0
        ),
        "max_coupling_over_diagonal": float(
            np.max(np.abs(full.coupling_terms)) / diagonal_stiffness
        ),
    }


def rao_impact(case_name: str, case: dict, omegas: np.ndarray, wave_angle: float) -> dict:
    """RAO amplitudes with the legacy and with the complete restoring matrix."""
    polygon = case["polygon"]
    draft = case["draft"]
    volume = polygon_area(polygon) * draft
    mass = RHO * volume
    hull = prism_hull(polygon, draft)
    cg = equilibrium_center_of_gravity(polygon, draft, case["z_g"])

    legacy = compute_hydrostatic_restoring_matrix(hull, mass, cg)
    full = compute_hydrostatic_restoring_matrix_full(hull, mass, cg)
    full_props = full.as_hydrostatic_properties()

    rows = []
    for omega in omegas:
        legacy_result = compute_rao_at_frequency(
            hull=hull,
            omega=float(omega),
            wave_angle=wave_angle,
            mass=mass,
            center_of_gravity=cg,
            hydrostatic_props=legacy,
        )
        full_result = compute_rao_at_frequency(
            hull=hull,
            omega=float(omega),
            wave_angle=wave_angle,
            mass=mass,
            center_of_gravity=cg,
            hydrostatic_props=full_props,
        )
        row = {"omega": float(omega)}
        for index, mode in enumerate(MODE_NAMES):
            legacy_amplitude = float(legacy_result.rao_amplitude[index])
            full_amplitude = float(full_result.rao_amplitude[index])
            row[mode] = {
                "legacy": legacy_amplitude,
                "complete": full_amplitude,
                "relative_error": relative_error(legacy_amplitude, full_amplitude),
            }
        rows.append(row)

    worst = max(
        (
            (row[mode]["relative_error"], mode, row["omega"])
            for row in rows
            for mode in MODE_NAMES
        ),
        default=(0.0, "-", 0.0),
    )
    return {
        "case": case_name,
        "wave_angle": float(wave_angle),
        "omegas": [float(value) for value in omegas],
        "rows": rows,
        "worst_relative_error": worst[0],
        "worst_mode": worst[1],
        "worst_omega": worst[2],
    }


def build_study(omega_grid: np.ndarray | None = None) -> dict:
    """Run every case plus the RAO impact study and return a JSON-able payload."""
    cases = {name: run_case(name, case) for name, case in CASES.items()}

    if omega_grid is None:
        omega_grid = np.array([0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.4])
    impact = [
        rao_impact("eccentric_barge", CASES["eccentric_barge"], omega_grid, math.pi / 2.0),
        rao_impact(
            "trapezoid_waterplane", CASES["trapezoid_waterplane"], omega_grid, 0.0
        ),
    ]
    return {
        "density": RHO,
        "gravity": G,
        "finite_difference_step": 2.0e-5,
        "cases": cases,
        "rao_impact": impact,
    }


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def _matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def write_figures(study: dict, output: Path) -> list[Path]:
    plt = _matplotlib()
    written: list[Path] = []

    # 1. matrix maps: legacy vs complete vs coupling, for the asymmetric case
    name = "trapezoid_waterplane"
    case = study["cases"][name]
    matrices = [
        ("legacy diagonal matrix", np.array(case["legacy_matrix"])),
        ("complete matrix (audit fix)", np.array(case["complete_matrix"])),
        ("coupling terms (complete - legacy)", np.array(case["coupling_terms"])),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))
    for axis, (title, matrix) in zip(axes, matrices):
        scale = np.max(np.abs(matrix)) or 1.0
        image = axis.imshow(matrix / scale, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
        axis.set_title(title, fontsize=10)
        axis.set_xticks(range(6))
        axis.set_xticklabels(MODE_NAMES, rotation=45, fontsize=8)
        axis.set_yticks(range(6))
        axis.set_yticklabels(MODE_NAMES, fontsize=8)
        for i in range(6):
            for j in range(6):
                value = matrix[i, j] / scale
                if abs(value) > 1.0e-6:
                    axis.text(
                        j,
                        i,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=6,
                        color="black",
                    )
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.suptitle(
        "Hydrostatic restoring matrix: trapezoidal (asymmetric) waterplane, "
        "values normalised by the matrix maximum",
        fontsize=11,
    )
    fig.tight_layout()
    path = output / "restoring_matrix_maps.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(path)

    # 2. parity plot: complete matrix vs finite-difference oracle
    fig, axis = plt.subplots(figsize=(6.0, 5.6))
    markers = {"symmetric_barge": "o", "eccentric_barge": "s",
               "trapezoid_waterplane": "^", "triangle_waterplane": "D"}
    for name, case in study["cases"].items():
        values = [entry["complete"] for entry in case["entries"]]
        oracle = [entry["oracle"] for entry in case["entries"]]
        axis.plot(
            oracle,
            values,
            markers.get(name, "o"),
            markersize=4.5,
            fillstyle="none",
            label=name.replace("_", " "),
        )
    all_values = [abs(entry["oracle"]) for case in study["cases"].values()
                  for entry in case["entries"]]
    limit = max(all_values) * 3.0
    axis.plot([-limit, limit], [-limit, limit], "k-", linewidth=0.8, label="y = x")
    axis.set_xscale("symlog", linthresh=1.0e2)
    axis.set_yscale("symlog", linthresh=1.0e2)
    axis.set_xlabel("finite-difference oracle  [N/m, N/rad, N·m/m, N·m/rad]")
    axis.set_ylabel("complete matrix (implementation)")
    axis.set_title("Complete restoring matrix vs independent oracle\n"
                   "(all non-zero entries, four waterplanes)", fontsize=10)
    axis.grid(True, which="both", alpha=0.3)
    axis.legend(fontsize=7)
    fig.tight_layout()
    path = output / "restoring_matrix_oracle_parity.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(path)

    # 3. relative error of the legacy matrix, per entry, symmetric vs eccentric
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    for axis, case_name in zip(axes, ("symmetric_barge", "eccentric_barge")):
        case = study["cases"][case_name]
        labels = [
            f"C[{entry['row'][:2]},{entry['column'][:2]}]"
            for entry in case["entries"]
        ]
        errors = [max(entry["legacy_relative_error"], 1.0e-18) for entry in case["entries"]]
        positions = np.arange(len(labels))
        axis.bar(positions, errors, color="#b23c3c")
        axis.set_xticks(positions)
        axis.set_xticklabels(labels, rotation=60, fontsize=7)
        axis.set_yscale("log")
        axis.axhline(0.01, color="k", linestyle="--", linewidth=0.8, label="1 % bar")
        axis.set_ylabel("|legacy - complete| / |complete|")
        axis.set_title(case_name.replace("_", " "), fontsize=10)
        axis.legend(fontsize=8)
        axis.grid(True, axis="y", alpha=0.3)
    fig.suptitle(
        "Error of the diagonal-only matrix per entry (left: inside its domain, "
        "right: outside)",
        fontsize=11,
    )
    fig.tight_layout()
    path = output / "legacy_diagonal_error.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(path)

    # 4. RAO impact curves
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    for axis, block in zip(axes, study["rao_impact"]):
        omegas = [row["omega"] for row in block["rows"]]
        for mode, style in (("heave", "-o"), ("roll", "-s"), ("pitch", "-^")):
            axis.plot(
                omegas,
                [row[mode]["legacy"] for row in block["rows"]],
                style,
                markersize=4,
                fillstyle="none",
                label=f"{mode}: legacy C",
            )
            axis.plot(
                omegas,
                [row[mode]["complete"] for row in block["rows"]],
                style.replace("-", "--"),
                markersize=4,
                label=f"{mode}: complete C",
            )
        axis.set_yscale("log")
        axis.set_xlabel("wave circular frequency ω [rad/s]")
        axis.set_ylabel("RAO amplitude [m/m or rad/m]")
        axis.set_title(
            f"{block['case'].replace('_', ' ')} (wave angle {math.degrees(block['wave_angle']):.0f}°)",
            fontsize=10,
        )
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=6.5, ncol=2)
    fig.suptitle(
        "RAO chain: diagonal-only vs complete hydrostatic restoring matrix",
        fontsize=11,
    )
    fig.tight_layout()
    path = output / "rao_impact_coupling.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(path)

    # 5. waterplane geometry with the coupling levers
    fig, axis = plt.subplots(figsize=(6.4, 5.6))
    for name, case in CASES.items():
        polygon = np.vstack([case["polygon"], case["polygon"][0]])
        axis.plot(polygon[:, 0], polygon[:, 1], "-", linewidth=1.2,
                  label=name.replace("_", " "))
    for name in ("eccentric_barge", "trapezoid_waterplane"):
        centre = study["cases"][name]["waterplane_center"]
        axis.plot([0.0, centre[0]], [0.0, centre[1]], "k--", linewidth=0.8)
        axis.plot([centre[0]], [centre[1]], "k*", markersize=11)
        axis.annotate(
            f"$(x_c,y_c)$ = ({centre[0]:.2f}, {centre[1]:.2f}) m",
            xy=centre,
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=8,
        )
    axis.plot([0.0], [0.0], "ko", markersize=6)
    axis.annotate("reference origin O", xy=(0.0, 0.0), xytext=(6, -12),
                  textcoords="offset points", fontsize=8)
    axis.set_aspect("equal")
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_title(
        "Waterplanes used in the study; coupling levers from O to the centroid",
        fontsize=10,
    )
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=7)
    fig.tight_layout()
    path = output / "waterplane_geometry.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(path)

    return written


def write_markdown_tables(study: dict, output: Path) -> Path:
    """Emit the report tables straight from the measured payload."""
    lines: list[str] = []

    lines.append("### Case geometry and hydrostatic properties\n")
    lines.append(
        "| case | A [m²] | ∇ [m³] | x_c [m] | y_c [m] | I_xx,O [m⁴] | I_xy,O [m⁴] "
        "| z_B [m] | z_G [m] | mass mismatch | diagonal adequate |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for name, case in study["cases"].items():
        i_xx, i_yy, i_xy = case["waterplane_moments_about_origin"]
        lines.append(
            f"| `{name}` | {case['waterplane_area']:.4f} | {case['displaced_volume']:.4f} "
            f"| {case['waterplane_center'][0]:+.4f} | {case['waterplane_center'][1]:+.4f} "
            f"| {i_xx:.4f} | {i_xy:+.4f} | {case['center_of_buoyancy'][2]:+.4f} "
            f"| {case['center_of_gravity'][2]:+.4f} "
            f"| {case['equilibrium_relative_mismatch']:.1e} "
            f"| {case['diagonal_adequate']} |"
        )

    lines.append("\n### Verification of the complete matrix\n")
    lines.append("| case | max rel. error vs closed form | max rel. error vs oracle | "
                 "max \|coupling\| / max \|diagonal\| |")
    lines.append("|---|---|---|---|")
    for name, case in study["cases"].items():
        lines.append(
            f"| `{name}` | {case['max_error_vs_closed_form']:.2e} "
            f"| {case['max_error_vs_oracle']:.2e} "
            f"| {case['max_coupling_over_diagonal']:.3f} |"
        )

    for name in ("symmetric_barge", "eccentric_barge", "trapezoid_waterplane",
                 "triangle_waterplane"):
        case = study["cases"][name]
        lines.append(f"\n### Entries, `{name}` (N/m, N/rad, N·m/m, N·m/rad)\n")
        lines.append("| entry | complete | legacy (diagonal) | coupling | legacy rel. error |")
        lines.append("|---|---|---|---|---|")
        for entry in case["entries"]:
            lines.append(
                f"| C[{entry['row']},{entry['column']}] "
                f"| {entry['complete']:.6e} | {entry['legacy']:.6e} "
                f"| {entry['complete'] - entry['legacy']:+.6e} "
                f"| {entry['legacy_relative_error']:.3e} |"
            )

    for block in study["rao_impact"]:
        lines.append(
            f"\n### RAO impact, `{block['case']}` "
            f"(wave angle {math.degrees(block['wave_angle']):.0f}°)\n"
        )
        lines.append("| ω [rad/s] | heave legacy | heave complete | Δ | roll legacy "
                     "| roll complete | Δ | pitch legacy | pitch complete | Δ |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for row in block["rows"]:
            lines.append(
                f"| {row['omega']:.2f} "
                f"| {row['heave']['legacy']:.4e} | {row['heave']['complete']:.4e} "
                f"| {row['heave']['relative_error']:.2%} "
                f"| {row['roll']['legacy']:.4e} | {row['roll']['complete']:.4e} "
                f"| {row['roll']['relative_error']:.2%} "
                f"| {row['pitch']['legacy']:.4e} | {row['pitch']['complete']:.4e} "
                f"| {row['pitch']['relative_error']:.2%} |"
            )

    path = output / "tables.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/m2424"))
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)

    study = build_study()
    payload = args.output / "hydrostatic_coupling_study.json"
    payload.write_text(json.dumps(study, indent=2), encoding="utf-8")
    figures = write_figures(study, args.output)
    tables = write_markdown_tables(study, args.output)
    print(f"wrote {payload}, {tables} and {len(figures)} figures to {args.output}")
    for name, case in study["cases"].items():
        print(
            f"{name:24s} diagonal_adequate={case['diagonal_adequate']!s:5s} "
            f"max|C_coupling|/|C_diag|={case['max_coupling_over_diagonal']:.3f} "
            f"max_err_vs_closed={case['max_error_vs_closed_form']:.2e} "
            f"max_err_vs_oracle={case['max_error_vs_oracle']:.2e}"
        )
    for block in study["rao_impact"]:
        print(
            f"RAO {block['case']:24s} worst relative change "
            f"{block['worst_relative_error']:.3%} ({block['worst_mode']} at "
            f"omega = {block['worst_omega']:.2f} rad/s)"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())