"""Dual-route cross-check of the hydrostatic restoring matrix (pressure route).

:func:`tensorbem.hydrostatic.compute_hydrostatic_restoring_matrix` builds the
6x6 hydrostatic restoring matrix from *waterplane integrals*;
:mod:`tensorbem.hydrostatic_pressure_route` builds the same matrix from
*direct pressure integration over the wetted body surface*.  This module runs
both routes on the same facets and writes the comparison table, the mesh
refinement study and a figure.

Usage
-----
    python -m tensorbem.benchmarks.hydrostatic_pressure_route_study --json out.json \
        --markdown out.md --figure out.png

Everything printed/written by this CLI is generated from the run itself: no
number is copied by hand into the report.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from ..geometry3d import ellipsoid_mesh
from ..hydrostatic import compute_hydrostatic_restoring_matrix
from ..hydrostatic_pressure_route import (
    DOF_NAMES,
    clip_to_fluid,
    pressure_loads,
    pressure_profile,
    richardson_restoring_matrix,
    wetted_hull_from_closed,
)

DENSITY = 1025.0
GRAVITY = 9.80665


# ---------------------------------------------------------------------------
# body meshes
# ---------------------------------------------------------------------------
def closed_box_mesh(
    length: float = 10.0,
    beam: float = 4.0,
    draft: float = 2.0,
    freeboard: float = 0.5,
    *,
    n_length: int = 4,
    n_beam: int = 3,
    n_height: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Closed rectangular box, ``x`` in ``[-L/2, L/2]``, ``y`` in ``[-B/2, B/2]``,
    ``z`` in ``[-draft, +freeboard]``, outward normals.

    A rectangular box is the sharpest test body for the two routes: its wetted
    surface consists of planar triangles exactly on the analytic surface, so the
    pressure integral and the waterplane integral are both exact and the only
    residual is the finite-difference truncation.
    """
    for name, value in (
        ("length", length),
        ("beam", beam),
        ("draft", draft),
        ("freeboard", freeboard),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be a positive finite number")
    vertices: list[np.ndarray] = []
    faces: list[tuple[int, int, int]] = []

    def patch(origin, e1, e2, nu, nv, outward):
        base = len(vertices)
        for i in range(nu + 1):
            for j in range(nv + 1):
                vertices.append(
                    np.asarray(origin, dtype=float)
                    + (i / nu) * np.asarray(e1, dtype=float)
                    + (j / nv) * np.asarray(e2, dtype=float)
                )
        out = np.asarray(outward, dtype=float)
        for i in range(nu):
            for j in range(nv):
                a = base + i * (nv + 1) + j
                b = a + 1
                c = a + (nv + 1)
                d = c + 1
                for tri in ((a, b, d), (a, d, c)):
                    p0, p1, p2 = (vertices[k] for k in tri)
                    normal = np.cross(p1 - p0, p2 - p0)
                    faces.append(tri if float(np.dot(normal, out)) > 0.0 else (tri[0], tri[2], tri[1]))

    height = draft + freeboard
    patch((-length / 2, -beam / 2, -draft), (length, 0, 0), (0, beam, 0), n_length, n_beam, (0, 0, -1))
    patch((-length / 2, -beam / 2, freeboard), (length, 0, 0), (0, beam, 0), n_length, n_beam, (0, 0, 1))
    patch((-length / 2, -beam / 2, -draft), (length, 0, 0), (0, 0, height), n_length, n_height, (0, -1, 0))
    patch((-length / 2, beam / 2, -draft), (length, 0, 0), (0, 0, height), n_length, n_height, (0, 1, 0))
    patch((-length / 2, -beam / 2, -draft), (0, beam, 0), (0, 0, height), n_beam, n_height, (-1, 0, 0))
    patch((length / 2, -beam / 2, -draft), (0, beam, 0), (0, 0, height), n_beam, n_height, (1, 0, 0))
    return np.asarray(vertices, dtype=float), np.asarray(faces, dtype=np.int64)


@dataclass(frozen=True)
class CrossCheckCase:
    name: str
    vertices: np.ndarray
    faces: np.ndarray
    center_of_gravity: tuple[float, float, float]
    note: str


def _translate(vertices: np.ndarray, offset: Sequence[float]) -> np.ndarray:
    return np.asarray(vertices, dtype=float) + np.asarray(offset, dtype=float)[None, :]


def build_cases() -> list[CrossCheckCase]:
    """The three verification bodies, all with the waterline at ``z = 0``."""
    barge_v, barge_f = closed_box_mesh(10.0, 4.0, 2.0, 0.5)
    ell = ellipsoid_mesh(3, np.array([5.0, 2.0, 1.5]))
    return [
        CrossCheckCase(
            name="barge-L10B4T2",
            vertices=barge_v,
            faces=barge_f,
            center_of_gravity=(0.0, 0.0, -1.5),
            note="planar wall-sided barge, symmetric waterplane, low centre of gravity",
        ),
        CrossCheckCase(
            name="spheroid-a5b2c1.5",
            vertices=ell.vertices,
            faces=ell.faces,
            center_of_gravity=(0.0, 0.0, 0.0),
            note="half-submerged ellipsoid: curved waterplane, non-degenerate roll/pitch",
        ),
        CrossCheckCase(
            name="spheroid-offset",
            vertices=_translate(ell.vertices, (1.2, 0.7, 0.0)),
            faces=ell.faces,
            center_of_gravity=(1.5, 1.0, 0.0),
            note="same spheroid displaced off the origin: non-zero waterplane first "
            "moments and a centre of gravity off the rotation axis",
        ),
    ]


# ---------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------
def _format_matrix(matrix: np.ndarray) -> list[list[float]]:
    return [[float(value) for value in row] for row in np.asarray(matrix)]


def compare_case(case: CrossCheckCase, *, steps: Sequence[float] = (2e-3, 1e-3, 5e-4)) -> dict:
    hull = wetted_hull_from_closed(case.vertices, case.faces)
    pressure = richardson_restoring_matrix(
        case.vertices,
        case.faces,
        center_of_gravity=case.center_of_gravity,
        density=DENSITY,
        gravity=GRAVITY,
        steps=steps,
    )
    waterplane = compute_hydrostatic_restoring_matrix(
        hull,
        mass=DENSITY * pressure.immersed_volume,
        center_of_gravity=np.asarray(case.center_of_gravity, dtype=float),
        density=DENSITY,
        gravity=GRAVITY,
    )
    c_pressure = np.asarray(pressure.matrix, dtype=float)
    c_waterplane = np.asarray(waterplane.restoring_matrix, dtype=float)
    scale = float(np.abs(c_waterplane).max())
    absolute = np.abs(c_pressure - c_waterplane)
    relative = absolute / np.maximum(np.abs(c_waterplane), 1e-300)
    # an entry is reported relatively only when it is a real entry of C, not
    # floating-point dust around an exactly zero coupling
    significant = np.abs(c_waterplane) > 1e-6 * scale
    worst = float(np.max(np.where(significant, relative, absolute / scale)))
    return {
        "name": case.name,
        "note": case.note,
        "waterplane_area": float(waterplane.waterplane_area),
        "displaced_volume_pressure": float(pressure.immersed_volume),
        "displaced_volume_waterplane": float(waterplane.displaced_volume),
        "center_of_buoyancy_pressure": [float(v) for v in pressure.center_of_buoyancy],
        "center_of_buoyancy_waterplane": [float(v) for v in waterplane.center_of_buoyancy],
        "matrix_pressure": _format_matrix(c_pressure),
        "matrix_waterplane": _format_matrix(c_waterplane),
        "absolute_difference": _format_matrix(absolute),
        "relative_difference": _format_matrix(relative),
        "significant": [[bool(v) for v in row] for row in significant],
        "max_relative_difference": worst,
        "diagonal_relative": {
            DOF_NAMES[k]: float(relative[k, k]) for k in range(6)
        },
    }


def pressure_profile_check(
    *, draft: float = 2.0, count: int = 20
) -> dict:
    """``p(z) = rho*g*(T - z)`` against the closed form at ``count`` depths."""
    depths = np.linspace(-draft, 0.0, count)
    points = np.column_stack([np.zeros(count), np.zeros(count), depths])
    numeric = pressure_profile(points, density=DENSITY, gravity=GRAVITY)
    analytic = DENSITY * GRAVITY * (0.0 - depths)
    relative = np.abs(numeric - analytic) / np.maximum(np.abs(analytic), 1e-300)
    return {
        "count": int(count),
        "depths": [float(z) for z in depths],
        "pressure": [float(p) for p in numeric],
        "analytic": [float(p) for p in analytic],
        "max_relative_error": float(relative.max()),
    }


def refinement_study(
    semi_axes: Sequence[float] = (5.0, 2.0, 1.5),
    subdivisions: Sequence[int] = (2, 3, 4),
) -> list[dict]:
    rows = []
    for sub in subdivisions:
        mesh = ellipsoid_mesh(int(sub), np.asarray(semi_axes, dtype=float))
        case = CrossCheckCase(
            name=f"spheroid-sub{sub}",
            vertices=mesh.vertices,
            faces=mesh.faces,
            center_of_gravity=(0.0, 0.0, 0.0),
            note="refinement",
        )
        row = compare_case(case)
        rows.append(
            {
                "subdivisions": int(sub),
                "panel_count": int(len(mesh.faces)),
                "max_relative_difference": row["max_relative_difference"],
                "heave_relative": row["diagonal_relative"]["heave"],
                "roll_relative": row["diagonal_relative"]["roll"],
                "pitch_relative": row["diagonal_relative"]["pitch"],
            }
        )
    return rows


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------
def _markdown_table(matrix: np.ndarray, fmt: str = "{:.3e}") -> str:
    header = "| row \\ col | " + " | ".join(DOF_NAMES) + " |\n"
    header += "|---" * 7 + "|\n"
    lines = [header]
    for i, name in enumerate(DOF_NAMES):
        lines.append(
            "| **" + name + "** | " + " | ".join(fmt.format(v) for v in matrix[i]) + " |\n"
        )
    return "".join(lines)


def build_report(payload: dict) -> str:
    lines = [
        "# Hydrostatic restoring matrix: pressure route vs waterplane route\n\n",
        "Generated by `python -m tensorbem.benchmarks.hydrostatic_pressure_route_study`. "
        "Every number below is emitted by that run (no hand-copied value).\n\n",
        f"density = {payload['density']} kg/m3, gravity = {payload['gravity']} m/s2, "
        f"Richardson steps = {payload['steps']} (m / rad).\n\n",
        "## Hydrostatic pressure profile (analytic check)\n\n",
        f"20 depths from z = -2 m to z = 0 m: max relative error "
        f"**{payload['pressure_profile']['max_relative_error']:.3e}** against "
        "`p = rho*g*(z_fs - z)`.\n\n",
    ]
    for case in payload["cases"]:
        lines.append(f"## {case['name']}\n\n{case['note']}\n\n")
        lines.append(
            f"- displaced volume: pressure route {case['displaced_volume_pressure']:.6f} m3, "
            f"waterplane route {case['displaced_volume_waterplane']:.6f} m3\n"
        )
        lines.append(
            f"- centre of buoyancy: pressure {np.round(case['center_of_buoyancy_pressure'][:2], 6).tolist()}, "
            f"waterplane {np.round(case['center_of_buoyancy_waterplane'][:2], 6).tolist()}\n"
        )
        lines.append(
            f"- **max element-wise relative difference {case['max_relative_difference']:.3e}**\n\n"
        )
        lines.append("### C from the pressure route\n\n")
        lines.append(_markdown_table(np.asarray(case["matrix_pressure"])))
        lines.append("\n### C from the waterplane route\n\n")
        lines.append(_markdown_table(np.asarray(case["matrix_waterplane"])))
        lines.append("\n### absolute difference\n\n")
        lines.append(_markdown_table(np.asarray(case["absolute_difference"])))
        lines.append("\n")
    lines.append("## Mesh refinement (half-submerged spheroid)\n\n")
    lines.append("| subdivisions | panels | max rel diff | heave | roll | pitch |\n")
    lines.append("|---|---|---|---|---|---|\n")
    for row in payload["refinement"]:
        lines.append(
            f"| {row['subdivisions']} | {row['panel_count']} | "
            f"{row['max_relative_difference']:.3e} | {row['heave_relative']:.3e} | "
            f"{row['roll_relative']:.3e} | {row['pitch_relative']:.3e} |\n"
        )
    return "".join(lines)


def make_figure(payload: dict, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cases = payload["cases"]
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    labels = [f"{i}-{j}" for i in range(6) for j in range(6)]
    width = 0.8 / len(cases)
    for k, case in enumerate(cases):
        relative = np.asarray(case["relative_difference"], dtype=float)
        significant = np.asarray(case["significant"], dtype=bool)
        scale = np.abs(np.asarray(case["matrix_waterplane"], dtype=float)).max()
        values = np.where(significant, relative, np.abs(np.asarray(case["absolute_difference"])) / scale)
        axes[0].bar(
            np.arange(36) + k * width,
            np.maximum(values.ravel(), 1e-18),
            width=width,
            label=case["name"],
        )
    axes[0].set_yscale("log")
    axes[0].axhline(1e-10, color="k", lw=0.8, ls="--", label="round-off floor 1e-10")
    axes[0].set_xticks(np.arange(36))
    axes[0].set_xticklabels(labels, rotation=90, fontsize=6)
    axes[0].set_ylabel("|C_pressure - C_waterplane| / |C_waterplane|")
    axes[0].set_title("element-wise route disagreement, 6x6 C")
    axes[0].legend(fontsize=7)

    rows = payload["refinement"]
    subs = [row["subdivisions"] for row in rows]
    axes[1].loglog(subs, [row["max_relative_difference"] for row in rows], "o-", label="max rel diff")
    axes[1].loglog(subs, [row["roll_relative"] for row in rows], "s--", label="roll")
    axes[1].loglog(subs, [row["heave_relative"] for row in rows], "^:", label="heave")
    axes[1].set_xlabel("icosphere subdivisions")
    axes[1].set_ylabel("relative difference")
    axes[1].set_title("mesh refinement, half-submerged spheroid")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def run(
    *,
    json_path: Optional[str] = None,
    markdown_path: Optional[str] = None,
    figure_path: Optional[str] = None,
    steps: Sequence[float] = (2e-3, 1e-3, 5e-4),
) -> dict:
    cases = [compare_case(case, steps=steps) for case in build_cases()]
    payload = {
        "density": DENSITY,
        "gravity": GRAVITY,
        "steps": [float(s) for s in steps],
        "cases": cases,
        "pressure_profile": pressure_profile_check(),
        "refinement": refinement_study(),
        "max_relative_difference": max(case["max_relative_difference"] for case in cases),
    }
    if json_path:
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
    if markdown_path:
        with open(markdown_path, "w", encoding="utf-8") as handle:
            handle.write(build_report(payload))
    if figure_path:
        make_figure(payload, figure_path)
    return payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", dest="json_path", default=None)
    parser.add_argument("--markdown", dest="markdown_path", default=None)
    parser.add_argument("--figure", dest="figure_path", default=None)
    parser.add_argument("--steps", dest="steps", default="2e-3,1e-3,5e-4")
    args = parser.parse_args(argv)
    steps = tuple(float(s) for s in args.steps.split(",") if s.strip())
    payload = run(
        json_path=args.json_path,
        markdown_path=args.markdown_path,
        figure_path=args.figure_path,
        steps=steps,
    )
    print(f"max relative difference pressure vs waterplane route: {payload['max_relative_difference']:.3e}")
    for case in payload["cases"]:
        print(
            f"  {case['name']:<20s} max rel diff {case['max_relative_difference']:.3e} "
            f"(heave {case['diagonal_relative']['heave']:.2e}, roll {case['diagonal_relative']['roll']:.2e}, "
            f"pitch {case['diagonal_relative']['pitch']:.2e})"
        )
    print(
        "pressure profile max rel error: "
        f"{payload['pressure_profile']['max_relative_error']:.3e}"
    )
    for row in payload["refinement"]:
        print(
            f"  refinement sub={row['subdivisions']} panels={row['panel_count']} "
            f"max rel diff {row['max_relative_difference']:.3e}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())