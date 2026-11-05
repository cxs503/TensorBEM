"""Six-degree-of-freedom radiation-matrix algebra: reciprocity, reflection
selection rules and reference-centre covariance.

Why this benchmark exists
-------------------------
:func:`tensorbem.free_surface_solver3d.solve_radiation_3d` returns the *full*
``(6, 6)`` added-mass and damping matrices of a floating body,

    A_ij + (i/omega) B_ij = -rho * int_{Gamma_B} phi_j n_i dS,

where ``phi_j`` is the radiation potential of mode ``j``.  Every verification
shipped so far reads only *diagonal* entries (Hulme's ``A11``/``A33`` for the
hemisphere, Lamb's added masses for the ellipsoid) or compares two entries that
symmetry already makes equal.  Nothing checks the *algebraic structure* of the
matrix, so a wrong moment arm, a swapped ``i``/``j`` index or a sign error in the
rotational normals could survive a green build as long as the diagonals were
right.  Four checks close that gap; the first three are properties of the
continuous problem, the fourth is an exact algebraic identity.

1. **Reciprocity.**  The added-mass and damping matrices are symmetric,
   ``A_ij = A_ji`` and ``B_ij = B_ji``.  In this solver ``A_ij`` and ``A_ji``
   are *different discrete integrals* (``phi_j n_i`` versus ``phi_i n_j`` over
   different radiation solutions), so the equality tests the assembled
   operators rather than the storage layout.

2. **Reflection selection rule.**  For a hull invariant under
   ``S_x: x -> -x`` and ``S_y: y -> -y`` the generalized normal velocity of each
   mode is even or odd under each reflection (:data:`REFLECTION_SIGNS`).  With
   ``eps_j(R) = +-1`` denoting that parity,

       A_ij = int phi_j n_i = int_{R Gamma_B} phi_j(Rx) n_i(Rx) = eps_i eps_j A_ij,

   so ``A_ij`` (and ``B_ij``) vanishes whenever ``eps_i eps_j = -1``.  The
   vanishing pairs are :func:`selection_mask`.

3. **Spin-axis identity.**  For a body of revolution about the vertical axis the
   azimuthal normal velocity satisfies ``(z_hat x r) . n = 0`` on the hull,
   because the normal of a surface of revolution has no azimuthal component.
   Hence ``phi_yaw == 0`` and ``A_66 = B_66 = 0`` exactly (Lamb 1932 s.114 gives
   the same as the ``a = b`` limit of the spheroid formula).  Unlike (2) this is
   an analytical *value* the discretisation has to converge to.

4. **Reference-centre covariance.**  Moving the reference point ``r_0 -> r_0 + d``
   changes the generalized velocities to ``u' = M u`` with
   ``M = [[I, -[d]_x], [0, I]]``.  Added mass is a tensor of that pair of vector
   spaces, so the same fields must reappear as ``A' = M^{-T} A M^{-1}`` (and the
   same for ``B``) for *any* body, wavenumber and mesh.  This is the only check
   here whose tolerance is roundoff instead of discretisation error, and it
   exercises the moment arm, the ``i``/``j`` order and the ``Re`` / ``omega*Im``
   split of ``A`` and ``B`` at once.

What the benchmark does not claim
---------------------------------
The pairs :func:`same_class_couplings` -- surge-pitch about the waterplane centre
of a hemisphere, and its image sway-roll -- have ``eps_i eps_j = +1``, so no
reflection forbids them, while reciprocity forces them to be *symmetric*.  A
Timman-Newman reading of the transverse mirror plane would instead make them
antisymmetric and hence exactly zero, but that relation does not follow from the
two reflections used above, and the measured sequence decays like ``h**1.85``
towards zero: the data cannot separate "exactly zero, residual is discretisation
error" from "small physical value below the discretisation error".  They are
therefore *reported with their observed order*, not asserted to vanish.

Run::

    python -m tensorbem.benchmarks.six_dof_matrix_algebra --write
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tensorbem.free_surface_solver3d import rigid_body_normals_3d, solve_radiation_3d
from tensorbem.geometry3d import SurfaceMesh3D, hemisphere_hull_mesh

FloatArray = NDArray[np.float64]

#: Mode order used everywhere in TensorBEM: ``(surge, sway, heave, roll, pitch, yaw)``.
MODES: tuple[str, ...] = ("surge", "sway", "heave", "roll", "pitch", "yaw")

#: Parity ``(eps(S_x), eps(S_y))`` of each mode's generalized normal velocity on a
#: hull invariant under both reflections.  Translations flip with the axis they
#: point along; rotations carry the extra ``det(R) = -1`` of an axial vector,
#: which is why roll (about ``x``) is even under ``S_x`` while pitch is odd there.
REFLECTION_SIGNS: dict[str, tuple[int, int]] = {
    "surge": (-1, +1),
    "sway": (+1, -1),
    "heave": (+1, +1),
    "roll": (+1, -1),
    "pitch": (-1, +1),
    "yaw": (-1, -1),
}

#: Default hull: Hulme's floating hemisphere (unit radius, waterline on ``z = 0``)
#: is invariant under both reflections, so every prediction above applies.
DEFAULT_WAVENUMBER = 1.0
DEFAULT_DENSITY = 1025.0
DEFAULT_LEVELS: tuple[int, ...] = (1, 2, 3)
DEFAULT_OFFSET: tuple[float, float, float] = (0.3, 0.0, 0.25)
_REPO_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT = _REPO_ROOT / "artifacts" / "six_dof_matrix_algebra.json"
PINNED_DOC = _REPO_ROOT / "docs" / "six_dof_matrix_algebra.md"

DERIVATION_NOTE = (
    "For a reflection R with det R = -1 a polar velocity maps as v(Rx) = R v(x) "
    "while an axial vector maps as w -> -R w, so each mode maps onto +-itself. "
    "Reading the sign off v_j(Rx) = eps_j v_j(x) gives REFLECTION_SIGNS; "
    "tests/test_six_dof_matrix_algebra.py re-derives it numerically from the "
    "assembled generalized normals."
)


# --------------------------------------------------------------------------
# structure predicted by the symmetries
# --------------------------------------------------------------------------
def selection_mask() -> NDArray[np.bool_]:
    """Return the ``(6, 6)`` mask of entries a reflection forces to zero.

    ``mask[i, j]`` is ``True`` when modes ``i`` and ``j`` have opposite parity
    under at least one of the two reflections, i.e. when ``A_ij = B_ij = 0``.
    The diagonal is never masked.
    """
    mask = np.zeros((6, 6), dtype=bool)
    for i, name_i in enumerate(MODES):
        for j, name_j in enumerate(MODES):
            if i == j:
                continue
            si, sj = REFLECTION_SIGNS[name_i], REFLECTION_SIGNS[name_j]
            mask[i, j] = (si[0] * sj[0] < 0) or (si[1] * sj[1] < 0)
    return mask


def forbidden_pairs() -> list[tuple[int, int]]:
    """Return the ``(i, j)`` pairs (both orders, ``i != j``) that must vanish."""
    mask = selection_mask()
    return [(i, j) for i in range(6) for j in range(6) if mask[i, j]]


def same_class_couplings() -> tuple[tuple[int, int], ...]:
    """Return the off-diagonal pairs *permitted* by the reflections (``i < j``)."""
    mask = selection_mask()
    return tuple((i, j) for i in range(6) for j in range(i + 1, 6) if not mask[i, j])


# --------------------------------------------------------------------------
# reference-centre transformation
# --------------------------------------------------------------------------
def centre_shift_matrix(offset: ArrayLike) -> FloatArray:
    """Return ``M`` with ``u' = M u`` for a reference point moved by ``offset``.

    ``u = (V, Omega)`` are the generalized velocities at the original reference
    point.  At the shifted point the translation velocity is
    ``V + Omega x d = V - [d]_x Omega``.  Hydrodynamic forces transform
    contravariantly, so the tensors follow ``X' = M^{-T} X M^{-1}``
    (:func:`covariant`).
    """
    d = np.asarray(offset, dtype=float)
    if d.shape != (3,) or not np.isfinite(d).all():
        raise ValueError("offset must be a finite 3-vector")
    skew = np.array(
        [[0.0, -d[2], d[1]], [d[2], 0.0, -d[0]], [-d[1], d[0], 0.0]], dtype=float
    )
    mat = np.eye(6)
    mat[:3, 3:] = -skew
    return mat


def covariant(tensor: ArrayLike, offset: ArrayLike) -> FloatArray:
    """Return the ``(6, 6)`` tensor referred to a reference point moved by ``offset``."""
    inv = np.linalg.inv(centre_shift_matrix(offset))
    return inv.T @ np.asarray(tensor, dtype=float) @ inv


def covariance_residual(reference: ArrayLike, shifted: ArrayLike, offset: ArrayLike) -> float:
    """Relative mismatch ``max|shifted - covariant(reference, offset)| / max|shifted|``."""
    shifted_arr = np.asarray(shifted, dtype=float)
    predicted = covariant(reference, offset)
    scale = float(np.max(np.abs(shifted_arr)))
    if scale <= 0.0:
        raise ValueError("shifted tensor is identically zero")
    return float(np.max(np.abs(shifted_arr - predicted)) / scale)


# --------------------------------------------------------------------------
# numerical probes
# --------------------------------------------------------------------------
def reflection_parity_residuals(
    mesh: SurfaceMesh3D, center: ArrayLike | None = None
) -> dict[str, Any]:
    """Check the parity table against the assembled generalized normals.

    Returns the largest mirror mismatch of the panel centroids and, per mode,
    ``max |n_j(R x) - eps_j n_j(x)| / max |n_j|``.  A mesh that is not
    mirror-symmetric about an axis cannot support the corresponding prediction,
    which is why the mesh residual is reported and asserted separately.
    """
    centroids = np.asarray(mesh.centroids, dtype=float)
    normals = rigid_body_normals_3d(mesh, center=center)
    out: dict[str, Any] = {"mesh": {}, "modes": {}}
    for axis, sign_vec, slot in (
        ("x", np.array([-1.0, 1.0, 1.0]), 0),
        ("y", np.array([1.0, -1.0, 1.0]), 1),
    ):
        reflected = centroids * sign_vec
        dist = np.linalg.norm(centroids[None, :, :] - reflected[:, None, :], axis=2)
        nearest = np.argmin(dist, axis=1)
        out["mesh"][axis] = float(np.max(dist[np.arange(dist.shape[0]), nearest]))
        per_mode = {}
        for index, name in enumerate(MODES):
            eps = REFLECTION_SIGNS[name][slot]
            scale = float(np.max(np.abs(normals[:, index])))
            residual = np.max(np.abs(normals[nearest, index] - eps * normals[:, index]))
            per_mode[name] = float(residual / scale) if scale > 0.0 else 0.0
        out["modes"][axis] = per_mode
    return out


def _observed_order(
    coarse_h: float, fine_h: float, coarse_e: float, fine_e: float
) -> float | None:
    """Return ``log(e_coarse/e_fine) / log(h_coarse/h_fine)`` when it is defined."""
    if coarse_e <= 0.0 or fine_e <= 0.0 or coarse_h <= fine_h:
        return None
    return float(math.log(coarse_e / fine_e) / math.log(coarse_h / fine_h))


def _panel_size(mesh: SurfaceMesh3D) -> float:
    """Return a characteristic panel size ``sqrt(2 * total area / N)`` (an ``h`` proxy)."""
    return float(math.sqrt(2.0 * float(np.sum(mesh.areas)) / mesh.areas.size))


def _worst_forbidden(tensor: FloatArray, mask: NDArray[np.bool_]) -> dict[str, Any]:
    """Return the largest reflection-forbidden entry, named for the report."""
    scale = float(np.max(np.abs(np.diag(tensor))))
    magnitudes = np.abs(np.where(mask, tensor, 0.0))
    flat = int(np.argmax(magnitudes))
    i, j = divmod(flat, 6)
    return {
        "pair": f"{MODES[i]}-{MODES[j]}",
        "value": float(tensor[i, j]),
        "rel": float(magnitudes[i, j] / scale),
    }


def _block(tensor: FloatArray, mask: NDArray[np.bool_]) -> dict[str, Any]:
    scale = float(np.max(np.abs(np.diag(tensor))))
    return {
        "scale": scale,
        "reciprocity_rel": float(np.max(np.abs(tensor - tensor.T))) / scale,
        "forbidden_rel": _worst_forbidden(tensor, mask)["rel"] if mask.any() else 0.0,
    }


def metrics(mesh: SurfaceMesh3D, radiation: Any, wall_time: float = 0.0) -> dict[str, Any]:
    """Reduce one six-mode radiation solve to the numbers the checks read."""
    added_mass = np.asarray(radiation.added_mass, dtype=float)
    damping = np.asarray(radiation.damping, dtype=float)
    mask = selection_mask()
    block_a = _block(added_mass, mask)
    block_b = _block(damping, mask)
    same_class: dict[str, Any] = {}
    for i, j in same_class_couplings():
        scale = block_a["scale"]
        same_class[f"{MODES[i]}-{MODES[j]}"] = {
            "added_mass": [float(added_mass[i, j]), float(added_mass[j, i])],
            "damping": [float(damping[i, j]), float(damping[j, i])],
            "symmetric_part_rel": float(0.5 * (added_mass[i, j] + added_mass[j, i]) / scale),
            "antisymmetric_part_rel": float(
                0.5 * (added_mass[i, j] - added_mass[j, i]) / scale
            ),
        }
    return {
        "panels": int(mesh.centroids.shape[0]),
        "panel_size": _panel_size(mesh),
        "added_mass": added_mass.tolist(),
        "damping": damping.tolist(),
        "added_mass_block": block_a,
        "damping_block": block_b,
        "forbidden_worst_added_mass": _worst_forbidden(added_mass, mask),
        "forbidden_worst_damping": _worst_forbidden(damping, mask),
        "same_class": same_class,
        "axisymmetry_rel": float(abs(added_mass[0, 0] - added_mass[1, 1]) / block_a["scale"]),
        "spin_axis_added_mass_rel": float(abs(added_mass[5, 5]) / block_a["scale"]),
        "spin_axis_damping_rel": float(abs(damping[5, 5]) / block_b["scale"]),
        "condition_number": float(radiation.condition_number),
        "relative_residual": float(radiation.relative_residual),
        "wall_time_s": float(wall_time),
    }


def run_level(
    subdivisions: int,
    wavenumber: float = DEFAULT_WAVENUMBER,
    *,
    density: float = DEFAULT_DENSITY,
    chief_points: Any = "auto",
) -> dict[str, Any]:
    """Solve all six modes on one hull mesh and reduce the matrices to metrics."""
    mesh = hemisphere_hull_mesh(int(subdivisions))
    start = time.perf_counter()
    radiation = solve_radiation_3d(
        mesh, wavenumber, density=density, chief_points=chief_points
    )
    wall = time.perf_counter() - start
    row = metrics(mesh, radiation, wall)
    row["subdivisions"] = int(subdivisions)
    row["chief_points"] = int(radiation.chief_points_used.shape[0])
    row["parity"] = reflection_parity_residuals(mesh)
    return row


def run_covariance(
    subdivisions: int = 2,
    wavenumber: float = DEFAULT_WAVENUMBER,
    *,
    density: float = DEFAULT_DENSITY,
    offset: Sequence[float] = DEFAULT_OFFSET,
) -> dict[str, Any]:
    """Verify ``A' = M^{-T} A M^{-1}`` (and the same for ``B``) under a centre shift."""
    mesh = hemisphere_hull_mesh(int(subdivisions))
    base = solve_radiation_3d(mesh, wavenumber, density=density)
    moved = solve_radiation_3d(
        mesh, wavenumber, density=density, center=np.asarray(offset, dtype=float)
    )
    return {
        "subdivisions": int(subdivisions),
        "panels": int(mesh.centroids.shape[0]),
        "offset": [float(v) for v in offset],
        "added_mass_residual_rel": covariance_residual(
            base.added_mass, moved.added_mass, offset
        ),
        "damping_residual_rel": covariance_residual(base.damping, moved.damping, offset),
        "added_mass_diagonal_origin": np.diag(np.asarray(base.added_mass)).tolist(),
        "added_mass_diagonal_shifted": np.diag(np.asarray(moved.added_mass)).tolist(),
    }


def _order_series(levels: list[dict[str, Any]], series: list[float]) -> list[float | None]:
    return [
        None,
        *[
            _observed_order(
                levels[k - 1]["panel_size"],
                levels[k]["panel_size"],
                series[k - 1],
                series[k],
            )
            for k in range(1, len(series))
        ],
    ]


def run(
    subdivisions: Sequence[int] = DEFAULT_LEVELS,
    wavenumber: float = DEFAULT_WAVENUMBER,
    *,
    density: float = DEFAULT_DENSITY,
    covariance_subdivisions: int = 2,
    offset: Sequence[float] = DEFAULT_OFFSET,
) -> dict[str, Any]:
    """Run the whole matrix-algebra study and return a JSON-serialisable payload."""
    levels = [run_level(s, wavenumber, density=density) for s in subdivisions]
    orders: dict[str, Any] = {}
    same_class_orders: dict[str, Any] = {}
    if len(levels) >= 2:
        for key in ("reciprocity_rel", "forbidden_rel"):
            for block in ("added_mass_block", "damping_block"):
                orders[f"{block}.{key}"] = _order_series(
                    levels, [row[block][key] for row in levels]
                )
        for name in levels[0]["same_class"]:
            same_class_orders[name] = _order_series(
                levels,
                [abs(row["same_class"][name]["symmetric_part_rel"]) for row in levels],
            )
        orders["spin_axis.added_mass_rel"] = _order_series(
            levels, [row["spin_axis_added_mass_rel"] for row in levels]
        )
        orders["spin_axis.damping_rel"] = _order_series(
            levels, [row["spin_axis_damping_rel"] for row in levels]
        )
    return {
        "case": "floating hemisphere, unit radius, waterline on z=0 (Hulme geometry)",
        "wavenumber": float(wavenumber),
        "density": float(density),
        "modes": list(MODES),
        "reflection_signs": {k: list(v) for k, v in REFLECTION_SIGNS.items()},
        "derivation": DERIVATION_NOTE,
        "forbidden_pairs": [f"{MODES[i]}-{MODES[j]}" for i, j in forbidden_pairs()],
        "same_class_pairs": [f"{MODES[i]}-{MODES[j]}" for i, j in same_class_couplings()],
        "levels": levels,
        "observed_order": orders,
        "same_class_orders": same_class_orders,
        "covariance": run_covariance(
            covariance_subdivisions, wavenumber, density=density, offset=offset
        ),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }


# --------------------------------------------------------------------------
# documentation rendering
# --------------------------------------------------------------------------
def render_tables(payload: dict[str, Any]) -> str:
    """Render the markdown tables that ``docs/six_dof_matrix_algebra.md`` quotes."""
    lines = [
        "| subdiv | panels | h | cond | residual | max|A-A^T|/|A| | max|B-B^T|/|B| "
        "| forbidden A | forbidden B | A66/A33 | B66/B33 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in payload["levels"]:
        a, b = row["added_mass_block"], row["damping_block"]
        lines.append(
            f"| {row['subdivisions']} | {row['panels']} | {row['panel_size']:.4f} "
            f"| {row['condition_number']:.3g} | {row['relative_residual']:.1e} "
            f"| {a['reciprocity_rel']:.3e} | {b['reciprocity_rel']:.3e} "
            f"| {a['forbidden_rel']:.3e} | {b['forbidden_rel']:.3e} "
            f"| {row['spin_axis_added_mass_rel']:.3e} "
            f"| {row['spin_axis_damping_rel']:.3e} |"
        )
    orders = payload.get("observed_order", {})
    lines += [
        "",
        "| metric | observed order per refinement step |",
        "|---|---|",
    ]
    for key in (
        "added_mass_block.reciprocity_rel",
        "damping_block.reciprocity_rel",
        "added_mass_block.forbidden_rel",
        "damping_block.forbidden_rel",
        "spin_axis.added_mass_rel",
        "spin_axis.damping_rel",
    ):
        series = orders.get(key, [])
        rendered = ", ".join(
            "n/a" if value is None else f"{value:.2f}" for value in series
        )
        lines.append(f"| `{key}` | {rendered} |")
    same_class_orders = payload.get("same_class_orders", {})
    if same_class_orders:
        lines += ["", "| permitted coupling | observed order per refinement step |", "|---|---|"]
        for name, series in same_class_orders.items():
            rendered = ", ".join(
                "n/a" if value is None else f"{value:.2f}" for value in series
            )
            lines.append(f"| `{name}` (symmetric part) | {rendered} |")
    covariance = payload["covariance"]
    lines += [
        "",
        "| centre-shift covariance (offset = "
        f"{covariance['offset']}) | relative residual |",
        "|---|---|",
        f"| added mass, {covariance['panels']} panels "
        f"| {covariance['added_mass_residual_rel']:.2e} |",
        f"| damping, {covariance['panels']} panels "
        f"| {covariance['damping_residual_rel']:.2e} |",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: run the study, print the tables and optionally write the artifact."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--subdivisions",
        type=int,
        nargs="+",
        default=list(DEFAULT_LEVELS),
        help="mesh refinement levels (panel count is 4 * 4**subdivisions)",
    )
    parser.add_argument("--wavenumber", type=float, default=DEFAULT_WAVENUMBER)
    parser.add_argument("--covariance-subdivisions", type=int, default=2)
    parser.add_argument(
        "--write",
        nargs="?",
        const=str(ARTIFACT),
        default=None,
        help="write the JSON payload (default: artifacts/six_dof_matrix_algebra.json)",
    )
    args = parser.parse_args(argv)
    payload = run(
        tuple(args.subdivisions),
        args.wavenumber,
        covariance_subdivisions=args.covariance_subdivisions,
    )
    print(render_tables(payload))
    if args.write:
        target = Path(args.write)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {target}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())