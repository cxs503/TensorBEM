"""Mixed Dirichlet/Neumann boundary-element solves on multiply connected 2-D domains.

The existing :mod:`tensorbem.solver` module solves the *pure Dirichlet* interior
problem on a single closed, counter-clockwise polygon.  ``docs/verification.md``
lists three capabilities as still missing (2026-10-24 audit):

* multiply connected domains (more than one boundary loop),
* mixed Dirichlet/Neumann boundary conditions,
* pure Neumann problems (compatibility condition and constant gauge).

This module closes exactly those three gaps for the interior 2-D Laplace
problem, with the same piecewise-constant panels and analytic straight-panel
integrals that the rest of the package uses.  The boundary integral equation at
a panel midpoint is

.. math::

    \\tfrac12 u_i - \\sum_j H_{ij} u_j + \\sum_j S_{ij} q_j = 0,

with :math:`S` the single-layer and :math:`H` the double-layer matrix
(:func:`tensorbem.solver.assemble_matrices` uses the identical convention).
Panels on which :math:`u` is prescribed (Dirichlet) carry an unknown flux, and
panels on which :math:`q = \\partial u/\\partial n` is prescribed (Neumann) carry
an unknown potential, so the assembled system is square.

Conventions
-----------
Normals follow the ``PanelMesh`` rule ``n = (t_y, -t_x)`` for an edge directed
along its tangent ``t``.  Each loop must therefore be ordered counter-clockwise
for an outer boundary and clockwise for a hole, so that every normal points out
of the computational domain.

Pure Neumann
------------
With no Dirichlet panel the system matrix :math:`\\tfrac12 I - H` is singular:
the constant vector is in its null space (row sums vanish).  The data are
solvable only if :math:`\\oint q\\,\\mathrm{d}s = 0`, and the potential is then
unique only up to a constant.  :func:`solve_mixed` returns the minimum-norm,
zero-gauge representative together with the rank deficiency and the
compatibility residual so the caller can check both conditions instead of
trusting the solve.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from .geometry import PanelMesh
from .kernels import (
    REFERENCE_LENGTH,
    integrate_double_layer_panel,
    integrate_single_layer_panel,
    validate_reference_length,
)

FloatArray = NDArray[np.float64]
# Singular values below this fraction of the largest one are treated as zero
# when the pure-Neumann system is solved and when the rank is reported.
_SINGULAR_RTOL = 1.0e-10


@dataclass(frozen=True)
class PanelSet:
    """Straight constant panels of one or more closed boundary loops.

    ``starts[j]`` is the first endpoint of panel ``j``, ``tangents[j]`` its unit
    direction, ``normals[j]`` the unit normal pointing out of the computational
    domain, and ``loop[j]`` the index of the boundary loop it belongs to.
    """

    starts: FloatArray
    tangents: FloatArray
    normals: FloatArray
    lengths: FloatArray
    loop: NDArray[np.int_]

    def __post_init__(self) -> None:
        starts = np.asarray(self.starts, dtype=float)
        tangents = np.asarray(self.tangents, dtype=float)
        normals = np.asarray(self.normals, dtype=float)
        lengths = np.asarray(self.lengths, dtype=float)
        loop = np.asarray(self.loop, dtype=int)
        n = len(lengths)
        if n < 3:
            raise ValueError("a boundary needs at least three panels")
        for name, array in (
            ("starts", starts), ("tangents", tangents), ("normals", normals)
        ):
            if array.shape != (n, 2):
                raise ValueError(f"{name} must have shape ({n}, 2)")
            if not np.isfinite(array).all():
                raise ValueError(f"{name} must all be finite")
        if loop.shape != (n,):
            raise ValueError(f"loop must have shape ({n},)")
        if not np.isfinite(lengths).all() or np.any(lengths <= 0.0):
            raise ValueError("panel lengths must be positive and finite")
        object.__setattr__(self, "starts", starts)
        object.__setattr__(self, "tangents", tangents)
        object.__setattr__(self, "normals", normals)
        object.__setattr__(self, "lengths", lengths)
        object.__setattr__(self, "loop", loop)

    @property
    def panel_count(self) -> int:
        return int(len(self.lengths))

    @property
    def centers(self) -> FloatArray:
        return self.starts + 0.5 * self.lengths[:, None] * self.tangents

    @property
    def perimeter(self) -> float:
        return float(np.sum(self.lengths))

    @classmethod
    def from_panels(
        cls,
        starts: Sequence[Sequence[float]],
        tangents: Sequence[Sequence[float]],
        normals: Sequence[Sequence[float]],
        lengths: Sequence[float],
        loop: Sequence[int] | None = None,
    ) -> "PanelSet":
        n = len(lengths)
        loop_index = np.zeros(n, dtype=int) if loop is None else np.asarray(loop, int)
        return cls(
            starts=np.asarray(starts, float),
            tangents=np.asarray(tangents, float),
            normals=np.asarray(normals, float),
            lengths=np.asarray(lengths, float),
            loop=loop_index,
        )

    @classmethod
    def from_mesh(cls, mesh: PanelMesh) -> "PanelSet":
        """Wrap a single closed counter-clockwise :class:`PanelMesh`."""
        tangents = mesh.vectors / mesh.lengths[:, None]
        return cls(
            starts=mesh.vertices,
            tangents=tangents,
            normals=mesh.normals,
            lengths=mesh.lengths,
            loop=np.zeros(mesh.panel_count, dtype=int),
        )

    @classmethod
    def from_loops(cls, loops: Iterable[Sequence[Sequence[float]]]) -> "PanelSet":
        """Build a multiply connected boundary from closed vertex loops.

        A loop's vertices define its segments implicitly (the last vertex closes
        back to the first).  Order each loop counter-clockwise for an outer
        boundary and clockwise for a hole.
        """
        starts: list[FloatArray] = []
        tangents: list[FloatArray] = []
        normals: list[FloatArray] = []
        lengths: list[float] = []
        loop_index: list[int] = []
        count = 0
        for k, raw in enumerate(loops):
            vertices = np.asarray(raw, dtype=float)
            if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 3:
                raise ValueError("each loop needs at least three (x, y) vertices")
            if not np.isfinite(vertices).all():
                raise ValueError("loop vertices must all be finite")
            ends = np.roll(vertices, -1, axis=0)
            edges = ends - vertices
            seg_lengths = np.linalg.norm(edges, axis=1)
            if np.any(seg_lengths <= 1.0e-14):
                raise ValueError("zero-length boundary panels are not allowed")
            unit = edges / seg_lengths[:, None]
            starts.extend(vertices)
            tangents.extend(unit)
            normals.extend(np.column_stack((unit[:, 1], -unit[:, 0])))
            lengths.extend(seg_lengths)
            loop_index.extend([k] * len(vertices))
            count += len(vertices)
        if count < 3:
            raise ValueError("a boundary needs at least three panels")
        return cls.from_panels(starts, tangents, normals, lengths, loop_index)


def circle_loop(
    panel_count: int, radius: float = 1.0, *, clockwise: bool = False
) -> FloatArray:
    """Vertices of a regular polygon inscribed in a circle of ``radius``.

    Counter-clockwise (outward normals) by default; pass ``clockwise=True`` for
    a hole boundary whose normals must point inward, towards the hole.
    """
    count = int(panel_count)
    if count < 3:
        raise ValueError("panel_count must be at least 3")
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be a positive finite number")
    theta = 2.0 * np.pi * np.arange(count) / count
    if clockwise:
        theta = -theta
    return radius * np.column_stack((np.cos(theta), np.sin(theta)))


def assemble_panels(
    panels: PanelSet, reference_length: float = REFERENCE_LENGTH
) -> tuple[FloatArray, FloatArray]:
    """Assemble analytic single- and double-layer matrices for ``panels``."""
    reference = validate_reference_length(reference_length)
    n = panels.panel_count
    single = np.empty((n, n), dtype=float)
    double = np.empty((n, n), dtype=float)
    for i, target in enumerate(panels.centers):
        for j in range(n):
            single[i, j] = integrate_single_layer_panel(
                target,
                panels.starts[j],
                panels.tangents[j],
                float(panels.lengths[j]),
                reference,
            )
            double[i, j] = integrate_double_layer_panel(
                target,
                panels.starts[j],
                panels.tangents[j],
                panels.normals[j],
                float(panels.lengths[j]),
            )
    return single, double


@dataclass(frozen=True)
class MixedBoundaryResult:
    """Boundary traces, system diagnostics and matrices from a mixed solve."""

    boundary_values: FloatArray
    normal_derivative: FloatArray
    dirichlet_panels: NDArray[np.int_]
    neumann_panels: NDArray[np.int_]
    system_matrix: FloatArray
    single_layer_matrix: FloatArray
    double_layer_matrix: FloatArray
    condition_number: float
    residual_l2: float
    relative_residual_l2: float
    rank_deficiency: int
    smallest_singular_value: float
    largest_singular_value: float
    solvability_residual: float | None = None
    gauge_removed: float = 0.0
    reference_length: float = REFERENCE_LENGTH
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def pure_neumann(self) -> bool:
        return len(self.dirichlet_panels) == 0



def evaluate_field(
    panels: PanelSet,
    boundary_values: FloatArray,
    normal_derivative: FloatArray,
    points: Sequence[Sequence[float]] | FloatArray,
    reference_length: float = REFERENCE_LENGTH,
) -> FloatArray:
    """Reconstruct the potential at interior points from its boundary traces.

    Uses the representation formula without the free-term coefficient, valid for
    points strictly inside the computational domain (the annular region between
    the loops, for a multiply connected boundary).
    """
    reference = validate_reference_length(reference_length)
    u = np.asarray(boundary_values, dtype=float)
    q = np.asarray(normal_derivative, dtype=float)
    n = panels.panel_count
    if u.shape != (n,) or q.shape != (n,):
        raise ValueError(f"boundary traces must have shape ({n},)")
    targets = np.atleast_2d(np.asarray(points, dtype=float))
    if targets.ndim != 2 or targets.shape[1] != 2:
        raise ValueError("points must have shape (m, 2)")
    values = np.empty(len(targets), dtype=float)
    for k, point in enumerate(targets):
        single = np.empty(n, dtype=float)
        double = np.empty(n, dtype=float)
        for j in range(n):
            single[j] = integrate_single_layer_panel(
                point,
                panels.starts[j],
                panels.tangents[j],
                float(panels.lengths[j]),
                reference,
            )
            double[j] = integrate_double_layer_panel(
                point,
                panels.starts[j],
                panels.tangents[j],
                panels.normals[j],
                float(panels.lengths[j]),
            )
        values[k] = float(single @ q - double @ u)
    return values


def _panels_from_mapping(
    mapping: Mapping[int, float] | None, n: int, name: str
) -> dict[int, float]:
    if mapping is None:
        return {}
    indices: dict[int, float] = {}
    for key, value in mapping.items():
        index = int(key)
        if not 0 <= index < n:
            raise ValueError(f"{name} panel index {index} is outside 0..{n - 1}")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{name} value on panel {index} must be finite")
        indices[index] = number
    return indices


def solve_mixed(
    panels: PanelSet,
    dirichlet: Mapping[int, float] | None = None,
    neumann: Mapping[int, float] | None = None,
    reference_length: float = REFERENCE_LENGTH,
    matrices: tuple[FloatArray, FloatArray] | None = None,
) -> MixedBoundaryResult:
    """Solve the interior Laplace problem with mixed or pure-Neumann data.

    ``dirichlet`` maps panel index to prescribed potential, ``neumann`` maps panel
    index to prescribed outward flux.  Every panel must appear in exactly one of
    the two mappings.  Passing an empty ``dirichlet`` solves a pure Neumann
    problem: the constant gauge is fixed by removing the mean and both the rank
    deficiency and the compatibility residual are reported instead of hidden.

    ``matrices`` optionally supplies a precomputed ``(single, double)`` pair from
    :func:`assemble_panels` for the same ``panels`` and ``reference_length``, so a
    convergence study does not reassemble the same matrices for every data set.
    """
    reference = validate_reference_length(reference_length)
    n = panels.panel_count
    fixed_values = _panels_from_mapping(dirichlet, n, "dirichlet")
    fixed_flux = _panels_from_mapping(neumann, n, "neumann")
    overlap = set(fixed_values) & set(fixed_flux)
    if overlap:
        raise ValueError(
            f"panels {sorted(overlap)} carry both Dirichlet and Neumann data"
        )
    missing = sorted(set(range(n)) - set(fixed_values) - set(fixed_flux))
    if missing:
        raise ValueError(f"panels {missing} carry neither Dirichlet nor Neumann data")

    d_index = np.array(sorted(fixed_values), dtype=int)
    n_index = np.array(sorted(fixed_flux), dtype=int)
    g_d = np.array([fixed_values[i] for i in d_index], dtype=float)
    q_n = np.array([fixed_flux[i] for i in n_index], dtype=float)

    if matrices is None:
        single, double = assemble_panels(panels, reference_length=reference)
    else:
        single = np.asarray(matrices[0], dtype=float)
        double = np.asarray(matrices[1], dtype=float)
        if single.shape != (n, n) or double.shape != (n, n):
            raise ValueError(f"supplied matrices must have shape ({n}, {n})")
    identity = np.eye(n)
    # Equation at every panel midpoint, in the package convention
    # (0.5 I + H) u - S q = 0 with H the analytic double-layer matrix.
    potential_operator = 0.5 * identity + double
    system = np.empty((n, n), dtype=float)
    system[:, : len(n_index)] = potential_operator[:, n_index]
    system[:, len(n_index):] = -single[:, d_index]
    rhs = -(potential_operator[:, d_index] @ g_d) + (single[:, n_index] @ q_n)

    singular_values = np.linalg.svd(system, compute_uv=False)
    largest = float(singular_values[0]) if len(singular_values) else 0.0
    threshold = _SINGULAR_RTOL * largest
    rank_deficiency = int(np.sum(singular_values <= threshold))
    condition = float(largest / singular_values[-1]) if singular_values[-1] > 0 else math.inf
    notes: list[str] = []
    solvability: float | None = None
    gauge = 0.0

    if rank_deficiency == 0:
        solution = np.linalg.solve(system, rhs)
    else:
        if len(d_index):
            raise ValueError(
                "mixed system is numerically singular "
                f"(deficiency {rank_deficiency}) although Dirichlet data are given; "
                "the boundary loops may be degenerate"
            )
        if rank_deficiency != 1:
            notes.append(
                f"unexpected rank deficiency {rank_deficiency} for a pure Neumann solve"
            )
        # Right null space (constants in exact arithmetic) and left null space
        # (the discrete indicator of the compatibility condition) from one SVD.
        u_svd, s_svd, vt_svd = np.linalg.svd(system)
        right_null = vt_svd[-1]
        left_null = u_svd[:, -1]
        rhs_norm = float(np.linalg.norm(rhs))
        if rhs_norm > 0.0:
            solvability = abs(float(left_null @ rhs)) / (
                float(np.linalg.norm(left_null)) * rhs_norm
            )
        else:
            solvability = 0.0
        pseudo = vt_svd.T @ np.diag(
            [1.0 / s if s > threshold else 0.0 for s in s_svd]
        ) @ u_svd.T
        solution = pseudo @ rhs
        # Remove the constant component so the gauge is unambiguous.
        if float(np.linalg.norm(right_null)) > 0.0:
            amplitude = float(right_null @ solution) / float(right_null @ right_null)
            solution = solution - amplitude * right_null
            gauge = amplitude
        notes.append("pure Neumann solve: minimum-norm, zero-constant gauge")

    potential = np.empty(n, dtype=float)
    flux = np.empty(n, dtype=float)
    potential[n_index] = solution[: len(n_index)]
    flux[d_index] = solution[len(n_index):]
    potential[d_index] = g_d
    flux[n_index] = q_n

    residual_vector = system @ solution - rhs
    residual = float(np.linalg.norm(residual_vector))
    rhs_norm = float(np.linalg.norm(rhs))
    return MixedBoundaryResult(
        boundary_values=potential,
        normal_derivative=flux,
        dirichlet_panels=d_index,
        neumann_panels=n_index,
        system_matrix=system,
        single_layer_matrix=single,
        double_layer_matrix=double,
        condition_number=condition,
        residual_l2=residual,
        relative_residual_l2=residual / rhs_norm if rhs_norm > 0.0 else residual,
        rank_deficiency=rank_deficiency,
        smallest_singular_value=float(singular_values[-1]),
        largest_singular_value=largest,
        solvability_residual=solvability,
        gauge_removed=gauge,
        reference_length=reference,
        notes=tuple(notes),
    )


def flux_integral(panels: PanelSet, flux: FloatArray) -> float:
    """Return ``∮ q ds``; it must vanish for solvable pure-Neumann data."""
    values = np.asarray(flux, dtype=float)
    if values.shape != (panels.panel_count,):
        raise ValueError(f"flux must have shape ({panels.panel_count},)")
    return float(values @ panels.lengths)