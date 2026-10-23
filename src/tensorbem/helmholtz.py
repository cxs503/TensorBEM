"""Two-dimensional exterior Helmholtz diffraction and vertical-pile wave loads.

Separation of variables for a bottom-mounted, surface-piercing vertical
cylinder of arbitrary horizontal cross-section ``Gamma`` in regular Airy waves
of wavenumber ``k`` and water depth ``h`` gives

    Phi(x, y, z) = -(i * g * A / omega) * (cosh(k*(z + h)) / cosh(k*h)) * varphi(x, y),

where ``omega**2 = g * k * tanh(k*h)`` and the dimensionless horizontal total
potential ``varphi = varphi_I + varphi_S`` satisfies

    (Laplacian_xy + k**2) varphi_S = 0     outside Gamma,
    d(varphi)/dn = 0                       on Gamma,
    sqrt(r) * (d(varphi_S)/dr - i*k*varphi_S) -> 0   as r -> infinity,

with ``varphi_I(x, y) = exp(i*k*(x*cos(beta) + y*sin(beta)))``.

With the outgoing Green function ``G_k(r) = (i/4)*H_0^{(1)}(k*r)`` satisfying
``-(Laplacian + k**2) G_k = delta`` (whose logarithmic singularity matches
``-(1/(2*pi))*log(r)`` from :mod:`tensorbem.kernels`), eliminating the unknown
scattered normal derivative via the interior Green identity for ``varphi_I``
yields the direct total-potential boundary integral equation

    (1/2 I - H_k) varphi = varphi_I          on Gamma,

and at any point ``x_int`` strictly inside the body cross-section,

    -H_k^{int} varphi = varphi_I(x_int).

Irregular frequencies
---------------------
On a closed boundary ``Gamma``, ``1/2 I - H_k`` loses rank whenever ``k**2`` is
an interior Dirichlet eigenvalue of ``-Laplacian`` inside ``Gamma`` (for a
circle of radius ``a``, the zeros of ``J_m(k*a) = 0``: ``ka = 2.40482556``,
``3.83170597``, ...). Appending a small set of interior null-field equations
(Schenck's 1968 CHIEF method, standard in WAMIT and Capytaine) restores full
column rank and ``O(1)`` conditioning across those resonances unless every
CHIEF point happens to lie on a nodal curve of the resonant interior mode.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.typing import NDArray

from .analytic import bessel_j01_y01
from .geometry import PanelMesh

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
_TWO_PI = 2.0 * np.pi


@dataclass(frozen=True)
class HelmholtzDiffractionResult:
    """Solution of a 2-D exterior rigid-body Helmholtz diffraction problem."""

    total_potential: ComplexArray
    scattered_potential: ComplexArray
    incident_potential: ComplexArray
    double_layer_matrix: ComplexArray
    wavenumber: float
    incident_angle: float
    chief_points: FloatArray
    condition_number: float
    boundary_relative_residual: float
    chief_relative_residual: float


@dataclass(frozen=True)
class VerticalCylinderDiffractionResult:
    """Three-dimensional wave loads on a depth-uniform vertical cylinder."""

    helmholtz: HelmholtzDiffractionResult
    force: ComplexArray
    moment_about_seabed: ComplexArray | None
    horizontal_integral: ComplexArray
    omega: float
    depth: float
    amplitude: float
    density: float


def _validate_positive(name: str, value: float) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive finite number")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return number


def _validate_panel_nodes(panel_nodes: int) -> int:
    if isinstance(panel_nodes, (bool, np.bool_)) or not isinstance(
        panel_nodes, (int, np.integer)
    ):
        raise ValueError("panel_nodes must be an integer >= 4")
    nodes = int(panel_nodes)
    if nodes < 4:
        raise ValueError("panel_nodes must be at least 4")
    return nodes


def points_inside_polygon(
    mesh: PanelMesh, points: FloatArray, *, min_clearance_ratio: float = 1.0e-3
) -> NDArray[np.bool_]:
    """Return True for points strictly inside ``mesh`` and clear of its edges."""
    if not isinstance(mesh, PanelMesh):
        raise TypeError("mesh must be a PanelMesh")
    targets = np.asarray(points, dtype=float)
    if targets.ndim != 2 or targets.shape[1] != 2:
        raise ValueError("points must have shape (M, 2)")
    if not np.isfinite(targets).all():
        raise ValueError("points must all be finite")

    v0 = mesh.vertices
    v1 = mesh.ends
    px = targets[:, 0][:, None]
    py = targets[:, 1][:, None]
    x0, y0 = v0[:, 0][None, :], v0[:, 1][None, :]
    x1, y1 = v1[:, 0][None, :], v1[:, 1][None, :]

    straddle = (y0 > py) != (y1 > py)
    safe_dy = np.where(straddle, y1 - y0, 1.0)
    x_intersect = x0 + (py - y0) * (x1 - x0) / safe_dy
    crossings = np.sum(straddle & (px < x_intersect), axis=1)
    inside = (crossings % 2) == 1

    # Minimum Euclidean distance from each point to every straight panel.
    edges = mesh.vectors
    len2 = np.sum(edges * edges, axis=1)
    delta = targets[:, None, :] - v0[None, :, :]
    param = np.clip(np.einsum("mpk,pk->mp", delta, edges) / len2[None, :], 0.0, 1.0)
    closest = v0[None, :, :] + param[:, :, None] * edges[None, :, :]
    min_dist = np.min(np.linalg.norm(targets[:, None, :] - closest, axis=2), axis=1)
    h_min = float(np.min(mesh.lengths))
    return inside & (min_dist >= min_clearance_ratio * h_min)


def default_chief_points(mesh: PanelMesh, count: int = 5) -> FloatArray:
    """Deterministic asymmetric interior CHIEF points inside ``mesh``.

    Points are placed at incommensurate radii and angles around the polygon's
    centroid so they do not simultaneously lie on a diameter or concentric
    nodal circle of low-order interior Dirichlet modes.
    """
    if not isinstance(mesh, PanelMesh):
        raise TypeError("mesh must be a PanelMesh")
    if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)):
        raise ValueError("count must be a positive integer")
    num = int(count)
    if num < 1:
        raise ValueError("count must be a positive integer")

    vertices = mesh.vertices
    v_next = mesh.ends
    cross = vertices[:, 0] * v_next[:, 1] - v_next[:, 0] * vertices[:, 1]
    area6 = 3.0 * float(np.sum(cross))
    centroid = np.array(
        [
            float(np.sum((vertices[:, 0] + v_next[:, 0]) * cross)) / area6,
            float(np.sum((vertices[:, 1] + v_next[:, 1]) * cross)) / area6,
        ]
    )

    edges = mesh.vectors
    len2 = np.sum(edges * edges, axis=1)
    param = np.clip(np.einsum("pk,pk->p", centroid[None, :] - vertices, edges) / len2, 0.0, 1.0)
    closest = vertices + param[:, None] * edges
    inradius = float(np.min(np.linalg.norm(centroid[None, :] - closest, axis=1)))
    if not points_inside_polygon(mesh, centroid[None, :])[0] or inradius <= 0.0:
        raise ValueError(
            "polygon centroid does not lie strictly inside the mesh; supply "
            "explicit interior chief_points for non-star-shaped cross-sections"
        )

    golden = math.pi * (3.0 - math.sqrt(5.0))
    candidates: list[FloatArray] = [centroid]
    for index in range(1, num):
        frac = 0.19 + 0.43 * ((index * 0.6180339887498949) % 1.0)
        angle = 0.37 + golden * index
        candidates.append(
            centroid + (frac * inradius) * np.array([math.cos(angle), math.sin(angle)])
        )
    points = np.asarray(candidates, dtype=float)
    if not np.all(points_inside_polygon(mesh, points)):
        raise ValueError("generated CHIEF points are too close to the boundary")
    return points


def assemble_helmholtz_double_layer(
    mesh: PanelMesh,
    wavenumber: float,
    targets: FloatArray | None = None,
    *,
    panel_nodes: int = 12,
) -> ComplexArray:
    """Assemble ``H_k[i, j] = integral_{panel j} dG_k(target_i, y)/dn_y ds``.

    Splits ``dG_k/dn_y`` into the static Laplace double-layer kernel (integrated
    analytically on straight panels) plus the bounded remainder

        [ (i*k/4)*H_1^{(1)}(k*r)/r - 1/(2*pi*r**2) ] * (target - y) . n_y,

    which vanishes identically on any straight panel containing ``target``.
    """
    if not isinstance(mesh, PanelMesh):
        raise TypeError("mesh must be a PanelMesh")
    k = _validate_positive("wavenumber", wavenumber)
    nodes_count = _validate_panel_nodes(panel_nodes)

    self_collocation = targets is None
    eval_points = mesh.centers if self_collocation else np.asarray(targets, dtype=float)
    if eval_points.ndim != 2 or eval_points.shape[1] != 2:
        raise ValueError("targets must have shape (M, 2)")
    if not np.isfinite(eval_points).all():
        raise ValueError("targets must all be finite")

    starts = mesh.vertices
    vecs = mesh.vectors
    lengths = mesh.lengths
    tangents = vecs / lengths[:, None]
    normals = mesh.normals

    delta_a = eval_points[:, None, :] - starts[None, :, :]
    xi = np.einsum("mpk,pk->mp", delta_a, tangents)
    eta = np.einsum("mpk,pk->mp", delta_a, normals)
    h = np.abs(eta)
    coplanar = h <= 1.0e-14
    h_safe = np.where(coplanar, 1.0, h)
    laplace_dl = np.where(
        coplanar,
        0.0,
        np.sign(eta)
        * (np.arctan((lengths[None, :] - xi) / h_safe) - np.arctan(-xi / h_safe))
        / _TWO_PI,
    )

    nodes, weights = leggauss(nodes_count)
    fraction = 0.5 * (nodes + 1.0)
    quad_points = starts[:, None, :] + fraction[None, :, None] * vecs[:, None, :]
    diff = eval_points[:, None, None, :] - quad_points[None, :, :, :]
    r = np.linalg.norm(diff, axis=3)
    dot_n = np.einsum("mpqk,pk->mpq", diff, normals)
    zero_mask = (r <= 1.0e-14) | coplanar[:, :, None]
    r_safe = np.where(zero_mask, 1.0, r)

    _, j1, _, y1 = bessel_j01_y01(k * r_safe)
    h1 = j1 + 1.0j * y1
    remainder = np.where(
        zero_mask,
        0.0j,
        (0.25j * k * h1 / r_safe - 1.0 / (_TWO_PI * r_safe * r_safe)) * dot_n,
    )
    smooth_dl = 0.5 * lengths[None, :] * np.sum(
        remainder * weights[None, None, :], axis=2
    )
    matrix = laplace_dl + smooth_dl
    if self_collocation:
        np.fill_diagonal(matrix, 0.0)
    return matrix


def solve_helmholtz_diffraction(
    mesh: PanelMesh,
    wavenumber: float,
    incident_angle: float = 0.0,
    *,
    panel_nodes: int = 12,
    chief_points: FloatArray | str | None = "auto",
) -> HelmholtzDiffractionResult:
    """Solve 2-D exterior Helmholtz plane-wave diffraction on ``mesh``.

    Parameters
    ----------
    mesh:
        Closed counter-clockwise cross-section polygon.
    wavenumber:
        Acoustic / surface-wave wavenumber ``k > 0``.
    incident_angle:
        Propagation angle ``beta`` (radians) of ``exp(i*k*(x*cos(beta)+y*sin(beta)))``.
    panel_nodes:
        Gauss-Legendre order for the smooth Hankel remainder on each panel.
    chief_points:
        ``"auto"`` uses :func:`default_chief_points` (5 interior points) to
        suppress irregular frequencies; ``None`` solves the unaugmented square
        boundary system; an ``(M, 2)`` array supplies explicit interior points.
    """
    if not isinstance(mesh, PanelMesh):
        raise TypeError("mesh must be a PanelMesh")
    k = _validate_positive("wavenumber", wavenumber)
    if isinstance(incident_angle, (bool, np.bool_)) or not math.isfinite(
        float(incident_angle)
    ):
        raise ValueError("incident_angle must be a finite number")
    beta = float(incident_angle)
    nodes_count = _validate_panel_nodes(panel_nodes)

    if isinstance(chief_points, str):
        if chief_points != "auto":
            raise ValueError("chief_points string option must be 'auto'")
        chief = default_chief_points(mesh, count=5)
    elif chief_points is None:
        chief = np.empty((0, 2), dtype=float)
    else:
        chief = np.asarray(chief_points, dtype=float)
        if chief.ndim != 2 or chief.shape[1] != 2:
            raise ValueError("chief_points must have shape (M, 2)")
        if len(chief) > 0 and not np.all(points_inside_polygon(mesh, chief)):
            raise ValueError("all chief_points must lie strictly inside the polygon")

    n = mesh.panel_count
    h_boundary = assemble_helmholtz_double_layer(
        mesh, k, targets=None, panel_nodes=nodes_count
    )
    boundary_system = 0.5 * np.eye(n, dtype=np.complex128) - h_boundary
    direction = np.array([math.cos(beta), math.sin(beta)])
    phi_inc = np.exp(1.0j * k * (mesh.centers @ direction))

    if len(chief) == 0:
        phi_total = np.linalg.solve(boundary_system, phi_inc)
        cond = float(np.linalg.cond(boundary_system))
        chief_rel_res = 0.0
    else:
        h_chief = assemble_helmholtz_double_layer(
            mesh, k, targets=chief, panel_nodes=nodes_count
        )
        phi_inc_chief = np.exp(1.0j * k * (chief @ direction))
        augmented_matrix = np.vstack((boundary_system, -h_chief))
        augmented_rhs = np.concatenate((phi_inc, phi_inc_chief))
        phi_total, *_ = np.linalg.lstsq(augmented_matrix, augmented_rhs, rcond=None)
        cond = float(np.linalg.cond(augmented_matrix))
        chief_res = -h_chief @ phi_total - phi_inc_chief
        chief_rel_res = float(
            np.linalg.norm(chief_res) / max(np.linalg.norm(phi_inc_chief), 1.0e-300)
        )

    boundary_res = boundary_system @ phi_total - phi_inc
    boundary_rel_res = float(
        np.linalg.norm(boundary_res) / max(np.linalg.norm(phi_inc), 1.0e-300)
    )
    return HelmholtzDiffractionResult(
        total_potential=phi_total,
        scattered_potential=phi_total - phi_inc,
        incident_potential=phi_inc,
        double_layer_matrix=h_boundary,
        wavenumber=k,
        incident_angle=beta,
        chief_points=chief,
        condition_number=cond,
        boundary_relative_residual=boundary_rel_res,
        chief_relative_residual=chief_rel_res,
    )


def solve_vertical_cylinder_diffraction(
    mesh: PanelMesh,
    wavenumber: float,
    depth: float = math.inf,
    amplitude: float = 1.0,
    density: float = 1025.0,
    *,
    gravity: float = 9.80665,
    incident_angle: float = 0.0,
    panel_nodes: int = 12,
    chief_points: FloatArray | str | None = "auto",
) -> VerticalCylinderDiffractionResult:
    """Solve 3-D wave diffraction loads on a bottom-mounted vertical cylinder.

    Integrates the horizontal Helmholtz diffraction potential ``varphi(x, y)``
    analytically over ``z in [-h, 0]`` with the linear vertical wave profile
    ``cosh(k*(z + h)) / cosh(k*h)`` to obtain the total horizontal wave force
    ``(F_x, F_y)`` and, for finite depth ``h``, the overturning moment
    ``(M_x, M_y)`` about ``(0, 0, -h)``.
    """
    k = _validate_positive("wavenumber", wavenumber)
    amp = _validate_positive("amplitude", amplitude)
    rho = _validate_positive("density", density)
    g = _validate_positive("gravity", gravity)
    if isinstance(depth, (bool, np.bool_)):
        raise ValueError("depth must be a positive number or inf")
    h_val = float(depth)
    if math.isnan(h_val) or h_val <= 0.0:
        raise ValueError("depth must be a positive number or inf")

    helmholtz = solve_helmholtz_diffraction(
        mesh,
        k,
        incident_angle=incident_angle,
        panel_nodes=panel_nodes,
        chief_points=chief_points,
    )
    horizontal_integral = (helmholtz.total_potential * mesh.lengths) @ mesh.normals

    if math.isinf(h_val):
        tanh_kh = 1.0
        omega = math.sqrt(g * k)
        moment: ComplexArray | None = None
    else:
        kh = k * h_val
        tanh_kh = math.tanh(kh) if kh < 50.0 else 1.0
        sech_kh = (1.0 / math.cosh(kh)) if kh < 50.0 else 0.0
        omega = math.sqrt(g * k * tanh_kh)
        moment_depth_integral = (kh * tanh_kh + sech_kh - 1.0) / (k * k)
        df_xy = -rho * g * amp * horizontal_integral
        # Moment about seabed (0, 0, -h): r x dF with r = (0, 0, z + h) gives
        # M_x = -(z + h) * dF_y, M_y = +(z + h) * dF_x.
        moment = moment_depth_integral * np.array([-df_xy[1], df_xy[0]], dtype=np.complex128)

    depth_integral = tanh_kh / k
    force = -rho * g * amp * depth_integral * horizontal_integral
    return VerticalCylinderDiffractionResult(
        helmholtz=helmholtz,
        force=force,
        moment_about_seabed=moment,
        horizontal_integral=horizontal_integral,
        omega=omega,
        depth=h_val,
        amplitude=amp,
        density=rho,
    )
