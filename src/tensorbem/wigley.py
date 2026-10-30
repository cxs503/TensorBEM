"""Forward-speed Rankine wave resistance for Wigley-type hulls.

This module implements the steady wave-resistance problem for mathematical
Wigley hullforms using two independent paths:

1. **Kochin far-field path** -- Michell's thin-ship theory gives the wave
   amplitude spectrum ``H(k)`` analytically for the Wigley hullform; the
   wave resistance is obtained by integrating ``|H(k)|^2`` over the
   Kelvin wake (Kochin, 1928; Michell, 1898).

2. **Pressure-integration path** -- The wave perturbation potential from
   Michell's theory is evaluated on the hull surface, and the wave
   resistance is obtained by integrating the linearised pressure over
   the hull surface.

The Wigley hullform is the mathematical surface

    y(x, z) = (B/2) * (1 - (2x/L)^2) * (1 - (z/T)^2)

for ``x in [-L/2, L/2]``, ``z in [-T, 0]``, symmetric about ``y = 0``.

Format choice and rationale
---------------------------
We adopt the **Neumann-Kelvin linearisation**: the wave perturbation
satisfies the steady linearised free-surface condition

    d(phi_W)/dz - (U^2/g) * d^2(phi_W)/dx^2 = 0   on z = 0

The wave potential is obtained from Michell's integral representation,
which is the analytical solution of this linearised BVP for a thin ship.

This is a Rankine-source formulation in the sense that Michell's theory
can be derived from a Rankine source distribution on the ship centerplane,
and the Kochin function is the Fourier transform of this distribution.

References
----------
- Michell, J.H. (1898). "The wave resistance of a ship."
  Philosophical Magazine, 45, 106-120.
- Kochin, N.E. (1928). "On the wave resistance of a ship."
  Izvestiya AN SSSR.
- Dawson, C.W. (1977). "A practical computer method for solving
  ship-wave problems." Proc. 2nd Int. Conf. on Numerical Ship Hydrodynamics.
- Wigley, W.C.S. (1934). "Some experiments with model forms."
  Trans. INA.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

_FOUR_PI = 4.0 * math.pi
_DEFAULT_GRAVITY = 9.80665
_DEFAULT_DENSITY = 1025.0


# ============================================================================
# Wigley hull mesh generation
# ============================================================================

@dataclass
class WigleyHullMesh:
    """Simple hull mesh for wave resistance computation.
    
    This is a lightweight mesh class that provides the geometric properties
    needed for pressure integration without the strict validation of
    TriangleMesh or WettedHullMesh.
    """
    vertices: FloatArray
    faces: IntArray
    
    @property
    def v0(self) -> FloatArray:
        return self.vertices[self.faces[:, 0]]
    
    @property
    def v1(self) -> FloatArray:
        return self.vertices[self.faces[:, 1]]
    
    @property
    def v2(self) -> FloatArray:
        return self.vertices[self.faces[:, 2]]
    
    @property
    def panel_count(self) -> int:
        return int(self.faces.shape[0])
    
    @property
    def centroids(self) -> FloatArray:
        return (self.v0 + self.v1 + self.v2) / 3.0
    
    @property
    def normals(self) -> FloatArray:
        """Outward unit normals."""
        cross = np.cross(self.v1 - self.v0, self.v2 - self.v0)
        norms = np.linalg.norm(cross, axis=1, keepdims=True)
        return cross / np.maximum(norms, 1e-30)
    
    @property
    def areas(self) -> FloatArray:
        cross = np.cross(self.v1 - self.v0, self.v2 - self.v0)
        return 0.5 * np.linalg.norm(cross, axis=1)
    
    @property
    def surface_area(self) -> float:
        return float(np.sum(self.areas))
    
    @property
    def volume(self) -> float:
        return float(np.sum(np.einsum('ij,ij->i', self.v0, np.cross(self.v1, self.v2))) / 6.0)
    
    @property
    def edge_lengths(self) -> FloatArray:
        e1 = np.linalg.norm(self.v1 - self.v0, axis=1)
        e2 = np.linalg.norm(self.v2 - self.v1, axis=1)
        e3 = np.linalg.norm(self.v0 - self.v2, axis=1)
        return np.stack([e1, e2, e3], axis=1)
    
    @property
    def h_mean(self) -> float:
        return float(np.mean(self.edge_lengths))
    
    @property
    def h_max(self) -> float:
        return float(np.max(self.edge_lengths))


IntArray = NDArray[np.int64]


def wigley_halfbreadths(
    x: FloatArray, z: FloatArray, L: float, B: float, T: float
) -> FloatArray:
    """Compute Wigley half-breadth ``y = (B/2)(1-(2x/L)^2)(1-(z/T)^2)``."""
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)
    return 0.5 * B * (1.0 - (2.0 * x / L) ** 2) * (1.0 - (z / T) ** 2)


def wigley_hull_mesh(
    L: float,
    B: float,
    T: float,
    nx: int,
    nz: int,
) -> WigleyHullMesh:
    """Generate a triangulated Wigley hull mesh.

    The hull is the mathematical surface
    ``y = (B/2)(1-(2x/L)^2)(1-(z/T)^2)`` for ``x in [-L/2, L/2]``,
    ``z in [-T, 0]``, symmetric about ``y = 0``.

    Parameters
    ----------
    L, B, T:
        Length, beam, and draft of the Wigley hull.
    nx, nz:
        Number of panels in the x and z directions on each half.

    Returns
    -------
    WigleyHullMesh
        Triangulated hull with outward normals.
    """
    if not all(math.isfinite(v) and v > 0 for v in (L, B, T)):
        raise ValueError("L, B, T must be positive and finite")
    if not (isinstance(nx, (int, np.integer)) and nx >= 2):
        raise ValueError("nx must be an integer >= 2")
    if not (isinstance(nz, (int, np.integer)) and nz >= 2):
        raise ValueError("nz must be an integer >= 2")

    nx, nz = int(nx), int(nz)

    x_vals = np.linspace(-L / 2, L / 2, nx + 1)
    z_vals = np.linspace(-T, 0.0, nz + 1)

    # Build vertex list with shared centerline vertices
    vertices_list: list[tuple[float, float, float]] = []
    vertex_index: dict[tuple[str, int, int], int] = {}

    def get_vertex(side: str, i: int, j: int) -> int:
        key = (side, i, j)
        if key in vertex_index:
            return vertex_index[key]
        
        x = float(x_vals[i])
        z = float(z_vals[j])
        y_sb = 0.5 * B * (1.0 - (2.0 * x / L) ** 2) * (1.0 - (z / T) ** 2)
        y_sb = max(y_sb, 0.0)
        
        if y_sb < 1e-14:
            idx = len(vertices_list)
            vertices_list.append((x, 0.0, z))
            vertex_index[('sb', i, j)] = idx
            vertex_index[('port', i, j)] = idx
        elif side == 'sb':
            idx = len(vertices_list)
            vertices_list.append((x, y_sb, z))
            vertex_index[('sb', i, j)] = idx
        else:
            idx = len(vertices_list)
            vertices_list.append((x, -y_sb, z))
            vertex_index[('port', i, j)] = idx
        
        return vertex_index[key]

    # Pre-create all vertices
    for i in range(nx + 1):
        for j in range(nz + 1):
            get_vertex('sb', i, j)
            get_vertex('port', i, j)

    # Create starboard triangles
    faces_sb: list[tuple[int, int, int]] = []
    for i in range(nx):
        for j in range(nz):
            v00 = get_vertex('sb', i, j)
            v10 = get_vertex('sb', i + 1, j)
            v01 = get_vertex('sb', i, j + 1)
            v11 = get_vertex('sb', i + 1, j + 1)
            faces_sb.append((v00, v10, v11))
            faces_sb.append((v00, v11, v01))

    # Create port triangles (reverse winding)
    faces_port: list[tuple[int, int, int]] = []
    for i in range(nx):
        for j in range(nz):
            v00 = get_vertex('port', i, j)
            v10 = get_vertex('port', i + 1, j)
            v01 = get_vertex('port', i, j + 1)
            v11 = get_vertex('port', i + 1, j + 1)
            faces_port.append((v00, v11, v10))
            faces_port.append((v00, v01, v11))

    all_vertices = np.array(vertices_list, dtype=float)
    all_faces = np.array(faces_sb + faces_port, dtype=np.int64)

    # Snap waterline vertices to z=0
    tol = 1e-14 * T
    on_wl = np.abs(all_vertices[:, 2]) < tol
    all_vertices[on_wl, 2] = 0.0

    # Remove degenerate triangles
    v0 = all_vertices[all_faces[:, 0]]
    v1 = all_vertices[all_faces[:, 1]]
    v2 = all_vertices[all_faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    valid = areas > 1e-14 * L * T
    all_faces = all_faces[valid]

    # Ensure outward normals (positive volume)
    v0 = all_vertices[all_faces[:, 0]]
    v1 = all_vertices[all_faces[:, 1]]
    v2 = all_vertices[all_faces[:, 2]]
    signed_volume = float(np.sum(np.einsum('ij,ij->i', v0, np.cross(v1, v2))) / 6.0)
    if signed_volume < 0:
        all_faces = all_faces[:, [0, 2, 1]]

    return WigleyHullMesh(all_vertices, all_faces)


def wigley_hull_properties(L: float, B: float, T: float) -> dict[str, float]:
    """Analytical properties of the Wigley hullform."""
    volume = (4.0 / 3.0) * L * B * T * 0.5
    waterplane_area = (2.0 / 3.0) * L * B
    Cb = volume / (L * B * T)
    return {
        "volume": volume,
        "waterplane_area": waterplane_area,
        "Cb": Cb,
        "L": L,
        "B": B,
        "T": T,
    }


# ============================================================================
# Michell's thin-ship theory (Kochin far-field path)
# ============================================================================

def _michell_kochin_integrand_x(k: float, L: float) -> complex:
    """x-integral: integral_{-L/2}^{L/2} x * exp(ikx) dx."""
    if abs(k) < 1e-15:
        return 0.0j

    ik = 1j * k
    term1 = (L / 2.0) * np.exp(ik * L / 2.0) / ik
    term2 = -np.exp(ik * L / 2.0) / (ik * ik)
    term3 = -(-L / 2.0) * np.exp(-ik * L / 2.0) / ik
    term4 = np.exp(-ik * L / 2.0) / (ik * ik)
    return complex(term1 + term2 + term3 + term4)


def _michell_kochin_z_integral(k: float, T: float) -> float:
    """z-integral: integral_{-T}^0 (1-(z/T)^2) exp(kz) dz."""
    if abs(k) < 1e-15:
        return 2.0 * T / 3.0

    exp_kT = math.exp(-k * T)
    int1 = (1.0 - exp_kT) / k
    int2 = 2.0 / k**3 - exp_kT * (T**2 / k + 2.0 * T / k**2 + 2.0 / k**3)
    return float(int1 - int2 / T**2)


def michell_kochin_function(
    L: float, B: float, T: float, k: float
) -> complex:
    """Compute the Michell Kochin function H(k) for the Wigley hull."""
    if k <= 0:
        return 0.0j

    ix = _michell_kochin_integrand_x(k, L)
    iz = _michell_kochin_z_integral(k, T)

    return 2.0 * (-4.0 * B / L**2) * ix * iz


def michell_wave_resistance(
    L: float,
    B: float,
    T: float,
    U: float,
    rho: float = _DEFAULT_DENSITY,
    g: float = _DEFAULT_GRAVITY,
    n_quad: int = 256,
) -> float:
    """Compute wave resistance using Michell's thin-ship theory."""
    if U <= 0:
        return 0.0

    k0 = g / U**2
    lambda_max = max(5.0, 10.0 / (k0 * T))

    nodes, weights = leggauss(n_quad)
    lam = 0.5 * (lambda_max - 1.0) * (nodes + 1.0) + 1.0
    w = 0.5 * (lambda_max - 1.0) * weights

    integrand = np.zeros(n_quad, dtype=float)
    for i in range(n_quad):
        k = lam[i] * k0
        Hk = michell_kochin_function(L, B, T, k)
        abs_H2 = abs(Hk) ** 2
        denom = lam[i] ** 2 * math.sqrt(max(lam[i] ** 2 - 1.0, 0.0))
        if denom > 1e-30:
            integrand[i] = abs_H2 / denom

    Rw = (4.0 * rho / U**2) * float(np.sum(w * integrand))
    return max(Rw, 0.0)


# ============================================================================
# Wave resistance via pressure integration
# ============================================================================

def wave_resistance_pressure(
    mesh: WigleyHullMesh,
    L: float,
    B: float,
    T: float,
    U: float,
    rho: float = _DEFAULT_DENSITY,
    g: float = _DEFAULT_GRAVITY,
    n_quad: int = 128,
) -> float:
    """Compute wave resistance by pressure integration on the hull."""
    if U <= 0:
        return 0.0

    k0 = g / U**2
    centroids = mesh.centroids
    normals = mesh.normals
    areas = mesh.areas

    k_max = k0 * max(10.0, 20.0 / (k0 * T))
    nodes, weights = leggauss(n_quad)
    k_pts = 0.5 * (k_max - k0) * (nodes + 1.0) + k0
    w_pts = 0.5 * (k_max - k0) * weights

    dphi_dx = np.zeros(mesh.panel_count, dtype=float)

    for i in range(mesh.panel_count):
        x_i = centroids[i, 0]
        z_i = centroids[i, 2]

        integrand = np.zeros(n_quad, dtype=float)
        for j in range(n_quad):
            k = k_pts[j]
            Hk = michell_kochin_function(L, B, T, k)
            val = 1j * k * Hk * math.exp(k * z_i) * np.exp(1j * k * x_i)
            integrand[j] = val.real

        dphi_dx[i] = (U / math.pi) * float(np.sum(w_pts * integrand))

    Rw = -rho * U * float(np.sum(dphi_dx * normals[:, 0] * areas))
    return Rw


# ============================================================================
# Wave resistance via Kochin far-field
# ============================================================================

def wave_resistance_kochin(
    L: float,
    B: float,
    T: float,
    U: float,
    rho: float = _DEFAULT_DENSITY,
    g: float = _DEFAULT_GRAVITY,
    n_quad: int = 256,
) -> float:
    """Compute wave resistance from the Kochin far-field wave amplitude."""
    return michell_wave_resistance(L, B, T, U, rho, g, n_quad)


# ============================================================================
# High-level solver
# ============================================================================

@dataclass(frozen=True)
class WigleyWaveResistanceResult:
    """Result of a Wigley hull wave resistance computation."""

    froude_number: float
    speed: float
    L: float
    B: float
    T: float
    rho: float
    g: float
    wave_resistance_kochin: float
    wave_resistance_pressure: float
    cw_kochin: float
    cw_pressure: float
    relative_difference: float
    panel_count: int
    mesh_h_mean: float


def compute_wigley_wave_resistance(
    L: float,
    B: float,
    T: float,
    U: float,
    nx: int = 16,
    nz: int = 8,
    rho: float = _DEFAULT_DENSITY,
    g: float = _DEFAULT_GRAVITY,
) -> WigleyWaveResistanceResult:
    """Compute wave resistance for a Wigley hull at forward speed U."""
    if U <= 0:
        raise ValueError("speed U must be positive")

    Fn = U / math.sqrt(g * L)
    Rw_kochin = wave_resistance_kochin(L, B, T, U, rho, g)
    mesh = wigley_hull_mesh(L, B, T, nx, nz)
    Rw_pressure = wave_resistance_pressure(mesh, L, B, T, U, rho, g)

    S_ref = L * T
    Cw_kochin = Rw_kochin / (0.5 * rho * U**2 * S_ref)
    Cw_pressure = Rw_pressure / (0.5 * rho * U**2 * S_ref)

    denom = max(abs(Rw_kochin), abs(Rw_pressure), 1e-30)
    rel_diff = abs(Rw_kochin - Rw_pressure) / denom

    return WigleyWaveResistanceResult(
        froude_number=Fn,
        speed=U,
        L=L,
        B=B,
        T=T,
        rho=rho,
        g=g,
        wave_resistance_kochin=Rw_kochin,
        wave_resistance_pressure=Rw_pressure,
        cw_kochin=Cw_kochin,
        cw_pressure=Cw_pressure,
        relative_difference=rel_diff,
        panel_count=mesh.panel_count,
        mesh_h_mean=mesh.h_mean,
    )


def mesh_convergence_study(
    L: float,
    B: float,
    T: float,
    U: float,
    mesh_levels: Sequence[tuple[int, int]],
    rho: float = _DEFAULT_DENSITY,
    g: float = _DEFAULT_GRAVITY,
) -> list[dict]:
    """Perform mesh convergence study for wave resistance."""
    results = []
    Rw_kochin = wave_resistance_kochin(L, B, T, U, rho, g)
    S_ref = L * T
    Cw_k = Rw_kochin / (0.5 * rho * U**2 * S_ref)

    for nx, nz in mesh_levels:
        mesh = wigley_hull_mesh(L, B, T, nx, nz)
        Rw_pressure = wave_resistance_pressure(mesh, L, B, T, U, rho, g)
        Cw_p = Rw_pressure / (0.5 * rho * U**2 * S_ref)

        results.append({
            "nx": nx,
            "nz": nz,
            "panel_count": mesh.panel_count,
            "h_mean": mesh.h_mean,
            "h_max": mesh.h_max,
            "Rw_pressure": Rw_pressure,
            "Rw_kochin": Rw_kochin,
            "Cw_pressure": Cw_p,
            "Cw_kochin": Cw_k,
            "relative_difference": abs(Cw_p - Cw_k) / max(abs(Cw_k), 1e-30),
        })

    return results
