"""2-D regular waves scattered by one free-edge, zero-draft elastic plate of finite length."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.optimize import brentq

from .geometry import PanelMesh
from .solver import assemble_matrices


@dataclass(frozen=True)
class FloeScattering:
    reflection: complex
    transmission: complex
    energy_defect: float
    relative_residual: float


@lru_cache(maxsize=4)
def _operators(length, depth, n):
    # Three equal surface regions, with the middle region covered by the floe.
    x = np.linspace(-length, 2 * length, 3 * n + 1)
    z = np.linspace(-depth, 0.0, max(6, round(n * depth / length)) + 1)

    def side(xs, zs):
        return np.column_stack(np.broadcast_arrays(xs, zs))

    mesh = PanelMesh(
        np.vstack(
            (
                side(x[:-1], -depth),
                side(x[-1], z[:-1]),
                side(x[:0:-1], 0.0),
                side(x[0], z[:0:-1]),
            )
        )
    )
    single, double = assemble_matrices(mesh, reference_length=4 * (length + depth))
    return mesh, single, double + 0.5 * np.eye(mesh.panel_count)


def _side_modes(z, k, depth, omega, gravity):
    # Midpoint-collocated exact vertical eigenfunctions of a constant-depth
    # exterior strip. The square modal basis gives both the DtN and its dual.
    mu = omega**2 * depth / gravity
    roots = [
        brentq(lambda s: s * np.sin(s) + mu * np.cos(s), (i - 0.5) * np.pi, i * np.pi)
        / depth
        for i in range(1, len(z))
    ]
    # Equivalent to cosh(k*(z+depth))/cosh(k*depth) at large depth as well.
    incident = (
        np.exp(k * z)
        * (1 + np.exp(-2 * k * (z + depth)))
        / (1 + np.exp(-2 * k * depth))
    )
    basis = np.column_stack([incident, np.cos(np.outer(z + depth, roots))])
    dual = np.linalg.solve(basis, np.eye(len(z)))
    dtn = (basis * np.r_[1j * k, -np.asarray(roots)]) @ dual
    return incident, dtn, dual[0]


def _plate(length, n, rigidity, mass_per_area, omega, rho, g):
    """Free-free Hermite beam; coupling integrates each basis over one fluid panel."""
    h = length / n
    k = (
        rigidity
        / h**3
        * np.array(
            [
                [12, 6 * h, -12, 6 * h],
                [6 * h, 4 * h * h, -6 * h, 2 * h * h],
                [-12, -6 * h, 12, -6 * h],
                [6 * h, 2 * h * h, -6 * h, 4 * h * h],
            ]
        )
    )
    m = (
        h
        / 420
        * np.array(
            [
                [156, 22 * h, 54, -13 * h],
                [22 * h, 4 * h * h, 13 * h, -3 * h * h],
                [54, 13 * h, 156, -22 * h],
                [-13 * h, -3 * h * h, -22 * h, 4 * h * h],
            ]
        )
    )
    block = k + (rho * g - mass_per_area * omega**2) * m
    beam = np.zeros((2 * (n + 1), 2 * (n + 1)))
    coupling = np.zeros((n, 2 * (n + 1)))
    for i in range(n):
        s = slice(2 * i, 2 * i + 4)
        beam[s, s] += block
        coupling[i, s] = [0.5, h / 12, 0.5, -h / 12]
    return beam, coupling


def scatter_floe(
    omega,
    *,
    length,
    depth,
    rigidity,
    mass_per_area,
    panels=24,
    density=1000.0,
    gravity=9.81,
):
    """Complex surface-elevation amplitudes at left/right ports for unit incidence at the left port.

    exp(-i omega t), inviscid water, constant depth, linear Euler–Bernoulli
    plate of unit span, no overwash, mooring load or imposed edge stiffness.
    Energy coefficients in the equal-depth ports are |R|² and |T|².
    """
    if not all(
        np.isfinite(v) and v > 0 for v in (omega, length, depth, density, gravity)
    ):
        raise ValueError(
            "positive finite frequency, dimensions, density and gravity required"
        )
    if not all(np.isfinite(v) and v >= 0 for v in (rigidity, mass_per_area)):
        raise ValueError(
            "plate rigidity and mass per area must be finite and non-negative"
        )
    if not isinstance(panels, (int, np.integer)) or panels < 4:
        raise ValueError("at least four plate panels required")
    k = brentq(
        lambda t: gravity * t * np.tanh(t * depth) - omega**2,
        0,
        omega**2 / gravity + omega / np.sqrt(gravity * depth) + 1 / depth,
    )
    mesh, single, jump = _operators(length, depth, panels)
    x, z = mesh.centers.T
    left = np.flatnonzero(x == -length)
    right = np.flatnonzero(x == 2 * length)
    covered = np.flatnonzero((z == 0) & (x > 0) & (x < length))[::-1]
    open_top = np.flatnonzero((z == 0) & ((x < 0) | (x > length)))
    lhs = jump.astype(complex)
    lhs[:, open_top] -= single[:, open_top] * (omega**2 / gravity)
    rhs = np.zeros(len(x), complex)
    for port in (left, right):
        mode, dtn, dual = _side_modes(z[port], k, depth, omega, gravity)
        lhs[:, port] -= single[:, port] @ dtn
        if port is left:
            rhs = single[:, port] @ (-2j * k * mode * gravity / omega**2)
            trace_left = dual
        else:
            trace_right = dual
    beam, coupling = _plate(
        length, panels, rigidity, mass_per_area, omega, density, gravity
    )
    # With displacement potential psi: d_n psi = eta on the plate and
    # D eta'''' + (rho g - m omega²) eta = rho omega² psi.
    lower = np.zeros((beam.shape[0], len(x)))
    lower[:, covered] = -density * omega**2 * (length / panels) * coupling.T
    matrix = np.block([[lhs, -single[:, covered] @ coupling], [lower, beam]])
    load = np.r_[rhs, np.zeros(len(beam))]
    solution = np.linalg.solve(matrix, load)
    refl = (trace_left @ solution[left]) * omega**2 / gravity - 1
    trans = (trace_right @ solution[right]) * omega**2 / gravity
    return FloeScattering(
        refl,
        trans,
        abs(abs(refl) ** 2 + abs(trans) ** 2 - 1),
        np.linalg.norm(matrix @ solution - load) / np.linalg.norm(load),
    )
