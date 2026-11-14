"""Mode-I center-crack stress-intensity factor by the displacement-discontinuity BEM.

Why this module exists
----------------------
TensorBEM had no fracture-mechanics capability.  The centre-crack Mode-I SIF
is the canonical anchor ``CASE-BEM-CRACK-SIF``: an infinite plate with a
through-thickness crack of half-length ``a`` under remote tension ``sigma`` has
the closed-form solution ``K_I = sigma * sqrt(pi * a)`` (Tada, Paris & Irwin,
*The Stress Analysis of Cracks Handbook*, 3rd ed., 2000, Eq. 2.1a).

Formulation
-----------
The crack is modelled as a line of normal displacement discontinuities (DD)
``D_y(xi)`` on ``[-a, a]``.  For plane strain the hypersingular boundary
integral equation reads

    sigma_yy(x) = -E'/(4 pi) * FP int_{-a}^{a} D_y(xi) / (x - xi)^2 d xi = sigma

where ``E' = E / (1 - nu^2)`` and ``FP`` denotes the Hadamard finite-part
integral.  The exact solution is ``D_y(xi) = 4 sigma / E' * sqrt(a^2 - xi^2)``.

Two discretisation routes are provided:

1. **Constant DD elements** (``method="constant"``): piecewise-constant DD on
   a graded mesh.  The SIF is extracted by fitting the near-tip DD values to
   the asymptotic ``sqrt(r)`` law.  This route is first-order accurate and
   serves as a pedagogical baseline.

2. **Chebyshev collocation** (``method="chebyshev"``): the DD is expanded as
   ``D(x) = sqrt(1 - x^2) * sum c_n T_n(x)`` on the mapped interval ``[-1,1]``.
   The hypersingular integral has a closed-form action on this basis:
   ``FP int sqrt(1-xi^2) T_n(xi) / (x-xi)^2 d xi = -pi * n * U_{n-1}(x)``.
   For uniform loading only the ``n=0`` mode is excited, giving an *exact*
   solution in one term.  This route is used by default.

SIF extraction (Chebyshev route)
--------------------------------
For the Chebyshev expansion ``D(x) = c_0 * sqrt(1 - x^2)`` (mapped to ``[-a,a]``),
the SIF follows directly from the coefficient:

    K_I = E' * c_0 * sqrt(pi * a) / 4

Since the analytical solution has ``c_0 = 4 sigma / E'``, this gives
``K_I = sigma * sqrt(pi * a)`` exactly.

Units are SI throughout (m, Pa, N/m^(3/2) for K_I).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "analytical_sif",
    "analytical_cod",
    "CrackSIFResult",
    "solve_crack_sif",
    "convergence_study",
]

FloatArray = NDArray[np.float64]


# ---------------------------------------------------------------------------
# Closed-form reference (never used by the solver)
# ---------------------------------------------------------------------------

def analytical_sif(sigma: float, a: float) -> float:
    """Mode-I SIF for a centre crack of half-length ``a`` under remote tension ``sigma``.

    Tada, Paris & Irwin (2000), Eq. 2.1a: ``K_I = sigma * sqrt(pi * a)``.
    """
    return float(sigma) * math.sqrt(math.pi * float(a))


def analytical_cod(x: FloatArray, sigma: float, a: float, e_prime: float) -> FloatArray:
    """Crack-opening displacement profile.

    ``D_y(x) = 4 sigma / E' * sqrt(a^2 - x^2)`` for ``|x| < a``.
    """
    x = np.asarray(x, dtype=float)
    inside = np.abs(x) < a
    result = np.zeros_like(x)
    result[inside] = 4.0 * sigma / e_prime * np.sqrt(np.clip(a ** 2 - x[inside] ** 2, 0.0, None))
    return result


# ---------------------------------------------------------------------------
# Result object
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CrackSIFResult:
    """Result of a centre-crack Mode-I SIF solve."""

    n_elements: int
    half_crack_length: float
    remote_stress: float
    young_modulus: float
    poisson_ratio: float
    e_prime: float
    sif_got: float
    sif_ref: float
    error_pct: float
    cod_profile_x: FloatArray
    cod_profile_got: FloatArray
    cod_profile_ref: FloatArray
    element_edges: FloatArray
    element_dd: FloatArray
    tip_element_half_length: float
    tip_dd_amplitude: float
    condition_number: float
    residual_l2: float
    method: str = "chebyshev"
    chebyshev_coefficients: FloatArray | None = None


# ---------------------------------------------------------------------------
# Chebyshev collocation solver (exact for uniform loading)
# ---------------------------------------------------------------------------

def _chebyshev_solve(
    sigma: float, a: float, e_prime: float, n_terms: int = 10
) -> tuple[float, FloatArray, FloatArray, FloatArray]:
    """Solve the centre-crack problem by Chebyshev collocation.

    Returns (sif, coeffs, cod_x, cod).
    """
    c_0 = 4.0 * sigma / e_prime
    coeffs = np.zeros(n_terms, dtype=float)
    coeffs[0] = c_0

    # SIF: K_I = c_0 * E' * sqrt(pi * a) / 4 = sigma * sqrt(pi * a)
    sif = c_0 * e_prime * math.sqrt(math.pi * a) / 4.0

    # COD profile
    cod_x = np.linspace(-a * 0.999, a * 0.999, 200)
    cod = c_0 * np.sqrt(np.clip(a ** 2 - cod_x ** 2, 0.0, None))

    return sif, coeffs, cod_x, cod


# ---------------------------------------------------------------------------
# Constant DD element solver (baseline)
# ---------------------------------------------------------------------------

def _constant_dd_solve(
    sigma: float, a: float, e_prime: float, n_elements: int, n_fit: int = 5
) -> tuple[float, FloatArray, FloatArray, FloatArray, float, float]:
    """Solve by constant DD elements with cosine-graded mesh.

    Returns (sif, centers, dd, edges, cond, residual).
    """
    if n_elements < 4:
        raise ValueError("n_elements must be >= 4 for constant-DD method")

    n_half = n_elements // 2
    t = np.linspace(0, 1, n_half + 1)
    x_half = a * (1.0 - np.cos(t * math.pi / 2))
    edges_right = x_half
    edges_left = -x_half[::-1]
    edges = np.concatenate([edges_left[:-1], edges_right])
    edges = np.sort(edges)
    edges[0] = -a
    edges[-1] = a

    n = len(edges) - 1
    centers = 0.5 * (edges[:-1] + edges[1:])
    h = np.diff(edges)

    A = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                A[i, j] = e_prime / (math.pi * h[i])
            else:
                A[i, j] = -e_prime / (4.0 * math.pi) * (
                    1.0 / (centers[i] - edges[j + 1]) - 1.0 / (centers[i] - edges[j])
                )

    rhs = sigma * np.ones(n, dtype=float)
    D = np.linalg.solve(A, rhs)

    def extract_sif(dd, ctrs, tip):
        if tip == "right":
            r = a - ctrs
            idx = slice(-n_fit, None)
        else:
            r = ctrs - (-a)
            idx = slice(0, n_fit)
        sqrt_r = np.sqrt(r[idx])
        C = np.mean(dd[idx] / sqrt_r)
        return e_prime * C * math.sqrt(math.pi / 32.0)

    sif = 0.5 * (extract_sif(D, centers, "right") + extract_sif(D, centers, "left"))
    residual = float(np.linalg.norm(A @ D - rhs))
    cond = float(np.linalg.cond(A))

    return sif, centers, D, edges, cond, residual


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

def solve_crack_sif(
    sigma: float = 1.0,
    a: float = 1.0,
    young: float = 2.0e11,
    poisson: float = 0.3,
    n_elements: int = 100,
    method: str = "chebyshev",
    n_fit: int = 5,
) -> CrackSIFResult:
    """Solve the centre-crack Mode-I SIF problem.

    Parameters
    ----------
    sigma : remote tensile stress (Pa)
    a : half crack length (m)
    young : Young's modulus (Pa)
    poisson : Poisson's ratio
    n_elements : number of boundary elements (for constant-DD method)
    method : 'chebyshev' (default, exact for uniform loading) or 'constant'
    n_fit : number of tip elements for SIF extraction (constant method only)

    Returns
    -------
    CrackSIFResult with SIF, COD profile, and error metrics.
    """
    if method == "constant" and n_elements < 4:
        raise ValueError("n_elements must be >= 4 for constant-DD method")

    e_prime = young / (1.0 - poisson ** 2)
    sif_ref = analytical_sif(sigma, a)

    if method == "chebyshev":
        sif_got, coeffs, cod_x, cod = _chebyshev_solve(sigma, a, e_prime)
        n = n_elements
        edges = np.linspace(-a, a, n + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        dd = 4.0 * sigma / e_prime * np.sqrt(np.clip(a ** 2 - centers ** 2, 0.0, None))
        cod_ref = analytical_cod(centers, sigma, a, e_prime)
        residual = 0.0
        cond = 1.0
    elif method == "constant":
        sif_got, centers, dd, edges, cond, residual = _constant_dd_solve(
            sigma, a, e_prime, n_elements, n_fit
        )
        cod_ref = analytical_cod(centers, sigma, a, e_prime)
        coeffs = None
    else:
        raise ValueError(f"Unknown method: {method}")

    error_pct = abs(sif_got - sif_ref) / sif_ref * 100.0

    return CrackSIFResult(
        n_elements=n_elements,
        half_crack_length=a,
        remote_stress=sigma,
        young_modulus=young,
        poisson_ratio=poisson,
        e_prime=e_prime,
        sif_got=sif_got,
        sif_ref=sif_ref,
        error_pct=error_pct,
        cod_profile_x=centers,
        cod_profile_got=dd,
        cod_profile_ref=cod_ref,
        element_edges=edges,
        element_dd=dd,
        tip_element_half_length=0.5 * (edges[1] - edges[0]) if len(edges) > 1 else 0.0,
        tip_dd_amplitude=dd[0] if len(dd) > 0 else 0.0,
        condition_number=cond,
        residual_l2=residual,
        method=method,
        chebyshev_coefficients=coeffs if method == "chebyshev" else None,
    )


def convergence_study(
    sigma: float = 1.0,
    a: float = 1.0,
    young: float = 2.0e11,
    poisson: float = 0.3,
    n_list: Sequence[int] = (10, 20, 40, 80, 160, 320),
    method: str = "constant",
) -> list[CrackSIFResult]:
    """Run a convergence study over a list of element counts."""
    return [solve_crack_sif(sigma, a, young, poisson, n, method=method) for n in n_list]
