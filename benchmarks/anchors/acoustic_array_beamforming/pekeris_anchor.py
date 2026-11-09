"""Pekeris two-layer waveguide anchor machinery for case ``acoustic-array-beamforming``.

Purpose
-------
This module implements the *verifiable* half of the external-anchor check demanded by
task ``paper-exp-acoustic-array-beamforming-1``:

1. ``derive_dispersion``      -- symbolic derivation of the Pekeris branch equation from
                                the Helmholtz equation + interface conditions, plus a
                                dimensional-homogeneity machine check.
2. ``pekeris_roots``          -- bracketed root solver for the trapped modes, with two
                                independent residual forms reported per root.
3. ``analytic_mode_count``    -- a closed-form trapped-mode count derived independently
                                of the solver, used to cross-check the solver.
4. ``fem_branch_roots``       -- a fully independent discretisation (1-D P1 finite
                                elements of the pressure modal problem) used as a
                                numerical cross-check + convergence evidence.
5. ``to_case_frame``          -- the conversion chain between the representations used by
                                the literature sources (k_r in rad/m, phase velocity,
                                k_r*H, grazing angle) and the case frame (s = k_r/k1),
                                with a round-trip identity residual.
6. ``validate_quote_record``  -- a structural gate: an external value is only eligible
                                for a ``VERIFIED`` anchor status if a *verbatim* quote, a
                                table/page locator and a source identifier are all
                                present.  Records without them are rejected, which is what
                                keeps hand-typed numbers out of claim statements.

Nothing in this module reads or embeds a literature *number*: every number produced here
is computed.  Literature numbers may only enter through ``quote_records`` and only
through the gate.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

# --------------------------------------------------------------------------------------
# dimensional algebra
# --------------------------------------------------------------------------------------

#: dimension vectors as exponents of (M, L, T)
DIM = {
    "dimensionless": (0, 0, 0),
    "length": (0, 1, 0),
    "inv_length": (0, -1, 0),
    "time": (0, 0, 1),
    "inv_time": (0, 0, -1),
    "speed": (0, 1, -1),
    "density": (1, -3, 0),
}


def dim_mul(*dims: Sequence[int]) -> tuple:
    out = [0, 0, 0]
    for d in dims:
        for i in range(3):
            out[i] += int(d[i])
    return tuple(out)


def dim_pow(d: Sequence[int], p: int) -> tuple:
    return tuple(int(x) * int(p) for x in d)


def dim_of_check(name: str, expected: str, actual: Sequence[int]) -> dict:
    exp = DIM[expected]
    act = tuple(int(x) for x in actual)
    return {
        "quantity": name,
        "expected_dimension": expected,
        "expected_vector_M_L_T": list(exp),
        "actual_vector_M_L_T": list(act),
        "homogeneous": act == tuple(int(x) for x in exp),
    }


def derive_dispersion() -> dict:
    """Derive the Pekeris branch equation symbolically and check it dimensionally.

    Layer ``0 <= z <= H``:   p1 = A sin(k_z z)          with k_z^2 = k1^2 - k_r^2
    Half-space ``z > H``:    p2 = B exp(-alpha (z-H))  with alpha^2 = k_r^2 - k2^2

    Interface conditions: continuity of pressure and of ``(1/rho) dp/dz``.
    Eliminating A and B gives ``tan(k_z H) = -(rho2/rho1) k_z / alpha``.
    """
    import sympy as sp

    z, H, kz, kr, k1, k2, alpha, rho1, rho2, A, B = sp.symbols(
        "z H k_z k_r k_1 k_2 alpha rho_1 rho_2 A B", positive=True
    )
    p1 = A * sp.sin(kz * z)
    p2 = B * sp.exp(-alpha * (z - H))

    helm1 = sp.diff(p1, z, 2) + (k1**2 - kr**2) * p1
    helm2 = sp.diff(p2, z, 2) + (k2**2 - kr**2) * p2
    helm1_def = sp.simplify(helm1.subs(k1**2, kz**2 + kr**2))
    helm2_def = sp.simplify(helm2.subs(kr**2, alpha**2 + k2**2))

    pres = sp.Eq(p1.subs(z, H), p2.subs(z, H))
    flux = sp.Eq(sp.diff(p1, z).subs(z, H) / rho1, sp.diff(p2, z).subs(z, H) / rho2)

    Bsol = sp.solve(pres, B)[0]
    flux_B = flux.subs(B, Bsol)
    # flux_B / A reads:  k_z cos(k_z H)/rho1 + alpha sin(k_z H)/rho2 = 0
    lhsA = sp.simplify(sp.expand(flux_B.lhs - flux_B.rhs) / A)
    # with sin = tan*cos (valid at a root, where cos != 0) the condition becomes
    #   cos(k_z H) * [ k_z/rho1 + alpha tan(k_z H)/rho2 ] = 0
    target = -(rho2 / rho1) * kz / alpha
    expr = sp.expand(lhsA.subs(sp.sin(kz * H), sp.tan(kz * H) * sp.cos(kz * H)))
    lhs_div_cos = sp.simplify(expr / sp.cos(kz * H))
    # substituting the branch equation into the flux condition must give identically 0
    residual = sp.simplify(expr.subs(sp.tan(kz * H), target))

    checks = [
        dim_of_check("k_z", "inv_length", DIM["inv_length"]),
        dim_of_check("k_r", "inv_length", DIM["inv_length"]),
        dim_of_check("alpha", "inv_length", DIM["inv_length"]),
        dim_of_check("k_z*H", "dimensionless", dim_mul(DIM["inv_length"], DIM["length"])),
        dim_of_check(
            "rho2/rho1", "dimensionless",
            dim_mul(DIM["density"], dim_pow(DIM["density"], -1)),
        ),
        dim_of_check(
            "k_z/alpha", "dimensionless",
            dim_mul(DIM["inv_length"], dim_pow(DIM["inv_length"], -1)),
        ),
        dim_of_check(
            "k1=omega/c1", "inv_length",
            dim_mul(DIM["inv_time"], dim_pow(DIM["speed"], -1)),
        ),
        dim_of_check(
            "s=k_r/k1", "dimensionless",
            dim_mul(DIM["inv_length"], dim_pow(DIM["inv_length"], -1)),
        ),
        dim_of_check(
            "c_ph=omega/k_r", "speed",
            dim_mul(DIM["inv_time"], dim_pow(DIM["inv_length"], -1)),
        ),
        dim_of_check(
            "k_r*H", "dimensionless", dim_mul(DIM["inv_length"], DIM["length"])
        ),
        dim_of_check(
            "lambda1=c1/f", "length", dim_mul(DIM["speed"], DIM["time"])
        ),
    ]

    return {
        "ansatz": {
            "layer": "p1(z) = A sin(k_z z), 0 <= z <= H",
            "halfspace": "p2(z) = B exp(-alpha (z-H)), z > H",
            "branch_definitions": "k_z^2 = k1^2 - k_r^2 ; alpha^2 = k_r^2 - k2^2",
        },
        "helmholtz_residual_layer": str(helm1_def),
        "helmholtz_residual_halfspace": str(helm2_def),
        "interface_conditions": [str(pres), str(flux)],
        "flux_after_pressure_elimination": str(lhsA),
        "flux_condition_over_cos": str(lhs_div_cos),
        "branch_equation_candidate": str(target),
        "interface_elimination_residual": str(residual),
        "branch_equation_tex": r"\tan(k_z H) = -\frac{\rho_2}{\rho_1}\,\frac{k_z}{\alpha}",
        "branch_equation_text": "tan(k_z H) + (rho2/rho1) k_z / alpha = 0",
        "residual_form_used": "F(k_z) = alpha sin(k_z H) + m k_z cos(k_z H), m = rho2/rho1",
        "sympy_residual_is_zero": bool(
            sp.simplify(helm1_def) == 0 and sp.simplify(helm2_def) == 0 and residual == 0
        ),
        "dimensional_checks": checks,
        "dimensional_checks_all_homogeneous": all(c["homogeneous"] for c in checks),
        "note": (
            "The branch equation is derived here, not transcribed; the derivation fixes the "
            "functional form used by the solver and the conversion chain and uses no "
            "literature number."
        ),
    }


# --------------------------------------------------------------------------------------
# waveguide / case definition
# --------------------------------------------------------------------------------------

@dataclass
class Waveguide:
    """Ideal two-layer (Pekeris) waveguide plus the case-frame array definition."""

    depth_m: float = 100.0
    c_water_m_s: float = 1500.0
    c_bottom_m_s: float = 1600.0
    rho_water_kg_m3: float = 1000.0
    rho_bottom_kg_m3: float = 1800.0
    n_elements: int = 32

    @property
    def density_ratio(self) -> float:
        return self.rho_bottom_kg_m3 / self.rho_water_kg_m3

    @property
    def critical_ratio(self) -> float:
        """Trapped modes require s = k_r/k1 in (c1/c2, 1)."""
        return self.c_water_m_s / self.c_bottom_m_s

    def k1(self, f: float) -> float:
        return 2.0 * math.pi * f / self.c_water_m_s

    def k2(self, f: float) -> float:
        return 2.0 * math.pi * f / self.c_bottom_m_s

    def to_dict(self) -> dict:
        return {
            "water_depth_m": self.depth_m,
            "c_water_m_s": self.c_water_m_s,
            "c_bottom_m_s": self.c_bottom_m_s,
            "rho_water_kg_m3": self.rho_water_kg_m3,
            "rho_bottom_kg_m3": self.rho_bottom_kg_m3,
            "density_ratio_rho2_over_rho1": self.density_ratio,
            "trapped_window_s_in": [self.critical_ratio, 1.0],
            "array_n_elements": self.n_elements,
        }


# --------------------------------------------------------------------------------------
# root solver
# --------------------------------------------------------------------------------------

def residual_F(kz: float, k1: float, k2: float, m: float, H: float) -> tuple:
    """Return (F, G, scale) for the branch equation.

    F(k_z) = alpha sin(k_z H) + m k_z cos(k_z H)        (regular, used for bracketing)
    G(k_z) = tan(k_z H) + m k_z / alpha                 (singular, used as 2nd form)
    """
    kr2 = k1 * k1 - kz * kz
    alpha2 = kr2 - k2 * k2
    alpha = math.sqrt(alpha2) if alpha2 > 0.0 else 0.0
    th = kz * H
    F = alpha * math.sin(th) + m * kz * math.cos(th)
    G = math.tan(th) + m * kz / alpha if alpha > 0.0 else math.inf
    scale = m * math.sqrt(max(k1 * k1 - k2 * k2, 0.0))
    return F, G, scale


def dresidual_F_dkz(kz: float, k1: float, k2: float, m: float, H: float) -> float:
    """Analytic dF/dk_z, used for the Newton polish of each bracketed root."""
    kz2max = k1 * k1 - k2 * k2
    alpha = math.sqrt(max(kz2max - kz * kz, 1e-300))
    th = kz * H
    dalpha = -kz / alpha
    return (dalpha * math.sin(th) + alpha * H * math.cos(th)
            + m * math.cos(th) - m * kz * H * math.sin(th))


@dataclass
class Root:
    freq_hz: float
    branch_index: int
    k_z: float
    k_r: float
    s: float
    grazing_angle_deg: float
    phase_velocity_m_s: float
    residual_F_abs: float = 0.0
    residual_F_rel: float = 0.0
    residual_G_rel: float = 0.0

    def as_dict(self) -> dict:
        return {
            "freq_hz": self.freq_hz,
            "branch_index": self.branch_index,
            "k_z_rad_m": self.k_z,
            "k_r_rad_m": self.k_r,
            "s_kr_over_k1": self.s,
            "grazing_angle_deg": self.grazing_angle_deg,
            "phase_velocity_m_s": self.phase_velocity_m_s,
            "residual_F_abs": self.residual_F_abs,
            "residual_F_rel": self.residual_F_rel,
            "residual_G_rel": self.residual_G_rel,
        }


def analytic_mode_count(wg: Waveguide, f: float) -> int:
    """Trapped-mode count from the bracket structure of the branch equation.

    Branch ``n`` sits in ``k_z H in (pi/2 + n pi, pi + n pi)``; a branch only exists if
    its left edge is below ``k_z_max H`` with ``k_z_max = sqrt(k1^2 - k2^2)``.
    Hence ``N = 1 + floor(X/pi - 1/2)`` for ``X = k_z_max H > pi/2``, else 0.
    """
    k1, k2 = wg.k1(f), wg.k2(f)
    if k2 >= k1:
        return 0
    X = math.sqrt(k1 * k1 - k2 * k2) * wg.depth_m
    if X <= 0.5 * math.pi:
        return 0
    return 1 + int(math.floor(X / math.pi - 0.5))


def pekeris_roots(wg: Waveguide, f: float) -> list:
    """All trapped modes at frequency ``f`` via bracketed Brent root finding."""
    from scipy.optimize import brentq

    k1, k2, m, H = wg.k1(f), wg.k2(f), wg.density_ratio, wg.depth_m
    if k2 >= k1:
        return []
    kz_max = math.sqrt(k1 * k1 - k2 * k2)
    theta_max = kz_max * H
    roots = []
    n = 0
    while 0.5 * math.pi + n * math.pi < theta_max:
        lo = (0.5 * math.pi + n * math.pi) * (1.0 + 1e-13) + 1e-12
        hi = min((n + 1) * math.pi, theta_max)
        hi = hi * (1.0 - 1e-13)
        if hi <= lo:
            break
        flo = residual_F(lo / H, k1, k2, m, H)[0]
        fhi = residual_F(hi / H, k1, k2, m, H)[0]
        if flo * fhi > 0.0:  # pragma: no cover - guarded by bracket structure
            n += 1
            continue
        theta = brentq(
            lambda t: residual_F(t / H, k1, k2, m, H)[0], lo, hi,
            xtol=1e-12, rtol=1e-12, maxiter=120,
        )
        kz = theta / H
        for _ in range(4):  # Newton polish: residual -> machine precision
            F = residual_F(kz, k1, k2, m, H)[0]
            dF = dresidual_F_dkz(kz, k1, k2, m, H)
            if dF == 0.0 or not math.isfinite(dF):
                break
            kz_new = kz - F / dF
            if not (0.0 < kz_new < kz_max):
                break
            if abs(kz_new - kz) <= 4.0 * sys.float_info.epsilon * max(kz, 1.0):
                kz = kz_new
                break
            kz = kz_new
        kr = math.sqrt(k1 * k1 - kz * kz)
        F, G, scale = residual_F(kz, k1, k2, m, H)
        roots.append(
            Root(
                freq_hz=f,
                branch_index=n,
                k_z=kz,
                k_r=kr,
                s=kr / k1,
                grazing_angle_deg=math.degrees(math.acos(kr / k1)),
                phase_velocity_m_s=2.0 * math.pi * f / kr,
                residual_F_abs=abs(F),
                residual_F_rel=abs(F) / scale if scale else math.inf,
                residual_G_rel=abs(G) / (1.0 + abs(m * kz)) if math.isfinite(G) else math.inf,
            )
        )
        n += 1
    return roots


# --------------------------------------------------------------------------------------
# independent discretisation: 1-D P1 finite elements for the pressure modal problem
# --------------------------------------------------------------------------------------

def _element_matrices(h: float, rho: float, omega: float, c: float) -> tuple:
    inv_r = 1.0 / rho
    Ke = inv_r / h * np.array([[1.0, -1.0], [-1.0, 1.0]])
    Me = inv_r * h / 6.0 * np.array([[2.0, 1.0], [1.0, 2.0]])
    Om = omega**2 * inv_r / c**2 * h / 6.0 * np.array([[2.0, 1.0], [1.0, 2.0]])
    return Ke, Me, Om


def fem_branch_roots(wg: Waveguide, f: float, n_layer: int = 400, growth: float = 1.04) -> list:
    """Wavenumbers from a 1-D P1 FEM of ``(1/rho p')' + (omega^2/(rho c^2) - k_r^2/rho) p = 0``.

    Weak form: ``K u + k_r^2 M u = Omega u`` with ``K = int (1/rho) p' v'``,
    ``M = int (1/rho) p v``, ``Omega = int omega^2/(rho c^2) p v``.  Dirichlet at the
    surface and at a truncation depth.  This route shares no algebra with the branch
    equation, so it is an independent check of the solver.
    """
    omega = 2.0 * math.pi * f
    H = wg.depth_m
    k1, k2 = wg.k1(f), wg.k2(f)
    if k2 >= k1:
        return []
    # truncation: 6 e-folding lengths of the slowest decaying trapped mode
    kr_min = k2
    alpha_min = math.sqrt(max(kr_min * kr_min - k2 * k2, 1e-12))
    L = max(6.0 / alpha_min, 3.0 * wg.c_bottom_m_s / f)

    z = list(np.linspace(0.0, H, n_layer + 1))
    h0 = H / n_layer
    h = h0
    while z[-1] < H + L:
        h *= growth
        z.append(min(z[-1] + h, H + L))
    z = np.array(z)
    n = len(z)

    K = np.zeros((n, n))
    M = np.zeros((n, n))
    Om = np.zeros((n, n))
    for e in range(n - 1):
        zc = 0.5 * (z[e] + z[e + 1])
        if zc <= H:
            rho, c = wg.rho_water_kg_m3, wg.c_water_m_s
        else:
            rho, c = wg.rho_bottom_kg_m3, wg.c_bottom_m_s
        he = z[e + 1] - z[e]
        Ke, Me, Oe = _element_matrices(he, rho, omega, c)
        idx = [e, e + 1]
        for a in range(2):
            for b in range(2):
                K[idx[a], idx[b]] += Ke[a, b]
                M[idx[a], idx[b]] += Me[a, b]
                Om[idx[a], idx[b]] += Oe[a, b]
    # Dirichlet: surface (z=0) and truncation (z=z_max)
    keep = np.arange(1, n - 1)
    K, M, Om = K[np.ix_(keep, keep)], M[np.ix_(keep, keep)], Om[np.ix_(keep, keep)]
    A = Om - K
    # generalised symmetric eigenproblem A u = lambda M u with M SPD -> Cholesky route
    from scipy.linalg import eigh
    Lc = np.linalg.cholesky(M)
    Linv = np.linalg.inv(Lc)
    S = Linv @ A @ Linv.T
    S = 0.5 * (S + S.T)
    n_keep = min(8, S.shape[0])
    lam = eigh(S, subset_by_index=[S.shape[0] - n_keep, S.shape[0] - 1],
               eigvals_only=True)      # largest k_r^2 only
    lam = lam[::-1]                       # descending k_r^2
    kr = np.sqrt(np.clip(lam, 0.0, None))
    # trapped modes: k_r^2 > k2^2 (energetically allowed in both media)
    return [float(v) for v in kr if v > k2 * 1.0000001]


# --------------------------------------------------------------------------------------
# conversion chain: literature representation -> case frame
# --------------------------------------------------------------------------------------

def to_case_frame(wg: Waveguide, f: float, representation: dict) -> dict:
    """Convert one literature-style representation of a modal wavenumber to the case frame.

    Supported representations (exactly one key must be present):

    ``k_r_rad_m``     horizontal wavenumber in rad/m
    ``c_ph_m_s``      modal phase velocity in m/s
    ``k_r_times_H``   dimensionless product k_r * H
    ``grazing_deg``   grazing angle of the modal plane wave in degrees

    The case frame is ``s = k_r / k1`` with ``k1 = 2 pi f / c1`` (dimensionless), from
    which every other representation is regenerated for the round-trip residual.
    """
    k1 = wg.k1(f)
    keys = [k for k in ("k_r_rad_m", "c_ph_m_s", "k_r_times_H", "grazing_deg")
            if k in representation and representation[k] is not None]
    if len(keys) != 1:
        raise ValueError(f"exactly one representation key required, got {keys}")
    key = keys[0]
    val = float(representation[key])
    if key == "k_r_rad_m":
        kr = val
        route = "s = k_r / (2 pi f / c1)"
    elif key == "c_ph_m_s":
        kr = 2.0 * math.pi * f / val
        route = "s = c1 / c_ph"
    elif key == "k_r_times_H":
        kr = val / wg.depth_m
        route = "s = (k_r H) c1 / (2 pi f H)"
    else:
        kr = k1 * math.cos(math.radians(val))
        route = "s = cos(theta_grazing)"
    s = kr / k1
    regen = {
        "k_r_rad_m": kr,
        "c_ph_m_s": 2.0 * math.pi * f / kr,
        "k_r_times_H": kr * wg.depth_m,
        "grazing_deg": math.degrees(math.acos(min(1.0, max(-1.0, s)))),
    }
    # round trip: from s back through every representation, residual vs the input
    two_pi_f = 2.0 * math.pi * f
    back = {
        "k_r_rad_m": regen["k_r_rad_m"] / k1,
        "c_ph_m_s": wg.c_water_m_s / regen["c_ph_m_s"],
        "k_r_times_H": regen["k_r_times_H"] * wg.c_water_m_s / (two_pi_f * wg.depth_m),
        "grazing_deg": math.cos(math.radians(regen["grazing_deg"])),
    }
    resid = {k: abs(back[k] - s) for k in back}
    return {
        "freq_hz": f,
        "input": {key: val},
        "conversion_route": route,
        "s_kr_over_k1": s,
        "regenerated": regen,
        "round_trip_s": back,
        "round_trip_abs_residual": resid,
        "round_trip_max_abs_residual": max(resid.values()),
        "trapped": bool(wg.critical_ratio < s < 1.0),
    }


def array_frame(wg: Waveguide, root: "Root", f: float) -> dict:
    """Case-frame array quantities implied by a modal wavenumber (all computed)."""
    lam1 = wg.c_water_m_s / f
    lam_r = 2.0 * math.pi / root.k_r
    return {
        "wavelength_water_m": lam1,
        "horizontal_wavelength_m": lam_r,
        "d_half_wavelength_m": 0.5 * lam1,
        "d_modal_nyquist_m": 0.5 * lam_r,
        "aperture_m": wg.n_elements * 0.5 * lam1,
        "phase_velocity_m_s": root.phase_velocity_m_s,
        "vertical_wavenumber_rad_m": root.k_z,
        "halfspace_decay_rad_m": math.sqrt(max(root.k_r**2 - wg.k2(f) ** 2, 0.0)),
    }


# --------------------------------------------------------------------------------------
# quote gate (anti-fabrication, structural)
# --------------------------------------------------------------------------------------

REQUIRED_QUOTE_FIELDS = (
    "source_id",
    "verbatim_quote",
    "locator",
    "table_or_page",
    "quantity",
    "value",
    "units",
)


def validate_quote_record(record: dict) -> dict:
    """Return a gate verdict for one candidate external value.

    A record is ``ELIGIBLE`` only when it carries a non-empty verbatim quote, a
    table/page locator, a source identifier and a numeric value with units.  Anything
    else is ``REJECTED`` with the reasons; rejected records must never be cited as an
    anchor for a claim.
    """
    problems = []
    for f in REQUIRED_QUOTE_FIELDS:
        if f not in record or record[f] in (None, "", [], {}):
            problems.append(f"missing_field:{f}")
    if "value" in record and record["value"] is not None:
        try:
            float(record["value"])
        except (TypeError, ValueError):
            problems.append("value_not_numeric")
    if "verbatim_quote" in record and isinstance(record["verbatim_quote"], str):
        if len(record["verbatim_quote"].strip()) < 8:
            problems.append("verbatim_quote_too_short")
    return {
        "record": record,
        "verdict": "ELIGIBLE" if not problems else "REJECTED",
        "reasons": problems,
    }


def anchor_status(records: Sequence[dict]) -> dict:
    """Aggregate gate verdicts into the anchor status used by the artifact."""
    verdicts = [validate_quote_record(r) for r in records]
    eligible = [v for v in verdicts if v["verdict"] == "ELIGIBLE"]
    return {
        "n_records": len(verdicts),
        "n_eligible": len(eligible),
        "status": "VERIFIED" if eligible else "WITHHELD",
        "verdicts": verdicts,
        "rule": (
            "status is VERIFIED only if at least one record passes validate_quote_record; "
            "otherwise WITHHELD and no external number may support a compliance claim"
        ),
    }