"""m17238 cross-audit: independent re-derivation of the CASE-BEM-WIGLEY3-WAVE-RES reference.

The audited case (m17206) grades TensorBEM's Kochin route and its centreplane panel
route against a "reference" that is itself produced by in-repo code
(``tensorbem.thin_ship_michell.wave_resistance_theta_form``).  That is a
self-consistency check unless somebody outside the audited code reproduces the
number.  This script is that outside check: a from-scratch implementation of the
classical Michell (1898) far-field integral, written only from the textbook form

    Rw = (4 rho g^2 / (pi U^2)) * int_1^inf  lam^2 |A(lam)|^2 / sqrt(lam^2-1) dlam,
    A(lam) = int int y_x(x, z) exp(k0 lam^2 z) exp(i k0 lam x) dx dz,

with ``y`` the half-breadth of the parabolic Wigley hull
``y = (B/2)(1-(2x/L)^2)(1-(z/T)^2)`` and ``y_x`` its longitudinal slope.  It
imports no ``tensorbem`` code, so agreement is evidence about the audited
numbers, not a restatement of them.

For the parabolic Wigley hull the (x, z) integral separates, so A(lam) is
evaluated with closed forms written here:

    int_{-a}^{a} x e^{i k x} dx      = 2i (sin(ka) - ka cos(ka)) / k^2,  a = L/2
    int_{-T}^{0} (1-(z/T)^2) e^{m z/T} dz
        = T [ (1-e^{-m})/m - (2 - e^{-m}(m^2+2m+2))/m^3 ],  m = k0 lam^2 T

Both closed forms are re-checked against brute-force Gauss quadrature inside
``_selfcheck_closed_forms`` (agreement < 1e-10), so a slip in the algebra here
cannot masquerade as a finding about the audited code.

The prefactor is the classical one (4 rho g^2 / (pi U^2)) applied to the
*one-side* half-breadth amplitude.  The audited report registers the absolute
scale / sides convention as WITHHELD; this script therefore also reports the
multiplicative factor between its own Rw and the audited Rw, so the convention
gap is measured instead of asserted.

Usage: python3 scripts/audit_m17238_michell_independent.py [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

# --- case parameters, read off the audited artifact (not recomputed from it) ---
L = 3.0
B = 0.3
T = 0.1875
RHO = 1025.0
G = 9.80665
FR = (0.20, 0.25, 0.30, 0.35, 0.40, 0.45)

N_LAM_PANELS = 500   # composite Gauss panels along s in [0, sqrt(lam_max-1)]
N_LAM_ORDER = 8      # Gauss order per panel  -> 4000 nodes
LAM_MAX = 30.0


def _composite_gauss(n_panels: int, order: int, a: float, b: float) -> tuple[np.ndarray, np.ndarray]:
    """Composite Gauss-Legendre (leggauss stays at low order: high orders are O(n^2) slow here)."""
    xi, wi = np.polynomial.legendre.leggauss(order)
    edges = np.linspace(a, b, n_panels + 1)
    lo, hi = edges[:-1], edges[1:]
    half = 0.5 * (hi - lo)
    mid = 0.5 * (hi + lo)
    nodes = (mid[:, None] + half[:, None] * xi[None, :]).ravel()
    weights = (half[:, None] * wi[None, :]).ravel()
    return nodes, weights


def _x_integral(k: np.ndarray) -> np.ndarray:
    """int_{-L/2}^{L/2} x e^{i k x} dx (odd integrand -> 2i * Im part)."""
    a = L / 2.0
    k = np.asarray(k, dtype=float)
    ka = k * a
    return 2j * (np.sin(ka) - ka * np.cos(ka)) / k**2


def _z_integral(m: np.ndarray) -> np.ndarray:
    """int_{-T}^{0} (1-(z/T)^2) e^{(m/T) z} dz, with m = k0 lam^2 T."""
    m = np.asarray(m, dtype=float)
    e = np.exp(-m)
    return T * ((1.0 - e) / m - (2.0 - e * (m * m + 2.0 * m + 2.0)) / m**3)


def _selfcheck_closed_forms() -> list[dict]:
    """Brute-force quadrature vs the closed forms above (independent of tensorbem).

    Panel counts are chosen from the local oscillation rate (k*a for x, m for z) so
    that the *check* cannot be the thing that is under-resolved.
    """
    a = L / 2.0
    rows = []
    for lam in (1.0, 2.0, 5.0, 12.0, 30.0):
        k = G / (0.2 * math.sqrt(G * L)) ** 2 * lam
        m = k * T * lam
        xq, wx = _composite_gauss(max(40, int(math.ceil(k * a / 2.0))), 8, -a, a)
        zq, wz = _composite_gauss(max(40, int(math.ceil(m / 2.0))), 8, -T, 0.0)
        num_x = complex((xq * np.exp(1j * k * xq)) @ wx)
        num_z = complex((((1.0 - (zq / T) ** 2) * np.exp(m * zq / T)) @ wz))
        cf_x = complex(_x_integral(np.array([k]))[0])
        cf_z = complex(_z_integral(np.array([m]))[0])
        rows.append({"lam": lam,
                     "x_rel_diff": abs(num_x - cf_x) / max(abs(cf_x), 1e-300),
                     "z_rel_diff": abs(num_z - cf_z) / max(abs(cf_z), 1e-300)})
    return rows


def amplitude(lam: np.ndarray, k0: float) -> np.ndarray:
    """Kochin amplitude A(lam) of the centreplane sheet (one-side half-breadth)."""
    lam = np.atleast_1d(np.asarray(lam, dtype=float))
    k = k0 * lam
    return -(4.0 * B / L**2) * _x_integral(k) * _z_integral(k0 * lam**2 * T)


def wave_resistance(fr: float, lam_max: float = LAM_MAX) -> dict:
    speed = fr * math.sqrt(G * L)
    k0 = G / speed**2
    # lam = 1 + s^2 removes the 1/sqrt(lam-1) endpoint singularity
    s_max = math.sqrt(lam_max - 1.0)
    ss, ws = _composite_gauss(N_LAM_PANELS, N_LAM_ORDER, 0.0, s_max)
    lam = 1.0 + ss**2
    a = amplitude(lam, k0)
    # dlam = 2 s ds ; sqrt(lam^2-1) = s sqrt(2+s^2) -> integrand 2 lam^2 |A|^2 / sqrt(2+s^2)
    integrand = 2.0 * lam**2 * np.abs(a) ** 2 / np.sqrt(2.0 + ss**2)
    integral = float(integrand @ ws)
    return {"froude_number": fr, "speed": speed, "k0": k0,
            "Rw_independent": 4.0 * RHO * G**2 / (math.pi * speed**2) * integral}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="artifacts/m17238_audit/audited_results.json",
                    help="copy of the audited results.json")
    ap.add_argument("--out", default="artifacts/m17238_audit/independent_michell.json")
    args = ap.parse_args(argv)

    ref_path = Path(args.ref)
    audited = json.loads(ref_path.read_text()) if ref_path.exists() else None
    ref_rows = {}
    if audited:
        for row in audited["reference"]["cases"]:
            ref_rows[round(row["froude_number"], 4)] = row["Rw"]

    rows = []
    for fr in FR:
        mine = wave_resistance(fr)
        theirs = ref_rows.get(round(fr, 4))
        rows.append({**mine, "Rw_audited_reference": theirs,
                     "rel_diff_vs_audited": (abs(mine["Rw_independent"] - theirs) / theirs
                                             if theirs else float("nan")),
                     "ratio_audited_over_independent": (theirs / mine["Rw_independent"]
                                                        if theirs else float("nan"))})

    trunc = []
    for fr in FR:
        hi = wave_resistance(fr, lam_max=4.0 * LAM_MAX)
        lo = next(r for r in rows if r["froude_number"] == fr)["Rw_independent"]
        trunc.append({"froude_number": fr,
                      "rel_drift_lam30_to_lam120": abs(hi["Rw_independent"] - lo) / lo})

    ratios = [r["ratio_audited_over_independent"] for r in rows]
    out = {
        "case_id": "CASE-BEM-WIGLEY3-WAVE-RES",
        "audit_mission": 17238,
        "audited_mission": 17206,
        "implementation": "from-scratch classical Michell far field, no tensorbem import",
        "quadrature": {"n_lambda": N_LAM_PANELS * N_LAM_ORDER, "lambda_max": LAM_MAX,
                       "mapping": "lambda = 1 + s^2"},
        "prefactor": "4 rho g^2 / (pi U^2) on the one-side half-breadth amplitude",
        "closed_form_selfcheck": _selfcheck_closed_forms(),
        "rows": rows,
        "ratio_audited_over_independent": {"min": min(ratios), "max": max(ratios),
                                           "mean": float(np.mean(ratios))},
        "max_rel_diff_vs_audited": max(r["rel_diff_vs_audited"] for r in rows),
        "own_truncation_check": trunc,
    }
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"closed_form_selfcheck_max_rel_diff":
                      max(max(r["x_rel_diff"], r["z_rel_diff"]) for r in out["closed_form_selfcheck"]),
                      "max_rel_diff_vs_audited": out["max_rel_diff_vs_audited"],
                      "ratio_band": out["ratio_audited_over_independent"]}, indent=2))
    for r in rows:
        print(f"Fr={r['froude_number']:.2f}  mine={r['Rw_independent']:.6e}  "
              f"audited={r['Rw_audited_reference']:.6e}  "
              f"rel={r['rel_diff_vs_audited']:.3e}  ratio={r['ratio_audited_over_independent']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())