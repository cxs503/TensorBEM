#!/usr/bin/env python3
"""Independent audit re-derivation for m17277 / CASE-BEM-QTF-DRIFT.

Deliberately does NOT import ``tensorbem.cylinder_drift``: the MacCamy-Fuchs
modes are rebuilt from ``scipy.special`` (jv / jvp / hankel1 / h1vp), the
waterline + surface azimuthal integrals and BOTH mean-drift anchors (near
field and reduced Maruo far field) are re-assembled here, then compared with

  * ``tensorbem.cylinder_qtf``  -- the delivered closed-form reference, and
  * ``tensorbem.cylinder_drift`` -- the in-repo anchors the delivery leans on.

Purpose: separate "the reference is a real machine-precision construction"
from "the reference is the same code re-exported" (the delivery's diagonal
identity shares ``cylinder_wall_field`` with the near-field anchor, so the
audit has to supply its own first-order field to make the check independent).

Run: PYTHONPATH=<TensorBEM>/src python3 docs/board_audit/m17328/audit_independent_qtf.py
"""
from __future__ import annotations

import json
import math
import sys

import numpy as np
from scipy.special import h1vp, hankel1, jv, jvp

RHO = 1025.0
G = 9.81
M_MAX = 14


def modes(ka: float, m_max: int = M_MAX) -> np.ndarray:
    """A_m = -i^m J_m'(ka) / H_m^(1)'(ka), m = -m_max..m_max (scipy)."""
    out = np.zeros(2 * m_max + 1, dtype=np.complex128)
    for m in range(-m_max, m_max + 1):
        out[m + m_max] = -1j**m * jvp(m, ka) / h1vp(m, ka)
    return out


def field(ka: float, theta: np.ndarray, m_max: int = M_MAX) -> dict[str, np.ndarray]:
    """psi and dpsi/dr on r = a (scipy Bessel/Hankel, independent code path)."""
    am = modes(ka, m_max)
    psi = np.zeros(theta.shape, dtype=complex)
    dr = np.zeros(theta.shape, dtype=complex)
    for m in range(-m_max, m_max + 1):
        a = am[m + m_max]
        psi += (1j**m * jv(m, ka) + a * hankel1(m, ka)) * np.exp(1j * m * theta)
        dr += (1j**m * jvp(m, ka) + a * h1vp(m, ka)) * ka * np.exp(1j * m * theta)
    return {"psi": psi, "dr": dr}


def dpsi_dtheta_own(ka: float, theta: np.ndarray, m_max: int = M_MAX) -> np.ndarray:
    am = modes(ka, m_max)
    out = np.zeros(theta.shape, dtype=complex)
    for m in range(-m_max, m_max + 1):
        out += 1j * m * (1j**m * jv(m, ka) + am[m + m_max] * hankel1(m, ka)) * np.exp(1j * m * theta)
    return out


def mean_drift_near_field(ka: float, n_theta: int = 4096) -> float:
    """Near-field (Pinkster) mean drift of the fixed cylinder, own quadrature."""
    theta = 2.0 * math.pi / n_theta * np.arange(n_theta)
    dth = 2.0 * math.pi / n_theta
    f = field(ka, theta)
    # d/dtheta via the modal sum (analytic in theta, no finite differences)
    am = modes(ka)
    dpsi_dtheta = np.zeros(theta.shape, dtype=complex)
    for m in range(-M_MAX, M_MAX + 1):
        a = am[m + M_MAX]
        dpsi_dtheta += 1j * m * (1j**m * jv(m, ka) + a * hankel1(m, ka)) * np.exp(1j * m * theta)
    grad_sq = np.abs(f["dr"]) ** 2 + np.abs(dpsi_dtheta) ** 2 + ka**2 * np.abs(f["psi"]) ** 2
    i1 = float(np.sum(grad_sq * np.cos(theta)) * dth)
    i2 = float(np.sum(np.abs(f["psi"]) ** 2 * np.cos(theta)) * dth)
    gkw = G / (ka / 1.0)  # (gA/omega)^2, A=1, a=1
    return 0.25 * RHO * gkw / (2.0 * ka) * 1.0 * i1 - 0.25 * RHO * G * 1.0 * i2


def kochin_own(ka: float, theta: np.ndarray) -> np.ndarray:
    am = modes(ka)
    s = np.zeros(theta.shape, dtype=complex)
    for m in range(-M_MAX, M_MAX + 1):
        s += am[m + M_MAX] * np.exp(-0.5j * m * math.pi) * np.exp(1j * m * theta)
    return s


def mean_drift_far_field(ka: float, n_theta: int = 4096) -> float:
    theta = 2.0 * math.pi / n_theta * np.arange(n_theta)
    dth = 2.0 * math.pi / n_theta
    pattern = np.abs(kochin_own(ka, theta)) ** 2
    integral = float(np.sum(pattern * (1.0 - np.cos(theta))) * dth)
    return RHO * G * integral / (2.0 * math.pi * (ka / 1.0))


def azimuthal_integrals(ka_i: float, ka_j: float, n_theta: int = 4096):
    """I1, I2 of the delivered closed form, own field + own quadrature."""
    theta = 2.0 * math.pi / n_theta * np.arange(n_theta)
    dth = 2.0 * math.pi / n_theta
    fi, fj = field(ka_i, theta)["psi"], field(ka_j, theta)["psi"]
    am_i, am_j = modes(ka_i), modes(ka_j)
    dti = np.zeros(theta.shape, dtype=complex)
    dtj = np.zeros(theta.shape, dtype=complex)
    for m in range(-M_MAX, M_MAX + 1):
        dti += 1j * m * (1j**m * jv(m, ka_i) + am_i[m + M_MAX] * hankel1(m, ka_i)) * np.exp(1j * m * theta)
        dtj += 1j * m * (1j**m * jv(m, ka_j) + am_j[m + M_MAX] * hankel1(m, ka_j)) * np.exp(1j * m * theta)
    prod = fi * np.conj(fj)
    i1 = complex(np.sum((dti * np.conj(dtj) + ka_i * ka_j * prod) * np.cos(theta)) * dth)
    i2 = complex(np.sum(prod * np.cos(theta)) * dth)
    return i1.real, i2.real


def main() -> int:
    from tensorbem.cylinder_drift import cylinder_wall_field, long_wave_drift_coefficient
    from tensorbem.cylinder_qtf import pair_integrals

    report: dict = {"density": RHO, "gravity": G, "m_max": M_MAX}

    # 1) first-order field: own scipy MF series vs module; no-flux BC residual
    rows = []
    for ka in (0.2, 0.4, 0.6, 0.8, 1.0, 1.4):
        theta = 2.0 * math.pi / 512 * np.arange(512)
        mine = field(ka, theta)
        theirs = cylinder_wall_field(ka, theta)
        scale = float(np.max(np.abs(mine["psi"])))
        bc = float(np.max(np.abs(mine["dr"])))
        rows.append({
            "ka": ka,
            "max_abs_psi_diff_vs_module": float(np.max(np.abs(mine["psi"] - theirs["psi"]))),
            "max_abs_dpsi_dtheta_diff_vs_module": float(np.max(np.abs(
                dpsi_dtheta_own(ka, theta) - theirs["dpsi_dtheta"]))),
            "no_flux_bc_residual_abs": bc,
            "no_flux_bc_residual_rel": bc / scale,
            "psi_scale": scale,
        })
    report["first_order_field_independent_check"] = rows

    # 2) azimuthal integrals of the closed form, own quadrature at 4096 pts
    rows = []
    for ki, kj in ((0.4, 0.4), (0.4, 0.6), (0.4, 0.8), (0.6, 0.8), (0.8, 0.8)):
        i1, i2 = azimuthal_integrals(ki, kj)
        p = pair_integrals(ki, kj, radius=1.0)
        rows.append({
            "ka_i": ki, "ka_j": kj,
            "I1_audit": i1, "I1_delivered": p.integral_surface,
            "I1_rel_diff": abs(i1 - p.integral_surface) / max(abs(i1), 1e-300),
            "I2_audit": i2, "I2_delivered": p.integral_waterline,
            "I2_rel_diff": abs(i2 - p.integral_waterline) / max(abs(i2), 1e-300),
        })
    report["azimuthal_integrals_independent_check"] = rows

    # 3) diagonal: closed form vs BOTH anchors re-assembled by the audit
    rows = []
    for ka in (0.2, 0.35, 0.5, 0.7, 0.9, 1.1, 1.4):
        cf = RHO * pair_integrals(ka, ka, radius=1.0).qtf_surge
        nf = mean_drift_near_field(ka)
        ff = mean_drift_far_field(ka)
        rows.append({
            "ka": ka, "closed_form": cf,
            "audit_near_field": nf, "audit_far_field_maruo": ff,
            "rel_diff_vs_audit_near": abs(cf - nf) / abs(nf),
            "rel_diff_vs_audit_far": abs(cf - ff) / abs(ff),
        })
    report["diagonal_anchor_independent_check"] = rows

    # 4) long-wave asymptote -> 5 pi^2 / 16 (independent of the module's law)
    rows = []
    for ka in (0.002, 0.005, 0.01, 0.02, 0.05):
        cf = RHO * pair_integrals(ka, ka, radius=1.0).qtf_surge
        measured = cf / (RHO * G * 1.0 * ka**3)
        rows.append({"ka": ka, "closed_form": cf, "measured_coefficient": measured,
                     "classical_5pi2_over_16": long_wave_drift_coefficient(),
                     "relative_error": abs(measured - long_wave_drift_coefficient())
                     / long_wave_drift_coefficient()})
    report["long_wave_asymptote_independent_check"] = rows

    # 5) off-diagonal: what independent support actually exists?
    off = []
    for ki, kj in ((0.4, 0.6), (0.4, 0.8), (0.6, 0.8)):
        p = pair_integrals(ki, kj, radius=1.0)
        pr = pair_integrals(kj, ki, radius=1.0)
        off.append({"ka_i": ki, "ka_j": kj, "real": p.qtf_surge, "imag": p.qtf_surge_imag,
                    "hermitian_residual": abs(complex(p.qtf_surge, p.qtf_surge_imag)
                                              - complex(pr.qtf_surge, -pr.qtf_surge_imag))})
    report["off_diagonal_available_checks"] = off
    report["off_diagonal_note"] = (
        "no independent anchor exists off the diagonal: neither the near-field nor the "
        "Maruo route produces T_ij for i != j, so the audit can only confirm Hermiticity "
        "and continuity-in-ka, not the absolute off-diagonal value. The delivery states "
        "this; the audit confirms the limitation."
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())