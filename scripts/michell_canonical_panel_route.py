"""m16365 -- canonical-spectrum Michell resistance for general hulls.

Generates every number and both figures of
``docs/michell_canonical_panel_route/`` (README + table + numbers.json + png).
Nothing here is hand-copied: the README text is assembled from ``numbers``.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorbem.michell import (
    canonical_resistance_nodes,
    ellipsoid_wave_resistance,
    ellipsoid_wave_resistance_canonical,
    michell_resistance,
    resistance_nodes,
    tandem_interference_factor,
    tandem_interference_factor_canonical,
    tandem_pair_resistance_canonical,
    wave_resistance_from_panels,
    wave_resistance_from_panels_canonical,
    wigley_panels,
)
from tensorbem.wigley import michell_wave_resistance_canonical

G = 9.80665
RHO = 1025.0
L, B, T = 1.0, 0.1, 1.0 / 16.0
FRUDES = (0.20, 0.25, 0.30, 0.35, 0.40)
OUT = Path("docs/michell_canonical_panel_route")


def speed(fr: float, length: float = L, g: float = G) -> float:
    return fr * math.sqrt(g * length)


def main() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    fine = wigley_panels(L, B, T, n_x=64, n_z=32, rule="gauss")
    coarse = wigley_panels(L, B, T, n_x=32, n_z=16, rule="gauss")

    # ---- 1. Froude sweep: closed form vs panel route vs legacy -------------
    sweep = []
    for fr in FRUDES:
        u = speed(fr)
        cl = michell_wave_resistance_canonical(L, B, T, u, g=G)
        p64 = wave_resistance_from_panels_canonical(fine, u, T, gravity=G)
        p32 = wave_resistance_from_panels_canonical(coarse, u, T, gravity=G)
        leg = wave_resistance_from_panels(fine, u, T, gravity=G)
        sweep.append(
            {
                "Fr": fr,
                "U": u,
                "Rw_canonical_closed_form": cl,
                "Rw_canonical_panels_32x16": p32,
                "Rw_canonical_panels_64x32": p64,
                "err_panels_32x16_pct": 100.0 * (p32 - cl) / cl,
                "err_panels_64x32_pct": 100.0 * (p64 - cl) / cl,
                "Rw_legacy_panels_64x32": leg,
                "legacy_over_canonical": leg / cl,
            }
        )

    # ---- 2. panel convergence at Fr = 0.20 --------------------------------
    u20 = speed(0.20)
    cl20 = michell_wave_resistance_canonical(L, B, T, u20, g=G)
    conv = []
    for n_x, n_z in ((8, 4), (16, 8), (32, 16), (64, 32)):
        p = wigley_panels(L, B, T, n_x=n_x, n_z=n_z, rule="gauss")
        r = wave_resistance_from_panels_canonical(p, u20, T, gravity=G)
        conv.append({"n_x": n_x, "n_z": n_z, "Rw": r, "err_pct": 100.0 * (r - cl20) / cl20})

    # ---- 3. similarity law (scale L,B,T,U,g by s) ------------------------
    def exponent(s: float = 2.0) -> dict:
        u = speed(0.30)
        b_c = wave_resistance_from_panels_canonical(fine, u, T, gravity=G)
        b_l = wave_resistance_from_panels(fine, u, T, gravity=G)
        scaled = wigley_panels(s * L, s * B, s * T, n_x=64, n_z=32, rule="gauss")
        s_c = wave_resistance_from_panels_canonical(scaled, s * u, s * T, gravity=s * G)
        s_l = wave_resistance_from_panels(scaled, s * u, s * T, gravity=s * G)
        return {
            "s": s,
            "canonical_exponent": math.log(s_c / b_c) / math.log(s),
            "legacy_exponent": math.log(s_l / b_l) / math.log(s),
        }

    sim = [exponent(1.5), exponent(2.0), exponent(3.0)]

    # ---- 4. quadrature weight identity -----------------------------------
    lam_l, w_l = resistance_nodes(speed(0.30), T, gravity=G, n_seg=4, n_quad=16)
    lam_c, w_c = canonical_resistance_nodes(speed(0.30), T, gravity=G, n_seg=4, n_quad=16)
    weight_check = {
        "max_abs_dev_of_lam": float(np.max(np.abs(lam_l - lam_c))),
        "max_rel_dev_of_lam4_ratio": float(np.max(np.abs(w_c / w_l / lam_c**4 - 1.0))),
        "n_quad32_vs_128_rel": abs(
            wave_resistance_from_panels_canonical(fine, speed(0.30), T, gravity=G, n_quad=32)
            / wave_resistance_from_panels_canonical(fine, speed(0.30), T, gravity=G, n_quad=128)
            - 1.0
        ),
    }

    # ---- 5. tandem interference on both spectra --------------------------
    tandem = []
    for fr in (0.25, 0.30, 0.35):
        u = speed(fr)
        for gap in (0.5, 1.0, 2.0):
            leg = tandem_interference_factor(fine.kochin, u, T, gap * L, gravity=G)
            can = tandem_interference_factor_canonical(fine.kochin_pair, u, T, gap * L, gravity=G)
            merged = tandem_pair_resistance_canonical(fine, u, T, gap * L, gravity=G)
            tandem.append(
                {
                    "Fr": fr,
                    "gap_over_L": gap,
                    "sigma_legacy": leg["sigma"],
                    "sigma_canonical": can["sigma"],
                    "sigma_canonical_merged": merged / (2.0 * can["r_single"]),
                    "shift_pct": 100.0 * (can["sigma"] / leg["sigma"] - 1.0),
                    "factored_vs_merged_rel": abs(merged / (2.0 * can["r_single"]) / can["sigma"] - 1.0),
                }
            )

    # ---- 6. ellipsoid ----------------------------------------------------
    ell = []
    for fr in (0.25, 0.30, 0.35):
        u = speed(fr)
        can = ellipsoid_wave_resistance_canonical(1.0, 0.1, T, u, gravity=G, n_psi=48, n_t=192)
        leg = ellipsoid_wave_resistance(1.0, 0.1, T, u, gravity=G, n_psi=48, n_t=192)
        ell.append({"Fr": fr, "Rw_canonical": can, "Rw_legacy": leg, "ratio": leg / can})
    ell_fine = ellipsoid_wave_resistance_canonical(1.0, 0.1, T, speed(0.30), gravity=G, n_psi=96, n_t=384)
    ell_ref = ell[1]["Rw_canonical"]
    ellipsoid_grid = {"n_psi48_n_t192": ell_ref, "n_psi96_n_t384": ell_fine,
                      "rel_dev_pct": 100.0 * (ell_fine - ell_ref) / ell_ref}

    # ---- 7. legacy general route == legacy panel route -------------------
    u30 = speed(0.30)
    legacy_identity = abs(
        michell_resistance(fine.kochin, u30, T, gravity=G)
        / wave_resistance_from_panels(fine, u30, T, gravity=G)
        - 1.0
    )

    numbers = {
        "geometry": {"hull": "parabolic Wigley hull", "L": L, "B": B, "T": T,
                     "Cb": 4.0 / 9.0, "rho": RHO, "g": G},
        "sweep": sweep,
        "panel_convergence_Fr020": conv,
        "similarity": sim,
        "weight_check": weight_check,
        "tandem": tandem,
        "ellipsoid": ell,
        "ellipsoid_grid": ellipsoid_grid,
        "legacy_general_vs_panel_rel": legacy_identity,
    }
    (OUT / "numbers.json").write_text(json.dumps(numbers, indent=2) + "\n")

    # ---- table.md --------------------------------------------------------
    lines = [
        "| Fr | U [m/s] | Rw canonical [N] | panel 32x16 [N] | err % | panel 64x32 [N] | err % | legacy panel [N] | legacy/canonical |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sweep:
        lines.append(
            f"| {r['Fr']:.2f} | {r['U']:.4f} | {r['Rw_canonical_closed_form']:.5f} | "
            f"{r['Rw_canonical_panels_32x16']:.5f} | {r['err_panels_32x16_pct']:+.3f} | "
            f"{r['Rw_canonical_panels_64x32']:.5f} | {r['err_panels_64x32_pct']:+.4f} | "
            f"{r['Rw_legacy_panels_64x32']:.5f} | {r['legacy_over_canonical']:.4f} |"
        )
    lines += ["", "| n_x x n_z | Rw (Fr=0.20) [N] | err % |", "|---|---:|---:|"]
    for c in conv:
        lines.append(f"| {c['n_x']}x{c['n_z']} | {c['Rw']:.6f} | {c['err_pct']:+.4f} |")
    lines += ["", "| Fr | gap/L | sigma legacy | sigma canonical | shift % | factored vs merged |",
              "|---:|---:|---:|---:|---:|---:|"]
    for t in tandem:
        lines.append(
            f"| {t['Fr']:.2f} | {t['gap_over_L']:.1f} | {t['sigma_legacy']:.5f} | "
            f"{t['sigma_canonical']:.5f} | {t['shift_pct']:+.2f} | {t['factored_vs_merged_rel']:.1e} |"
        )
    lines += ["", "| Fr | ellipsoid canonical [N] | ellipsoid legacy [N] | legacy/canonical |",
              "|---:|---:|---:|---:|"]
    for e in ell:
        lines.append(f"| {e['Fr']:.2f} | {e['Rw_canonical']:.5f} | {e['Rw_legacy']:.5f} | {e['ratio']:.4f} |")
    (OUT / "table.md").write_text("\n".join(lines) + "\n")

    # ---- figures ---------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax = axes[0]
    frs = [r["Fr"] for r in sweep]
    ax.semilogy(frs, [r["Rw_canonical_closed_form"] for r in sweep], "o-", label="canonical (closed form, a)")
    ax.semilogy(frs, [r["Rw_canonical_panels_64x32"] for r in sweep], "s--", label="canonical panel route 64x32 (b)")
    ax.semilogy(frs, [r["Rw_legacy_panels_64x32"] for r in sweep], "^:", label="legacy single-wavenumber panel (c)")
    ax.set_xlabel("Fr = U/sqrt(gL)")
    ax.set_ylabel("Rw [N]")
    ax.set_title("Michell wave resistance, Wigley L/B=10, B/T=1.6")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot([c["n_x"] for c in conv], [abs(c["err_pct"]) for c in conv], "o-", label="panel route vs closed form (Fr=0.20)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n_x (n_z = n_x/2)")
    ax.set_ylabel("|relative error| [%]")
    ax.set_title("spectral convergence of the canonical panel route")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_canonical_vs_legacy.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for fr, marker in ((0.25, "o"), (0.30, "s"), (0.35, "^")):
        rows = [t for t in tandem if t["Fr"] == fr]
        ax.plot([t["gap_over_L"] for t in rows], [t["sigma_legacy"] for t in rows], marker + "--",
                label=f"legacy sigma (Fr={fr})")
        ax.plot([t["gap_over_L"] for t in rows], [t["sigma_canonical"] for t in rows], marker + "-",
                label=f"canonical sigma (Fr={fr})")
    ax.axhline(1.0, color="k", lw=0.8, alpha=0.5)
    ax.set_xlabel("tandem gap s / L")
    ax.set_ylabel("interference factor sigma = R_pair/(2 R_single)")
    ax.set_title("tandem interference is spectrum dependent")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "fig_tandem_sigma_spectrum.png", dpi=150)
    plt.close(fig)

    return numbers


if __name__ == "__main__":
    n = main()
    print(json.dumps({k: n[k] for k in ("similarity", "weight_check", "ellipsoid_grid")}, indent=2))
    print("legacy general vs panel rel:", n["legacy_general_vs_panel_rel"])
