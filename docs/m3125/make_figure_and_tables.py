"""Figures and machine-generated tables for the Wigley normalization audit (m3125).

Everything numeric in the report comes from
``docs/m3125/outputs/wigley_normalization_audit.json`` through this script:
the tables are written by code, not transcribed.
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "outputs")
FIG = os.path.join(os.path.dirname(__file__), "figures")


def load():
    with open(os.path.join(OUT, "wigley_normalization_audit.json"), encoding="utf-8") as fh:
        return json.load(fh)


def fig_dimensional(a, path):
    fl = a["froude_ladder"]
    rows = fl["rows"]
    fr = [r["Fr"] for r in rows]
    cw_doc = [r["Cw_kochin_S_wet"] for r in rows]
    band = [(r["Cw_corrected_band"]["C=4"], r["Cw_corrected_band"]["C=1"]) for r in rows]
    lo = [b[0] for b in band]
    hi = [b[1] for b in band]
    cwp = [abs(r["Cw_pressure_S_wet"]) for r in rows]

    fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.2))
    ax[0].fill_between(fr, lo, hi, color="tab:blue", alpha=0.25,
                       label="dimensionally forced band  $g^2/C$, $C\\in[1,4]$")
    ax[0].plot(fr, cw_doc, "o-", color="tab:red", label="documented Kochin $C_w$ (as implemented)")
    ax[0].plot(fr, cwp, "s--", color="tab:green", label="pressure path $|C_w|$ (negative)")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("Froude number $Fr$")
    ax[0].set_ylabel("$C_w$  (wetted area $S=%.4f$ m$^2$)" % fl["S_wet"])
    ax[0].set_title("Kochin path is not homogeneous: the documented\n$C_w$ lies below the admissible band")
    ax[0].grid(alpha=0.3, which="both")
    ax[0].legend(fontsize=8, loc="lower right")

    # right: unit-rescaling response bar chart
    u = a["implied_output_dimensions"]
    names = ["mass", "length", "time"]
    x = np.arange(len(names))
    ax[1].bar(x - 0.2, [u["kochin"][n] for n in names], 0.4, label="Kochin path", color="tab:red")
    ax[1].bar(x + 0.2, [u["pressure"][n] for n in names], 0.4, label="pressure path", color="tab:green")
    ax[1].axhline(0, color="k", lw=0.8)
    for i, n in enumerate(names):
        target = a["force_unit_response"][n]
        ax[1].plot([i - 0.45, i + 0.45], [target, target], "k--", lw=1.2)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(names)
    ax[1].set_ylabel("measured unit-change exponent")
    ax[1].set_title("Unit-rescaling test: black dashes = force $kg\\,m\\,s^{-2}$\nKochin responds as $kg\\,s^2\\,m^{-1}$")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_pressure(a, path):
    lad = a["pressure_mesh_ladder"]
    panels = [r["panels"] for r in lad["rows"]]
    rw = [r["Rw_pressure"] for r in lad["rows"]]
    rk = lad["rows"][0]["Rw_kochin"]

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(panels, rw, "s-", color="tab:green", label="pressure path $R_w$")
    ax.axhline(0.0, color="k", lw=0.9)
    ax.axhline(rk, color="tab:red", ls="--", label="Kochin path $R_w$ (m=%.3f N)" % rk)
    ax.set_xscale("log")
    ax.set_xlabel("panels on the hull")
    ax.set_ylabel("$R_w$ [N]  ($Fr=%.2f$)" % lad["Fr"])
    ax.set_title("Pressure path: negative at every mesh and\nnot converging (sign is the defect)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def tables(a) -> str:
    L = []
    L.append("### T1. Unit-rescaling test (the decisive measurement)\n")
    L.append("| path | measured response `sm^a sl^b st^c` | force requires | homogeneous? |")
    L.append("|---|---|---|---|")
    for k in ("kochin", "pressure"):
        u = a["implied_output_dimensions"][k]
        L.append(
            f"| {k} | `sm^{u['mass']:.0f} sl^{u['length']:.0f} st^{u['time']:.0f}` "
            f"= {fmt_dims(u)} | `sm sl st^-2` = kg m s^-2 | "
            f"{'yes' if a['unit_rescaling'][k]['homogeneous'] else '**NO**'} |"
        )
    L.append(f"\nunit change used: mass x{a['unit_change']['mass']:.0f}, "
             f"length x{a['unit_change']['length']:.0f}, time x{a['unit_change']['time']:.0f}; "
             f"Kochin ratio = {a['unit_rescaling']['kochin']['ratio']:.6g} vs required "
             f"{a['unit_rescaling']['kochin']['required_ratio']:.6g} "
             f"(ratio of ratios = {a['unit_rescaling']['kochin']['ratio']/a['unit_rescaling']['kochin']['required_ratio']:.4g} = "
             f"$g^2$-scaled), pressure ratio = {a['unit_rescaling']['pressure']['ratio']:.6g}.\n")

    L.append("### T2. Geometric-similarity test (why no convergence study could see it)\n")
    L.append("| path | $R_w$ base [N] | $R_w$ at $s=2$, $U\\to U\\sqrt2$ [N] | measured ratio | required $s^3$ | rel. error |")
    L.append("|---|---|---|---|---|---|")
    for k in ("kochin", "pressure"):
        r = a["scale_invariance"][k]
        L.append(f"| {k} | {r['Rw']:.6g} | {r['Rw_scaled']:.6g} | {r['ratio']:.10f} | "
                 f"{r['expected']:.1f} | {r['rel_error']:.2e} |")
    L.append("\nBoth paths pass: geometric similarity is blind to the defect.\n")

    L.append(f"### T3. Froude ladder ($\\rho=1025$, $g=9.80665$, $L=2$, $B=0.2$, $T=0.125$, "
             f"$S_{{wet}}={a['froude_ladder']['S_wet']:.4f}$ m$^2$)\n")
    L.append("| $Fr$ | $U$ [m/s] | $R_w$ Kochin [N] | $R_w$ pressure [N] | pressure/Kochin | "
             "$C_w$ Kochin (doc) | $C_w$ pressure |")
    L.append("|---:|---:|---:|---:|---:|---:|---:|")
    for r in a["froude_ladder"]["rows"]:
        L.append(f"| {r['Fr']:.2f} | {r['U']:.4f} | {r['Rw_kochin']:.4e} | {r['Rw_pressure']:.4e} | "
                 f"{r['ratio_pressure_over_kochin']:+.3f} | {r['Cw_kochin_S_wet']:.3e} | "
                 f"{r['Cw_pressure_S_wet']:.3e} |")

    L.append("\n### T4. Dimensionally forced correction band for $C_w$ (wetted area)\n")
    L.append("| $Fr$ | $C_w$ as implemented | $g^2/C$, $C=4\\pi$ | $C=2\\pi$ | $C=\\pi$ | $C=4$ | $C=2$ | $C=1$ |")
    L.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    keys = ["C=4pi", "C=2pi", "C=pi", "C=4", "C=2", "C=1"]
    for r in a["froude_ladder"]["rows"]:
        b = r["Cw_corrected_band"]
        L.append(f"| {r['Fr']:.2f} | {r['Cw_kochin_S_wet']:.3e} | "
                 + " | ".join(f"{b[k]:.3e}" for k in keys) + " |")

    L.append(f"\n### T5. Pressure-path mesh ladder ($Fr={a['pressure_mesh_ladder']['Fr']}$)\n")
    L.append("| $n_x\\times n_z$ | panels | $R_w$ pressure [N] | $R_w$ Kochin [N] | pressure/Kochin |")
    L.append("|---|---:|---:|---:|---:|")
    for r in a["pressure_mesh_ladder"]["rows"]:
        L.append(f"| {r['nx']}$\\times${r['nz']} | {r['panels']} | {r['Rw_pressure']:.4e} | "
                 f"{r['Rw_kochin']:.4e} | {r['ratio_pressure_over_kochin']:+.4f} |")
    return "\n".join(L) + "\n"


def fmt_dims(u: dict) -> str:
    parts = []
    for n, sym in (("mass", "kg"), ("length", "m"), ("time", "s")):
        e = u[n]
        if abs(e) < 1e-12:
            continue
        parts.append(sym if abs(abs(e) - 1) < 1e-12 and e > 0 else
                     (f"{sym}^-1" if abs(abs(e) - 1) < 1e-12 else f"{sym}^{e:.0f}"))
    return " ".join(parts)


def main():
    os.makedirs(FIG, exist_ok=True)
    a = load()
    fig_dimensional(a, os.path.join(FIG, "fig_dimensional_response.png"))
    fig_pressure(a, os.path.join(FIG, "fig_pressure_mesh.png"))
    with open(os.path.join(OUT, "tables.md"), "w", encoding="utf-8") as fh:
        fh.write(tables(a))
    print("wrote figures + tables.md")


if __name__ == "__main__":
    main()