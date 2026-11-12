"""Generate the Michell exit arbitration artefacts (mission m16345).

Run::

    PYTHONPATH=src python scripts/m16345_michell_exit_arbitration.py

Writes ``docs/michell_exit_arbitration/numbers.json``, ``table.md`` and
``figure.png``.  No number in ``docs/michell_exit_arbitration.md`` is typed by
hand: the markdown tables are copied from ``table.md``, which this script emits.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from tensorbem import rankine_wave as RW
from tensorbem import wigley as W

OUT = Path("docs/michell_exit_arbitration")
L, B, T = 10.0, 1.0, 0.625  # Wigley III
RHO, G = 1025.0, 9.80665
FR = (0.20, 0.25, 0.30, 0.35, 0.40)


def speed(fr: float, length: float = L, gravity: float = G) -> float:
    return fr * math.sqrt(gravity * length)


def similarity_exponent(fn, fr: float = 0.30) -> float:
    base = fn(L, B, T, speed(fr), rho=RHO, g=G)
    s = 4.0
    got = fn(L * s, B * s, T * s, speed(fr, L * s, G * s), rho=RHO, g=G * s)
    return math.log(got / base) / math.log(s)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for fr in FR:
        u = speed(fr)
        canon = W.michell_wave_resistance_canonical(L, B, T, u, rho=RHO, g=G)
        legacy = RW.michell_wave_resistance_legacy(L, B, T, u, rho=RHO, g=G)
        alt = RW.michell_wave_resistance_legacy(L, B, T, u, rho=RHO, g=G, n_quad=1024)
        rows.append(
            {
                "Fr": fr,
                "U": u,
                "Rw_canonical_N": canon,
                "Rw_legacy_256_N": legacy,
                "Rw_legacy_1024_N": alt,
                "legacy_over_canonical": legacy / canon,
                "legacy_quad_drift_256_to_1024": (alt - legacy) / legacy,
                "public_entry_N": RW.michell_wave_resistance(L, B, T, u, rho=RHO, g=G),
            }
        )

    quad = [
        {"n_quad": nq, "Rw_legacy_N": RW.michell_wave_resistance_legacy(L, B, T, speed(0.30), rho=RHO, g=G, n_quad=nq)}
        for nq in (64, 128, 256, 512, 1024)
    ]

    k0 = G / speed(0.30) ** 2
    kochan = [
        {
            "k_over_k0": k / k0,
            "rankine_H": abs(RW.michell_kochin_function(k, L, B, T, speed(0.30), G)),
            "wigley_H": abs(W.michell_kochin_function(L, B, T, k, alpha=k)),
            "ratio": abs(RW.michell_kochin_function(k, L, B, T, speed(0.30), G))
            / abs(W.michell_kochin_function(L, B, T, k, alpha=k)),
        }
        for k in (0.5 * k0, k0, 1.5 * k0, 3.0 * k0, 8.0 * k0)
    ]

    payload = {
        "geometry": {"L": L, "B": B, "T": T, "rho": RHO, "g": G},
        "rows": rows,
        "quadrature_drift_legacy": quad,
        "kochin_ratio": kochan,
        "similarity_exponent": {
            "public_entry": similarity_exponent(RW.michell_wave_resistance),
            "legacy": similarity_exponent(RW.michell_wave_resistance_legacy),
        },
        "energy_flux_relative_error": RW.wave_energy_flux(L, B, T, speed(0.30), rho=RHO, g=G)["relative_error"],
    }
    (OUT / "numbers.json").write_text(json.dumps(payload, indent=2))

    lines = [
        "| Fr | U [m/s] | Rw canonical [N] | Rw legacy (256) [N] | legacy/canonical | legacy quad drift 256->1024 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['Fr']:.2f} | {row['U']:.4f} | {row['Rw_canonical_N']:.4f} | "
            f"{row['Rw_legacy_256_N']:.4f} | {row['legacy_over_canonical']:.4f} | "
            f"{100 * row['legacy_quad_drift_256_to_1024']:+.3f} % |"
        )
    lines += ["", "| n_quad | Rw legacy [N] |", "|---:|---:|"]
    for q in quad:
        lines.append(f"| {q['n_quad']} | {q['Rw_legacy_N']:.4f} |")
    lines += ["", "| k/k0 | |H_rankine| | |H_wigley| | ratio |", "|---:|---:|---:|---:|"]
    for r in kochan:
        lines.append(f"| {r['k_over_k0']:.2f} | {r['rankine_H']:.6g} | {r['wigley_H']:.6g} | {r['ratio']:.4f} |")
    lines += [
        "",
        f"similarity exponent (Fr-preserving, expected 4 for a force): "
        f"public entry = {payload['similarity_exponent']['public_entry']:.6f}, "
        f"legacy = {payload['similarity_exponent']['legacy']:.6f}",
        "",
        f"energy_flux relative_error = {payload['energy_flux_relative_error']!r} (tautology)",
    ]
    (OUT / "table.md").write_text("\n".join(lines) + "\n")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    fr_dense = np.linspace(0.18, 0.42, 25)
    canon = [W.michell_wave_resistance_canonical(L, B, T, speed(float(f)), rho=RHO, g=G) for f in fr_dense]
    legacy = [RW.michell_wave_resistance_legacy(L, B, T, speed(float(f)), rho=RHO, g=G) for f in fr_dense]
    axes[0].plot(fr_dense, canon, "-", label="canonical spectrum (Rw, N)")
    axes[0].plot(fr_dense, legacy, "--", label="legacy single-wavenumber")
    axes[0].plot([r["Fr"] for r in rows], [r["public_entry_N"] for r in rows], "o", ms=4, label="rankine_wave entry (after fix)")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Fr")
    axes[0].set_ylabel("Rw [N]")
    axes[0].set_title("Wigley III wave resistance: arbitration")
    axes[0].legend(fontsize=8)

    axes[1].plot(fr_dense, [c / l for c, l in zip(canon, legacy)], "-")
    axes[1].axhline(1.0, color="k", lw=0.8)
    axes[1].set_xlabel("Fr")
    axes[1].set_ylabel("canonical / legacy")
    axes[1].set_title("pre-repair defect is Froude-dependent (0.7x .. 9x)")
    fig.tight_layout()
    fig.savefig(OUT / "figure.png", dpi=140)
    print((OUT / "table.md").read_text())


if __name__ == "__main__":
    main()