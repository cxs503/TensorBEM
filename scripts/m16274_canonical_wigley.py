"""m16274 -- canonical-spectrum Wigley wave resistance: numbers, figure, ledger.

Run: ``PYTHONPATH=src python3 scripts/m16274_canonical_wigley.py``
Writes ``docs/m16274/numbers.json`` and ``docs/m16274/fig_canonical_vs_legacy.png``.
Every number quoted in ``docs/m16274/README.md`` is read back from numbers.json.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorbem import thin_ship_michell, wigley

OUT = Path("docs/m16274")
G = 9.80665
RHO = 1025.0
HULLS = {
    "m3125 case (L=3.0)": (3.0, 0.3, 0.1875),
    "Wigley III (L=10.0)": (10.0, 1.0, 0.625),
}
FROUDES = (0.20, 0.25, 0.30, 0.35, 0.40)


def speed(fn: float, L: float) -> float:
    return fn * math.sqrt(G * L)


def theta_form(L: float, B: float, T: float, U: float) -> float:
    hull = thin_ship_michell.WigleyHull(L=L, B=B, T=T)
    return float(
        thin_ship_michell.michell_wave_resistance(
            thin_ship_michell.wigley_amplitude_function(hull),
            U,
            lam_max=40.0,
            n_lam=240,
            rho=RHO,
            g=G,
        )["Rw"]
    )


def unit_exponents(func, L: float, B: float, T: float, fr: float):
    """(mass, length, time) exponents under a 2x rescaling of each base unit."""
    base = func(L, B, T, speed(fr, L))
    out = []
    for axis in ("mass", "length", "time"):
        s = 2.0
        sm, sl, st = (s, 1.0, 1.0) if axis == "mass" else (
            (1.0, s, 1.0) if axis == "length" else (1.0, 1.0, s)
        )
        # physical values in the rescaled unit system
        Ls, Bs, Ts = L / sl, B / sl, T / sl
        rhos = RHO * sl**3 / sm
        gs = G * st**2 / sl
        Us = speed(fr, L) * st / sl
        val = func(Ls, Bs, Ts, Us, rhos, gs)
        out.append(math.log(val / base) / math.log(s))
    return tuple(float(v) for v in out)


def run() -> dict:
    record: dict = {"g": G, "rho": RHO, "hulls": {}, "froudes": list(FROUDES)}

    for name, (L, B, T) in HULLS.items():
        rows = []
        for fn in FROUDES:
            U = speed(fn, L)
            legacy = wigley.michell_wave_resistance(L, B, T, U, RHO, G)
            canon = wigley.michell_wave_resistance_canonical(L, B, T, U, RHO, G)
            theta = theta_form(L, B, T, U)
            rows.append(
                {
                    "Fr": fn,
                    "U": U,
                    "legacy_N": legacy,
                    "canonical_N": canon,
                    "theta_form_N": theta,
                    "canonical_vs_theta_rel": abs(canon - theta) / theta,
                    "legacy_vs_theta_rel": abs(legacy - theta) / theta,
                    "amplification": canon / legacy,
                }
            )
        record["hulls"][name] = {"L": L, "B": B, "T": T, "rows": rows}

    L, B, T = HULLS["m3125 case (L=3.0)"]
    record["unit_exponents"] = {
        "legacy_default": unit_exponents(wigley.michell_wave_resistance, L, B, T, 0.30),
        "canonical": unit_exponents(wigley.michell_wave_resistance_canonical, L, B, T, 0.30),
        "theta_form": unit_exponents(
            lambda l, b, t, u, rho=RHO, g=G: theta_form(l, b, t, u), 3.0, 0.3, 0.1875, 0.30
        ),
        "force_target": (-1.0, -1.0, 2.0),
    }

    # dispersion identity: H evaluated on the canonical pair vs the legacy pair
    L, B, T = HULLS["m3125 case (L=3.0)"]
    U = speed(0.30, L)
    k0 = G / U**2
    lam = np.array([1.0, 1.5, 2.0, 4.0, 10.0])
    record["dispersion"] = {
        "lambda": lam.tolist(),
        "abs_H_canonical": [
            abs(wigley.michell_kochin_function(L, B, T, k0 * l**2, k0 * l)) for l in lam
        ],
        "abs_H_legacy_pair": [
            abs(wigley.michell_kochin_function(L, B, T, k0 * l, k0 * l)) for l in lam
        ],
        "residual_legacy_pair": (np.abs(lam * k0 - (lam * k0) ** 2 / k0) / (lam * k0)).tolist(),
    }

    # zero-speed / robustness gates
    record["gates"] = {
        "zero_speed": wigley.michell_wave_resistance_canonical(L, B, T, 0.0),
        "negative_speed": wigley.michell_wave_resistance_canonical(L, B, T, -1.0),
        "n_quad_128_vs_256": abs(
            wigley.michell_wave_resistance_canonical(L, B, T, U, n_quad=128)
            - wigley.michell_wave_resistance_canonical(L, B, T, U, n_quad=256)
        )
        / wigley.michell_wave_resistance_canonical(L, B, T, U, n_quad=256),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "numbers.json").write_text(json.dumps(record, indent=2) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for name, data in record["hulls"].items():
        fr = [r["Fr"] for r in data["rows"]]
        axes[0].plot(fr, [r["legacy_N"] for r in data["rows"]], "o--",
                     label=f"{name}: legacy default")
        axes[0].plot(fr, [r["canonical_N"] for r in data["rows"]], "s-",
                     label=f"{name}: canonical")
        axes[0].plot(fr, [r["theta_form_N"] for r in data["rows"]], "x",
                     label=f"{name}: theta form")
    axes[0].set_xlabel("Froude number")
    axes[0].set_ylabel("wave resistance  Rw [N]")
    axes[0].set_yscale("log")
    axes[0].set_title("Wigley Rw: legacy vs canonical vs theta form")
    axes[0].legend(fontsize=6)
    axes[0].grid(alpha=0.3)

    for name, data in record["hulls"].items():
        fr = [r["Fr"] for r in data["rows"]]
        axes[1].plot(fr, [r["canonical_vs_theta_rel"] for r in data["rows"]], "s-",
                     label=f"{name}: canonical vs theta")
        axes[1].plot(fr, [r["legacy_vs_theta_rel"] for r in data["rows"]], "o--",
                     label=f"{name}: legacy vs theta")
    axes[1].axhline(3e-2, color="k", ls=":", lw=1)
    axes[1].text(0.2, 3.4e-2, "3% gate", fontsize=7)
    axes[1].set_xlabel("Froude number")
    axes[1].set_ylabel("relative difference")
    axes[1].set_yscale("log")
    axes[1].set_title("agreement with the independent theta form")
    axes[1].legend(fontsize=6)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_canonical_vs_legacy.png", dpi=150)
    return record


if __name__ == "__main__":
    rec = run()
    for name, data in rec["hulls"].items():
        print(f"\n## {name}  L={data['L']} B={data['B']} T={data['T']}")
        print("| Fr | Rw legacy [N] | Rw canonical [N] | Rw theta [N] | canon vs theta | legacy vs theta | canon/legacy |")
        print("|---:|---:|---:|---:|---:|---:|---:|")
        for r in data["rows"]:
            print(
                f"| {r['Fr']:.2f} | {r['legacy_N']:.5f} | {r['canonical_N']:.5f} | "
                f"{r['theta_form_N']:.5f} | {100*r['canonical_vs_theta_rel']:.3f}% | "
                f"{100*r['legacy_vs_theta_rel']:.1f}% | {r['amplification']:.2f} |"
            )
    print("\nunit exponents:", json.dumps(rec["unit_exponents"]))
    print("gates:", json.dumps(rec["gates"]))
    print("dispersion:", json.dumps(rec["dispersion"]))