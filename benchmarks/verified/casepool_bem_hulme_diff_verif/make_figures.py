"""Figures for CASE-BEM-HULME-DIFF-VERIF, rendered from results.json only."""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = json.loads((HERE / "results.json").read_text())
OUT = Path("docs/figures")
OUT.mkdir(parents=True, exist_ok=True)


def _c(entry):
    return complex(entry["re"], entry["im"])


def main() -> None:
    ka_band = DATA["band_ka"]
    case_sub = DATA["case_subdivisions"]
    case = {r["ka"]: r for r in DATA["ladder"] if r["subdivisions"] == case_sub}
    indep = {(r["ka"], r["dof"]): r for r in DATA["independent"]}

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8))
    for col, (dof, label) in enumerate((("surge", "surge  F1"), ("heave", "heave  F3"))):
        ref = [_c(case[ka][dof]["reference"]) for ka in ka_band]
        bem = [_c(case[ka][dof]["bem"]) for ka in ka_band]
        ind = [_c(indep[(ka, dof)]["total_finest"]) for ka in ka_band]
        ax = axes[0][col]
        ax.plot(ka_band, [abs(v) for v in ref], "k-o", label="Hulme 1982 multipole (ref)")
        ax.plot(ka_band, [abs(v) for v in bem], "s--", color="tab:red",
                label=f"TensorBEM BIE, {case[ka_band[0]]['panels']} panels")
        ax.plot(ka_band, [abs(v) for v in ind], "^:", color="tab:blue",
                label="capytaine diffraction + exact FK")
        ax.set_xlabel("ka"); ax.set_ylabel(r"$|F|/(\rho g A a^2)$")
        ax.set_title(f"{label}: magnitude"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        ax = axes[1][col]
        ax.plot(ka_band, [math.degrees(math.atan2(v.imag, v.real)) for v in ref],
                "k-o", label="reference")
        ax.plot(ka_band, [math.degrees(math.atan2(v.imag, v.real)) for v in bem],
                "s--", color="tab:red", label="TensorBEM")
        ax.plot(ka_band, [math.degrees(math.atan2(v.imag, v.real)) for v in ind],
                "^:", color="tab:blue", label="capytaine + exact FK")
        ax.set_xlabel("ka"); ax.set_ylabel("phase [deg]")
        ax.set_title(f"{label}: phase (gate G2)"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("CASE-BEM-HULME-DIFF-VERIF: floating-hemisphere exciting force, three routes")
    fig.tight_layout()
    fig.savefig(OUT / "casepool_bem_hulme_diff_verif_routes.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    for ax, dof in zip(axes, ("surge", "heave")):
        for ka in ka_band:
            rows = sorted((r for r in DATA["ladder"] if r["ka"] == ka),
                          key=lambda r: r["panels"])
            ax.loglog([r["panels"] for r in rows],
                      [r[dof]["complex_relative"] for r in rows], "o-",
                      label=f"ka={ka}")
        ax.set_xlabel("panels"); ax.set_ylabel("complex relative error")
        ax.set_title(f"{dof}: mesh convergence (min order "
                     f"{DATA['gates']['G3_mesh_convergence']['lowest_order']:.2f})")
        ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "casepool_bem_hulme_diff_verif_convergence.png", dpi=150)
    plt.close(fig)
    print("figures written")


if __name__ == "__main__":
    main()
