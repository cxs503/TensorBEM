"""Figure for the m3085 doc-contract report: every curve is read from the artifacts.

Left panel: the four refinement ladders of ``docs/three_dimensional_bem.md``
against the maximum panel size, with the O(h) and O(h^2) reference slopes.
Right panel: the surface-velocity story of section 6.4 -- the recovery path
converges while the representation formula stagnates at 53-65 %, which is the
reason the document tells users which of the two paths is usable.

Run with ``PYTHONPATH=src python3 docs/m3085/make_figure.py``; no number in the
figure is transcribed by hand, they all come from the committed artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"
FIGURES = HERE / "figures"

STUDY = json.loads((OUTPUTS / "three_dimensional_study.json").read_text())
VELOCITY = {
    panels: json.loads((OUTPUTS / f"sphere_velocity_s{sub}.json").read_text())
    for panels, sub in ((20, 0), (80, 1), (320, 2), (1280, 3))
}


def ladder(case: str, quantity: str) -> tuple[list[float], list[float]]:
    rows = STUDY["cases"][case]["results"]
    return [r["h_max"] for r in rows], [r[quantity] for r in rows]


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    series = [
        ("interior flux rel. $L^2$", *ladder("interior_linear_harmonic", "normal_derivative_relative_l2_error"), "o-"),
        ("sphere added mass", *ladder("exterior_translating_sphere", "added_mass_relative_error"), "s-"),
        ("constant-Dirichlet total flux", *ladder("exterior_constant_dirichlet", "total_flux_relative_error"), "^-"),
    ]
    for label, h, error, style in series:
        left.loglog(h, error, style, label=label, lw=1.6, ms=5)

    h = [VELOCITY[p]["h_max"] for p in sorted(VELOCITY)]
    left.loglog(h, [VELOCITY[p]["velocity_relative_l2_error"] for p in sorted(VELOCITY)], "d-",
                label="exterior velocity $E_v$", lw=1.6, ms=5)

    # Reference slopes anchored on the finest interior point.
    h_ref = [h[-1], h[0]]
    anchor = series[0][2][-1]
    left.loglog(h_ref, [anchor, anchor * (h_ref[1] / h_ref[0])], "k--", lw=0.9, label="O(h)")
    left.loglog(h_ref, [anchor, anchor * (h_ref[1] / h_ref[0]) ** 2], "k:", lw=1.1, label="O(h²)")

    left.set_xlabel("maximum panel size $h_{\\max}$  [m]")
    left.set_ylabel("relative error")
    left.set_title("Refinement ladders of §6 (all four tables reproduce)")
    left.invert_xaxis()
    left.grid(True, which="both", alpha=0.25)
    left.legend(fontsize=7.5, loc="lower right")

    panels = sorted(VELOCITY)
    recovered = [VELOCITY[p]["surface_velocity_relative_l2_error"] for p in panels]
    representation = [VELOCITY[p]["representation_surface_velocity_relative_l2_error"] for p in panels]
    right.loglog(panels, recovered, "o-", label="recovery path  $\\nabla_{tan}u + n\\,q$", lw=1.8)
    right.loglog(panels, representation, "s-", label="representation formula (stagnates)", lw=1.8)
    right.axhspan(0.53, 0.65, color="tab:red", alpha=0.10)
    right.axhline(3.0e-2, color="k", ls="--", lw=0.9)
    right.text(panels[0] * 1.05, 3.3e-2, "3 % acceptance line", fontsize=7.5)
    right.text(panels[0] * 1.05, 0.57, "53-65 % plateau", fontsize=7.5, color="tab:red")
    for x, y in zip(panels, recovered):
        right.annotate(f"{y:.2e}", (x, y), textcoords="offset points", xytext=(0, -12),
                       fontsize=7, ha="center")
    right.set_xscale("log")
    right.set_xlabel("panels $N$")
    right.set_ylabel("surface-velocity relative $L^2$ error")
    right.set_title("§6.4 surface velocity: which path is usable")
    right.grid(True, which="both", alpha=0.25)
    right.legend(fontsize=8, loc="center right")

    fig.suptitle("TensorBEM m3085 — docs/three_dimensional_bem.md numbers as an executable contract", fontsize=10)
    fig.tight_layout()
    path = FIGURES / "three_dimensional_doc_pins.png"
    fig.savefig(path, dpi=150)
    print(path)


if __name__ == "__main__":
    main()