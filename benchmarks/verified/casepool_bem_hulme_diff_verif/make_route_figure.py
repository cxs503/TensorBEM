"""Figure for the independent-route refinement study, rendered from the JSON only.

Produces ``docs/figures/casepool_bem_hulme_diff_verif_independent_route.png``:

  (a) complex relative deviation of ``FK_exact + capytaine_scattered`` against
      the multipole reference vs mean wetted-edge length ``h`` (log-log), with
      the ecosystem 3 % line, the case's 5 % gate, and the reference's own
      multipole truncation band (from the ``n_terms`` ladder) drawn for scale;
  (b) the ka = 2.0 heave cell that used to fail, showing real and imaginary
      parts approaching the reference rung by rung.

No number is typed in: everything is read back from
``independent_route_convergence.json``.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = json.loads((HERE / "independent_route_convergence.json").read_text())
OUT = Path("docs/figures")
OUT.mkdir(parents=True, exist_ok=True)
NAME = "casepool_bem_hulme_diff_verif_independent_route.png"

KA_COLOURS = {"1.0": "#1f77b4", "1.5": "#ff7f0e", "2.0": "#d62728"}


def main() -> None:
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12.5, 5.0))

    # ---- (a) error ladder ------------------------------------------------ #
    for dof, marker, ls in (("surge", "s", "--"), ("heave", "o", "-")):
        for key, colour in KA_COLOURS.items():
            rungs = DATA["per_ka"][key]["rungs"]
            hs = [r["h_mean"] for r in rungs]
            err = [100.0 * r["relative"][dof] for r in rungs]
            ax_a.plot(hs, err, ls, marker=marker, color=colour, ms=5,
                      label=f"ka={key}  {dof}")
    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    ax_a.set_xlabel("mean wetted-edge length h  [m]")
    ax_a.set_ylabel("|F_independent - F_reference| / |F_reference|  [%]")
    ax_a.set_title("(a) independent-route refinement ladder")
    ax_a.axhline(3.0, color="k", lw=1.2, ls="-.")
    ax_a.text(0.042, 3.1, "3 % ecosystem line", fontsize=8)
    ax_a.axhline(5.0, color="grey", lw=1.0, ls=":")
    ax_a.text(0.042, 5.15, "5 % case gate G4", fontsize=8)
    sens = max(
        100.0 * v["max_relative_spread"]
        for e in DATA["per_ka"].values()
        for v in e["reference_truncation_sensitivity"].values()
    )
    ax_a.axhline(max(sens, 1e-4), color="green", lw=1.0, ls="--")
    ax_a.text(0.042, max(sens, 1e-4) * 1.15,
              f"reference multipole truncation band ({sens:.1e} %)", fontsize=8,
              color="green")
    ax_a.grid(alpha=0.3, which="both")
    ax_a.legend(fontsize=8, ncol=2)

    # ---- (b) the ka=2.0 heave cell --------------------------------------- #
    entry = DATA["per_ka"]["2.0"]
    ref = entry["reference"]["heave"]
    rungs = entry["rungs"]
    faces = [r["wetted_faces"] for r in rungs]
    for part, marker in (("re", "o"), ("im", "D")):
        vals = [r["total"]["heave"][part] for r in rungs]
        ax_b.plot(faces, vals, "-", marker=marker, ms=5,
                  label=f"independent route, {part}")
        ax_b.axhline(ref[part], color="k", ls=":", lw=1.0,
                     label=f"reference, {part}")
    for i, r in enumerate(rungs):
        ax_b.annotate(f"{100 * r['relative']['heave']:.2f}%",
                      (faces[i], r["total"]["heave"]["re"]),
                      textcoords="offset points", xytext=(0, 9), fontsize=8,
                      ha="center")
    ax_b.set_xscale("log")
    ax_b.set_xlabel("wetted faces")
    ax_b.set_ylabel("heave exciting force component  [N]")
    ax_b.set_title("(b) ka = 2.0 heave: the cell that used to fail G4's 3 % line")
    ax_b.grid(alpha=0.3, which="both")
    ax_b.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT / NAME, dpi=150)
    print(f"wrote {OUT / NAME}")


if __name__ == "__main__":
    main()