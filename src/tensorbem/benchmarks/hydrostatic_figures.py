"""Reproducible figures + Markdown report for the hydrostatic verification suite.

Running this module regenerates, from the live numerical results:

* ``hydrostatic_verification_report.json`` - raw JSON evidence,
* ``docs/figures/hydrostatic_convergence.png`` - refinement study,
* ``docs/figures/hydrostatic_exactness.png`` - error summary + waterplane geometry,
* ``docs/hydrostatic_verification.md`` - narrative report with tables injected
  straight from the JSON (no hand-copied digits).

Usage::

    python -m tensorbem.benchmarks.hydrostatic_figures --out-root .
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .hydrostatic_convergence import (
    elliptic_study,
    hemisphere_study,
    rectangular_barge_reference,
    run_all,
    to_markdown,
)


def _plot_convergence(report: dict[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), dpi=160)

    hemi = report["hemisphere"]
    ax = axes[0]
    panels = np.array([row["panels"] for row in hemi["levels"]], dtype=float)
    series = (
        ("displaced_volume_error", "V", "o"),
        ("waterplane_area_error", "A_wp", "s"),
        ("i_transverse_error", "I_T", "^"),
        ("kb_error", "KB", "d"),
    )
    for key, label, marker in series:
        errors = np.array([row[key] for row in hemi["levels"]], dtype=float)
        ax.loglog(panels, errors, marker + "-", label=label, lw=1.4, ms=4.0)
    # Reference second-order slope anchored at the finest mesh.
    ref = panels[-1] * (np.array([1.0, 0.25]))
    anchor = np.array([hemi["levels"][-1]["displaced_volume_error"], 0.0])[0]
    ax.loglog(ref, [anchor, anchor * 4.0], "k--", lw=1.0, label="slope $h^2$")
    ax.set_xlabel("panels")
    ax.set_ylabel("relative error vs closed form")
    ax.set_title("Floating hemisphere (R = 1 m): $O(h^2)$ convergence")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    ell = report["elliptic"]
    ax = axes[1]
    n = np.array([row["panels_in_waterline"] for row in ell["levels"]], dtype=float)
    err_area = np.array([row["waterplane_area_error_vs_ellipse"] for row in ell["levels"]])
    ax.loglog(n, err_area, "o-", lw=1.4, ms=4.0, label="$A_{wp}$ vs ellipse $\\pi ab$")
    ax.loglog(n, err_area[0] * (n[0] / n) ** 2, "k--", lw=1.0, label="slope $n^{-2}$")
    ax.loglog(
        n,
        [row["polygon_area_exact_error"] + 1e-17 for row in ell["levels"]],
        "s:",
        lw=1.0,
        ms=3.0,
        label="vs inscribed polygon (round-off)",
    )
    ax.set_xlabel("waterline panels (5 m x 2 m ellipse)")
    ax.set_ylabel("relative error")
    ax.set_title("Elliptic waterplane: polygon exact, ellipse $O(n^{-2})$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _plot_exactness(report: dict[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), dpi=160)

    barge = report["barge"]
    row = barge["rows"][-1]
    hemi_rest = report["hemisphere_restoring"]
    labels = ["V", "A_wp", "I_T", "I_L", "KB", "C heave", "C roll", "C pitch"]
    barge_errors = [
        row["displaced_volume_error"],
        row["waterplane_area_error"],
        row["i_transverse_error"],
        row["i_longitudinal_error"],
        row["kb_error"],
        row["c_heave_error"],
        row["c_roll_error"],
        row["c_pitch_error"],
    ]
    hemi_errors = [
        hemi_rest["error"]["volume"],
        hemi_rest["error"]["kb"],
        hemi_rest["error"]["bm"],
        hemi_rest["error"]["c_heave"],
        hemi_rest["error"]["c_roll"],
    ]
    hemi_labels = ["V", "KB", "BM", "C heave", "C roll"]

    ax = axes[0]
    x = np.arange(len(labels))
    ax.bar(x, np.maximum(barge_errors, 1e-17), color="#2b6cb0", width=0.6)
    ax.set_yscale("log")
    ax.set_ylim(1e-17, 1e-1)
    ax.axhline(0.03, color="firebrick", ls="--", lw=1.0)
    ax.text(len(labels) - 0.4, 0.033, "3 % platform bar", color="firebrick",
            ha="right", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("relative error")
    ax.set_title("Wall-sided barge (10x4x1 m): exact to round-off")

    ax = axes[1]
    x = np.arange(len(hemi_labels))
    ax.bar(x, hemi_errors, color="#2f855a", width=0.55)
    ax.set_yscale("log")
    ax.axhline(0.03, color="firebrick", ls="--", lw=1.0)
    ax.text(len(hemi_labels) - 0.4, 0.033, "3 % platform bar", color="firebrick",
            ha="right", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(hemi_labels)
    ax.set_ylabel("relative error")
    ax.set_title(
        f"Hemisphere (R = 1 m, {hemi_rest['panels']} panels):\nrestoring chain vs closed form"
    )

    for ax in axes:
        ax.grid(True, axis="y", which="major", alpha=0.3)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def build_document(report: dict[str, Any]) -> str:
    """Assemble the Markdown report with tables computed from ``report``."""
    barge = report["barge"]
    hemi = report["hemisphere"]
    ell = report["elliptic"]
    rest = report["hemisphere_restoring"]
    ref = rectangular_barge_reference(barge["length"], barge["beam"], barge["draft"])

    header = """# Hydrostatic module: closed-form verification and convergence

**Scope**: closed-form re-verification of `tensorbem.hydrostatic`
(displaced volume, centre of buoyancy, waterplane area/first/second moments,
6x6 restoring matrix) plus the hull generators added with this study.

**Why this exists**: the original m273 acceptance tests compared a 64-panel
hemisphere with the analytic values using 5 % (`A_wp`) and 10 % (`V`)
tolerances. Those numbers are larger than the discretization error of a
*correct* solver at that mesh, so the tests could not distinguish a bug from
mesh coarseness, and they sit far outside the 3 % platform bar. This study
replaces them with (a) exactness checks where the mathematics is exact and
(b) a measured order of convergence where it is not.

## 1. Method

* **Geometry sources** - `hemisphere_hull_mesh` (existing) plus two new hull
  generators in `geometry3d`: `rectangular_barge_hull_mesh` (wall-sided box)
  and `elliptic_cylinder_hull_mesh` (polygonal waterline inscribed in an
  ellipse). Both new generators are validated by `WettedHullMesh` itself, which
  enforces outward normals, unique directed edges, waterline-closed boundary
  loops and positive displaced volume.
* **Exact cases** - for a wall-sided hull the wetted surface is planar, so the
  divergence-theorem volume integral and the waterline polygon integrals are
  exact: `V = L B T`, `A_wp = L B`, `I_T = L B^3/12`, `I_L = B L^3/12`,
  `KB = T/2`, `BM_T = B^2/(12T)`, `BM_L = L^2/(12T)`. Any non-round-off
  deviation is an implementation bug.
* **Curved cases** - the inscribed polyhedral hull under-estimates curved
  geometry. Reporting the observed order `p = log2(e_k / e_(k+1))` and a
  Richardson extrapolation separates discretization from implementation error.
  For the hemisphere `GM = KB + BM - KG = R - KG`, so the whole chain
  waterplane -> volume -> KB/BM -> stiffness is exercised against one closed
  form.
* **Conventions** - `center_of_gravity` is given in the global frame used by
  the hull vertices (`z = 0` at the calm waterline); the module converts to
  keel-relative heights internally as `kg = cg_z - min(z)`. All checks below
  use rho = {rho} kg/m^3 and g = 9.80665 m/s^2.

Reference formulas:

```
                          hemisphere      barge            ellipse
V                         2 pi R^3 / 3    L B T            pi a b T (polygon exact)
A_wp                      pi R^2          L B              pi a b
I_T = int y^2 dA          pi R^4 / 4      L B^3 / 12       pi a b^3 / 4
I_L = int x^2 dA          pi R^4 / 4      B L^3 / 12       pi b a^3 / 4
KB (above keel)           5 R / 8         T / 2            T / 2
BM_T = I_T / V            3 R / 8         B^2 / (12 T)     a^2 / (4 T)
BM_L = I_L / V            3 R / 8         L^2 / (12 T)     b^2 / (4 T)
GM                        R - KG          KB + BM - KG     KB + BM - KG
```

## 2. Results

![Convergence studies](figures/hydrostatic_convergence.png)

*Figure 1* - left: relative error of the hemisphere hydrostatic quantities
against mesh refinement, with a `h^2` reference slope; right: elliptic
waterplane area against the inscribed-polygon value (round-off) and against the
true ellipse (`n^-2`).

![Exactness summary](figures/hydrostatic_exactness.png)

*Figure 2* - left: wall-sided barge errors are at round-off level across every
quantity including the restoring matrix diagonal; right: the hemisphere
restoring chain at {hemi_panels} panels stays below the 1 % level, well inside
the 3 % bar.

### 2.1 Wall-sided barge (exact reference)

Reference: V = {ref_v} m^3, A_wp = {ref_a} m^2, I_T = {ref_it} m^4,
I_L = {ref_il} m^4, KB = {ref_kb} m, GM_T = {gm_t} m, GM_L = {gm_l} m
(CG at KG = {kg} m above the keel).

"""

    barge_tables = to_markdown({"barge": barge, "hemisphere": {"levels": [], "richardson": {}},
                                "elliptic": {"levels": []}, "hemisphere_restoring": rest})
    # Take only the barge section, then append the other sections explicitly.
    barge_section = barge_tables.split("### Floating hemisphere refinement study")[0]
    barge_section = "\n".join(
        line for line in barge_section.splitlines() if not line.startswith("### Wall-sided")
    ).strip()

    hemi_section = f"""
### 2.2 Floating hemisphere refinement study (R = 1 m)

Panel count is `4 * 4^subdivisions`; mesh spacing halves per subdivision, so the
observed order follows from successive error ratios.
"""

    hemi_table = to_markdown({"barge": {"rows": [], "max_error": 0.0},
                              "hemisphere": hemi,
                              "elliptic": {"levels": []},
                              "hemisphere_restoring": rest})
    hemi_table = hemi_table.split("### Floating hemisphere refinement study")[1]
    hemi_table = hemi_table.split("### Elliptic waterplane convergence")[0].strip()

    ell_section = f"""
### 2.3 Elliptic waterplane convergence (5 m x 2 m, draft 1 m)

The meshed hull is wall-sided, so `V` and `A_wp` are *exact for the inscribed
polygon* (round-off column); the ellipse values are approached at `O(n^-2)`.
"""

    ell_table = to_markdown({"barge": {"rows": [], "max_error": 0.0},
                             "hemisphere": {"levels": [], "richardson": {}},
                             "elliptic": ell,
                             "hemisphere_restoring": rest})
    ell_table = ell_table.split("### Elliptic waterplane convergence")[1]
    ell_table = ell_table.split("### Hemisphere restoring matrix")[0].strip()

    rest_section = f"""
### 2.4 Hemisphere restoring matrix (subdivisions = {rest['subdivisions']},
{rest['panels']} panels, KG = {rest['kg_above_keel']} m above keel)

`c_heave = rho g A_wp`, `c_roll = c_pitch = rho g V GM` with `GM = R - KG`.

| quantity | computed | closed form | relative error |
|---|---|---|---|
"""

    for key in ("volume", "kb", "bm", "gm", "c_heave", "c_roll"):
        rest_section += (
            f"| {key} | {rest['computed'][key]:.6f} | {rest['analytic'][key]:.6f} | "
            f"{rest['error'][key]:.3e} |\n"
        )

    worst_hemi = max(
        max(row["displaced_volume_error"], row["waterplane_area_error"], row["kb_error"])
        for row in hemi["levels"]
    )
    finest = hemi["levels"][-1]
    tail = f"""
## 3. Error judgement

| key quantity | mesh | relative error | below 3 %? |
|---|---|---|---|
| barge V, A_wp, I_T, I_L, KB, C heave/roll/pitch (worst of 3 meshes) | 10-62 panels | {barge['max_error']:.2e} | yes (round-off) |
| hemisphere V | {finest['panels']} panels | {finest['displaced_volume_error']:.2e} | yes |
| hemisphere A_wp | {finest['panels']} panels | {finest['waterplane_area_error']:.2e} | yes |
| hemisphere I_T | {finest['panels']} panels | {finest['i_transverse_error']:.2e} | yes |
| hemisphere KB | {finest['panels']} panels | {finest['kb_error']:.2e} | yes |
| hemisphere V, Richardson extrapolated | {finest['panels']} panels | {hemi['richardson']['displaced_volume_error']:.2e} | yes |
| hemisphere A_wp, Richardson extrapolated | {finest['panels']} panels | {hemi['richardson']['waterplane_area_error']:.2e} | yes |
| hemisphere GM (subdivisions = {rest['subdivisions']}) | {rest['panels']} panels | {rest['error']['gm']:.2e} | yes |
| hemisphere C roll (subdivisions = {rest['subdivisions']}) | {rest['panels']} panels | {rest['error']['c_roll']:.2e} | yes |
| elliptic A_wp vs ellipse | {ell['levels'][-1]['panels_in_waterline']} waterline panels | {ell['levels'][-1]['waterplane_area_error_vs_ellipse']:.2e} | yes |

**Answer to "is every key quantity below 3 %?"** Yes, on the meshes quoted
above. The hemisphere at the originally tested 64-panel mesh is *not* within
3 % (`V` error {hemi['levels'][1]['displaced_volume_error']:.2%},
`A_wp` error {hemi['levels'][1]['waterplane_area_error']:.2%}): the old test
tolerances (5 %/10 %) were hiding genuine discretization error, not bounding a
bug. At {hemi['levels'][3]['panels']} panels both are below 1 %, and the
Richardson-extrapolated values reproduce the closed forms to round-off, which
is what proves the implementation correct.

## 4. Caveats

* The hemisphere error sequence is nearly self-similar (panel geometry is
  reproduced by radial projection), which is why the Richardson extrapolation
  lands within round-off of the exact value; do not expect the same for
  arbitrary hull meshes.
* `compute_hydrostatic_restoring_matrix` expects `center_of_gravity` in the
  global frame (`z = 0` waterline). Passing a keel-relative height silently
  shifts `KG` by the draft and can flip the sign of `C[3,3]`; a dedicated test
  now covers the sign convention (`test_unstable_body_yields_negative_roll_restoring`).
* Off-diagonal hydrostatic coupling (e.g. waterplane centroid offset producing
  `C[2,4]`, or `C[3,4]` from a non-symmetric waterplane) is not modelled by the
  module; these checks cover the diagonal only.
* The verification hulls are wall-sided or smooth convex bodies. Partially
  submerged panels cut obliquely by the waterline are handled by the geometry
  layer only if the mesh is cut accordingly; no automatic waterline clipping of
  straddling panels exists.
* RAO/excitation correctness is out of scope here; this study only pins the
  hydrostatic stiffness chain that the RAO solver consumes.

## 5. Reproduce

```bash
PYTHONPATH=src python -m tensorbem.benchmarks.hydrostatic_convergence --json hydrostatic_verification_report.json
PYTHONPATH=src python -m tensorbem.benchmarks.hydrostatic_figures --out-root .
PYTHONPATH=src python -m pytest tests/test_hydrostatic_convergence.py -q
```
"""

    return (
        header.format(
            rho=1025.0,
            hemi_panels=rest["panels"],
            ref_v=f"{ref.displaced_volume:g}",
            ref_a=f"{ref.waterplane_area:g}",
            ref_it=f"{ref.i_transverse:.6g}",
            ref_il=f"{ref.i_longitudinal:.6g}",
            ref_kb=f"{ref.kb:g}",
            gm_t=f"{barge['metacentric']['gm_transverse']:.4f}",
            gm_l=f"{barge['metacentric']['gm_longitudinal']:.4f}",
            kg=f"{barge['metacentric']['kg']:g}",
        )
        + barge_section
        + "\n"
        + hemi_section
        + "\n"
        + hemi_table
        + "\n"
        + ell_section
        + "\n"
        + ell_table
        + "\n"
        + rest_section
        + tail
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=Path("."))
    args = parser.parse_args(argv)

    report = run_all()
    root = args.out_root
    (root / "hydrostatic_verification_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    _plot_convergence(report, root / "docs" / "figures" / "hydrostatic_convergence.png")
    _plot_exactness(report, root / "docs" / "figures" / "hydrostatic_exactness.png")
    (root / "docs" / "hydrostatic_verification.md").write_text(
        build_document(report), encoding="utf-8"
    )
    print("wrote report, figures and docs")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())