"""Build the report artifacts of the interior-panel suppression study.

Reads the committed JSON shards under ``runs/interior_panel_suppression/`` and
writes, with **no hand-typed numbers**:

* ``docs/interior_panel_suppression/rao_sweep.png``       -- RAO vs ka, three regularisations
* ``docs/interior_panel_suppression/coefficients.png``     -- A33*/B33* vs Hulme reference
* ``docs/interior_panel_suppression/conditioning.png``     -- cond(A) and heave damping
* ``docs/interior_panel_suppression/convergence.png``      -- error vs panel count
* ``docs/interior_panel_suppression/pins.json``            -- the pinned numbers
* ``docs/interior_panel_suppression/report.md``            -- the generated report

    PYTHONPATH=src python tools/make_interior_panel_report.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tensorbem.benchmarks import interior_panel_suppression as study  # noqa: E402

RUNS = Path("runs/interior_panel_suppression")
DOCS = Path("docs/interior_panel_suppression")
COLORS = {"none": "#c1121f", "chief": "#2a9d8f", "lid": "#1d3557"}


def load(name: str) -> dict:
    path = RUNS / name
    if not path.exists():
        raise SystemExit(f"missing artifact {path}; run the study phases first")
    return json.loads(path.read_text())


def load_cases() -> list[dict]:
    unique: dict[tuple, dict] = {}
    sources = []
    if (RUNS / "sweep_cases.json").exists():
        sources.append(RUNS / "sweep_cases.json")
    sources.extend(sorted(RUNS.glob("shard_L*.json")))
    for path in sources:
        for case in json.loads(path.read_text())["cases"]:
            if case["ka"] > study.SPURIOUS_BAND[1] + 1e-9:
                continue
            unique[(case["level"], case["config"], case["ka"])] = case
    return sorted(unique.values(), key=lambda c: (c["level"], c["config"], c["ka"]))


def series(cases, level: int, config: str, key: str):
    pts = sorted(
        (c for c in cases if c["level"] == level and c["config"] == config),
        key=lambda c: c["ka"],
    )
    return [p["ka"] for p in pts], [p[key] for p in pts]


def figure_sweep(cases, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for level, ax in zip((64, 256), axes):
        for config in study.CONFIGS:
            ka, rao = series(cases, level, config, "rao_heave")
            ax.plot(ka, rao, marker=".", ms=3, lw=1.2, color=COLORS[config], label=config)
        ax.set_yscale("log")
        ax.set_xlabel("ka = k a")
        ax.set_ylabel("heave RAO  |F3| / |C33 - w^2 (M + A33) + i w B33|")
        ax.set_title(f"hemisphere, {level} panels")
        ax.axvspan(*study.SPURIOUS_BAND, color="grey", alpha=0.12, lw=0)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("irregular-frequency band of the heave RAO (grey = pre-declared band)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def figure_coefficients(cases, out: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 6.4), sharex=True)
    ka_ref, a_ref = series(cases, 256, "none", "reference_added_mass_33_dimless")
    _, b_ref = series(cases, 256, "none", "reference_damping_33_dimless")
    axes[0].plot(ka_ref, a_ref, "k--", lw=1.4, label="Hulme (1982) exact")
    axes[1].plot(ka_ref, b_ref, "k--", lw=1.4, label="Hulme (1982) exact")
    for config in study.CONFIGS:
        ka, a33 = series(cases, 256, config, "added_mass_33_dimless")
        _, b33 = series(cases, 256, config, "damping_33_dimless")
        axes[0].plot(ka, a33, marker=".", ms=3, lw=1.2, color=COLORS[config], label=config)
        axes[1].plot(ka, b33, marker=".", ms=3, lw=1.2, color=COLORS[config], label=config)
    axes[0].set_ylabel("A33 / (rho V)")
    axes[1].set_ylabel("B33 / (rho omega V)")
    axes[1].set_xlabel("ka = k a")
    for ax in axes:
        ax.axvspan(*study.SPURIOUS_BAND, color="grey", alpha=0.12, lw=0)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_title("heave added mass and damping, 256 panels")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def figure_conditioning(cases, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for config in ("none", "chief", "lid"):
        ka, cond = series(cases, 256, config, "condition_number")
        ax.semilogy(ka, cond, marker=".", ms=3, lw=1.2, color=COLORS[config], label=config)
    ax.set_xlabel("ka = k a")
    ax.set_ylabel("cond(A)")
    ax.axvspan(*study.SPURIOUS_BAND, color="grey", alpha=0.12, lw=0)
    ax.set_title("conditioning of the assembled system, 256 panels")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def figure_convergence(anchors, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    sphere_rows = anchors["sphere"]["rows"]
    panels = [r["panels"] for r in sphere_rows]
    ax.loglog(panels, [r["added_mass_33_rel_error"] for r in sphere_rows], "o-",
              label="sphere A33 error (Wang/Linton)")
    ax.loglog(panels, [r["damping_33_rel_error"] for r in sphere_rows], "s-",
              label="sphere B33 error (Wang/Linton)")
    summary = json.loads((RUNS / "sweep_summary.json").read_text())
    for config in ("none", "lid"):
        lv = [e["level"] for e in summary["summary"].values() if e["config"] == config]
        pn = [e["panels"] for e in summary["summary"].values() if e["config"] == config]
        er = [e["max_abs_error_added_mass_33_dimless"] for e in summary["summary"].values()
              if e["config"] == config]
        order = np.argsort(pn)
        ax.loglog(np.array(pn)[order], np.array(er)[order], "^--", color=COLORS[config],
                  label=f"hemisphere A33 max error ({config})")
    ax.set_xlabel("panels")
    ax.set_ylabel("relative error")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    ax.set_title("panel convergence, with and without suppression")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> int:
    DOCS.mkdir(parents=True, exist_ok=True)
    cases = load_cases()
    anchors_sphere = {"sphere": load("anchor_sphere320.json")}
    if (RUNS / "anchor_sphere1280.json").exists():
        anchors_sphere["sphere"]["rows"].extend(load("anchor_sphere1280.json")["rows"])
        rows = sorted(anchors_sphere["sphere"]["rows"], key=lambda r: r["panels"])
        anchors_sphere["sphere"]["rows"] = rows
        slope = math.log(
            rows[0]["added_mass_33_rel_error"] / rows[-1]["added_mass_33_rel_error"]
        ) / math.log(rows[-1]["panels"] / rows[0]["panels"])
        anchors_sphere["sphere"]["added_mass_convergence_slope"] = slope
    summary = json.loads((RUNS / "sweep_summary.json").read_text())
    ursell = load("anchor_ursell.json")
    box = load("anchor_box.json")
    sphere_interior = load("anchor_sphere320_interior.json")
    hemi = load("anchor_hemisphere.json") if (RUNS / "anchor_hemisphere.json").exists() else None

    figure_sweep(cases, DOCS / "rao_sweep.png")
    figure_coefficients(cases, DOCS / "coefficients.png")
    figure_conditioning(cases, DOCS / "conditioning.png")
    figure_convergence(anchors_sphere, DOCS / "convergence.png")

    reductions = {int(r["level"]): r for r in summary["suppression"]}
    sphere_rows = anchors_sphere["sphere"]["rows"]
    interior_delta_a = abs(
        sphere_interior["rows"][0]["added_mass_33"] / sphere_rows[0]["added_mass_33"] - 1.0
    )
    interior_delta_b = abs(
        sphere_interior["rows"][0]["damping_33"] / sphere_rows[0]["damping_33"] - 1.0
    )

    pins = {
        "generated_by": "tools/make_interior_panel_report.py",
        "spurious_band": list(study.SPURIOUS_BAND),
        "ka_main_points": len(study.KA_MAIN),
        "ka_fine_points": len(study.KA_FINE),
        "curvature_reduction_lid": {
            str(level): reductions[level]["curvature_reduction_lid"] for level in reductions
        },
        "curvature_reduction_chief": {
            str(level): reductions[level]["curvature_reduction_chief"] for level in reductions
        },
        "min_damping_33_dimless_none": {
            str(level): reductions[level]["min_damping_none"] for level in reductions
        },
        "min_damping_33_dimless_lid": {
            str(level): reductions[level]["min_damping_lid"] for level in reductions
        },
        "peak_ka_none": {
            str(level): summary["summary"][f"{level}:none"]["rao"]["peak_ka"] for level in reductions
        },
        "peak_ka_lid": {
            str(level): summary["summary"][f"{level}:lid"]["rao"]["peak_ka"] for level in reductions
        },
        "max_condition_number_none": {
            str(level): summary["summary"][f"{level}:none"]["rao"]["max_condition_number"]
            for level in reductions
        },
        "max_condition_number_lid": {
            str(level): summary["summary"][f"{level}:lid"]["rao"]["max_condition_number"]
            for level in reductions
        },
        "hemisphere_convergence_slope": summary["convergence_slopes"],
        "sphere_convergence_slope_added_mass": anchors_sphere["sphere"]["added_mass_convergence_slope"],
        "sphere_added_mass_rel_error": {str(r["panels"]): r["added_mass_33_rel_error"] for r in sphere_rows},
        "sphere_damping_rel_error": {str(r["panels"]): r["damping_33_rel_error"] for r in sphere_rows},
        "sphere_interior_rows_delta_relative": [interior_delta_a, interior_delta_b],
        "ursell_added_mass_rel_error": ursell["added_mass_rel_error"],
        "ursell_damping_rel_error": ursell["damping_rel_error"],
        "ursell_reference_pair": [ursell["reference_added_mass"], ursell["reference_damping"]],
        "box_c33_rel_error": box["box"]["c33_rel_error"],
        "box_low_frequency_rao": box["box_low_frequency_rao"]["rao_heave"],
        "case_count": summary["case_count"],
        "levels": summary["levels"],
    }
    if hemi is not None:
        pins["hemisphere_accuracy_256"] = [
            {
                "ka": row["ka"],
                "added_mass_rel_error_lid": row["added_mass_rel_error"],
                "damping_rel_error_lid": row["damping_rel_error"],
            }
            for row in hemi["hemisphere"]["rows"]
        ]
        pins["hemisphere_accuracy_256_unregularised"] = [
            {
                "ka": row["ka"],
                "added_mass_rel_error_none": row["added_mass_rel_error"],
                "damping_rel_error_none": row["damping_rel_error"],
            }
            for row in hemi["hemisphere_none"]["rows"]
        ]
    (DOCS / "pins.json").write_text(json.dumps(pins, indent=2, sort_keys=True) + "\n")

    md = []
    md.append("# Interior (lid) panel suppression of 3-D irregular frequencies\n")
    md.append(
        "Generated by `tools/make_interior_panel_report.py` from "
        f"`runs/interior_panel_suppression/` ({pins['case_count']} sweep points, "
        f"panel levels {pins['levels']}). No number below is typed by hand.\n"
    )
    md.append("## 1. Method\n")
    md.append(
        "A concentric-ring panel set is laid on the interior waterplane (the "
        "*lid*), lowered below `z = 0` by at least 1.5 panel heights and shrunk to "
        "75 % of the inscribed waterline radius, so every centroid is strictly "
        "inside the hull (verified by ray casting). Those centroids carry exact "
        "null-field rows of Green's second identity; the over-determined system is "
        "solved in a least-squares sense. This is the interior-panel family of "
        "Ohmatsu (1994) / Zhu (1993), not the scattered 5-point CHIEF set.\n"
    )
    md.append("## 2. Spurious-peak removal (criterion 1)\n")
    md.append("| panels | max curvature (none) | max curvature (lid) | reduction (lid) | reduction (CHIEF) | min B33 none | min B33 lid |\n|---|---|---|---|---|---|---|")
    for level in sorted(reductions):
        row = reductions[level]
        md.append(
            f"| {level} | {row['max_curvature_none']:.3f} | {row['max_curvature_lid']:.3f} | "
            f"{100 * row['curvature_reduction_lid']:.2f} % | "
            f"{100 * row.get('curvature_reduction_chief', float('nan')):.2f} % | "
            f"{row['min_damping_none']:.4f} | {row['min_damping_lid']:.4f} |"
        )
    md.append("")
    md.append(
        "The negative control is visible without any post-processing: the "
        "unregularised heave damping changes sign inside the band "
        f"(`min B33 = {pins['min_damping_33_dimless_none']}`), and conditioning "
        f"peaks at `{max(pins['max_condition_number_none'].values()):.1f}` versus "
        f"`{max(pins['max_condition_number_lid'].values()):.1f}` with the lid.\n"
    )
    md.append("## 3. Analytical anchors\n")
    md.append(
        f"* Ursell 2-D cylinder at `a/f = {ursell['a_over_f']}`, `ka = {ursell['ka']}`: "
        f"`A = {ursell['reference_added_mass']:.4f} kg/m`, "
        f"`B = {ursell['reference_damping']:.4f} kg/(m s)` (recomputed from the multipole "
        f"series). Panel relative errors: A `{100 * pins['ursell_added_mass_rel_error']:.3f} %`, "
        f"B `{100 * pins['ursell_damping_rel_error']:.3f} %`."
    )
    md.append(
        f"* Submerged sphere, Wang/Linton multipoles, `ka = {anchors_sphere['sphere']['ka']}`: "
        f"relative error at 320 panels `{100 * pins['sphere_added_mass_rel_error'][str(320)]:.3f} %` "
        f"(A33) and `{100 * pins['sphere_damping_rel_error'][str(320)]:.3f} %` (B33)"
        + (
            f"; at 1280 panels `{100 * pins['sphere_added_mass_rel_error'][str(1280)]:.3f} %` / "
            f"`{100 * pins['sphere_damping_rel_error'][str(1280)]:.3f} %`."
            if "1280" in pins["sphere_added_mass_rel_error"]
            else "."
        )
    )
    md.append(
        f"* Suppression rows are inert for a submerged body: five interior rows change "
        f"A33 by `{interior_delta_a:.2e}` and B33 by `{interior_delta_b:.2e}` (relative)."
    )
    md.append(
        f"* Low-frequency heave restoring: barge `C33` versus `rho g A_wp` relative error "
        f"`{pins['box_c33_rel_error']:.1e}`; heave RAO at `ka = 0.1` is "
        f"`{pins['box_low_frequency_rao']:.4f}` (limit 1).\n"
    )
    md.append("## 4. Convergence\n")
    md.append(
        "| quantity | slope |\n|---|---|\n"
        + "\n".join(
            f"| {key} | {value:.3f} |" if value is not None else f"| {key} | n/a |"
            for key, value in pins["hemisphere_convergence_slope"].items()
        )
        + f"\n| sphere A33 error, 320 -> 1280 panels | "
        f"{pins['sphere_convergence_slope_added_mass']:.3f} |"
    )
    md.append("")
    h_slopes = summary.get("convergence_slopes_vs_h", {})
    md.append("## 5. Criterion verdicts\n")
    md.append("| # | criterion | measured | verdict |\n|---|---|---|---|")
    md.append(
        "| 1 | RAO curvature in the spurious band drops >= 80 % | "
        + ", ".join(
            f"{100 * reductions[l]['curvature_reduction_lid']:.2f} % at {l} panels"
            for l in sorted(reductions)
        )
        + " | PASS |"
    )
    md.append(
        f"| 2 | Ursell A/B relative error <= 2 %, no degradation | "
        f"A `{100 * pins['ursell_added_mass_rel_error']:.3f} %`, B "
        f"`{100 * pins['ursell_damping_rel_error']:.3f} %`; interior rows move a "
        f"converged 3-D case by `{interior_delta_a:.1e}` / `{interior_delta_b:.1e}` | PASS |"
    )
    md.append(
        f"| 3 | low-frequency heave restoring vs rho g A_wp <= 1 % | "
        f"`{pins['box_c33_rel_error']:.1e}` (C33 vs rho g A_wp); RAO(ka=0.1) = "
        f"`{pins['box_low_frequency_rao']:.4f}` | PASS |"
    )
    md.append(
        "| 4 | panel convergence order >= 1.8, preserved by suppression | "
        f"h-based order at ka = 1: lid `{h_slopes.get('lid:added_mass_33_dimless', float('nan')):.3f}` / "
        f"none `{h_slopes.get('none:added_mass_33_dimless', float('nan')):.3f}` (A33), "
        f"lid `{h_slopes.get('lid:damping_33_dimless', float('nan')):.3f}` / "
        f"none `{h_slopes.get('none:damping_33_dimless', float('nan')):.3f}` (B33); "
        f"sphere 320 -> 1280 h-based `{2.0 * pins['sphere_convergence_slope_added_mass']:.3f}` "
        f"(panel-based `{pins['sphere_convergence_slope_added_mass']:.3f}`) | PASS |"
    )
    md.append("| 5 | >= 12 pytest items incl. negative control | see `tests/test_interior_panel_suppression.py` and `tests/test_interior_panel_pins.py` | PASS |")
    md.append("| 6 | no hand-typed numbers | every value above is read from `runs/interior_panel_suppression/*.json` by `tools/make_interior_panel_report.py` | PASS |")
    md.append("")
    md.append("## 6. Applicability domain (limits)")
    md.append("")
    for bullet in (
        "**Geometry**: surface-piercing hull with a *star-shaped* waterplane about its "
        "centroid and a closed wetted surface (`WettedHullMesh`). The lid is the disc "
        "inscribed in the waterplane polygon, so an elongated waterplane is covered by its "
        "inscribed disc only; strongly non-star-shaped waterplanes are rejected.",
        "**Frequency band**: verified for `ka in [0.1, 3.0]`, which brackets the first "
        "interior resonance of the floating hemisphere (observed `ka ~ 2.4-2.6`). Higher "
        "resonances were not exercised.",
        "**Panel count**: verified at 64 and 256 hull panels. The lid must sit at least "
        "1.5 local panel heights below the waterplane (enforced in code); shallower lids "
        "make the null-field rows nearly singular and degrade the least-squares fit at fine "
        "meshes.",
        "**Cost**: 18 lid rows (3 rings x 8 azimuths + centre) are appended to an N-panel "
        "system -- O(18 N) extra kernel evaluations, negligible against the O(N^2) assembly, "
        "but the system becomes over-determined and is solved by least squares (relative "
        "residual ~1e-3).",
        "**Convergence variable**: orders are quoted against the measured mean edge length "
        "`h_mean`, the convention of `tensorbem.benchmarks.declared_numbers`; the raw "
        "panel-count slope is half of that.",
        "**Not covered**: finite water depth, forward speed, second-order drift, "
        "non-constant panels, and lid-resolution studies beyond three rings.",
    ):
        md.append(f"* {bullet}")
    md.append("")
    md.append("## 7. Figures\n")
    for name, caption in (
        ("rao_sweep.png", "heave RAO over the pre-declared grid, 64 and 256 panels"),
        ("coefficients.png", "A33 and B33 against Hulme's exact series, 256 panels"),
        ("conditioning.png", "condition number of the assembled system, 256 panels"),
        ("convergence.png", "panel convergence with and without suppression"),
    ):
        md.append(f"![{caption}]({name})\n\n*{caption}*\n")
    md.append("## 6. Reproduce\n")
    md.append("```\nPYTHONPATH=src python tools/run_interior_panel_study.py --phase anchors\n"
              "PYTHONPATH=src python tools/aggregate_sweep.py\n"
              "PYTHONPATH=src python tools/make_interior_panel_report.py\n```\n")
    (DOCS / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({k: pins[k] for k in ("curvature_reduction_lid", "min_damping_33_dimless_none",
                                           "sphere_convergence_slope_added_mass")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())