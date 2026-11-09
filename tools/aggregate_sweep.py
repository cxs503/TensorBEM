"""Aggregate shard JSON files into the study summary and print the verdicts.

    PYTHONPATH=src python tools/aggregate_sweep.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from tensorbem.benchmarks import interior_panel_suppression as study

RUNS = Path("runs/interior_panel_suppression")


def load_cases() -> list[dict]:
    cases: list[dict] = []
    consolidated = RUNS / "sweep_cases.json"
    if consolidated.exists():
        cases.extend(json.loads(consolidated.read_text())["cases"])
    for path in sorted(RUNS.glob("shard_L*.json")):
        cases.extend(json.loads(path.read_text())["cases"])
    # Keep the last occurrence of a duplicated (level, config, ka) key.
    unique: dict[tuple, dict] = {}
    for case in cases:
        unique[(case["level"], case["config"], case["ka"])] = case
    return sorted(unique.values(), key=lambda c: (c["level"], c["config"], c["ka"]))


def convergence_slope(summary: dict, config: str, key: str) -> float | None:
    levels = sorted({entry["level"] for entry in summary.values() if entry["config"] == config})
    if len(levels) < 2:
        return None
    lo, hi = levels[0], levels[-1]
    if key == "rao":
        e_lo = summary[f"{lo}:{config}"]["rao"]["max_curvature"]
        e_hi = summary[f"{hi}:{config}"]["rao"]["max_curvature"]
        p_lo = summary[f"{lo}:{config}"]["panels"]
        p_hi = summary[f"{hi}:{config}"]["panels"]
        if e_lo <= 0.0 or e_hi <= 0.0:
            return None
        return float(math.log(e_lo / e_hi) / math.log(p_hi / p_lo))
    e_lo = summary[f"{lo}:{config}"][key]
    e_hi = summary[f"{hi}:{config}"][key]
    p_lo = summary[f"{lo}:{config}"]["panels"]
    p_hi = summary[f"{hi}:{config}"]["panels"]
    if e_lo <= 0.0 or e_hi <= 0.0:
        return None
    return float(math.log(e_lo / e_hi) / math.log(p_hi / p_lo))


def level_h_mean(level: int) -> float:
    """Measured mean panel edge length of the hemisphere level (convergence variable)."""
    from tensorbem.geometry3d import hemisphere_hull_mesh

    return float(hemisphere_hull_mesh(study.HEMISPHERE_LEVELS[level]).h_mean)


def main() -> int:
    cases = load_cases()
    summary = study.summarise_sweep(cases)
    reductions = [
        study.suppression_reduction(summary, level)
        for level in sorted({c["level"] for c in cases})
        if f"{level}:none" in summary and f"{level}:lid" in summary
    ]
    slopes = {
        f"{config}:{key}": convergence_slope(summary, config, key)
        for config in study.CONFIGS
        if f"{min(c['level'] for c in cases)}:{config}" in summary
        for key in (
            "max_abs_error_added_mass_33_dimless",
            "max_abs_error_damping_33_dimless",
        )
    }
    # Reference-style slope: the analytic error at a *fixed* ka (1.0), the same
    # estimator the published Hulme refinement table uses.
    def slope_at_ka(config: str, key: str, ka: float = 1.0):
        pts = [c for c in cases if c["config"] == config and abs(c["ka"] - ka) < 1e-9]
        pts.sort(key=lambda c: c["panels"])
        if len(pts) < 2:
            return None
        errs = []
        for c in pts:
            ref = c["reference_added_mass_33_dimless"] if key.startswith("added") else c["reference_damping_33_dimless"]
            val = c[key]
            errs.append(abs(val / ref - 1.0))
        if min(errs) <= 0.0:
            return None
        return float(
            math.log(errs[0] / errs[-1]) / math.log(pts[-1]["panels"] / pts[0]["panels"])
        )

    slopes_at_ka = {
        f"{config}:{key}": slope_at_ka(config, key)
        for config in study.CONFIGS
        for key in ("added_mass_33_dimless", "damping_33_dimless")
    }
    # h-refinement slopes: the convergence variable is the *measured* mean edge
    # length, not the panel count (the octahedral subdivision does not halve h).
    h_by_level = {level: level_h_mean(level) for level in sorted({c["level"] for c in cases})}

    def slope_vs_h(config: str, key: str, ka: float = 1.0):
        pts = [c for c in cases if c["config"] == config and abs(c["ka"] - ka) < 1e-9]
        pts.sort(key=lambda c: h_by_level[c["level"]], reverse=True)
        if len(pts) < 2:
            return None
        errs = []
        for c in pts:
            ref = (
                c["reference_added_mass_33_dimless"]
                if key.startswith("added")
                else c["reference_damping_33_dimless"]
            )
            errs.append(abs(c[key] / ref - 1.0))
        hs = [h_by_level[c["level"]] for c in pts]
        if min(errs) <= 0.0:
            return None
        return float(math.log(errs[0] / errs[-1]) / math.log(hs[0] / hs[-1]))

    slopes_vs_h = {
        f"{config}:{key}": slope_vs_h(config, key)
        for config in study.CONFIGS
        for key in ("added_mass_33_dimless", "damping_33_dimless")
    }
    payload = {
        "h_mean_by_level": h_by_level,
        "convergence_slopes_at_ka1": slopes_at_ka,
        "convergence_slopes_vs_h": slopes_vs_h,
        "case_count": len(cases),
        "levels": sorted({c["level"] for c in cases}),
        "summary": summary,
        "suppression": reductions,
        "convergence_slopes": slopes,
    }
    out = RUNS / "sweep_summary.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"reductions": reductions, "slopes": slopes}, indent=2))
    print(f"[wrote] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())