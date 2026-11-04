"""Executable registry for the *remaining* published milestone numbers.

``declared_numbers.py`` (m2136) and ``doc_claims.py`` (m2186) already pin the
cylinder and Ursell headline numbers.  The m2186 report explicitly left the
rest of the documentation's quoted numbers as prose:

    "其余里程碑发布数字（球体 20–1280 面元、Hulme 半球、Wang/Linton 潜没球）
     同样只有散文声明，可按同一 claim 注册表模式批量纳管。"

This module closes that gap.  Every number quoted in ``README.md`` and
``docs/marine_bem_benchmarks.md`` for the ellipse, rigid-body, Dean,
MacCamy-Fuchs, sphere, Hulme, Wang/Linton and interior-harmonic benchmarks is
registered with its source, the exact parametric setting, and a drift
tolerance.  :func:`check` re-runs the benchmarks and judges each quoted
number ``PASS``/``DRIFT``; ``python -m tensorbem.benchmarks.milestone_claims``
prints the same table as strict JSON and ``--strict`` gates on it.

Tiers follow the same cost classes as ``declared_numbers``:

* ``fast``     -- 2-D Laplace / free-surface solves, seconds each;
* ``standard`` -- small 3-D Laplace solves (sphere 20/80/320 panels);
* ``heavy``    -- 3-D free-surface or 1280-panel assemblies; registered so the
  claim is executable and CI-gated, but not run by the default ``check``.

A claim that no test reads is a claim that can silently drift; the companion
``tests/test_milestone_claims.py`` keeps the registry honest.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from tensorbem.benchmarks import (
    dean_diffraction,
    ellipse_added_mass,
    interior_harmonics,
    maccamy_fuchs,
    rigid_body_added_mass,
    sphere_added_mass,
)

__all__ = [
    "CLAIMS",
    "MilestoneClaim",
    "TIERS",
    "check",
    "measure",
]

REPO_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT = REPO_ROOT / "artifacts" / "milestone_claims.json"

#: Cost classes, cheapest first.
TIERS: tuple[str, ...] = ("fast", "standard", "heavy")


@dataclass(frozen=True)
class MilestoneClaim:
    """One quoted number and the setting that must reproduce it."""

    key: str
    description: str
    source: str
    group: str
    quantity: str
    parameters: dict[str, Any] = field(default_factory=dict)
    declared: float = 0.0
    tolerance: float = 5.0e-3
    tier: str = "fast"
    kind: str = "value"  # "value" or "upper"


CLAIMS: tuple[MilestoneClaim, ...] = (
    # --- 4.3 elliptic cylinder, Table 3 (128 panels) -------------------
    MilestoneClaim(
        key="ellipse_added_mass_128_x",
        description="ellipse added-mass relative error, x-axis, 128 panels",
        source="docs/marine_bem_benchmarks.md section 4.3 Table 3; README",
        group="ellipse",
        quantity="added_mass_relative_error_x",
        parameters={"panels": 128, "axis": "x"},
        declared=7.5356e-4,
    ),
    MilestoneClaim(
        key="ellipse_added_mass_128_y",
        description="ellipse added-mass relative error, y-axis, 128 panels",
        source="docs/marine_bem_benchmarks.md section 4.3 Table 3; README",
        group="ellipse",
        quantity="added_mass_relative_error_y",
        parameters={"panels": 128, "axis": "y"},
        declared=8.9977e-4,
    ),
    # --- 4.7 rigid-body 2-D Neumann, Table 5 (128 panels) --------------
    MilestoneClaim(
        key="rigid_body_2d_128_A11",
        description="2-D ellipse rigid-body A11 relative error, 128 panels",
        source="docs/marine_bem_benchmarks.md section 4.7 Table 5",
        group="rigid_body_2d",
        quantity="A11_relative_error",
        parameters={"panels": 128},
        declared=4.94e-5,
    ),
    MilestoneClaim(
        key="rigid_body_2d_128_A22",
        description="2-D ellipse rigid-body A22 relative error, 128 panels",
        source="docs/marine_bem_benchmarks.md section 4.7 Table 5",
        group="rigid_body_2d",
        quantity="A22_relative_error",
        parameters={"panels": 128},
        declared=9.70e-5,
    ),
    MilestoneClaim(
        key="rigid_body_2d_128_A66",
        description="2-D ellipse rigid-body rotational A66 relative error, 128 panels",
        source="docs/marine_bem_benchmarks.md section 4.7 Table 5",
        group="rigid_body_2d",
        quantity="A66_relative_error",
        parameters={"panels": 128},
        declared=5.70e-4,
    ),
    # --- 4.8 Dean diffraction, Table 7 (64 panels) ----------------------
    MilestoneClaim(
        key="dean_cylinder_transmission_64",
        description="Dean cylinder transmission-coefficient error, 64 panels",
        source="docs/marine_bem_benchmarks.md section 4.8 Table 7",
        group="dean",
        quantity="panel_transmission_error",
        parameters={"panels": 64},
        declared=8.39e-3,
    ),
    MilestoneClaim(
        key="dean_cylinder_polarisation_64",
        description="Dean cylinder circular-polarisation error, 64 panels",
        source="docs/marine_bem_benchmarks.md section 4.8 Table 7",
        group="dean",
        quantity="panel_circular_polarisation_error",
        parameters={"panels": 64},
        declared=1.10e-14,
        tolerance=1.0,  # absolute-scale quantity; judged as an upper bound
        kind="upper",
    ),
    MilestoneClaim(
        key="dean_mfs_force_64",
        description="Dean cylinder MFS horizontal-force relative error, 64 panels",
        source="docs/marine_bem_benchmarks.md section 4.8 Table 7",
        group="dean",
        quantity="mfs_force_x_relative_error",
        parameters={"panels": 64},
        declared=1.51e-3,
        tolerance=5.0e-2,
    ),
    # --- 4.9 MacCamy-Fuchs CHIEF, Table 9 (64 panels) -------------------
    MilestoneClaim(
        key="maccamy_fuchs_chief_j11",
        description="MacCamy-Fuchs CHIEF-stabilised force error at J1 resonance",
        source="docs/marine_bem_benchmarks.md section 4.9 Table 9",
        group="maccamy_fuchs",
        quantity="chief_force_relative_error_j11",
        parameters={"panels": 64},
        declared=2.81e-3,
        tolerance=5.0e-2,
    ),
    MilestoneClaim(
        key="maccamy_fuchs_unaugmented_j11",
        description=(
            "MacCamy-Fuchs unaugmented BIE force error at J1 resonance "
            "(the 10.35% spurious peak that CHIEF removes)"
        ),
        source="docs/marine_bem_benchmarks.md section 4.9 Table 9",
        group="maccamy_fuchs",
        quantity="unaugmented_force_relative_error_j11",
        parameters={"panels": 64},
        declared=1.035e-1,
        tolerance=5.0e-2,
    ),
    # --- 3-D sphere added mass, README (20/80/320/1280) -----------------
    MilestoneClaim(
        key="sphere_added_mass_20",
        description="sphere added-mass relative error, 20 panels",
        source="README.md (3-D sphere benchmark)",
        group="sphere",
        quantity="added_mass_relative_error",
        parameters={"subdivisions": 0},
        declared=2.13e-1,
        tier="standard",
    ),
    MilestoneClaim(
        key="sphere_added_mass_80",
        description="sphere added-mass relative error, 80 panels",
        source="README.md (3-D sphere benchmark)",
        group="sphere",
        quantity="added_mass_relative_error",
        parameters={"subdivisions": 1},
        declared=6.48e-2,
        tier="standard",
    ),
    MilestoneClaim(
        key="sphere_added_mass_320",
        description="sphere added-mass relative error, 320 panels",
        source="README.md (3-D sphere benchmark)",
        group="sphere",
        quantity="added_mass_relative_error",
        parameters={"subdivisions": 2},
        declared=1.74e-2,
        tier="standard",
    ),
    MilestoneClaim(
        key="sphere_added_mass_1280",
        description="sphere added-mass relative error, 1280 panels",
        source="README.md (3-D sphere benchmark)",
        group="sphere",
        quantity="added_mass_relative_error",
        parameters={"subdivisions": 3},
        declared=4.47e-3,
        tier="heavy",
    ),
    # --- 4.10 Hulme floating hemisphere, Table 10 (256 panels) ---------
    MilestoneClaim(
        key="hulme_256_A33",
        description="Hulme heave added-mass A33' relative error, 256 panels",
        source="docs/marine_bem_benchmarks.md section 4.10 Table 10",
        group="hulme",
        quantity="a33_relative_error",
        parameters={"subdivisions": 3},
        declared=7.41e-3,
        tier="heavy",
    ),
    MilestoneClaim(
        key="hulme_256_B33",
        description="Hulme heave damping B33' relative error, 256 panels",
        source="docs/marine_bem_benchmarks.md section 4.10 Table 10",
        group="hulme",
        quantity="b33_relative_error",
        parameters={"subdivisions": 3},
        declared=1.18e-2,
        tier="heavy",
    ),
    MilestoneClaim(
        key="hulme_256_A11",
        description="Hulme surge added-mass A11' relative error, 256 panels",
        source="docs/marine_bem_benchmarks.md section 4.10 Table 10",
        group="hulme",
        quantity="a11_relative_error",
        parameters={"subdivisions": 3},
        declared=1.30e-2,
        tier="heavy",
    ),
    MilestoneClaim(
        key="hulme_256_B11",
        description="Hulme surge damping B11' relative error, 256 panels",
        source="docs/marine_bem_benchmarks.md section 4.10 Table 10",
        group="hulme",
        quantity="b11_relative_error",
        parameters={"subdivisions": 3},
        declared=3.53e-2,
        tier="heavy",
    ),
    MilestoneClaim(
        key="hulme_256_Fz",
        description="Hulme vertical wave force Fz relative error, 256 panels",
        source="docs/marine_bem_benchmarks.md section 4.10 Table 10",
        group="hulme",
        quantity="fz_relative_error",
        parameters={"subdivisions": 3},
        declared=5.67e-3,
        tier="heavy",
    ),
    # --- 4.11 Wang/Linton submerged sphere, Table 12 (320 panels) ------
    MilestoneClaim(
        key="wang_linton_320_A33",
        description="Wang/Linton heave added-mass A33' relative error, 320 panels",
        source="docs/marine_bem_benchmarks.md section 4.11 Table 12",
        group="wang_linton",
        quantity="a33_relative_error",
        parameters={"subdivisions": 2},
        declared=1.85e-2,
        tier="heavy",
    ),
    MilestoneClaim(
        key="wang_linton_320_A11",
        description="Wang/Linton surge added-mass A11' relative error, 320 panels",
        source="docs/marine_bem_benchmarks.md section 4.11 Table 12",
        group="wang_linton",
        quantity="a11_relative_error",
        parameters={"subdivisions": 2},
        declared=2.08e-2,
        tier="heavy",
    ),
    # --- 4.5 interior solid harmonics, 1280 panels ----------------------
    MilestoneClaim(
        key="interior_harmonics_1280_flux_deg1",
        description="interior-harmonic flux L2 error, degree 1, 1280 panels",
        source="docs/marine_bem_benchmarks.md section 4.5; README",
        group="interior_harmonics",
        quantity="flux_error_degree_1",
        parameters={"subdivisions": 3, "degree": 1},
        declared=3.76e-3,
        tier="heavy",
    ),
    MilestoneClaim(
        key="interior_harmonics_1280_flux_deg3",
        description="interior-harmonic flux L2 error, degree 3, 1280 panels",
        source="docs/marine_bem_benchmarks.md section 4.5; README",
        group="interior_harmonics",
        quantity="flux_error_degree_3",
        parameters={"subdivisions": 3, "degree": 3},
        declared=1.19e-2,
        tier="heavy",
    ),
)


def _first_harmonic_name(degree: int) -> str:
    """Name of the first solid harmonic of a given degree (degree-1 is 'x')."""
    from tensorbem.benchmarks.interior_harmonics import solid_harmonic_names

    names = solid_harmonic_names(int(degree))
    if not names:
        raise ValueError(f"no solid harmonics for degree {degree}")
    return names[0]


def _measure_group(group: str, parameters: dict[str, Any]) -> dict[str, float]:
    """Run one benchmark and return the flat quantities the registry may quote."""
    if group == "ellipse":
        row = ellipse_added_mass.run(
            int(parameters["panels"]), axis=str(parameters["axis"])
        )
        return {
            f"added_mass_relative_error_{parameters['axis']}": float(
                row["added_mass_relative_error"]
            )
        }
    if group == "rigid_body_2d":
        report = rigid_body_added_mass.run_ellipse_neumann_study(
            panel_counts=(int(parameters["panels"]),)
        )
        level = report["levels"][-1]
        errors = level["diagonal_relative_errors"]
        modes = report["modes"]
        out: dict[str, float] = {}
        for idx, mode in enumerate(modes):
            out[f"{mode}_relative_error"] = float(errors[mode])
        # canonical A11/A22/A66 aliases by position (axial, transverse, spin)
        for alias, idx in (("A11", 0), ("A22", 1), ("A66", 2)):
            if idx < len(modes):
                out[f"{alias}_relative_error"] = float(errors[modes[idx]])
        return out
    if group == "dean":
        row = dean_diffraction.run(int(parameters["panels"]))
        circ = row["circular_cylinder"]
        return {
            "panel_transmission_error": float(circ["panel_transmission_error"]),
            "panel_circular_polarisation_error": float(
                circ["panel_circular_polarisation_error"]
            ),
            "mfs_force_x_relative_error": float(circ["mfs_force_x_relative_error"]),
        }
    if group == "maccamy_fuchs":
        rows = maccamy_fuchs.run_irregular_frequency_check(
            panel_count=int(parameters["panels"])
        )
        out = {}
        for row in rows:
            tag = "j01" if row["resonance"].startswith("j0") else "j11"
            out[f"chief_force_relative_error_{tag}"] = float(
                row["chief_force_relative_error"]
            )
            out[f"unaugmented_force_relative_error_{tag}"] = float(
                row["unaugmented_force_relative_error"]
            )
        return out
    if group == "sphere":
        row = sphere_added_mass.run(int(parameters["subdivisions"]))
        return {"added_mass_relative_error": float(row["added_mass_relative_error"])}
    if group == "hulme":
        from tensorbem.benchmarks import floating_hemisphere

        report = floating_hemisphere.run_refinement_study(
            subdivisions=(int(parameters["subdivisions"]),)
        )
        level = report["levels"][-1]
        return {
            "a33_relative_error": float(level["heave_added_mass_rel_error"]),
            "b33_relative_error": float(level["heave_damping_rel_error"]),
            "a11_relative_error": float(level["surge_added_mass_rel_error"]),
            "b11_relative_error": float(level["surge_damping_rel_error"]),
            "fz_relative_error": float(level["excitation_force_z_rel_error"]),
        }
    if group == "wang_linton":
        from tensorbem.benchmarks import submerged_sphere_wave

        report = submerged_sphere_wave.run_refinement_study(
            subdivisions=(int(parameters["subdivisions"]),)
        )
        level = report["levels"][-1]
        return {
            "a33_relative_error": float(level["heave_added_mass_rel_error"]),
            "a11_relative_error": float(level["surge_added_mass_rel_error"]),
        }
    if group == "interior_harmonics":
        degree = int(parameters["degree"])
        report = interior_harmonics.run(
            subdivisions=(int(parameters["subdivisions"]),), degrees=(degree,)
        )
        name = _first_harmonic_name(degree)
        row = report["cases"][name]["results"][-1]
        return {
            f"flux_error_degree_{degree}": float(
                row["normal_derivative_relative_l2_error"]
            )
        }
    raise ValueError(f"unknown benchmark group: {group!r}")


def _judge(claim: MilestoneClaim, measured: float) -> tuple[float, str]:
    if not math.isfinite(measured):
        return math.inf, "DRIFT"
    if claim.declared == 0.0:
        deviation = abs(measured)
        return deviation, "PASS" if deviation == 0.0 else "DRIFT"
    deviation = (measured - claim.declared) / abs(claim.declared)
    if claim.kind == "upper":
        return deviation, "PASS" if measured <= claim.declared else "DRIFT"
    if claim.kind != "value":
        raise ValueError(f"unknown comparison kind: {claim.kind!r}")
    return deviation, "PASS" if abs(deviation) <= claim.tolerance else "DRIFT"


def measure(
    tiers: Sequence[str] = ("fast", "standard"),
) -> dict[str, float]:
    """Run the benchmarks for the selected tiers.

    Returns a dict keyed by *claim key* (not by quantity), so that two claims
    sharing a quantity name but differing in parameters (e.g. the sphere at
    20/80/320 panels) are each measured with their own setting.
    """
    selected = set(tiers)
    unknown = selected.difference(TIERS)
    if unknown:
        raise ValueError(f"unknown tiers: {sorted(unknown)}")
    measured: dict[str, float] = {}
    cache: dict[tuple[str, str], dict[str, float]] = {}
    for claim in CLAIMS:
        if claim.tier not in selected:
            continue
        if claim.key in measured:
            continue
        cache_key = (claim.group, json.dumps(claim.parameters, sort_keys=True))
        if cache_key not in cache:
            cache[cache_key] = _measure_group(claim.group, claim.parameters)
        values = cache[cache_key]
        if claim.quantity in values:
            measured[claim.key] = float(values[claim.quantity])
    return measured


def check(
    tiers: Sequence[str] = ("fast", "standard"),
) -> dict[str, Any]:
    """Re-run the selected tiers and judge every quoted number in them."""
    measured = measure(tiers)
    rows = []
    for claim in CLAIMS:
        if claim.tier not in set(tiers):
            continue
        if claim.key not in measured:
            rows.append(
                {
                    "key": claim.key,
                    "tier": claim.tier,
                    "declared": claim.declared,
                    "measured": None,
                    "relative_deviation": None,
                    "status": "MISSING",
                }
            )
            continue
        deviation, status = _judge(claim, float(measured[claim.key]))
        rows.append(
            {
                "key": claim.key,
                "tier": claim.tier,
                "declared": claim.declared,
                "measured": float(measured[claim.key]),
                "relative_deviation": deviation,
                "status": status,
            }
        )
    return {
        "benchmark": "tensorbem_milestone_published_numbers",
        "environment": {
            "python": platform.python_version(),
            "numpy": __import__("numpy").__version__,
            "platform": platform.platform(),
        },
        "tiers": sorted(selected := set(tiers)),
        "rows": rows,
        "all_pass": all(r["status"] == "PASS" for r in rows),
    }


def write_artifact(report: dict[str, Any], path: Path = ARTIFACT) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tiers",
        default="fast,standard",
        help="comma-separated subset of fast,standard,heavy",
    )
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--output", type=Path, default=ARTIFACT)
    args = parser.parse_args()
    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]
    report = check(tiers)
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    write_artifact(report, args.output)
    print(rendered)
    if args.strict and not report["all_pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
