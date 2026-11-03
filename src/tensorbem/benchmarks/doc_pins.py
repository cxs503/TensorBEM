"""Machine-checked pins for the numbers quoted in the documentation.

The benchmark tables in ``README.md`` and ``docs/marine_bem_benchmarks.md`` are
hand-written prose, while the numbers in them come from code that keeps moving.
The two drift apart silently: a quadrature change in the indirect radiation
solver moved the documented 128-panel Ursell added-mass error from
``7.72e-3`` to ``8.51e-3`` and nothing in CI noticed for several commits
(see ``docs/ursell_quadrature_provenance.md``). This module closes that hole by
turning every documented number into a pinned, executable claim.

The workflow is

* ``python -m tensorbem.benchmarks.doc_pins --write`` recomputes every pinned
  case and rewrites ``docs/benchmark_pins.json``;
* ``python -m tensorbem.benchmarks.doc_pins --check`` recomputes them and fails
  (exit status 1) if any value moved by more than its tolerance;
* ``python -m tensorbem.benchmarks.doc_pins --table ursell`` prints the
  documentation table rendered straight from the pinned values, so the table in
  the docs can be regenerated instead of retyped;
* ``tests/test_documented_benchmarks.py`` runs the fast subset in CI.

A failure here means one of two things and both are worth knowing: either the
code changed and the docs are now stale (rerun ``--write`` *and* update the
prose), or the code regressed and the documented accuracy is no longer
delivered. The pin file records the settings and the producing commit so the
difference can be told apart.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping

from . import (
    cylinder_added_mass,
    ellipse_added_mass,
    ellipsoid_added_mass,
    interior_harmonics,
    sphere_added_mass,
    three_dimensional_study,
    ursell_cylinder,
)

PIN_SCHEMA = "tensorbem-doc-pins/1"
DEFAULT_PIN_PATH = Path(__file__).resolve().parents[3] / "docs" / "benchmark_pins.json"

Runner = Callable[..., Mapping[str, object]]


@dataclass(frozen=True)
class PinCase:
    """One documented benchmark case and the metrics quoted from it."""

    name: str
    doc: str
    runner: str
    run: Runner
    settings: Mapping[str, object]
    metrics: tuple[str, ...]
    rtol: float = 1.0e-6
    slow: bool = False
    caption: str = ""
    extra: Mapping[str, str] = field(default_factory=dict)

    def evaluate(self) -> dict[str, float]:
        """Run the benchmark and return the pinned metrics as floats.

        Metric names may address nested output by a dotted path, e.g.
        ``axes.0.added_mass_relative_error`` for three-axis ellipsoid rows.
        """
        values = self.run(**dict(self.settings))
        return {metric: float(_lookup(values, metric, self.name)) for metric in self.metrics}


def _cylinder(panels: int) -> dict[str, object]:
    return {"panel_count": panels}


def _ursell(panels: int) -> dict[str, object]:
    return {"panel_count": panels}


def _sphere(subdivisions: int) -> dict[str, object]:
    return {"subdivisions": subdivisions}


def _ellipsoid(subdivisions: int) -> dict[str, object]:
    return {"subdivisions": subdivisions}


def _ellipse(panels: int, axis: str) -> dict[str, object]:
    return {"panel_count": panels, "axis": axis}


def _run_interior_harmonic(harmonic: str, subdivisions: int) -> dict[str, object]:
    """One solid harmonic of one degree on one mesh level."""
    return interior_harmonics.run_case(harmonic, subdivisions)


def _run_constant_dirichlet(subdivisions: int) -> dict[str, object]:
    """Exterior constant Dirichlet problem; the reference flux is 4*pi*a."""
    return three_dimensional_study.run_constant_dirichlet_case(subdivisions)


#: Metrics quoted for the 3-D sphere table (docs/marine_bem_benchmarks.md table 2).
_SPHERE_METRICS = (
    "added_mass_relative_error",
    "normal_derivative_relative_l2_error",
    "surface_area_ratio",
    "volume_ratio",
)

#: Metrics quoted for the 2-D ellipse table (table 3).
_ELLIPSE_METRICS = (
    "added_mass_per_unit_span",
    "added_mass_relative_error",
    "normal_derivative_relative_l2_error",
)

#: Metrics quoted for the ellipsoid table (table 4): three axes plus mesh volume.
_ELLIPSOID_METRICS = (
    "axes.0.exact_added_mass",
    "axes.1.exact_added_mass",
    "axes.2.exact_added_mass",
    "axes.0.added_mass_relative_error",
    "axes.1.added_mass_relative_error",
    "axes.2.added_mass_relative_error",
    "volume_ratio",
)

#: Metrics quoted for the interior solid-harmonic suite (table 5).
_INTERIOR_METRICS = ("normal_derivative_relative_l2_error", "field_rmse")

_URSELL_METRICS = (
    "reference_added_mass",
    "reference_damping",
    "panel_added_mass",
    "panel_added_mass_relative_error",
    "panel_damping",
    "panel_damping_relative_error",
    "near_over_far_field_damping",
    "reference_solution_added_mass_relative_error",
)

CASES: tuple[PinCase, ...] = (
    PinCase(
        name="cylinder_added_mass_2d_16",
        doc="docs/marine_bem_benchmarks.md (2-D circle, 16 panels)",
        runner="cylinder_added_mass.run",
        run=cylinder_added_mass.run,
        settings=_cylinder(16),
        metrics=(
            "added_mass_per_unit_span",
            "added_mass_relative_error",
            "single_layer_condition_number",
        ),
        caption="2.62e-2 at 16 panels",
    ),
    PinCase(
        name="cylinder_added_mass_2d_64",
        doc="docs/marine_bem_benchmarks.md (2-D circle, 64 panels)",
        runner="cylinder_added_mass.run",
        run=cylinder_added_mass.run,
        settings=_cylinder(64),
        metrics=(
            "added_mass_per_unit_span",
            "added_mass_relative_error",
            "single_layer_condition_number",
        ),
        caption="1.61e-3 at 64 panels",
    ),
    PinCase(
        name="cylinder_added_mass_2d_128",
        doc="README.md line 66 + docs/marine_bem_benchmarks.md (2-D circle, 128 panels)",
        runner="cylinder_added_mass.run",
        run=cylinder_added_mass.run,
        settings=_cylinder(128),
        metrics=(
            "added_mass_per_unit_span",
            "added_mass_relative_error",
            "single_layer_condition_number",
        ),
        caption="4.02e-4 at 128 panels",
    ),
    PinCase(
        name="ursell_cylinder_64",
        doc="docs/marine_bem_benchmarks.md section 4.6 (Ursell, 64 panels)",
        runner="ursell_cylinder.run",
        run=ursell_cylinder.run,
        settings=_ursell(64),
        metrics=_URSELL_METRICS,
        caption="first row of the Ursell table",
    ),
    PinCase(
        name="ursell_cylinder_128",
        doc="README.md line 13 + docs/marine_bem_benchmarks.md section 4.6 (Ursell, 128 panels)",
        runner="ursell_cylinder.run",
        run=ursell_cylinder.run,
        settings=_ursell(128),
        metrics=_URSELL_METRICS,
        caption="headline Ursell number",
    ),
    PinCase(
        name="ursell_cylinder_256",
        doc="docs/marine_bem_benchmarks.md section 4.6 (Ursell, 256 panels)",
        runner="ursell_cylinder.run",
        run=ursell_cylinder.run,
        settings=_ursell(256),
        metrics=_URSELL_METRICS,
        slow=True,
        caption="refinement row; takes about 90 s, excluded from the default check",
    ),

    # --- 3-D translating sphere (Laplace, no free surface) -----------------
    PinCase(
        name="sphere_added_mass_3d_20",
        doc="docs/marine_bem_benchmarks.md table 2 (3-D sphere, 20 panels)",
        runner="sphere_added_mass.run",
        run=sphere_added_mass.run,
        settings=_sphere(0),
        metrics=_SPHERE_METRICS,
        caption="2.13e-1 at 20 panels",
    ),
    PinCase(
        name="sphere_added_mass_3d_80",
        doc="docs/marine_bem_benchmarks.md table 2 (3-D sphere, 80 panels)",
        runner="sphere_added_mass.run",
        run=sphere_added_mass.run,
        settings=_sphere(1),
        metrics=_SPHERE_METRICS,
        caption="6.48e-2 at 80 panels",
    ),
    PinCase(
        name="sphere_added_mass_3d_320",
        doc="docs/marine_bem_benchmarks.md table 2 (3-D sphere, 320 panels)",
        runner="sphere_added_mass.run",
        run=sphere_added_mass.run,
        settings=_sphere(2),
        metrics=_SPHERE_METRICS,
        slow=True,
        caption="1.74e-2 at 320 panels",
    ),
    PinCase(
        name="sphere_added_mass_3d_1280",
        doc="README.md line 70 + docs/marine_bem_benchmarks.md table 2 (3-D sphere, 1280 panels)",
        runner="sphere_added_mass.run",
        run=sphere_added_mass.run,
        settings=_sphere(3),
        metrics=_SPHERE_METRICS,
        slow=True,
        caption="headline 4.47e-3 at 1280 panels",
    ),
    PinCase(
        name="constant_dirichlet_3d_1280",
        doc="docs/three_dimensional_bem.md section 5 (exterior constant Dirichlet, 1280 panels)",
        runner="three_dimensional_study.run_constant_dirichlet_case",
        run=_run_constant_dirichlet,
        settings=_sphere(3),
        metrics=("total_flux_relative_error", "exact_total_flux"),
        slow=True,
        caption="3.32e-3 total-flux error, capacitance 4*pi*a",
    ),
    # --- 2-D translating ellipse (Laplace, no free surface) ----------------
    PinCase(
        name="ellipse_added_mass_2d_128_x",
        doc="README.md line 70 + docs/marine_bem_benchmarks.md table 3 (ellipse, 128 panels, x)",
        runner="ellipse_added_mass.run",
        run=ellipse_added_mass.run,
        settings=_ellipse(128, "x"),
        metrics=_ELLIPSE_METRICS,
        caption="7.54e-4 along the major axis",
    ),
    PinCase(
        name="ellipse_added_mass_2d_128_y",
        doc="README.md line 70 + docs/marine_bem_benchmarks.md table 3 (ellipse, 128 panels, y)",
        runner="ellipse_added_mass.run",
        run=ellipse_added_mass.run,
        settings=_ellipse(128, "y"),
        metrics=_ELLIPSE_METRICS,
        caption="9.00e-4 along the minor axis",
    ),
    # --- 3-D translating ellipsoid (Laplace, no free surface) --------------
    PinCase(
        name="ellipsoid_added_mass_3d_20",
        doc="docs/marine_bem_benchmarks.md table 4 (ellipsoid, 20 panels)",
        runner="ellipsoid_added_mass.run",
        run=ellipsoid_added_mass.run,
        settings=_ellipsoid(0),
        metrics=_ELLIPSOID_METRICS,
        caption="2.1e-1 on all three axes at 20 panels",
    ),
    PinCase(
        name="ellipsoid_added_mass_3d_1280",
        doc="README.md line 70 + docs/marine_bem_benchmarks.md table 4 (ellipsoid, 1280 panels)",
        runner="ellipsoid_added_mass.run",
        run=ellipsoid_added_mass.run,
        settings=_ellipsoid(3),
        metrics=_ELLIPSOID_METRICS,
        slow=True,
        caption="4.17e-3 / 4.46e-3 / 4.94e-3 at 1280 panels",
    ),
    # --- interior solid harmonics, degree 1-3 (Laplace, no free surface) ---
    PinCase(
        name="interior_harmonic_degree1_1280",
        doc="README.md line 70 + docs/marine_bem_benchmarks.md table 5 (u = x, 1280 panels)",
        runner="interior_harmonics.run_case",
        run=_run_interior_harmonic,
        settings={"harmonic": "x", "subdivisions": 3},
        metrics=_INTERIOR_METRICS,
        slow=True,
        caption="3.76e-3 flux error, degree 1",
    ),
    PinCase(
        name="interior_harmonic_degree2_1280",
        doc="docs/marine_bem_benchmarks.md table 5 (u = x^2-y^2, 1280 panels)",
        runner="interior_harmonics.run_case",
        run=_run_interior_harmonic,
        settings={"harmonic": "x2-y2", "subdivisions": 3},
        metrics=_INTERIOR_METRICS,
        slow=True,
        caption="7.34e-3 flux error, degree 2",
    ),
    PinCase(
        name="interior_harmonic_degree3_1280",
        doc="README.md line 70 + docs/marine_bem_benchmarks.md table 5 (u = x^3-3xy^2, 1280 panels)",
        runner="interior_harmonics.run_case",
        run=_run_interior_harmonic,
        settings={"harmonic": "x3-3xy2", "subdivisions": 3},
        metrics=_INTERIOR_METRICS,
        slow=True,
        caption="1.19e-2 flux error, degree 3",
    ),
)

CASES_BY_NAME = {case.name: case for case in CASES}


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:  # pragma: no cover - git missing
        return "unknown"
    commit = result.stdout.strip()
    return commit or "unknown"


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _lookup(values: object, dotted: str, case_name: str) -> object:
    """Resolve a dotted path such as ``axes.0.added_mass`` inside benchmark output."""
    node = values
    for part in dotted.split("."):
        if isinstance(node, Mapping):
            if part not in node:
                raise KeyError(f"{case_name}: benchmark output lacks {dotted!r}")
            node = node[part]
        elif isinstance(node, (list, tuple)):
            try:
                node = node[int(part)]
            except (IndexError, ValueError) as error:
                raise KeyError(f"{case_name}: cannot index {dotted!r}") from error
        else:
            raise KeyError(f"{case_name}: cannot resolve {dotted!r}")
    return node


def compare(
    case: PinCase, pinned: Mapping[str, object], measured: Mapping[str, float]
) -> list[str]:
    """Return one message per metric that drifted outside ``case.rtol``."""
    problems: list[str] = []
    for metric in case.metrics:
        if metric not in pinned:
            problems.append(f"{case.name}: no pinned value for {metric}")
            continue
        reference = pinned[metric]
        if not _is_number(reference):
            problems.append(f"{case.name}: pinned {metric} is not a number")
            continue
        expected = float(reference)  # type: ignore[arg-type]
        actual = measured[metric]
        if not math.isfinite(actual):
            problems.append(f"{case.name}.{metric}: measured {actual!r} is not finite")
            continue
        scale = max(abs(expected), 1.0e-300)
        drift = abs(actual - expected) / scale
        if drift > case.rtol:
            problems.append(
                f"{case.name}.{metric}: pinned {expected!r}, measured {actual!r}, "
                f"relative drift {drift:.3e} > rtol {case.rtol:.1e}"
            )
    return problems


def check(
    pin_path: Path | None = None,
    *,
    include_slow: bool = False,
    cases: tuple[PinCase, ...] | None = None,
) -> tuple[list[str], dict[str, dict[str, float]]]:
    """Recompute the pinned cases and compare them with the pin file.

    Returns the list of drift messages (empty when everything matches) and the
    measured values, so callers can render them.
    """
    path = DEFAULT_PIN_PATH if pin_path is None else pin_path
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    pinned_cases = document.get("pins", {})
    if not isinstance(pinned_cases, dict):
        raise ValueError(f"{path}: expected a mapping under 'pins'")

    selected = CASES if cases is None else cases
    measured_all: dict[str, dict[str, float]] = {}
    problems: list[str] = []
    for case in selected:
        if case.slow and not include_slow:
            continue
        if case.name not in pinned_cases:
            problems.append(f"{case.name}: not present in {path}")
            continue
        measured = case.evaluate()
        measured_all[case.name] = measured
        problems.extend(compare(case, pinned_cases[case.name].get("metrics", {}), measured))

    unknown = sorted(set(pinned_cases) - {case.name for case in CASES})
    if unknown:
        problems.append(f"{path}: pins for unknown cases {unknown}")
    return problems, measured_all


def build_document(*, include_slow: bool = True) -> dict[str, object]:
    """Recompute every case and return a fresh pin document."""
    pins: dict[str, object] = {}
    for case in CASES:
        if case.slow and not include_slow:
            continue
        started = time.perf_counter()
        measured = case.evaluate()
        print(
            f"{case.name}: {time.perf_counter() - started:.1f}s", file=sys.stderr, flush=True
        )
        pins[case.name] = {
            "doc": case.doc,
            "runner": case.runner,
            "settings": dict(case.settings),
            "metrics": measured,
            "rtol": case.rtol,
            "slow": case.slow,
            "caption": case.caption,
        }
    return {
        "schema": PIN_SCHEMA,
        "generated_by": "python -m tensorbem.benchmarks.doc_pins --write",
        "commit": _git_commit(),
        "note": (
            "Numbers quoted in README.md and docs/marine_bem_benchmarks.md. "
            "Regenerate with --write after any intentional change; --check is "
            "what CI and tests/test_documented_benchmarks.py enforce."
        ),
        "pins": pins,
    }


def write(pin_path: Path | None = None, *, include_slow: bool = True) -> Path:
    path = DEFAULT_PIN_PATH if pin_path is None else Path(pin_path)
    document = build_document(include_slow=include_slow)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def render_ursell_table(pin_path: Path | None = None) -> str:
    """Render the Ursell comparison table from pinned values.

    Kept next to the pins so the documentation table is generated, not typed.
    """
    path = DEFAULT_PIN_PATH if pin_path is None else Path(pin_path)
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    pins = document["pins"]
    rows = ["| panels | $A$ | relative error | $B$ | relative error | near/far damping |",
            "|---|---|---|---|---|---|"]
    for count in (64, 128, 256):
        metrics = pins[f"ursell_cylinder_{count}"]["metrics"]
        rows.append(
            "| {n} | {a:.2f} | ${ea:.2e}$ | {b:.2f} | ${eb:.2e}$ | {ratio:.4f} |".format(
                n=count,
                a=metrics["panel_added_mass"],
                ea=metrics["panel_added_mass_relative_error"],
                b=metrics["panel_damping"],
                eb=metrics["panel_damping_relative_error"],
                ratio=metrics["near_over_far_field_damping"],
            )
        )
    reference = pins["ursell_cylinder_128"]["metrics"]
    rows.append(
        "| Ursell reference | {a:.2f} | — | {b:.2f} | — | — |".format(
            a=reference["reference_added_mass"], b=reference["reference_damping"]
        )
    )
    panel = pins["ursell_cylinder_128"]["metrics"]
    rows.append(
        "| MFS (24 sources, 128 panels) | — | ${e:.2e}$ | — | — | — |".format(
            e=panel["reference_solution_added_mass_relative_error"]
        )
    )
    return "\n".join(rows)


def render_three_dimensional_tables(pin_path: Path | None = None) -> str:
    """Render the 3-D documentation tables straight from the pinned values.

    Same motivation as :func:`render_ursell_table`: the three-dimensional tables
    in ``docs/marine_bem_benchmarks.md`` sections 4.2-4.5 are regenerated here
    instead of being retyped from a run report.
    """
    path = DEFAULT_PIN_PATH if pin_path is None else Path(pin_path)
    pins = json.loads(Path(path).read_text(encoding="utf-8"))["pins"]

    def sci(value: float) -> str:
        return f"${value:.4e}$"

    lines = [
        "**Table 2. Sphere added-mass convergence (3-D)**",
        "",
        "| panels | added-mass relative error | flux relative L2 | area ratio | volume ratio |",
        "|---:|---:|---:|---:|---:|",
    ]
    for name, panels in (
        ("sphere_added_mass_3d_20", 20),
        ("sphere_added_mass_3d_80", 80),
        ("sphere_added_mass_3d_320", 320),
        ("sphere_added_mass_3d_1280", 1280),
    ):
        metrics = pins[name]["metrics"]
        lines.append(
            f"| {panels} | {sci(metrics['added_mass_relative_error'])} | "
            f"{sci(metrics['normal_derivative_relative_l2_error'])} | "
            f"{metrics['surface_area_ratio']:.6f} | {metrics['volume_ratio']:.6f} |"
        )

    lines += [
        "",
        "**Table 3. Ellipse added mass at 128 panels (2-D, per unit span)**",
        "",
        "| motion | added mass | relative error | flux relative L2 |",
        "|---|---:|---:|---:|",
    ]
    for axis, name in (("x", "ellipse_added_mass_2d_128_x"), ("y", "ellipse_added_mass_2d_128_y")):
        metrics = pins[name]["metrics"]
        lines.append(
            f"| ${axis}$ | {metrics['added_mass_per_unit_span']:.6f} | "
            f"{sci(metrics['added_mass_relative_error'])} | "
            f"{sci(metrics['normal_derivative_relative_l2_error'])} |"
        )

    ellipsoid = pins["ellipsoid_added_mass_3d_1280"]["metrics"]
    lines += [
        "",
        "**Table 4. Ellipsoid added mass at 1280 panels (2-D axes 1.5/1.0/0.75 m)**",
        "",
        "| axis | exact added mass | relative error |",
        "|---|---:|---:|",
    ]
    for index, axis in enumerate("xyz"):
        lines.append(
            f"| ${axis}$ | {ellipsoid[f'axes.{index}.exact_added_mass']:.4f} | "
            f"{sci(ellipsoid[f'axes.{index}.added_mass_relative_error'])} |"
        )
    lines.append(f"| mesh $V/V_{{\\text{{exact}}}}$ | — | {ellipsoid['volume_ratio']:.6f} |")

    lines += [
        "",
        "**Table 5. Interior solid harmonics at 1280 panels**",
        "",
        "| degree | flux relative L2 | interior probe RMSE |",
        "|---:|---:|---:|",
    ]
    for degree, name in (
        (1, "interior_harmonic_degree1_1280"),
        (2, "interior_harmonic_degree2_1280"),
        (3, "interior_harmonic_degree3_1280"),
    ):
        metrics = pins[name]["metrics"]
        lines.append(
            f"| {degree} | {sci(metrics['normal_derivative_relative_l2_error'])} | "
            f"{sci(metrics['field_rmse'])} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true", help="rewrite the pin file")
    action.add_argument("--check", action="store_true", help="verify the pin file")
    action.add_argument(
        "--table",
        choices=("ursell", "threedim"),
        help="print a documentation table from the pins",
    )
    parser.add_argument("--pin-path", type=Path, default=None)
    parser.add_argument("--include-slow", action="store_true",
                        help="also run the cases marked slow when checking")
    parser.add_argument(
        "--fast-only",
        action="store_true",
        help=(
            "skip the slow cases when writing; without this flag --write "
            "recomputes every pinned case, because writing a fast-only pin file "
            "would silently un-pin the documented slow rows"
        ),
    )
    args = parser.parse_args(argv)

    if args.table == "ursell":
        print(render_ursell_table(args.pin_path))
        return 0
    if args.table == "threedim":
        print(render_three_dimensional_tables(args.pin_path))
        return 0
    if args.write:
        path = write(args.pin_path, include_slow=not args.fast_only)
        print(f"wrote {path}")
        return 0

    problems, measured = check(args.pin_path, include_slow=args.include_slow)
    for name, values in measured.items():
        summary = ", ".join(f"{key}={value:.6g}" for key, value in values.items())
        print(f"{name}: {summary}")
    if problems:
        for problem in problems:
            print(f"DRIFT {problem}", file=sys.stderr)
        print(f"{len(problems)} documented number(s) no longer match the code", file=sys.stderr)
        return 1
    print(f"all {len(measured)} pinned case(s) match docs/benchmark_pins.json")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())