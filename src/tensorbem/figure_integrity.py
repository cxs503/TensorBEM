"""Figure integrity: verify that verification figures are reproducible from committed data.

Background
----------
The platform push gate drops ``*.png`` outside ``docs/`` (``tp_pusher._EXCLUDE2``),
so verification bundles that commit figures under ``benchmarks/verified/<case>/figures/``
arrive on GitHub without them.  Figure-presence assertions then fail on ``main`` --
the case records are complete but their figures are dangling.

A committed binary is therefore not a durable invariant in this repository; the
durable invariant is that each case's figures are **reproducible from the committed
numbers**, with no hand-typed value.

This module provides:
- Discovery of all verification cases with figure builders
- Regeneration of figures from committed ``result.json`` / ``results.json``
- Validation of regenerated figures (size, format, report linkage)
- A CLI entry point for batch integrity checks

Usage::

    python -m tensorbem.figure_integrity                          # check all cases
    python -m tensorbem.figure_integrity --case bem_finite_depth  # single case
    python -m tensorment.figure_integrity --output /tmp/figures   # custom output dir
"""

from __future__ import annotations

import argparse
import importlib
import json
import pathlib
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Case registry: maps case directory name → (module, builder, result_file)
# ---------------------------------------------------------------------------

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_VERIFIED = _REPO_ROOT / "benchmarks" / "verified"


@dataclass
class FigureResult:
    """Outcome of a single figure integrity check."""

    case: str
    figure_name: str
    passed: bool
    size_bytes: int = 0
    error: str = ""
    output_path: str = ""


@dataclass
class CaseIntegrityReport:
    """Integrity report for one verification case."""

    case: str
    result_file: str
    figures_checked: int = 0
    figures_passed: int = 0
    figures_failed: int = 0
    details: list[FigureResult] = field(default_factory=list)
    error: str = ""

    @property
    def all_passed(self) -> bool:
        return self.figures_failed == 0 and self.error == ""


# ---------------------------------------------------------------------------
# Figure builders registry
# ---------------------------------------------------------------------------

def _discover_cases() -> dict[str, dict[str, Any]]:
    """Discover verification cases that have figure builders.

    Returns a dict mapping case directory name → info dict with keys:
    - module: dotted module path for the benchmark
    - builder: function name that generates figures
    - result_file: name of the result JSON file
    - figure_names: expected figure file names
    """
    cases: dict[str, dict[str, Any]] = {}

    # bem_finite_depth: has make_figures function
    fd_dir = _VERIFIED / "bem_finite_depth"
    if fd_dir.is_dir() and (fd_dir / "result.json").is_file():
        cases["bem_finite_depth"] = {
            "module": "tensorbem.benchmarks.bem_finite_depth_anchors",
            "builder": "make_figures",
            "result_file": "result.json",
            "figure_names": [
                "fig_energy_identities.png",
                "fig_kramers_kronig.png",
                "fig_deep_water_limit.png",
                "fig_section_matrix.png",
            ],
        }

    # thin-ship studies: have _figure_* functions
    for study, fig_func in [
        ("bem_wigley_hydrostatics", "_figure_hydrostatics"),
        ("bem_ellipsoid_wave", "_figure_ellipsoid"),
        ("bem_multihull_interference", "_figure_multihull"),
    ]:
        study_dir = _VERIFIED / study
        if study_dir.is_dir() and (study_dir / "results.json").is_file():
            cases[study] = {
                "module": "tensorbem.benchmarks.thin_ship_verification",
                "builder": fig_func,
                "result_file": "results.json",
                "figure_names": [f"{study}.png"],
            }

    # CASE-BEM-ELLIPSOID-KOCHIN: pooled case, ellipsoid Kochin far field
    ko_dir = _VERIFIED / "bem_ellipsoid_kochin"
    if ko_dir.is_dir() and (ko_dir / "results.json").is_file():
        cases["bem_ellipsoid_kochin"] = {
            "module": "tensorbem.benchmarks.ellipsoid_kochin_case",
            "builder": "_figure_kochin",
            "result_file": "results.json",
            "figure_names": ["bem_ellipsoid_kochin.png"],
        }

    return cases


def _load_result(case_dir: pathlib.Path, result_file: str) -> dict:
    """Load the committed result JSON for a case."""
    path = case_dir / result_file
    if not path.is_file():
        raise FileNotFoundError(f"Missing result file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _get_builder(module_path: str, func_name: str) -> Callable:
    """Import and return a figure builder function."""
    mod = importlib.import_module(module_path)
    func = getattr(mod, func_name, None)
    if func is None or not callable(func):
        raise ImportError(f"Builder {func_name} not found in {module_path}")
    return func


# ---------------------------------------------------------------------------
# Core integrity check
# ---------------------------------------------------------------------------

MINIMUM_FIGURE_BYTES = 10_000  # matplotlib renders are typically >10 kB


def check_case_integrity(
    case: str,
    *,
    output_dir: pathlib.Path | None = None,
    minimum_bytes: int = MINIMUM_FIGURE_BYTES,
) -> CaseIntegrityReport:
    """Check that all figures for a case can be regenerated from committed data.

    Parameters
    ----------
    case : str
        Case directory name under benchmarks/verified/
    output_dir : Path, optional
        Where to write regenerated figures. Default: a temp subdirectory.
    minimum_bytes : int
        Minimum file size for a valid figure (rejects blank/aborted renders).

    Returns
    -------
    CaseIntegrityReport
    """
    cases = _discover_cases()
    if case not in cases:
        return CaseIntegrityReport(
            case=case,
            result_file="",
            error=f"Case '{case}' not found in registry or missing result file",
        )

    info = cases[case]
    case_dir = _VERIFIED / case
    out_dir = output_dir or (pathlib.Path("/tmp") / "figure_integrity" / case)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = CaseIntegrityReport(
        case=case,
        result_file=info["result_file"],
    )

    try:
        result = _load_result(case_dir, info["result_file"])
    except FileNotFoundError as e:
        report.error = str(e)
        return report

    try:
        builder = _get_builder(info["module"], info["builder"])
    except ImportError as e:
        report.error = str(e)
        return report

    # Check if builder takes (result, outdir) or (result, path) signature
    # bem_finite_depth make_figures returns list[Path]
    # thin_ship _figure_* take (result, path) for single figure

    if info["builder"] == "make_figures":
        # Multi-figure builder: returns list of paths
        try:
            generated = builder(result, out_dir)
            for fig_path in generated:
                fig_path = pathlib.Path(fig_path)
                name = fig_path.name
                size = fig_path.stat().st_size if fig_path.is_file() else 0
                passed = fig_path.is_file() and size > minimum_bytes
                report.details.append(FigureResult(
                    case=case,
                    figure_name=name,
                    passed=passed,
                    size_bytes=size,
                    output_path=str(fig_path),
                ))
                if passed:
                    report.figures_passed += 1
                else:
                    report.figures_failed += 1
                    if not fig_path.is_file():
                        report.details[-1].error = "file not created"
                    else:
                        report.details[-1].error = f"size {size} < {minimum_bytes}"
            report.figures_checked = len(generated)
        except Exception as e:
            report.error = f"Builder failed: {e}"
    else:
        # Single-figure builder: takes (result, path)
        for fig_name in info["figure_names"]:
            fig_path = out_dir / fig_name
            try:
                builder(result, fig_path)
                size = fig_path.stat().st_size if fig_path.is_file() else 0
                passed = fig_path.is_file() and size > minimum_bytes
                report.details.append(FigureResult(
                    case=case,
                    figure_name=fig_name,
                    passed=passed,
                    size_bytes=size,
                    output_path=str(fig_path),
                ))
                if passed:
                    report.figures_passed += 1
                else:
                    report.figures_failed += 1
                    if not fig_path.is_file():
                        report.details[-1].error = "file not created"
                    else:
                        report.details[-1].error = f"size {size} < {minimum_bytes}"
                report.figures_checked += 1
            except Exception as e:
                report.figures_failed += 1
                report.details.append(FigureResult(
                    case=case,
                    figure_name=fig_name,
                    passed=False,
                    error=str(e),
                ))
                report.figures_checked += 1

    return report


def check_all_cases(
    *,
    output_root: pathlib.Path | None = None,
    minimum_bytes: int = MINIMUM_FIGURE_BYTES,
) -> list[CaseIntegrityReport]:
    """Check figure integrity for all registered cases.

    Returns a list of CaseIntegrityReport, one per case.
    """
    cases = _discover_cases()
    results = []
    for case_name in sorted(cases.keys()):
        out_dir = None
        if output_root is not None:
            out_dir = output_root / case_name
        results.append(check_case_integrity(case_name, output_dir=out_dir, minimum_bytes=minimum_bytes))
    return results


# ---------------------------------------------------------------------------
# Report linkage check
# ---------------------------------------------------------------------------

def check_report_linkage(case: str, figure_names: Sequence[str]) -> dict[str, bool]:
    """Verify that each figure is referenced in the case's report.md.

    Returns a dict mapping figure name → whether it's linked in the report.
    """
    case_dir = _VERIFIED / case
    report_path = case_dir / "report.md"
    if not report_path.is_file():
        return {name: False for name in figure_names}

    report_text = report_path.read_text(encoding="utf-8")
    return {name: name in report_text for name in figure_names}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    """CLI: check figure integrity for verification cases."""
    parser = argparse.ArgumentParser(
        description="Verify that verification figures are reproducible from committed data."
    )
    parser.add_argument(
        "--case",
        default=None,
        help="Check a single case (default: all registered cases)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory for regenerated figures",
    )
    parser.add_argument(
        "--minimum-bytes",
        type=int,
        default=MINIMUM_FIGURE_BYTES,
        help=f"Minimum figure size in bytes (default: {MINIMUM_FIGURE_BYTES})",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    args = parser.parse_args(argv)

    output_dir = pathlib.Path(args.output) if args.output else None

    if args.case:
        reports = [check_case_integrity(args.case, output_dir=output_dir, minimum_bytes=args.minimum_bytes)]
    else:
        reports = check_all_cases(output_root=output_dir, minimum_bytes=args.minimum_bytes)

    if args.json:
        output = []
        for r in reports:
            output.append({
                "case": r.case,
                "result_file": r.result_file,
                "figures_checked": r.figures_checked,
                "figures_passed": r.figures_passed,
                "figures_failed": r.figures_failed,
                "all_passed": r.all_passed,
                "error": r.error,
                "details": [
                    {
                        "figure_name": d.figure_name,
                        "passed": d.passed,
                        "size_bytes": d.size_bytes,
                        "error": d.error,
                        "output_path": d.output_path,
                    }
                    for d in r.details
                ],
            })
        print(json.dumps(output, indent=2))
    else:
        all_ok = True
        for r in reports:
            status = "PASS" if r.all_passed else "FAIL"
            if not r.all_passed:
                all_ok = False
            print(f"[{status}] {r.case}: {r.figures_passed}/{r.figures_checked} figures passed")
            if r.error:
                print(f"  Error: {r.error}")
            for d in r.details:
                if not d.passed:
                    print(f"  FAIL {d.figure_name}: {d.error}")
        if not all_ok:
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
