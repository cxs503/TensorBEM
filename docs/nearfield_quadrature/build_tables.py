"""Inject every number of ``docs/nearfield_quadrature/report.md`` from the artifacts.

The report is written once with ``<!-- BEGIN:NAME -->`` / ``<!-- END:NAME -->``
markers; running this script replaces the block between each pair with a table
generated from ``artifacts/nearfield_quadrature/*``.  No number in the report is
typed by hand, so re-running the benchmark and this script is enough to refresh
the document.

Usage::

    PYTHONPATH=src python3 docs/nearfield_quadrature/build_tables.py --inject
    PYTHONPATH=src python3 docs/nearfield_quadrature/build_tables.py --check
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts" / "nearfield_quadrature"
REPORT = Path(__file__).resolve().parent / "report.md"


def _as_number(value: str) -> Any:
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _read_csv(name: str) -> list[dict[str, Any]]:
    path = ARTIFACTS / name
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            {key: _as_number(value) for key, value in row.items() if value not in ("", "nan")}
            for row in csv.DictReader(handle)
        ]


def _read_json(name: str) -> dict[str, Any]:
    path = ARTIFACTS / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _table(header: Iterable[str], rows: Iterable[Iterable[str]]) -> str:
    head = "| " + " | ".join(header) + " |"
    rule = "|" + "|".join("---" for _ in header) + "|"
    body = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join([head, rule, *body])


def _sci(value: float, digits: int = 2) -> str:
    if value == 0.0:
        return "0"
    return f"{value:.{digits}e}"


def block_gates() -> str:
    payload = _read_json("gates_verified.json")
    rows = []
    for gate in payload.get("gates", []):
        value = gate["value"]
        if isinstance(value, float):
            value_text = _sci(value)
        elif isinstance(value, dict):
            value_text = ", ".join(f"{k}={_sci(v) if isinstance(v, float) else v}" for k, v in value.items())
        else:
            value_text = str(value)
        rows.append(
            [
                gate["id"],
                gate["claim"],
                value_text,
                str(gate["threshold"]),
                "PASS" if gate["pass"] else "FAIL",
            ]
        )
    return _table(["gate", "claim", "measured", "threshold", "verdict"], rows)


def block_quadrature_law() -> str:
    rows = _read_csv("quadrature_error_law.csv")
    out = []
    for route in ("adaptive", "polar", "plain_duffy"):
        for row in rows:
            if row["route"] != route:
                continue
            out.append(
                [
                    route,
                    row["footprint"],
                    f"{row['clearance_ratio']:.0e}",
                    _sci(row["single_rel_error"]),
                    _sci(row["double_rel_error"]),
                    f"{row['kernel_evaluations']:.0f}",
                ]
            )
    return _table(["route", "footprint", "eps/h", "single rel. err.", "double rel. err.", "kernel evals"], out)


def block_quadrature_summary() -> str:
    study = _read_json("quadrature_study.json")
    worst = study.get("worst_error", {})
    orders = study.get("observed_orders", {})
    negative = study.get("negative_control", {})
    rows = [
        ["adaptive subdivision", _sci(worst.get("adaptive", float("nan"))), f"{orders.get('adaptive', float('nan')):.2f}"],
        ["graded polar (sinh)", _sci(worst.get("polar", float("nan"))), f"{orders.get('polar', float('nan')):.2f}"],
        ["plain Duffy (negative control)", _sci(worst.get("plain_duffy", float("nan"))), f"{orders.get('plain_duffy', float('nan')):.3f}"],
    ]
    table = _table(["route", "worst rel. error over eps/h in [1e-3, 1]", "e(N) log-log slope"], rows)
    extra = (
        f"\nNegative control: error at ``eps/h = 1`` is {_sci(negative.get('error_at_clearance_1', float('nan')))}, "
        f"at ``eps/h = 1e-3`` it is {_sci(negative.get('error_at_clearance_1e-3', float('nan')))}, "
        f"i.e. a degradation factor of {negative.get('degradation_factor', float('nan')):.2e}."
    )
    return table + extra


def block_quadrature_convergence() -> str:
    rows = [
        row
        for row in _read_csv("quadrature_convergence.csv")
        if abs(float(row["clearance_ratio"]) - 1e-3) < 1e-12
    ]
    out = []
    for route in ("adaptive", "polar", "plain_duffy"):
        subset = sorted(
            [r for r in rows if r["route"] == route], key=lambda r: float(r["kernel_evaluations"])
        )
        for row in subset:
            out.append(
                [
                    route,
                    row["resolution"],
                    f"{float(row['kernel_evaluations']):.0f}",
                    _sci(float(row["single_rel_error"])),
                    _sci(float(row["double_rel_error"])),
                ]
            )
    return _table(["route", "resolution", "kernel evals", "single rel. err.", "double rel. err."], out)


def block_a33_b33() -> str:
    rows = _read_csv("a33_b33.csv")
    rows.sort(key=lambda r: (r["ka"], r["panels"]))
    out = []
    for row in rows:
        out.append(
            [
                f"{row['ka']:.2f}",
                f"{row['panels']:.0f}",
                f"{row['a33_over_rhoR3_bem']:.5f}",
                f"{row['a33_over_rhoR3_hulme1982']:.5f}",
                f"{row['a33_rel_error']:+.3%}",
                f"{row['b33_over_rhowR3_bem']:.5f}",
                f"{row['b33_over_rhowR3_hulme1982']:.5f}",
                f"{row['b33_rel_error']:+.3%}",
                _sci(row["b33_near_far_residual"]),
                "yes" if int(row.get("certified", 0)) else "",
            ]
        )
    return _table(
        [
            "Ka",
            "panels",
            "A33/(rho R^3) BEM",
            "A33/(rho R^3) Hulme",
            "err",
            "B33/(rho w R^3) BEM",
            "B33/(rho w R^3) Hulme",
            "err",
            "BEM near/far resid.",
            "certified",
        ],
        out,
    )


def block_refinement() -> str:
    payload = _read_json("gates_verified.json")
    refinement = payload.get("hemisphere", {}).get("refinement", {})
    rows = []
    for ka_key in sorted(refinement, key=float):
        entry = refinement[ka_key]
        for panels, a33, b33 in zip(entry["panels"], entry["a33_rel_error"], entry["b33_rel_error"]):
            rows.append([ka_key, f"{panels}", f"{a33:+.3%}", f"{b33:+.3%}"])
    return _table(["Ka", "panels", "A33 rel. error", "B33 rel. error"], rows)


def block_energy() -> str:
    rows = _read_csv("energy_identity.csv")
    if not rows:
        return "_energy identity artifact missing_"
    out = []
    for row in sorted(rows, key=lambda r: r["ka"]):
        out.append(
            [
                f"{row['ka']:.2f}",
                f"{row['n_terms']:.0f}",
                f"{row['damping_near_dimless']:.6f}",
                f"{row['damping_far_dimless']:.6f}",
                _sci(row["residual"]),
                _sci(row["kochin_relative_spread"]),
            ]
        )
    return _table(
        [
            "Ka",
            "n_terms",
            "B33 near (rho w V)",
            "B33 far (rho w V)",
            "residual",
            "K(theta) spread",
        ],
        out,
    )


def block_energy_truncation() -> str:
    rows = _read_csv("energy_identity_truncation.csv")
    out = []
    for row in sorted(rows, key=lambda r: (r["ka"], r["n_terms"])):
        out.append([f"{row['ka']:.1f}", f"{row['n_terms']:.0f}", _sci(row["residual"])])
    return _table(["Ka", "n_terms", "near/far residual"], out)


def block_bem_energy() -> str:
    rows = _read_csv("bem_energy_identity.csv")
    out = []
    for row in sorted(rows, key=lambda r: r["panels"]):
        out.append(
            [
                f"{row['panels']:.0f}",
                f"{row['damping_near']:.2f}",
                f"{row['damping_far']:.2f}",
                _sci(row["near_far_residual"]),
                f"{row['damping_rel_error_vs_hulme']:+.3%}",
                f"{row['condition_number']:.2f}",
            ]
        )
    payload = _read_json("bem_energy_identity.json")
    orders = payload.get("residual_orders_in_h", [])
    table = _table(
        [
            "panels",
            "B33 near (N s/m)",
            "B33 far (N s/m)",
            "near/far residual",
            "B33 error vs Hulme",
            "cond",
        ],
        out,
    )
    return table + "\n\nObserved order of the residual in ``h``: " + ", ".join(f"{o:.2f}" for o in orders)


def block_headline() -> str:
    payload = _read_json("gates_verified.json")
    hemisphere = payload.get("hemisphere", {})
    energy = payload.get("energy_identity", {})
    quad = payload.get("quadrature", {})
    lines = [
        f"* gates passing: **{sum(1 for g in payload.get('gates', []) if g['pass'])}/"
        f"{len(payload.get('gates', []))}** (``all_gates_pass = {payload.get('all_gates_pass')}``)",
        f"* quadrature worst relative error, adaptive route: {_sci(quad.get('worst_error', {}).get('adaptive', float('nan')))}",
        f"* quadrature worst relative error, graded polar route: {_sci(quad.get('worst_error', {}).get('polar', float('nan')))}",
        f"* negative control (plain Duffy) worst relative error: {_sci(quad.get('worst_error', {}).get('plain_duffy', float('nan')))}",
        f"* hemisphere heave vs Hulme (1982), Ka = {hemisphere.get('ka_range', ['?', '?'])[0]} .. "
        f"{hemisphere.get('ka_range', ['?', '?'])[1]}: max |A33 err| = {_sci(hemisphere.get('max_abs_a33_rel_error', float('nan')))}, "
        f"max |B33 err| = {_sci(hemisphere.get('max_abs_b33_rel_error', float('nan')))}",
        f"* energy identity (near-field pressure vs far-field Kochin), worst residual: "
        f"{_sci(energy.get('worst_residual', float('nan')))} at Ka = {energy.get('worst_residual_ka', '?')}",
    ]
    return "\n".join(lines)


BLOCKS = {
    "HEADLINE": block_headline,
    "GATES": block_gates,
    "QUAD_SUMMARY": block_quadrature_summary,
    "QUAD_LAW": block_quadrature_law,
    "QUAD_CONV": block_quadrature_convergence,
    "REFINEMENT": block_refinement,
    "A33B33": block_a33_b33,
    "ENERGY": block_energy,
    "ENERGY_TRUNCATION": block_energy_truncation,
    "BEM_ENERGY": block_bem_energy,
}


def render(text: str) -> str:
    out = text
    for name, builder in BLOCKS.items():
        begin = f"<!-- BEGIN:{name} -->"
        end = f"<!-- END:{name} -->"
        if begin not in out or end not in out:
            continue
        pre, rest = out.split(begin, 1)
        _, post = rest.split(end, 1)
        out = f"{pre}{begin}\n{builder()}\n{end}{post}"
    return out


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inject", action="store_true", help="rewrite report.md in place")
    parser.add_argument("--check", action="store_true", help="exit 1 if report.md is stale")
    args = parser.parse_args(list(argv) if argv is not None else None)
    current = REPORT.read_text(encoding="utf-8")
    updated = render(current)
    if args.check:
        return 0 if current == updated else 1
    if args.inject:
        REPORT.write_text(updated, encoding="utf-8")
        print(f"injected {len(BLOCKS)} blocks into {REPORT}")
    else:
        print(updated)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())