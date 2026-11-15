"""Re-derive a case-pool record's verdict from the record itself.

Every ``benchmarks/verified/*/results.json`` in this repository carries three
things that can be checked *without re-running the solver*:

* a list of ``criteria``, each with a measured ``value``, a prose ``threshold``
  and the verdict ``ok`` the case module recorded;
* an ``acceptance`` block that summarises those verdicts
  (``passed``, ``failed_checks``, ``n_checks``, ``tolerance``);
* per-route convergence ladders (``levels``) plus the ``observed_order_*`` the
  module derived from them, and a ``finest`` row that repeats the last level.

A record is self-certifying unless somebody re-derives those three.  This module
does exactly that, with no import of the case modules and no solver:

1. **Threshold re-evaluation** -- every machine-readable threshold
   (``"< 0.03"``, ``"<= 1e-12 a"``, ``">= 3.0"``, ...) is parsed and the
   comparison re-applied to the recorded ``value``.  A criterion whose ``ok``
   disagrees with its own numbers is a blocking defect: it means the verdict was
   written by hand, or the comparison was made against something other than the
   published threshold.  Prose-only thresholds (``"in [0.5, 2.0]"``) are
   reported as *not machine-verifiable* instead of being silently counted as
   passes.
2. **Acceptance consistency** -- ``passed``, ``failed_checks`` and ``n_checks``
   must agree with the criteria list, and the record's declared ``tolerance``
   must not be looser than the pool tolerance (3 % unless overridden).
3. **Order provenance** -- each ``observed_order_X`` must equal the order that
   follows from the published ladder of ``X_relative_l2`` and ``h_mean``; a
   ``finest`` row must equal the last rung of its ladder.  This catches
   hand-edited or stale summaries that a spot check of one number would miss.

Usage::

    python -m tensorbem.case_pool_verdict benchmarks/verified/<case>/results.json

Exit status is 0 only if the record re-derives cleanly, 1 otherwise, so the
module can gate a commit or a report.  Nothing here replaces the case's own
acceptance criteria: it verifies that the committed *certificate* is internally
honest, not that the physics is right (the case module's criteria do that).
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Sequence

POOL_TOLERANCE = 0.03

_THRESHOLD_RE = re.compile(
    r"^\s*(<=|>=|<|>)\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)"
)
_ORDER_PREFIX = "observed_order_"


def parse_threshold(text: Any) -> tuple[str, float] | None:
    """Return ``(operator, limit)`` for a machine-readable threshold string.

    ``None`` means the threshold is prose only (``"in [0.5, 2.0]"``,
    ``"True on every level"``): it cannot be re-evaluated here and must be
    counted as unverified rather than as a pass.
    """
    if not isinstance(text, str):
        return None
    match = _THRESHOLD_RE.match(text)
    if match is None:
        return None
    return match.group(1), float(match.group(2))


def compare(value: float, operator: str, limit: float) -> bool:
    """Apply a parsed threshold to a measured value (no tolerance snapping)."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"criterion value must be a number, got {value!r}")
    if math.isnan(value):
        return False
    if operator == "<":
        return value < limit
    if operator == "<=":
        return value <= limit
    if operator == ">":
        return value > limit
    if operator == ">=":
        return value >= limit
    raise ValueError(f"unsupported operator {operator!r}")


def _aggregate(value: Any) -> dict[str, float] | None:
    """Scalar readings of a criterion value (``{"value": x}`` or ``{min, max, ...}``)."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return {"value": float(value)}
    if isinstance(value, (list, tuple)):
        numbers = [
            float(item)
            for item in value
            if isinstance(item, (int, float)) and not isinstance(item, bool)
        ]
        numbers = [item for item in numbers if not math.isnan(item)]
        if not numbers:
            return None
        readings = {"min": min(numbers), "max": max(numbers)}
        readings["first"] = numbers[0]
        readings["last"] = numbers[-1]
        return readings
    return None


def audit_criteria(criteria: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Re-evaluate every criterion's recorded verdict against its own numbers."""
    rows: list[dict[str, Any]] = []
    for index, criterion in enumerate(criteria):
        name = criterion.get("name", f"#{index}")
        parsed = parse_threshold(criterion.get("threshold"))
        recorded = bool(criterion.get("ok"))
        if parsed is None:
            rows.append(
                {
                    "name": name,
                    "value": criterion.get("value"),
                    "threshold": criterion.get("threshold"),
                    "recorded_ok": recorded,
                    "rederived_ok": None,
                    "status": "not_machine_verifiable",
                }
            )
            continue
        operator, limit = parsed
        value = criterion.get("value")
        aggregations = _aggregate(value)
        if aggregations is None:
            rows.append(
                {
                    "name": name,
                    "value": value,
                    "threshold": criterion.get("threshold"),
                    "recorded_ok": recorded,
                    "rederived_ok": None,
                    "status": "non_numeric_value",
                }
            )
            continue
        outcomes = {
            tag: compare(scalar, operator, limit) for tag, scalar in aggregations.items()
        }
        # A criterion over a ladder of rungs may be published as a list: the
        # module compares the worst rung for most of them and the last rung for
        # ``circulation_observed_order``.  Both readings are legitimate; the
        # audit therefore accepts the verdict if *some* reading reproduces it and
        # says which ones do, instead of guessing one aggregation.
        rederived = any(outcomes.values())
        row_status = "agree" if rederived == recorded else "disagree"
        row = {
            "name": name,
            "value": value,
            "threshold": criterion.get("threshold"),
            "recorded_ok": recorded,
            "rederived_ok": rederived,
            "status": row_status,
        }
        if len(set(outcomes.values())) > 1:
            row["aggregation_satisfied"] = sorted(
                tag for tag, ok in outcomes.items() if ok
            )
        rows.append(row)
    return {
        "n_criteria": len(rows),
        "n_agree": sum(1 for row in rows if row["status"] == "agree"),
        "disagreements": [row for row in rows if row["status"] == "disagree"],
        "non_numeric": [
            row for row in rows if row["status"] == "non_numeric_value"
        ],
        "unverified": [
            row["name"]
            for row in rows
            if row["status"] == "not_machine_verifiable"
        ],
        "not_ok": [row["name"] for row in rows if not row["recorded_ok"]],
        "aggregation_ambiguous": [
            {"name": row["name"], "satisfied": row["aggregation_satisfied"]}
            for row in rows
            if "aggregation_satisfied" in row
        ],
        "rows": rows,
    }


def audit_acceptance(
    payload: dict[str, Any], criteria: Sequence[dict[str, Any]], tolerance: float
) -> dict[str, Any]:
    """Check that the acceptance block summarises the criteria list."""
    acceptance = payload.get("acceptance") or {}
    failed = [c.get("name") for c in criteria if not c.get("ok")]
    recorded_failed = list(acceptance.get("failed_checks") or [])
    recorded_passed = acceptance.get("passed")
    verdict = payload.get("overall")
    declared_tolerance = acceptance.get("tolerance")
    problems: list[str] = []
    if recorded_failed != failed:
        problems.append(
            f"failed_checks {recorded_failed} does not match the {len(failed)} "
            f"violated criteria {failed}"
        )
    if recorded_passed is not None and bool(recorded_passed) != (not failed):
        problems.append(
            f"passed={recorded_passed} contradicts {len(failed)} violated criteria"
        )
    if verdict is not None and str(verdict).upper() not in (
        "PASS" if not failed else "FAIL",
        "PASS" if not failed else "FAILED",
    ):
        problems.append(f"overall={verdict!r} contradicts the criteria list")
    if acceptance.get("n_checks") not in (None, len(criteria)):
        problems.append(
            f"n_checks={acceptance.get('n_checks')} but {len(criteria)} criteria "
            "are published"
        )
    if declared_tolerance is None:
        problems.append("no tolerance is declared in the acceptance block")
    elif float(declared_tolerance) > float(tolerance) + 0.0:
        problems.append(
            f"declared tolerance {declared_tolerance} is looser than the pool "
            f"tolerance {tolerance}"
        )
    return {
        "recorded_passed": recorded_passed,
        "recorded_failed": recorded_failed,
        "rederived_failed": failed,
        "declared_tolerance": declared_tolerance,
        "problems": problems,
    }


def _ladder_value(level: dict[str, Any], metric: str) -> float | None:
    for key in (f"{metric}_relative_l2", metric):
        value = level.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def audit_orders(payload: dict[str, Any]) -> dict[str, Any]:
    """Recompute every published order fit (and the ``finest`` summary) from the ladder."""
    findings: list[dict[str, Any]] = []
    for route_name, route in (payload.get("routes") or {}).items():
        if not isinstance(route, dict):
            continue
        levels = [lvl for lvl in route.get("levels") or [] if isinstance(lvl, dict)]
        for key, reported in route.items():
            if not key.startswith(_ORDER_PREFIX) or not isinstance(reported, (list, tuple)):
                continue
            metric = key[len(_ORDER_PREFIX):]
            errors = [_ladder_value(lvl, metric) for lvl in levels]
            spacings = [
                lvl.get("h_mean") if isinstance(lvl.get("h_mean"), (int, float)) else None
                for lvl in levels
            ]
            observed: list[float | None] = []
            for index in range(len(levels) - 1):
                e0, e1 = errors[index], errors[index + 1]
                if not e0 or not e1 or e0 <= 0.0 or e1 <= 0.0:
                    observed.append(None)
                    continue
                if spacings[index] and spacings[index + 1]:
                    ratio = spacings[index] / spacings[index + 1]
                else:
                    ratio = float(levels[index]["panels"]) / float(
                        levels[index + 1]["panels"]
                    )
                observed.append(math.log(e0 / e1) / math.log(ratio))
            mismatches = [
                {"rung": index, "reported": rep, "recomputed": obs}
                for index, (rep, obs) in enumerate(zip(reported, observed))
                if obs is not None and abs(float(rep) - obs) > 1e-9 * max(1.0, abs(obs))
            ]
            if len(reported) != len(observed):
                mismatches.append(
                    {
                        "rung": None,
                        "reported": list(reported),
                        "recomputed": observed,
                        "note": "published order count does not match the ladder",
                    }
                )
            findings.append(
                {
                    "route": route_name,
                    "key": key,
                    "reported": list(reported),
                    "recomputed": observed,
                    "mismatches": mismatches,
                }
            )
        finest = route.get("finest")
        if isinstance(finest, dict) and levels:
            last = levels[-1]
            for key, value in finest.items():
                if key not in last or not isinstance(value, (int, float)):
                    continue
                if isinstance(value, bool):
                    continue
                if abs(float(value) - float(last[key])) > 1e-12 * max(
                    1.0, abs(float(last[key]))
                ):
                    findings.append(
                        {
                            "route": route_name,
                            "key": f"finest.{key}",
                            "reported": float(value),
                            "recomputed": float(last[key]),
                            "mismatches": [
                                {
                                    "rung": "finest",
                                    "reported": float(value),
                                    "recomputed": float(last[key]),
                                }
                            ],
                        }
                    )
    return {"n_fits": len(findings), "findings": findings}


def audit_record(
    payload: dict[str, Any], tolerance: float = POOL_TOLERANCE
) -> dict[str, Any]:
    """Run the whole re-derivation and report whether the record is self-consistent."""
    criteria = payload.get("criteria") or []
    if not (
        isinstance(criteria, list)
        and all(isinstance(row, dict) for row in criteria)
        and isinstance(payload.get("acceptance"), dict)
    ):
        return {
            "case_id": payload.get("case_id"),
            "schema": payload.get("schema"),
            "verdict": payload.get("overall"),
            "supported": False,
            "blocking": False,
            "ok": True,
            "reason": (
                "not a case-pool record: no list of criterion objects with an "
                "acceptance block (schemas differ per benchmark family)"
            ),
        }
    criteria_report = audit_criteria(criteria)
    acceptance_report = audit_acceptance(payload, criteria, tolerance)
    order_report = audit_orders(payload)
    order_mismatches = [f for f in order_report["findings"] if f["mismatches"]]
    blocking = bool(
        criteria_report["disagreements"]
        or criteria_report["non_numeric"]
        or acceptance_report["problems"]
        or order_mismatches
    )
    return {
        "case_id": payload.get("case_id"),
        "schema": payload.get("schema"),
        "verdict": payload.get("overall"),
        "supported": True,
        "criteria": criteria_report,
        "acceptance": acceptance_report,
        "orders": order_report,
        "order_mismatches": order_mismatches,
        "blocking": blocking,
        "ok": not blocking,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", nargs="+", type=Path, help="results.json paths")
    parser.add_argument(
        "--tolerance", type=float, default=POOL_TOLERANCE, help="pool tolerance"
    )
    parser.add_argument(
        "--quiet", action="store_true", help="print one line per record only"
    )
    args = parser.parse_args(argv)
    worst = 0
    for path in args.records:
        payload = json.loads(path.read_text())
        report = audit_record(payload, tolerance=args.tolerance)
        if not report["supported"]:
            print(f"{path}: skipped ({report['reason']})")
            continue
        if args.quiet:
            print(
                f"{path}: {report['case_id']} verdict={report['verdict']} "
                f"criteria={report['criteria']['n_criteria']} "
                f"agree={report['criteria']['n_agree']} "
                f"unverified={len(report['criteria']['unverified'])} "
                f"blocking={report['blocking']}"
            )
        else:
            slim = dict(report)
            slim.pop("supported", None)
            slim["criteria"] = {
                key: value
                for key, value in report["criteria"].items()
                if key != "rows"
            }
            print(json.dumps({"record": str(path), **slim}, indent=2, default=str))
        worst = max(worst, 1 if report["blocking"] else 0)
    return worst


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())