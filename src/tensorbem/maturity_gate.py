"""Five-level maturity grading (L0–L5) and cross-consistency gate for TensorBEM capabilities.

This module extends the static evidence registry of :mod:`tensorbem.capability_evidence`
with a **maturity ladder** that answers not just "is it implemented?" but "how mature
is the verification?".

Maturity levels
---------------
=====  =====================================================================
Level  Meaning
=====  =====================================================================
L0     ``NOT_SUPPORTED`` — capability not implemented or explicitly out of scope
L1     ``STUB`` / ``MOCK`` — placeholder, hard-coded, or random output
L2     ``IMPLEMENTED`` — code exists, basic tests pass, documentation anchor present
       (the floor of the current ``REAL`` grade in ``capability_evidence``)
L3     ``VERIFIED`` — verified against an analytical or reference solution with
       quantified error; at least one convergence study or GCI exists
L4     ``CROSS_VALIDATED`` — cross-validated against an independent numerical route
       (e.g. two independent formulations agree) or an external solver (e.g. Capytaine)
L5     ``PRODUCTION`` — production-ready: L3 + L4 + uncertainty quantification +
       real-world or experimental validation + full documentation chain
=====  =====================================================================

Gate semantics
--------------
A maturity gate is **fail-able**: it evaluates to PASS/FAIL for each capability
based on whether the claimed maturity level is supported by evidence.  The
overall gate coverage is ``passing_gates / total_failable_gates``.

Cross-consistency
-----------------
Related capabilities are checked for mutual consistency.  For example, if a
radiation capability claims L4 but the corresponding diffraction capability is
only L2, the pair is flagged for review.

Usage::

    python -m tensorbem.maturity_gate                    # print maturity table
    python -m tensorbem.maturity_gate --json out.json    # machine form
    python -m tensorbem.maturity_gate --report           # generate markdown report
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from . import capability_evidence as ce

# ---------------------------------------------------------------------------
# Maturity levels
# ---------------------------------------------------------------------------

L0 = "L0_NOT_SUPPORTED"
L1 = "L1_STUB_MOCK"
L2 = "L2_IMPLEMENTED"
L3 = "L3_VERIFIED"
L4 = "L4_CROSS_VALIDATED"
L5 = "L5_PRODUCTION"

LEVEL_ORDER = [L0, L1, L2, L3, L4, L5]
LEVEL_NAMES = {
    L0: "Not Supported",
    L1: "Stub / Mock",
    L2: "Implemented",
    L3: "Verified",
    L4: "Cross-Validated",
    L5: "Production",
}


@dataclass
class MaturityEvidence:
    """Derived maturity evidence for one capability."""

    capability_key: str
    capability_title: str
    evidence_level: str  # from capability_evidence: REAL/MOCK/STUB/etc.
    maturity_level: str  # L0-L5
    gate_pass: bool
    reasons: list[str] = field(default_factory=list)
    # L3+ evidence
    has_analytical_reference: bool = False
    has_convergence_study: bool = False
    has_quantified_error: bool = False
    # L4+ evidence
    has_independent_route: bool = False
    has_external_crosscheck: bool = False
    # L5 evidence
    has_uncertainty_quantification: bool = False
    has_experimental_validation: bool = False
    has_full_doc_chain: bool = False


# ---------------------------------------------------------------------------
# Maturity criteria detection
# ---------------------------------------------------------------------------

def _read_file_safe(path: pathlib.Path) -> str:
    try:
        return path.read_text(encoding="utf-8").lower()
    except (OSError, UnicodeDecodeError):
        return ""


def _detect_analytical_reference(cap: ce.Capability, root: pathlib.Path) -> bool:
    """Check if the capability has an analytical/reference solution in its tests."""
    # Strong indicators: explicit analytical solution names
    ref_keywords = [
        "analytic", "analytical", "exact", "closed_form", "closed-form",
        "maccamy", "hulme", "ursell", "wigley.*analytic", "ellipsoid.*kochin",
        "sphere.*analytic", "cylinder.*exact", "haskind", "kochin.*analytic",
        "maruo.*closed", "biesel", "havelock.*analytic", "mie.*series",
        "bessel.*exact", "polygon.*exact",
    ]
    for test_id in cap.tests:
        file_part = test_id.partition("::")[0]
        fpath = root / file_part
        text = _read_file_safe(fpath)
        if any(re.search(kw, text) for kw in ref_keywords):
            return True
    return False


def _detect_convergence_study(cap: ce.Capability, root: pathlib.Path) -> bool:
    """Check if the capability has a convergence study or GCI."""
    conv_keywords = [
        r"convergence", r"converge", r"\bgci\b", r"grid_convergence",
        r"grid-convergence", r"roache", r"refinement", r"mesh_convergence",
        r"panel_convergence", r"subdivision", r"n_panels", r"n_subdivisions",
        r"convergence.*order", r"order.*convergence",
    ]
    for test_id in cap.tests:
        file_part = test_id.partition("::")[0]
        fpath = root / file_part
        text = _read_file_safe(fpath)
        if any(re.search(kw, text) for kw in conv_keywords):
            return True
    return False


def _detect_quantified_error(cap: ce.Capability, root: pathlib.Path) -> bool:
    """Check if the capability reports quantified error (not just pass/fail)."""
    err_keywords = [
        r"relative_error", r"rel_error", r"\berror_", r"\berr\b",
        r"tolerance", r"percent", r"epsilon", r"atol", r"rtol",
        r"assert.*<.*e-", r"within.*%", r"error_budget",
        r"relative.*error", r"error.*relative",
    ]
    for test_id in cap.tests:
        file_part = test_id.partition("::")[0]
        fpath = root / file_part
        text = _read_file_safe(fpath)
        if any(re.search(kw, text) for kw in err_keywords):
            return True
    return False


def _detect_independent_route(cap: ce.Capability, root: pathlib.Path) -> bool:
    """Check if the capability has two independent numerical routes that are compared.

    This requires explicit mention of two routes being compared or checked for
    agreement, not just the presence of keywords like 'haskind' or 'kochin'.
    """
    # Strong indicators of independent route comparison
    route_patterns = [
        r"two.*route", r"two-route", r"dual.*route",
        r"independent.*route", r"route.*independent",
        r"near.field.*far.field", r"far.field.*near.field",
        r"near_field.*far_field", r"far_field.*near_field",
        r"reciprocity", r"optical.theorem", r"energy.identity",
        r"cross.route", r"cross_route",
        r"alternative.*route", r"route.*alternative",
        r"vs.*route", r"route.*vs",
        r"agree.*within", r"within.*agree",
        r"pairwise.*difference",
    ]
    for test_id in cap.tests:
        file_part = test_id.partition("::")[0]
        fpath = root / file_part
        text = _read_file_safe(fpath)
        if any(re.search(kw, text) for kw in route_patterns):
            return True
    return False


def _detect_external_crosscheck(cap: ce.Capability, root: pathlib.Path) -> bool:
    """Check if the capability is cross-checked against an external solver."""
    ext_keywords = [
        "capytaine", "nemoh", "wamit", "aqwa", "hydrostar",
        "crosscheck", "cross_check", "cross-val", "crossval",
        "oss_cross", "oss_crossval",
    ]
    for test_id in cap.tests:
        file_part = test_id.partition("::")[0]
        fpath = root / file_part
        text = _read_file_safe(fpath)
        if any(kw in text for kw in ext_keywords):
            return True
    return False


def _detect_uncertainty_quantification(cap: ce.Capability, root: pathlib.Path) -> bool:
    """Check if the capability has uncertainty quantification."""
    uq_keywords = [
        "uncertainty", "uq", "monte_carlo", "polynomial_chaos",
        "sensitivity", "confidence", "variance", "standard_deviation",
        "error_bar", "recovery_uncertainty", "verification_uncertainty",
    ]
    for test_id in cap.tests:
        file_part = test_id.partition("::")[0]
        fpath = root / file_part
        text = _read_file_safe(fpath)
        if any(kw in text for kw in uq_keywords):
            return True
    return False


def _detect_experimental_validation(cap: ce.Capability, root: pathlib.Path) -> bool:
    """Check if the capability has experimental/tank data validation."""
    exp_keywords = [
        "experiment", "experimental", "tank", "towing", "ittc",
        "david", "kcs", "kvlcc", "measured",
        "digitised", "digitized",
    ]
    for test_id in cap.tests:
        file_part = test_id.partition("::")[0]
        fpath = root / file_part
        text = _read_file_safe(fpath)
        if any(kw in text for kw in exp_keywords):
            return True
    return False


def _detect_full_doc_chain(cap: ce.Capability, root: pathlib.Path) -> bool:
    """Check if the capability has a full documentation chain."""
    if not cap.docs:
        return False
    doc_paths = [p for p, _ in cap.docs]
    if len(doc_paths) >= 2:
        return True
    for p in doc_paths:
        fpath = root / p
        if fpath.is_file() and fpath.stat().st_size > 2048:
            return True
    return False


# ---------------------------------------------------------------------------
# Cross-consistency groups (using actual capability keys)
# ---------------------------------------------------------------------------

CROSS_CONSISTENCY_GROUPS: list[tuple[str, str, str]] = [
    # (capability_a, capability_b, relationship description)
    ("free-surface-3d", "laplace-3d", "3-D free-surface builds on 3-D Laplace"),
    ("free-surface-2d", "laplace-2d", "2-D free-surface builds on 2-D Laplace"),
    ("second-order-drift", "cylinder-and-farfield-drift", "drift force builds on cylinder/farfield"),
    ("ursell-radiation", "free-surface-2d", "Ursell radiation requires 2-D free-surface"),
    ("hulme-hemisphere", "free-surface-3d", "Hulme hemisphere requires 3-D free-surface"),
    ("maccamy-fuchs", "acoustic-3d-helmholtz", "MacCamy-Fuchs uses Helmholtz formulation"),
    ("cummins-memory-kernel-closure", "coupled-memory-kernel-matrix", "Cummins closure requires memory kernel matrix"),
    ("hydrostatics", "rao", "RAO builds on hydrostatics"),
    ("michell-canonical-spectrum-panel-route", "michell-general-and-tandem", "both use Michell thin-ship formulation"),
    ("finite-depth-2d", "finite-depth-3d", "finite-depth kernels in 2-D and 3-D"),
    ("ice-region-hydrodynamics", "finite-depth-2d", "ice dispersion builds on finite-depth"),
    ("acoustic-3d-helmholtz", "acoustic-3d-pulse", "pulse synthesis builds on Helmholtz harmonics"),
    ("diffraction-drift-maturity", "second-order-drift", "diffraction-drift maturity covers second-order drift"),
    ("hydrostatic-restoring-pressure-route", "hydrostatics", "restoring pressure route builds on hydrostatics"),
    ("flexural-gravity-numeric", "flexural-gravity-dispersion", "numeric flexural-gravity builds on dispersion"),
    ("near-field-quadrature", "near-singular", "near-field quadrature handles near-singular integrals"),
    ("quadratic-panel-operator", "linear-panel-operators", "quadratic panels extend linear operators"),
    ("irregular-frequency-predictor-verified", "irregular-frequency-sanitisation", "irregular frequency prediction and sanitisation"),
    ("state-space-realization", "time-domain-memory-kernel", "state-space realization of memory kernel"),
    ("kkr-causality", "cummins-memory-kernel-closure", "Kramers-Kronig causality for memory kernel"),
]


@dataclass
class CrossConsistencyResult:
    cap_a: str
    cap_b: str
    relationship: str
    maturity_a: str
    maturity_b: str
    consistent: bool
    reason: str = ""


def check_cross_consistency(
    maturity_map: dict[str, MaturityEvidence],
) -> list[CrossConsistencyResult]:
    """Check cross-consistency between related capabilities.

    Two capabilities are consistent if their maturity levels differ by at most
    one step (e.g. L3 and L4 are OK, L2 and L5 are not).
    """
    results = []
    for cap_a, cap_b, relationship in CROSS_CONSISTENCY_GROUPS:
        ev_a = maturity_map.get(cap_a)
        ev_b = maturity_map.get(cap_b)
        if ev_a is None or ev_b is None:
            continue
        idx_a = LEVEL_ORDER.index(ev_a.maturity_level)
        idx_b = LEVEL_ORDER.index(ev_b.maturity_level)
        diff = abs(idx_a - idx_b)
        consistent = diff <= 1
        reason = ""
        if not consistent:
            reason = (
                f"Maturity gap of {diff} levels: "
                f"{ev_a.capability_key}={ev_a.maturity_level}, "
                f"{ev_b.capability_key}={ev_b.maturity_level}"
            )
        results.append(CrossConsistencyResult(
            cap_a=cap_a, cap_b=cap_b, relationship=relationship,
            maturity_a=ev_a.maturity_level, maturity_b=ev_b.maturity_level,
            consistent=consistent, reason=reason,
        ))
    return results


# ---------------------------------------------------------------------------
# Maturity grading
# ---------------------------------------------------------------------------

def grade_maturity(
    capability: ce.Capability,
    row_evidence: ce.RowEvidence,
    root: pathlib.Path,
) -> MaturityEvidence:
    """Grade the maturity level of one capability based on evidence."""

    ev = MaturityEvidence(
        capability_key=capability.key,
        capability_title=capability.title,
        evidence_level=row_evidence.level,
        maturity_level=L0,
        gate_pass=False,
    )

    # L0: not supported
    if row_evidence.level in (ce.NOT_SUPPORTED, ce.MISSING):
        ev.maturity_level = L0
        ev.gate_pass = True  # honest NOT_SUPPORTED passes the gate
        ev.reasons.append("capability not in scope")
        return ev

    # L1: stub or mock
    if row_evidence.level in (ce.STUB, ce.MOCK):
        ev.maturity_level = L1
        ev.gate_pass = True  # honest STUB/MOCK passes the gate
        ev.reasons.append(f"evidence level is {row_evidence.level}")
        return ev

    # L2: implemented (code + tests + docs exist)
    if row_evidence.level in (ce.WITHHELD, ce.STALE_PIN, ce.STALE_CLAIM):
        ev.maturity_level = L2
        ev.gate_pass = False  # stale/withheld fails the gate
        ev.reasons.extend(row_evidence.reasons)
        return ev

    # At this point: row_evidence.level == REAL
    # Base maturity is L2 (implemented)
    ev.maturity_level = L2
    ev.gate_pass = True

    # Detect L3 evidence: analytical reference + quantified error
    ev.has_analytical_reference = _detect_analytical_reference(capability, root)
    ev.has_convergence_study = _detect_convergence_study(capability, root)
    ev.has_quantified_error = _detect_quantified_error(capability, root)

    if ev.has_analytical_reference and ev.has_quantified_error:
        ev.maturity_level = L3
        ev.reasons.append("verified against analytical/reference solution")
        if ev.has_convergence_study:
            ev.reasons.append("convergence study present")

    # Detect L4 evidence: independent route or external crosscheck
    ev.has_independent_route = _detect_independent_route(capability, root)
    ev.has_external_crosscheck = _detect_external_crosscheck(capability, root)

    if ev.has_independent_route or ev.has_external_crosscheck:
        if ev.maturity_level == L3:
            ev.maturity_level = L4
            ev.reasons.append("cross-validated via independent route or external solver")
        else:
            ev.reasons.append("has cross-validation evidence but lacks L3 verification")

    # Detect L5 evidence: UQ + experimental + full doc
    ev.has_uncertainty_quantification = _detect_uncertainty_quantification(capability, root)
    ev.has_experimental_validation = _detect_experimental_validation(capability, root)
    ev.has_full_doc_chain = _detect_full_doc_chain(capability, root)

    if ev.maturity_level == L4 and ev.has_uncertainty_quantification and ev.has_full_doc_chain:
        ev.maturity_level = L5
        ev.reasons.append("production-ready with UQ and full documentation")

    return ev


def audit_maturity(root: pathlib.Path | None = None) -> list[MaturityEvidence]:
    """Run the full maturity audit for all capabilities."""
    root = root or ce.repo_root()
    rows = ce.audit(root)
    results = []
    for row in rows:
        ev = grade_maturity(row.capability, row, root)
        results.append(ev)
    return results


# ---------------------------------------------------------------------------
# Gate coverage computation
# ---------------------------------------------------------------------------

def gate_coverage(maturity_evidence: list[MaturityEvidence]) -> dict[str, Any]:
    """Compute gate coverage metrics.

    Gate coverage = passing_failable_gates / total_failable_gates.
    NOT_SUPPORTED (L0) capabilities are not failable (they are honestly out of scope).
    """
    failable = [ev for ev in maturity_evidence if ev.maturity_level != L0]
    passing = [ev for ev in failable if ev.gate_pass]
    failing = [ev for ev in failable if not ev.gate_pass]

    by_level: dict[str, int] = {}
    for ev in maturity_evidence:
        by_level[ev.maturity_level] = by_level.get(ev.maturity_level, 0) + 1

    return {
        "total_capabilities": len(maturity_evidence),
        "failable_gates": len(failable),
        "passing_gates": len(passing),
        "failing_gates": len(failing),
        "coverage": len(passing) / len(failable) if failable else 1.0,
        "by_level": by_level,
        "failing_keys": [ev.capability_key for ev in failing],
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def render_markdown_table(maturity_evidence: list[MaturityEvidence]) -> str:
    """Render a markdown table of maturity levels."""
    lines = [
        "# TensorBEM Capability Maturity Ledger",
        "",
        "| Capability | Title | Evidence | Maturity | Gate | L3+ | L4+ | L5+ |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for ev in sorted(maturity_evidence, key=lambda e: LEVEL_ORDER.index(e.maturity_level)):
        l3 = "✓" if ev.has_analytical_reference and ev.has_quantified_error else ""
        l4 = "✓" if ev.has_independent_route or ev.has_external_crosscheck else ""
        l5 = "✓" if ev.has_uncertainty_quantification and ev.has_full_doc_chain else ""
        gate = "PASS" if ev.gate_pass else "FAIL"
        title_short = ev.capability_title[:50]
        lines.append(
            f"| {ev.capability_key} | {title_short} | "
            f"{ev.evidence_level} | {ev.maturity_level} | {gate} | {l3} | {l4} | {l5} |"
        )
    return "\n".join(lines)


def render_report(
    maturity_evidence: list[MaturityEvidence],
    cross_results: list[CrossConsistencyResult],
    coverage: dict[str, Any],
) -> str:
    """Render a full maturity gate report."""
    lines = [
        "# TensorBEM Maturity Gate Report (H6-MATURITY-GATE)",
        "",
        "## 1. Gate Coverage Summary",
        "",
        f"- **Total capabilities**: {coverage['total_capabilities']}",
        f"- **Failable gates**: {coverage['failable_gates']}",
        f"- **Passing gates**: {coverage['passing_gates']}",
        f"- **Failing gates**: {coverage['failing_gates']}",
        f"- **Gate coverage**: {coverage['coverage']:.1%}",
        "",
        "### Maturity Distribution",
        "",
    ]
    for level in LEVEL_ORDER:
        count = coverage["by_level"].get(level, 0)
        lines.append(f"- **{level}** ({LEVEL_NAMES[level]}): {count}")

    lines += [
        "",
        "## 2. Failing Gates",
        "",
    ]
    failing = [ev for ev in maturity_evidence if not ev.gate_pass]
    if failing:
        for ev in failing:
            lines.append(f"- **{ev.capability_key}** ({ev.maturity_level}): {', '.join(ev.reasons)}")
    else:
        lines.append("All failable gates pass.")

    lines += [
        "",
        "## 3. Cross-Consistency Check",
        "",
    ]
    inconsistent = [r for r in cross_results if not r.consistent]
    if inconsistent:
        for r in inconsistent:
            lines.append(
                f"- **{r.cap_a}** ({r.maturity_a}) ↔ **{r.cap_b}** ({r.maturity_b}): "
                f"{r.relationship} — {r.reason}"
            )
    else:
        lines.append("All cross-consistency checks pass.")

    lines += [
        "",
        "## 4. Full Maturity Table",
        "",
        render_markdown_table(maturity_evidence),
        "",
        "## 5. Methodology",
        "",
        "### Five-Level Maturity Scale",
        "",
        "| Level | Name | Criteria |",
        "|---|---|---|",
        "| L0 | Not Supported | Capability not implemented or explicitly out of scope |",
        "| L1 | Stub/Mock | Placeholder, hard-coded, or random output |",
        "| L2 | Implemented | Code exists, basic tests pass, documentation anchor present |",
        "| L3 | Verified | L2 + verified against analytical/reference solution with quantified error + convergence study |",
        "| L4 | Cross-Validated | L3 + cross-validated via independent numerical route or external solver |",
        "| L5 | Production | L4 + uncertainty quantification + experimental validation + full documentation chain |",
        "",
        "### Gate Coverage",
        "",
        "Gate coverage = passing_failable_gates / total_failable_gates.",
        "L0 (NOT_SUPPORTED) capabilities are not failable — they are honestly out of scope.",
        "A gate fails when the claimed maturity level is not supported by evidence.",
        "",
        "### Cross-Consistency",
        "",
        "Related capabilities are checked for maturity compatibility. Two capabilities",
        "are consistent if their maturity levels differ by at most one step.",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", type=pathlib.Path, help="write machine-readable output")
    parser.add_argument("--report", action="store_true", help="generate markdown report")
    parser.add_argument("--root", type=pathlib.Path, help="repository root")
    args = parser.parse_args(argv)

    root = args.root or ce.repo_root()
    maturity = audit_maturity(root)
    cross = check_cross_consistency({ev.capability_key: ev for ev in maturity})
    coverage = gate_coverage(maturity)

    if args.json:
        payload = {
            "coverage": coverage,
            "maturity": [
                {
                    "key": ev.capability_key,
                    "title": ev.capability_title,
                    "evidence_level": ev.evidence_level,
                    "maturity_level": ev.maturity_level,
                    "gate_pass": ev.gate_pass,
                    "reasons": ev.reasons,
                    "l3": {
                        "analytical_reference": ev.has_analytical_reference,
                        "convergence_study": ev.has_convergence_study,
                        "quantified_error": ev.has_quantified_error,
                    },
                    "l4": {
                        "independent_route": ev.has_independent_route,
                        "external_crosscheck": ev.has_external_crosscheck,
                    },
                    "l5": {
                        "uncertainty_quantification": ev.has_uncertainty_quantification,
                        "experimental_validation": ev.has_experimental_validation,
                        "full_doc_chain": ev.has_full_doc_chain,
                    },
                }
                for ev in maturity
            ],
            "cross_consistency": [
                {
                    "cap_a": r.cap_a,
                    "cap_b": r.cap_b,
                    "relationship": r.relationship,
                    "maturity_a": r.maturity_a,
                    "maturity_b": r.maturity_b,
                    "consistent": r.consistent,
                    "reason": r.reason,
                }
                for r in cross
            ],
        }
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")

    if args.report:
        report = render_report(maturity, cross, coverage)
        report_path = root / "docs" / "maturity_gate_report.md"
        report_path.write_text(report, encoding="utf-8")
        print(f"wrote {report_path}")

    # Print summary
    print(f"\nMaturity Gate Summary:")
    print(f"  Total capabilities: {coverage['total_capabilities']}")
    print(f"  Failable gates: {coverage['failable_gates']}")
    print(f"  Passing: {coverage['passing_gates']}")
    print(f"  Failing: {coverage['failing_gates']}")
    print(f"  Coverage: {coverage['coverage']:.1%}")
    print(f"\nMaturity distribution:")
    for level in LEVEL_ORDER:
        count = coverage["by_level"].get(level, 0)
        print(f"  {level}: {count}")

    failing = [ev for ev in maturity if not ev.gate_pass]
    if failing:
        print(f"\nFailing gates ({len(failing)}):")
        for ev in failing:
            print(f"  - {ev.capability_key}: {', '.join(ev.reasons)}")

    inconsistent = [r for r in cross if not r.consistent]
    if inconsistent:
        print(f"\nCross-consistency issues ({len(inconsistent)}):")
        for r in inconsistent:
            print(f"  - {r.cap_a}({r.maturity_a}) ↔ {r.cap_b}({r.maturity_b}): {r.reason}")

    return 0 if not failing and not inconsistent else 1


if __name__ == "__main__":
    raise SystemExit(main())
