"""Executable contract for the *conclusion-class prose* numbers of the 3-D document.

Why this module exists
----------------------
:mod:`tensorbem.benchmarks.three_dimensional_doc_pins` pinned the four result
tables of ``docs/three_dimensional_bem.md`` (86 numeric cells).  The numbers a
reader actually quotes, however, live *outside* those tables: the abstract, the
diagnosis of section 5.5, item 9 of the discussion and the conclusion all restate
the same three headline facts

* the representation-formula surface velocity is wrong by ``53%--65%`` and does
  not improve under refinement,
* the same formula is off by ``48%`` at ``10^{-3}`` off the surface,
* and by ``1.6%`` at the body centre,

eight times, in eight independently worded sentences.  Prose is where a repo
drifts silently: the table can be regenerated while four restatements keep the
stale number.  This repository has already been bitten three times by numbers
that no build checked (the Ursell ``7.7e-3 -> 8.51e-3`` column, the MFS column,
the Wigley ``2LBT/3 -> 4LBT/9`` displacement).

What is checked, and how
------------------------
Every restatement is registered as a :class:`ProseClaim` with three parts:

1. ``anchor`` -- a literal sentence fragment that must still occur verbatim in
   the document.  Reworded or deleted: the claim fails, because the audit trail
   it belongs to is gone;
2. ``tokens`` -- the printed numbers carried inside that fragment.  They are
   parsed from the anchor itself, never hand-copied, and the tolerance is derived
   from the digits actually printed (``0.518`` is a claim to three significant
   digits, ``53`` to two);
3. ``group`` -- the measurement that produces them.  Measurements come from the
   committed artifacts under ``docs/m3085/outputs`` (both the surface-velocity
   study and the per-subdivision ``sphere_velocity`` runs) or, for the near-body
   layer, from a cheap live solve (80 panels, milliseconds).

Restatements of the same fact share a group, so the ledger additionally requires
the eight sentences to *agree with each other*: updating one and forgetting the
others is a DRIFT, which is the failure mode that motivates the module.

Numbers that are qualitative rather than numeric (``"the error does not decrease
under refinement"``, ``"the recovered path does not have that defect"``) are
registered in :data:`QUALITATIVE_CLAIMS` and checked as booleans on the same
measurements.

Usage
-----
``python -m tensorbem.benchmarks.prose_claims --check`` prints strict JSON and
exits non-zero on drift; ``--coverage`` reports how many previously unguarded
prose numbers are now bound; ``--write-doc-section`` renders the table used in
``docs/m3556/README.md`` from the measurement side, not from a transcript.

Scope: this is a documentation contract.  No solver code is touched and no new
physics is claimed; the numbers themselves are the ones the artifacts already
hold.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

from .three_dimensional_doc_pins import (
    ARTIFACTS,
    DOC,
    REPO_ROOT,
    cell_to_float,
    printed_tolerance,
)

SCHEMA_VERSION = 1

#: Per-subdivision exterior translating-sphere artifacts (20/80/320/1280 panels).
SPHERE_VELOCITY = ARTIFACTS / "sphere_velocity_s{subdivisions}.json"
SURFACE_STUDY = ARTIFACTS / "surface_velocity_study.json"
SUBDIVISIONS = {20: 0, 80: 1, 320: 2, 1280: 3}

#: The refinement levels quoted by the document's velocity table (§6.4) and by
#: the band claim: the 20-panel level (81%) is printed in the table but never
#: quoted in prose, so the band is taken over the three quoted levels.
BAND_PANELS = (80, 320, 1280)

#: Panels used by the live near-body-layer probe.  The document's §5.5 numbers
#: come from the coarsest icosphere (subdivisions=1, 80 panels); the solve is
#: small enough to redo instead of trusting a stored copy.
LAYER_SUBDIVISIONS = 1
LAYER_PROBE_INDEX = 7
LAYER_OFFSET = 1.0e-3


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------


def _load(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"missing artifact {path}; regenerate with --recompute")
    return json.loads(path.read_text())


def representation_ladder(artifacts: Path = ARTIFACTS) -> list[dict[str, float]]:
    """Representation-formula vs recovered surface velocity, per panel count."""
    rows = []
    for panels, subdivisions in sorted(SUBDIVISIONS.items()):
        payload = _load(artifacts / SPHERE_VELOCITY.name.format(subdivisions=subdivisions))
        rows.append(
            {
                "panels": panels,
                "representation_relative_l2_error": float(
                    payload["representation_surface_velocity_relative_l2_error"]
                ),
                "recovered_relative_l2_error": float(
                    payload["surface_velocity_relative_l2_error"]
                ),
            }
        )
    return rows


def representation_band(artifacts: Path = ARTIFACTS) -> tuple[float, float]:
    """The quoted ``53%--65%`` band, as (low, high) in percent.

    The band is the span of the representation-formula error over the levels the
    document quotes (80/320/1280 panels): ``0.5316`` to ``0.6531``.
    """
    rows = [row for row in representation_ladder(artifacts) if row["panels"] in BAND_PANELS]
    values = [100.0 * row["representation_relative_l2_error"] for row in rows]
    return min(values), max(values)


def representation_orders(artifacts: Path = ARTIFACTS) -> list[float]:
    """Observed convergence orders of the representation path (both cases)."""
    study = _load(artifacts / SURFACE_STUDY.name)
    orders = study["observed_orders"]["representation_translating_sphere"]
    orders = orders + study["observed_orders"]["representation_linear_trace"]
    return [float(value) for value in orders]


def recovered_error_decreases(artifacts: Path = ARTIFACTS) -> bool:
    """The recovered path's error falls at every refinement."""
    errors = [row["recovered_relative_l2_error"] for row in representation_ladder(artifacts)]
    return all(later < earlier for earlier, later in zip(errors, errors[1:]))


def near_body_layer() -> dict[str, float]:
    """Live probe of the representation formula inside the ``O(h)`` layer.

    Interior harmonic extension of ``u = x`` on the coarsest icosphere: the exact
    velocity is the constant ``(1, 0, 0)``, so the x-component already is the
    quantity the document quotes.  Two evaluation points, both taken from §5.5:

    * ``10^{-3}`` inside the surface below panel 7 (inside the boundary layer),
    * the body centre (outside it).

    Cost: one 80-panel dense solve, i.e. milliseconds -- cheap enough to run on
    every build instead of pinning a stored copy of the answer.
    """
    from tensorbem.geometry3d import icosphere
    from tensorbem.solver3d import evaluate_surface_velocity, solve_laplace_surface

    mesh = icosphere(LAYER_SUBDIVISIONS, radius=1.0)
    boundary = mesh.centroids[:, 0].copy()
    result = solve_laplace_surface(mesh, boundary, domain="interior")
    index = LAYER_PROBE_INDEX
    inward = mesh.centroids[index] - LAYER_OFFSET * mesh.normals[index]
    approaching = evaluate_surface_velocity(
        mesh, result, inward[None, :], minimum_clearance_factor=0.0
    )[0]
    centre = evaluate_surface_velocity(
        mesh, result, np.array([[0.0, 0.0, 0.0]]), minimum_clearance_factor=0.0
    )[0]
    exact = 1.0
    return {
        "layered_velocity": float(approaching[0]),
        "centre_velocity": float(centre[0]),
        "exact_velocity": exact,
        "layered_percent_error": 100.0 * abs(float(approaching[0]) - exact) / exact,
        "centre_percent_error": 100.0 * abs(float(centre[0]) - exact) / exact,
        "panels": float(mesh.panel_count),
    }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProseClaim:
    """One restatement of a headline number, bound to its measurement.

    ``group`` selects both the measurement function and the *order* in which it
    reports its numbers, so ``tokens`` is read positionally: ``("53", "65")`` is
    ``(low, high)`` of the representation band.
    """

    id: str
    section: str
    anchor: str
    tokens: tuple[str, ...]
    group: str
    note: str = ""


def _measure(group: str, artifacts: Path, layer: Mapping[str, float]) -> list[float] | bool:
    if group == "representation_band":
        return list(representation_band(artifacts))
    if group == "layered_probe":
        return [
            layer["layered_velocity"],
            layer["exact_velocity"],
            layer["layered_percent_error"],
        ]
    if group == "centre_probe":
        return [layer["centre_velocity"], layer["centre_percent_error"]]
    raise KeyError(f"unknown claim group {group!r}")


#: The eight restatements.  ``anchor`` strings are reproduced from the document
#: verbatim, including the en dash in ``53%–65%``; a reformulation breaks the
#: claim on purpose, because the sentence *is* the audit trail.
PROSE_CLAIMS: tuple[ProseClaim, ...] = (
    ProseClaim(
        id="abstract-band",
        section="摘要",
        anchor="表示公式路径在紧贴表面处重构误差达 53%–65%",
        tokens=("53", "65"),
        group="representation_band",
        note="The headline claim, first stated in the abstract.",
    ),
    ProseClaim(
        id="s55-band",
        section="5.5",
        anchor="在面元质心（配点）上重构的速度有 53%–65% 的误差",
        tokens=("53", "65"),
        group="representation_band",
        note="The diagnosis itself.",
    ),
    ProseClaim(
        id="s55-layer",
        section="5.5",
        anchor="速度为 0.518 而正确值为 1.000（相对误差 48%）",
        tokens=("0.518", "1.000", "48"),
        group="layered_probe",
        note="Live 80-panel solve, probe 1e-3 inside panel 7.",
    ),
    ProseClaim(
        id="s55-centre",
        section="5.5",
        anchor="在体心处则为 0.984（1.6%）",
        tokens=("0.984", "1.6"),
        group="centre_probe",
        note="Same solve, probe at the body centre.",
    ),
    ProseClaim(
        id="s64-band",
        section="6.4",
        anchor="停在 53%–65% 且几乎不降",
        tokens=("53", "65"),
        group="representation_band",
        note="Read against the table's own last column.",
    ),
    ProseClaim(
        id="s7-band",
        section="7",
        anchor="重构误差为 53%–65% 且不随加密下降",
        tokens=("53", "65"),
        group="representation_band",
        note="Discussion item 9.",
    ),
    ProseClaim(
        id="s7-layer",
        section="7",
        anchor="紧贴表面误差 48%，体心处 1.6%",
        tokens=("48", "1.6"),
        group="centre_probe_plus_layered",
        note="Discussed as one sentence; checked as two numbers.",
    ),
    ProseClaim(
        id="s9-band",
        section="9",
        anchor="停在 53%–65%",
        tokens=("53", "65"),
        group="representation_band",
        note="Conclusion.",
    ),
    ProseClaim(
        id="s9-layer",
        section="9",
        anchor="紧贴表面误差 48%，体心处 1.6%",
        tokens=("48", "1.6"),
        group="centre_probe_plus_layered",
        note="Conclusion, same sentence shape as item 9.",
    ),
    ProseClaim(
        id="s11-index",
        section="11",
        anchor="如 53%–65%、48%、1.6%",
        tokens=("53", "65", "48", "1.6"),
        group="band_plus_layered",
        note="The pointer that promised this ledger.",
    ),
)

#: Qualitative claims: prose that asserts a behaviour instead of a number.
QUALITATIVE_CLAIMS: tuple[tuple[str, str, str, Callable[[Path], bool]], ...] = (
    (
        "representation-does-not-converge",
        "5.5",
        "且不随加密下降",
        lambda artifacts: max(representation_orders(artifacts)) < 0.5,
    ),
    (
        "recovered-path-has-no-defect",
        "5.5",
        "恢复路径没有这一缺陷",
        recovered_error_decreases,
    ),
)


def _group_values(group: str, artifacts: Path, layer: Mapping[str, float]) -> list[float]:
    if group == "representation_band":
        return list(representation_band(artifacts))
    if group == "layered_probe":
        return [layer["layered_velocity"], layer["exact_velocity"], layer["layered_percent_error"]]
    if group == "centre_probe":
        return [layer["centre_velocity"], layer["centre_percent_error"]]
    if group == "centre_probe_plus_layered":
        return [layer["layered_percent_error"], layer["centre_percent_error"]]
    if group == "band_plus_layered":
        low, high = representation_band(artifacts)
        return [low, high, layer["layered_percent_error"], layer["centre_percent_error"]]
    raise KeyError(f"unknown claim group {group!r}")


NUMBER = re.compile(r"\b\d+\.\d+|(?<![\d.])\d+(?:\.\d+)?")


def _token_in(token: str, anchor: str) -> bool:
    """Does ``anchor`` still print ``token`` as a standalone number?"""
    if token not in anchor:
        return False
    for match in NUMBER.finditer(anchor):
        if match.group(0) == token:
            return True
    return False


def check(
    text: str | None = None,
    artifacts: Path = ARTIFACTS,
    layer: Mapping[str, float] | None = None,
) -> dict[str, object]:
    """Recompute every registered restatement and judge it against its anchor.

    Returns a strict-JSON-serialisable dict: one finding per claim with
    ``PASS``/``DRIFT``, the printed value, the measured value and the deviation.
    ``DRIFT`` covers two distinct failures and says which one happened:
    ``anchor missing`` -- the sentence was reworded, deleted, **or the number
    inside it was edited** (the anchors carry their numbers verbatim, so those
    three cases are indistinguishable by construction and all mean "this claim
    is no longer tracked"); and ``value drift`` -- the sentence is intact but the
    number no longer reproduces.  A third branch, ``token not standalone``, can
    only fire on a registry mistake (a token that is not a standalone number in
    its own anchor) and is unit-tested as an invariant.
    """
    text = DOC.read_text() if text is None else text
    layer = near_body_layer() if layer is None else layer
    cache: dict[str, list[float]] = {}

    def values(group: str) -> list[float]:
        if group not in cache:
            cache[group] = [float(value) for value in _group_values(group, artifacts, layer)]
        return cache[group]

    findings = []
    for claim in PROSE_CLAIMS:
        finding: dict[str, object] = {
            "id": claim.id,
            "section": claim.section,
            "group": claim.group,
            "anchor": claim.anchor,
            "tokens": list(claim.tokens),
            "status": "PASS",
        }
        if claim.anchor not in text:
            finding["status"] = "DRIFT"
            finding["reason"] = (
                "anchor missing: the sentence was reworded or removed, or the number inside it was edited"
            )
            findings.append(finding)
            continue
        measured = values(claim.group)
        rows = []
        for index, token in enumerate(claim.tokens):
            entry: dict[str, object] = {"token": token}
            if not _token_in(token, claim.anchor):
                entry["status"] = "DRIFT"
                entry["reason"] = "token not standalone: registry/anchor inconsistency"
                finding["status"] = "DRIFT"
                rows.append(entry)
                continue
            printed = cell_to_float(token)
            measured_value = measured[index]
            tolerance = printed_tolerance(token)
            deviation = abs(measured_value - printed) / abs(printed) if printed else abs(measured_value)
            entry.update(
                {
                    "printed": printed,
                    "measured": measured_value,
                    "relative_deviation": deviation,
                    "tolerance": tolerance,
                    "status": "PASS" if deviation <= tolerance else "DRIFT",
                }
            )
            if entry["status"] == "DRIFT":
                finding["status"] = "DRIFT"
            rows.append(entry)
        finding["cells"] = rows
        findings.append(finding)

    qualitative = []
    for name, section, anchor, predicate in QUALITATIVE_CLAIMS:
        if anchor not in text:
            qualitative.append(
                {
                    "id": name,
                    "section": section,
                    "anchor": anchor,
                    "status": "DRIFT",
                    "reason": "anchor missing",
                }
            )
            continue
        holds = bool(predicate(artifacts))
        qualitative.append(
            {
                "id": name,
                "section": section,
                "anchor": anchor,
                "status": "PASS" if holds else "DRIFT",
                "measured": holds,
            }
        )

    drifted = [item for item in findings + qualitative if item["status"] != "PASS"]
    return {
        "schema_version": SCHEMA_VERSION,
        "document": str(DOC.relative_to(REPO_ROOT)),
        "claims": len(findings),
        "numeric_tokens_bound": sum(len(claim.tokens) for claim in PROSE_CLAIMS),
        "qualitative_claims": len(qualitative),
        "drift": len(drifted),
        "verdict": "PASS" if not drifted else "DRIFT",
        "findings": findings,
        "qualitative": qualitative,
    }


def bound_tokens(text: str | None = None) -> list[str]:
    """The printed numbers this ledger binds, de-duplicated and in document order."""
    text = DOC.read_text() if text is None else text
    seen: list[str] = []
    for claim in PROSE_CLAIMS:
        if claim.anchor not in text:
            continue
        for token in claim.tokens:
            if token not in seen:
                seen.append(token)
    return seen


def coverage(text: str | None = None) -> dict[str, object]:
    """How much of the document's prose this ledger newly guards.

    The baseline is :func:`tensorbem.benchmarks.three_dimensional_doc_pins.coverage`
    (table cells are protected, everything else is not).  This function adds the
    *distinct printed numbers* bound by the claims above and reports the delta,
    so the honest sentence "the tables are guarded, most prose is not" can be
    replaced by a counted statement.
    """
    from .three_dimensional_doc_pins import coverage as table_coverage

    text = DOC.read_text() if text is None else text
    base = table_coverage(text)
    bound = bound_tokens(text)
    table_tokens = int(base["protected_in_tables"])
    total = int(base["numeric_tokens_total"])
    return {
        "numeric_tokens_total": total,
        "protected_in_tables": table_tokens,
        "bound_by_prose_claims": len(bound),
        "bound_tokens": bound,
        "guarded_total": table_tokens + len(bound),
        "guarded_fraction": (table_tokens + len(bound)) / total if total else 1.0,
        "unguarded_before": int(base["unguarded"]),
        "unguarded_after": int(base["unguarded"]) - len(bound),
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _fmt(value: float) -> str:
    return f"{value:.6g}"


def render_section(report: Mapping[str, object] | None = None) -> str:
    """Markdown for the ledger table, rendered from the measurement side."""
    report = check() if report is None else report
    lines = [
        "| # | 节 | 锚句（文档原文片段） | 印出值 | 实测值 | 相对偏差 | 容差 | 判定 |",
        "|---:|---|---|---|---:|---:|---:|---|",
    ]
    for index, finding in enumerate(report["findings"], start=1):  # type: ignore[index]
        cells = finding.get("cells", [])
        if not cells:
            lines.append(
                f"| {index} | {finding['section']} | `{finding['anchor']}` | — | — | — | — | {finding['status']} |"
            )
            continue
        for cell in cells:
            lines.append(
                "| {} | {} | `{}` | {} | {} | {:.2e} | {:.1e} | {} |".format(
                    index,
                    finding["section"],
                    finding["anchor"],
                    cell["token"],
                    _fmt(float(cell["measured"])) if "measured" in cell else "—",
                    float(cell.get("relative_deviation", float("nan"))),
                    float(cell.get("tolerance", float("nan"))),
                    cell["status"],
                )
            )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="judge every registered claim (default)")
    parser.add_argument("--coverage", action="store_true", help="report the coverage delta")
    parser.add_argument("--write-doc-section", action="store_true", help="print the markdown table")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.coverage:
        payload: object = coverage()
    elif args.write_doc_section:
        print(render_section())
        return 0
    else:
        payload = check()
    if args.json or not args.write_doc_section:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    if isinstance(payload, dict) and payload.get("verdict") == "DRIFT":
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())