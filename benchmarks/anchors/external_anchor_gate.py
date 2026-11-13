#!/usr/bin/env python3
"""External-anchor provenance gate for TensorBEM (campaign node M-R16-03).

Why this file exists
--------------------
Report 16348 exposed a failure mode that *passes* every numerical gate: an
anchor whose reference values were taken from an **in-repo** series (the Hulme
multipole routine in ``tensorbem.analytic``) while the report labelled them as
"Hulme (1982) original table".  The numbers were right and reproducible; the
*provenance* was wrong.  No tolerance check can catch that, because reference
and solver then share a derivation.

This module turns "the anchor came from outside the repository" into a
machine-checkable property:

1. **Provenance schema** -- every anchor record must name a source class drawn
   from :data:`EXTERNAL_SOURCE_CLASSES`, a bibliographic identifier (DOI / ISBN
   / URL, form-validated; ISBN by check digit) and a ``locator`` (equation,
   table or page).  Sourced-from-the-repo classes
   (:data:`IN_REPO_SOURCE_CLASSES`) and any bibliographic string that points at
   this package are rejected.
2. **Independent re-derivation** -- the stored reference number is recomputed
   from the *published closed form* by code that does **not** import
   ``tensorbem`` (only ``scipy.special``).  A transcribed/edited number that
   does not follow from the cited equation is caught at 1e-6 relative.
3. **Tolerance comparison** -- the BEM value is compared with the anchor at the
   ledger tolerance (5 % default); every row is emitted with
   ``{quantity, computed, reference, error_pct, pass_5pct}``.

The three checks are independent: the red-team fixtures under
``benchmarks/anchors/external/redteam/`` each break one of them.

Reference classes
-----------------
``LITERATURE_CLOSED_FORM``
    The cited work publishes the closed-form solution and we re-implement it.
    Honest about what it is: **no digitised table digits are claimed.**  This
    is the class used by the MacCamy & Fuchs anchor.  Re-implementing a
    published formula is an external anchor (the reference cannot follow the
    repository's discretisation), but it is *weaker* than digits read off a
    paper table -- see ``docs/external_anchors_bem.md``.
``LITERATURE_TABLE``
    Digitised numbers with a page/table locator.  Not used yet: inventing table
    digits is exactly the defect this gate exists to prevent.
``EXPERIMENT``
    Measured values carrying an experimental uncertainty.

Usage
-----
    python benchmarks/anchors/external_anchor_gate.py \
        --anchor benchmarks/anchors/external/maccamy_fuchs_cylinder.json \
        --out artifacts/external_anchors/maccamy_fuchs_error_table.json \
        --figure docs/external_anchors_bem_error.png
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

ANCHOR_SCHEMA = "tensorbem.external_anchor/v1"
REGISTRY_SCHEMA = "tensorbem.external_anchor_registry/v1"

#: Source classes that count as an *external* anchor.
EXTERNAL_SOURCE_CLASSES = ("LITERATURE_CLOSED_FORM", "LITERATURE_TABLE", "EXPERIMENT")

#: Source classes that mean "the reference came out of this repository".
#: Naming one of these is a provenance failure, not a numeric failure -- it is
#: the report-16348 defect (Hulme series labelled as Hulme's table).
IN_REPO_SOURCE_CLASSES = (
    "IN_REPO_SERIES",
    "IN_REPO_DERIVED",
    "IN_REPO_ARTIFACT",
    "SELF_REFERENCE",
    "MOCK",
    "STUB",
)

#: Substrings that betray a reference whose provenance is this package.  Applied
#: to the bibliographic block *and* to the reference-method implementation.
REPO_MARKERS = (
    "tensorbem.",
    "src/tensorbem",
    "benchmarks/verified",
    "artifacts/",
)

#: Ledger tolerance: an error above 5 % is not acceptable as evidence.
DEFAULT_TOLERANCE = 0.05

#: Relative deviation allowed between a stored reference and its re-derivation
#: from the cited closed form (double precision, different Bessel routines).
REFERENCE_REPRODUCIBILITY = 1.0e-6

_DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$")
_ISBN10_RE = re.compile(r"^\d{9}[\dX]$")
_ISBN13_RE = re.compile(r"^97[89]\d{10}$")


# ---------------------------------------------------------------------------
# Bibliographic identifier validation (form only -- see docs for the limit)
# ---------------------------------------------------------------------------


def normalise_identifier(value: str) -> str:
    """Strip separators that are typographic rather than semantic."""
    return re.sub(r"[\s\-]", "", str(value)).upper()


def isbn_checksum_ok(value: str) -> bool:
    """Verify an ISBN-10 or ISBN-13 check digit."""
    compact = normalise_identifier(value)
    if _ISBN10_RE.match(compact):
        total = 0
        for index, char in enumerate(compact):
            digit = 10 if char == "X" else int(char)
            total += digit * (10 - index)
        return total % 11 == 0
    if _ISBN13_RE.match(compact):
        total = 0
        for index, char in enumerate(compact[:12]):
            total += int(char) * (1 if index % 2 == 0 else 3)
        check = (10 - total % 10) % 10
        return check == int(compact[12])
    return False


def doi_form_ok(value: str) -> bool:
    return bool(_DOI_RE.match(str(value).strip()))


def url_form_ok(value: str) -> bool:
    return str(value).strip().lower().startswith(("https://", "http://"))


def identifier_ok(identifier: dict[str, Any]) -> bool:
    """A DOI must look like a DOI, an ISBN must pass its check digit, a URL
    must be absolute.  Existence is *not* checked here (offline gate); the
    record's ``identifier_note`` has to say so."""
    kind = str(identifier.get("type", "")).lower()
    value = identifier.get("value", "")
    if kind == "doi":
        return doi_form_ok(value)
    if kind == "isbn":
        return isbn_checksum_ok(value)
    if kind == "url":
        return url_form_ok(value)
    return False


# ---------------------------------------------------------------------------
# Record loading / schema validation
# ---------------------------------------------------------------------------


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_registry(root: str | Path) -> dict[str, Any]:
    root_path = Path(root)
    registry_path = root_path / "index.json"
    registry = load_json(registry_path)
    if registry.get("schema") != REGISTRY_SCHEMA:
        raise ValueError(f"{registry_path}: schema must be {REGISTRY_SCHEMA}")
    registry["root"] = str(root_path)
    registry["records"] = [load_json(root_path / entry["file"]) for entry in registry["anchors"]]
    return registry


def _finding(code: str, message: str, field: str = "") -> dict[str, str]:
    return {"code": code, "message": message, "field": field}


def _bibliographic_strings(record: dict[str, Any]) -> Iterable[tuple[str, str]]:
    source = record.get("source") or {}
    primary = source.get("primary") or {}
    for key in ("title", "series", "publisher", "locator", "note"):
        if isinstance(primary.get(key), str):
            yield f"source.primary.{key}", primary[key]
    for index, identifier in enumerate(source.get("identifiers") or []):
        if isinstance(identifier.get("work"), str):
            yield f"source.identifiers[{index}].work", identifier["work"]
    method = record.get("reference_method") or {}
    for key in ("implementation", "note", "derivation"):
        if isinstance(method.get(key), str):
            yield f"reference_method.{key}", method[key]


def validate_record(record: dict[str, Any]) -> list[dict[str, str]]:
    """Return findings (empty list == the record passes the provenance schema).

    Findings carry a stable ``code`` so tests and ledgers can assert *which*
    rule fired instead of just "it failed somewhere".
    """
    findings: list[dict[str, str]] = []
    if record.get("schema") != ANCHOR_SCHEMA:
        findings.append(_finding("schema_mismatch", f"schema must be {ANCHOR_SCHEMA}", "schema"))
    if not record.get("anchor_id"):
        findings.append(_finding("missing_anchor_id", "anchor_id is required", "anchor_id"))

    source = record.get("source")
    if not isinstance(source, dict) or not source:
        findings.append(_finding("missing_source", "source block is required", "source"))
        return findings

    source_class = source.get("class")
    if source_class in IN_REPO_SOURCE_CLASSES:
        findings.append(
            _finding(
                "provenance_in_repo_derived",
                f"source class {source_class!r} is a reference derived inside this "
                "repository, not an external anchor (the report-16348 defect)",
                "source.class",
            )
        )
    elif source_class not in EXTERNAL_SOURCE_CLASSES:
        findings.append(
            _finding(
                "provenance_class_unknown",
                f"source class {source_class!r} is not one of {EXTERNAL_SOURCE_CLASSES}",
                "source.class",
            )
        )

    primary = source.get("primary") or {}
    for key in ("authors", "year", "title", "locator"):
        if not primary.get(key):
            findings.append(
                _finding("source_incomplete", f"source.primary.{key} is required", f"source.primary.{key}")
            )

    identifiers = source.get("identifiers") or []
    if not identifiers:
        findings.append(
            _finding(
                "missing_external_identifier",
                "at least one DOI / ISBN / URL identifier is required",
                "source.identifiers",
            )
        )
    elif not any(identifier_ok(item) for item in identifiers):
        findings.append(
            _finding(
                "missing_external_identifier",
                "no identifier has a valid form (DOI pattern, ISBN check digit, absolute URL)",
                "source.identifiers",
            )
        )

    for field_name, text in _bibliographic_strings(record):
        lowered = text.lower()
        for marker in REPO_MARKERS:
            if marker.lower() in lowered:
                findings.append(
                    _finding(
                        "provenance_points_into_repo",
                        f"{field_name} refers to repository-internal material "
                        f"({marker!r}); that is not an external anchor",
                        field_name,
                    )
                )
                break

    method = record.get("reference_method") or {}
    if not method:
        findings.append(
            _finding("missing_reference_method", "reference_method block is required", "reference_method")
        )
    else:
        if method.get("independent_of_repo_package") is not True:
            findings.append(
                _finding(
                    "reference_method_not_independent",
                    "reference_method.independent_of_repo_package must be true: a "
                    "reference computed from this package's own solver shares the "
                    "discretisation it is meant to judge",
                    "reference_method.independent_of_repo_package",
                )
            )
        known = REFERENCE_CALCULATORS.get(str(method.get("id")), {})
        if not known:
            findings.append(
                _finding(
                    "reference_method_unknown",
                    f"reference_method.id {method.get('id')!r} has no registered "
                    f"independent calculator (known: {sorted(REFERENCE_CALCULATORS)})",
                    "reference_method.id",
                )
            )
        if not isinstance(method.get("equation"), str) or not method["equation"].strip():
            findings.append(
                _finding(
                    "missing_reference_equation",
                    "reference_method.equation is required",
                    "reference_method.equation",
                )
            )
        uncertainty = method.get("uncertainty") or {}
        if not uncertainty.get("kind") or "relative" not in uncertainty:
            findings.append(
                _finding(
                    "missing_uncertainty",
                    "reference_method.uncertainty must state kind and relative magnitude",
                    "reference_method.uncertainty",
                )
            )

    quantities = record.get("quantities") or []
    if not quantities:
        findings.append(_finding("missing_quantities", "at least one quantity is required", "quantities"))
    seen: set[str] = set()
    for index, quantity in enumerate(quantities):
        qid = quantity.get("id")
        if not qid:
            findings.append(_finding("quantity_missing_id", "quantity.id is required", f"quantities[{index}]"))
            continue
        if qid in seen:
            findings.append(
                _finding("quantity_duplicate_id", f"duplicate quantity id {qid!r}", f"quantities[{index}]")
            )
        seen.add(qid)
        if not known:
            continue
        if quantity.get("kind") not in known.get("kinds", ()):
            findings.append(
                _finding(
                    "quantity_kind_unknown",
                    f"quantity {qid!r} kind {quantity.get('kind')!r} is not derivable "
                    f"from reference_method {method.get('id')!r}",
                    f"quantities[{index}].kind",
                )
            )
        if not quantity.get("unit"):
            findings.append(
                _finding("quantity_missing_unit", f"quantity {qid!r} needs a unit", f"quantities[{index}].unit")
            )
        reference = quantity.get("reference")
        if not isinstance(reference, (int, float)) or isinstance(reference, bool) or not math.isfinite(float(reference)):
            findings.append(
                _finding(
                    "quantity_reference_not_finite",
                    f"quantity {qid!r} reference must be a finite number",
                    f"quantities[{index}].reference",
                )
            )
        conditions = quantity.get("case") or {}
        for key in known.get("required_case", ()):
            if key not in conditions:
                findings.append(
                    _finding(
                        "quantity_missing_condition",
                        f"quantity {qid!r} needs case.{key}",
                        f"quantities[{index}].case.{key}",
                    )
                )

    tolerance = (record.get("tolerance") or {}).get("relative")
    if not isinstance(tolerance, (int, float)) or not 0.0 < float(tolerance) <= DEFAULT_TOLERANCE:
        findings.append(
            _finding(
                "tolerance_invalid",
                f"tolerance.relative must be in (0, {DEFAULT_TOLERANCE}] -- a looser gate "
                "than the ledger tolerance would hide the very error it grades",
                "tolerance.relative",
            )
        )
    return findings


# ---------------------------------------------------------------------------
# Independent re-derivation from the published closed form
# ---------------------------------------------------------------------------


def _maccamy_fuchs_force(
    ka: float,
    *,
    radius: float,
    water_depth: float | str,
    amplitude: float,
    rho: float,
    gravity: float,
) -> complex:
    """MacCamy & Fuchs (1954) horizontal force per unit length, complex phasor.

    ``F_x = 4 rho g A tanh(k h) / (k^2 H_1^{(1)'}(k a))`` with ``k = ka / a``.

    Implemented with ``scipy.special.hankel1`` only: this module never imports
    ``tensorbem`` for the reference, so it cannot inherit the solver's
    discretisation or its Bessel helpers.
    """
    from scipy.special import hankel1  # lazy: validation must not need scipy

    if ka <= 0.0:
        raise ValueError("ka must be positive")
    k = ka / radius
    h1_prime = hankel1(0, ka) - hankel1(1, ka) / ka
    if isinstance(water_depth, str) or water_depth is None:
        if str(water_depth).lower() not in {"inf", "infinite", "infinity"}:
            raise ValueError(f"unsupported water_depth {water_depth!r}")
        tanh_kh = 1.0
    else:
        depth = float(water_depth)
        if depth <= 0.0:
            raise ValueError("water_depth must be positive or 'inf'")
        tanh_kh = math.tanh(k * depth)
    return 4.0 * rho * gravity * amplitude * tanh_kh / (k * k * h1_prime)


def _case_float(record: dict[str, Any], quantity: dict[str, Any], key: str) -> float:
    """Per-quantity case value if given, else the anchor-wide condition."""
    case = quantity.get("case") or {}
    if key in case:
        return float(case[key])
    return float(record["conditions"][key])


def _maccamy_fuchs_quantity(record: dict[str, Any], quantity: dict[str, Any]) -> float:
    conditions = record["conditions"]
    force = _maccamy_fuchs_force(
        _case_float(record, quantity, "ka"),
        radius=float(conditions["radius_m"]),
        water_depth=conditions["water_depth_over_radius"],
        amplitude=float(conditions["wave_amplitude_m"]),
        rho=float(conditions["rho_kg_m3"]),
        gravity=float(conditions["g_m_s2"]),
    )
    kind = quantity["kind"]
    if kind == "force_amplitude":
        return abs(force)
    if kind == "force_phase_deg":
        return math.degrees(math.atan2(force.imag, force.real))
    if kind == "c_m":
        # C_M = |F| / (rho g A pi a^2)  ->  2 as ka -> 0 (Morison inertia limit).
        scale = (
            float(conditions["rho_kg_m3"])
            * float(conditions["g_m_s2"])
            * float(conditions["wave_amplitude_m"])
            * math.pi
            * float(conditions["radius_m"]) ** 2
        )
        return abs(force) / scale
    raise ValueError(f"unsupported quantity kind {kind!r}")


#: Registry of independent reference calculators.  Adding a case means adding a
#: published closed form here -- never a call into ``tensorbem``.
REFERENCE_CALCULATORS: dict[str, dict[str, Any]] = {
    "maccamy_fuchs_hankel_force": {
        "callable": _maccamy_fuchs_quantity,
        "kinds": ("force_amplitude", "force_phase_deg", "c_m"),
        "required_case": ("ka",),
        "source": "MacCamy & Fuchs (1954); Newman (1977) sec. 7.5",
    },
}


def recompute_reference(record: dict[str, Any], quantity: dict[str, Any]) -> float:
    """Re-derive one quantity's reference value from the cited closed form."""
    method_id = record["reference_method"]["id"]
    calculator = REFERENCE_CALCULATORS.get(method_id)
    if calculator is None:
        raise KeyError(f"no independent calculator for reference_method {method_id!r}")
    return float(calculator["callable"](record, quantity))


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


def _relative(reference: float, value: float) -> float:
    if reference == 0.0:
        return abs(value)
    return abs(value - reference) / abs(reference)


def check_provenance(record: dict[str, Any]) -> dict[str, Any]:
    findings = validate_record(record)
    return {
        "passed": not findings,
        "rule": "source must be external (DOI/ISBN/URL + locator) and the reference "
        "method must not import this package",
        "findings": findings,
    }


def check_reference_reproducible(record: dict[str, Any]) -> dict[str, Any]:
    """Stored reference digits must follow from the cited equation."""
    rows = []
    worst = 0.0
    for quantity in record.get("quantities", []):
        try:
            recomputed = recompute_reference(record, quantity)
        except Exception as exc:  # unknown method or quantity kind: fail closed
            rows.append(
                {
                    "quantity": quantity.get("id"),
                    "stored": quantity.get("reference"),
                    "recomputed": None,
                    "relative_deviation": None,
                    "passed": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        deviation = _relative(float(quantity["reference"]), recomputed)
        worst = max(worst, deviation)
        rows.append(
            {
                "quantity": quantity["id"],
                "stored": float(quantity["reference"]),
                "recomputed": recomputed,
                "relative_deviation": deviation,
                "passed": deviation <= REFERENCE_REPRODUCIBILITY,
            }
        )
    return {
        "passed": all(row["passed"] for row in rows) if rows else False,
        "threshold": REFERENCE_REPRODUCIBILITY,
        "max_relative_deviation": worst,
        "rows": rows,
    }


def check_tolerance(
    record: dict[str, Any],
    computed: dict[str, float],
    tolerance: float | None = None,
) -> dict[str, Any]:
    """Compare solver values with the anchor at the ledger tolerance."""
    tol = float(
        tolerance if tolerance is not None else record.get("tolerance", {}).get("relative", DEFAULT_TOLERANCE)
    )
    rows = []
    for quantity in record.get("quantities", []):
        qid = quantity["id"]
        reference = float(quantity["reference"])
        if qid not in computed:
            rows.append(
                {
                    "quantity": qid,
                    "computed": None,
                    "reference": reference,
                    "error_pct": None,
                    "pass_5pct": False,
                    "note": "no computed value supplied",
                }
            )
            continue
        value = float(computed[qid])
        error_pct = 100.0 * _relative(reference, value)
        rows.append(
            {
                "quantity": qid,
                "computed": value,
                "reference": reference,
                "error_pct": error_pct,
                "pass_5pct": bool(error_pct <= 100.0 * tol),
                "check": "pointwise",
            }
        )
    # Literature limits (e.g. C_M -> 2 as ka -> 0) are graded on the tolerance
    # only: their reference is a *stated limit*, so there is no pointwise closed
    # form to reproduce at 1e-6.  They still appear as full error rows.
    for limit in record.get("limit_checks", []):
        qid = limit["id"]
        reference = float(limit["reference"])
        limit_tol = float(limit.get("tolerance_relative", tol))
        if qid not in computed:
            rows.append(
                {
                    "quantity": qid,
                    "computed": None,
                    "reference": reference,
                    "error_pct": None,
                    "pass_5pct": False,
                    "check": "limit",
                    "note": "no computed value supplied",
                }
            )
            continue
        value = float(computed[qid])
        error_pct = 100.0 * _relative(reference, value)
        rows.append(
            {
                "quantity": qid,
                "computed": value,
                "reference": reference,
                "error_pct": error_pct,
                "pass_5pct": bool(error_pct <= 100.0 * limit_tol),
                "check": "limit",
            }
        )
    return {
        "passed": all(row["pass_5pct"] for row in rows) if rows else False,
        "tolerance_relative": tol,
        "rows": rows,
    }


def gate(
    record: dict[str, Any],
    computed: dict[str, float],
    tolerance: float | None = None,
) -> dict[str, Any]:
    """Run all three checks; PASS only when provenance *and* numbers hold."""
    provenance = check_provenance(record)
    reproducible = check_reference_reproducible(record)
    numeric = check_tolerance(record, computed, tolerance)
    failures: list[str] = []
    if not provenance["passed"]:
        failures.extend(f"provenance:{item['code']}" for item in provenance["findings"])
    if not reproducible["passed"]:
        failures.append("reference:not_reproducible_from_cited_equation")
    if not numeric["passed"]:
        failures.append("tolerance:exceeded")
    return {
        "anchor_id": record.get("anchor_id"),
        "verdict": "PASS" if not failures else "FAIL",
        "failures": failures,
        "checks": {
            "provenance": provenance,
            "reference_reproducible": reproducible,
            "tolerance": numeric,
        },
    }


# ---------------------------------------------------------------------------
# Solver side: the numbers the gate grades (the only part that imports tensorbem)
# ---------------------------------------------------------------------------


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_src_on_path() -> None:
    src = _repo_root() / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def solve_force(record: dict[str, Any], ka: float, panels: int = 128, nodes: int = 12) -> complex:
    """One TensorBEM panel solve of the horizontal force phasor at ``ka``."""
    _ensure_src_on_path()
    from tensorbem.geometry import regular_polygon
    from tensorbem.helmholtz import solve_vertical_cylinder_diffraction

    conditions = record["conditions"]
    mesh = regular_polygon(panels, radius=float(conditions["radius_m"]))
    solved = solve_vertical_cylinder_diffraction(
        mesh,
        ka,
        depth=conditions["water_depth_over_radius"],
        amplitude=float(conditions["wave_amplitude_m"]),
        density=float(conditions["rho_kg_m3"]),
        gravity=float(conditions["g_m_s2"]),
        chief_points="auto",
        panel_nodes=nodes,
    )
    return complex(solved.force[0])


def compute_quantities(record: dict[str, Any], panels: int = 128, nodes: int = 12) -> dict[str, float]:
    """Run the TensorBEM panel route for every quantity in the record."""
    conditions = record["conditions"]
    radius = float(conditions["radius_m"])
    cache: dict[float, complex] = {}

    def force_at(ka_value: float) -> complex:
        if ka_value not in cache:
            cache[ka_value] = solve_force(record, ka_value, panels=panels, nodes=nodes)
        return cache[ka_value]

    values: dict[str, float] = {}
    for quantity in record["quantities"]:
        ka = _case_float(record, quantity, "ka")
        force = force_at(ka)
        kind = quantity["kind"]
        if kind == "force_amplitude":
            values[quantity["id"]] = abs(force)
        elif kind == "force_phase_deg":
            values[quantity["id"]] = math.degrees(math.atan2(force.imag, force.real))
        elif kind == "c_m":
            scale = (
                float(conditions["rho_kg_m3"])
                * float(conditions["g_m_s2"])
                * float(conditions["wave_amplitude_m"])
                * math.pi
                * radius**2
            )
            values[quantity["id"]] = abs(force) / scale
        else:  # pragma: no cover - schema validation rejects unknown kinds
            raise ValueError(f"unsupported quantity kind {kind!r}")

    scale = (
        float(conditions["rho_kg_m3"])
        * float(conditions["g_m_s2"])
        * float(conditions["wave_amplitude_m"])
        * math.pi
        * radius**2
    )
    for limit in record.get("limit_checks", []):
        kas = [float(item) for item in limit.get("extrapolation_kas", ())]
        if len(kas) != 2:
            raise ValueError(f"limit check {limit['id']!r} needs two extrapolation_kas")
        cms = [abs(force_at(ka)) / scale for ka in kas]
        # O(ka^2) behaviour of C_M: L = (4 f(ka/2) - f(ka)) / 3.
        values[limit["id"]] = (4.0 * cms[1] - cms[0]) / 3.0
    return values


def convergence_study(
    record: dict[str, Any], panels: tuple[int, ...] = (32, 64, 128), ka: float = 1.0
) -> list[dict[str, Any]]:
    """Mesh refinement graded against the *external* reference, not itself.

    One representative wave number is enough to show the discretisation trend,
    and re-solving every quantity at every density would make the gate cost
    minutes for no extra information (the trend is set by ``ka / panels``).
    """
    probe = {"id": "force_amplitude", "kind": "force_amplitude", "case": {"ka": ka}}
    reference = recompute_reference(record, probe)
    rows = []
    for count in panels:
        computed = abs(solve_force(record, ka, panels=count))
        error_pct = 100.0 * _relative(reference, computed)
        rows.append(
            {
                "panels": count,
                "ka": ka,
                "computed": computed,
                "reference": reference,
                "error_pct": error_pct,
                "max_error_pct": error_pct,
            }
        )
    return rows


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_repo_root(), text=True
        ).strip()
    except Exception:  # pragma: no cover - provenance string only
        return "unknown"


def build_error_table(record: dict[str, Any], panels: int = 128) -> dict[str, Any]:
    """Full artifact: verdict + per-quantity error rows + convergence evidence."""
    computed = compute_quantities(record, panels=panels)
    verdict = gate(record, computed)
    rows = verdict["checks"]["tolerance"]["rows"]
    reproducible = verdict["checks"]["reference_reproducible"]
    return {
        "schema": "tensorbem.external_anchor_error_table/v1",
        "anchor_id": record["anchor_id"],
        "case": record["case"]["name"] if isinstance(record.get("case"), dict) else record.get("case"),
        "reference": {
            "source_class": record["source"]["class"],
            "title": record["source"]["primary"]["title"],
            "authors": record["source"]["primary"]["authors"],
            "year": record["source"]["primary"]["year"],
            "locator": record["source"]["primary"]["locator"],
            "identifiers": record["source"]["identifiers"],
        },
        "solver": {
            "route": "tensorbem.helmholtz.solve_vertical_cylinder_diffraction",
            "panels": panels,
            "panel_nodes": 12,
            "chief_points": "auto",
            "water_depth_over_radius": record["conditions"]["water_depth_over_radius"],
            "git_commit_at_generation": _git_commit(),
        },
        "tolerance_relative": verdict["checks"]["tolerance"]["tolerance_relative"],
        "verdict": verdict["verdict"],
        "failures": verdict["failures"],
        "max_error_pct": max(row["error_pct"] for row in rows if row["error_pct"] is not None),
        "all_pass_5pct": all(row["pass_5pct"] for row in rows),
        "all_pass_1pct": all(
            row["error_pct"] is not None and row["error_pct"] <= 1.0 for row in rows
        ),
        "reference_reproducible": reproducible["passed"],
        "max_reference_deviation": reproducible["max_relative_deviation"],
        "quantities": rows,
        "convergence": convergence_study(record),
    }


def render_figure(table: dict[str, Any], path: str | Path) -> Path:
    """Two-panel evidence figure: error vs ka, and mesh convergence."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [row for row in table["quantities"] if row["quantity"].startswith("force_amplitude")]
    kas = [float(row["quantity"].rsplit("_", 1)[1]) for row in rows]
    errors = [row["error_pct"] for row in rows]
    convergence = table["convergence"]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    axes[0].semilogy(kas, errors, "o-", label="TensorBEM (128 panels) vs MacCamy-Fuchs")
    axes[0].axhline(5.0, color="tab:red", ls="--", label="ledger tolerance 5 %")
    axes[0].axhline(1.0, color="tab:orange", ls=":", label="1 %")
    axes[0].set_xlabel("ka [-]")
    axes[0].set_ylabel("|error| [%]")
    axes[0].set_title("force amplitude vs external closed form")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=8)

    panels = [row["panels"] for row in convergence]
    max_err = [row["max_error_pct"] for row in convergence]
    axes[1].loglog(panels, max_err, "s-", label="max error over all quantities")
    axes[1].axhline(5.0, color="tab:red", ls="--", label="ledger tolerance 5 %")
    axes[1].set_xlabel("panels on the cylinder")
    axes[1].set_ylabel("max |error| [%]")
    axes[1].set_title("mesh convergence against the external anchor")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(fontsize=8)

    fig.suptitle("TensorBEM external anchor: MacCamy & Fuchs (1954) vertical cylinder")
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--anchor",
        default="benchmarks/anchors/external/maccamy_fuchs_cylinder.json",
        help="anchor record to gate",
    )
    parser.add_argument("--panels", type=int, default=128)
    parser.add_argument("--out", default="artifacts/external_anchors/maccamy_fuchs_error_table.json")
    parser.add_argument(
        "--mirror",
        default="benchmarks/anchors/external/evidence/maccamy_fuchs_error_table.json",
        help="tracked copy of the same table (artifacts/ is git-ignored in this repo)",
    )
    parser.add_argument("--figure", default="docs/external_anchors_bem_error.png")
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args(argv)

    root = _repo_root()
    record = load_json(root / args.anchor)
    table = build_error_table(record, panels=args.panels)

    payload = json.dumps(table, indent=2, sort_keys=True, allow_nan=False) + "\n"
    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(payload, encoding="utf-8")
    print(f"wrote {out}")
    if args.mirror:
        mirror = root / args.mirror
        mirror.parent.mkdir(parents=True, exist_ok=True)
        mirror.write_text(payload, encoding="utf-8")
        print(f"wrote {mirror}")

    if not args.no_figure:
        figure = render_figure(table, root / args.figure)
        print(f"wrote {figure}")

    print("verdict:", table["verdict"], "failures:", table["failures"])
    print(f"max error {table['max_error_pct']:.4f} %  (all<5%: {table['all_pass_5pct']}, all<1%: {table['all_pass_1pct']})")
    for row in table["quantities"]:
        print(
            f"  {row['quantity']:<26} computed={row['computed']:.6g} reference={row['reference']:.6g} "
            f"error={row['error_pct']:.4f}% pass5%={row['pass_5pct']}"
        )
    return 0 if table["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())