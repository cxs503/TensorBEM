"""Executable contract for the numbers quoted in ``docs/lifting_foil.md``.

Why this module exists
----------------------
``docs/lifting_foil.md`` is the verification note for the analytic Joukowski
reference and the 2-D exterior-Neumann discretisation built on it.  Section 3
says "all computed by ``benchmarks/lifting_foil.py``" and prints two tables:
the analytic self-consistency of the Kutta lift (circulation route against
pressure route, four incidences) and the panel-refinement ladder of the
non-lifting field (four meshes, four columns).  Sections 2, 4 and 5 quote a
further nine numbers in prose (decay offset and its residual spread, the
thin-aerofoil limit, the Kutta-closure agreement at 128 panels, the sensitivity
to the singular point, and the negative control that rejects the layer-gradient
closure).

Until this module existed **nothing** in ``tests/`` or ``src/`` read any of
those numbers.  The tests around the lifting code assert *qualitative*
properties ("refinement reduces the error", "lift within three percent"), so
every digit in the note could drift while the build stayed green -- the failure
mode that produced the Ursell ``7.7e-3 -> 8.51e-3`` incident recorded in
``docs/ursell_quadrature_provenance.md``.

What is checked
---------------
The document text *is* the claim, so the numbers are parsed back out of the
markdown (:func:`parse_document`) instead of being transcribed here; a claim
that is not in the document cannot be silently satisfied, and a claim that is
*removed* from the document breaks the parse rather than the check.  Two
settings do come from the document, because a note that changes which
configuration it reports should have that configuration recomputed: the two
singular-point fractions and the panel count of the negative control.

:func:`measure` then recomputes every claimed quantity from the current code
and :func:`compare` issues one verdict per number, with the tolerance taken
from the printed precision (``4.80e-3`` is a claim to three significant
digits, ``61`` to the integer).  There is no intermediate artifact, so there is
no second place for the numbers to go stale::

    PYTHONPATH=src python3 -m tensorbem.benchmarks.lifting_doc_pins --check
    PYTHONPATH=src python3 -m tensorbem.benchmarks.lifting_doc_pins --json out.json

``tests/test_lifting_doc_pins.py`` runs :func:`check` in CI and carries the
negative controls (a mutated document cell and a mutated measurement must both
be reported as ``DRIFT``), so a gate that stopped checking cannot pass.

Two audit findings this gate produced
-------------------------------------
Both were fixed in the document by the commit that added this module; they are
listed here because they are what the gate is for, and both are now claims in
their own right so they cannot come back:

* the note said the rms potential error "halves with each refinement (second
  order in the panel size)".  Halving the panel size is exactly what the ladder
  does, so second order would quarter the error and the printed column does not
  quarter it.  The measured observed orders are 0.88, 0.91 and 0.92, i.e. first
  order in the panel size, and the note now says that (:func:`_parse_order`).
* the residual spread of the boundary values was printed as ``+/-0.0017`` with
  no mesh and no statistic named, so it was not reproducible as written.  At
  128 panels -- the mesh that reproduces every other number in the note -- one
  standard deviation of the same difference field is ``0.0014``, and the note
  now states the mesh and the statistic (:func:`_parse_spread`).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[3]
DOC = REPO_ROOT / "docs" / "lifting_foil.md"

PIN_SCHEMA = "tensorbem-lifting-doc-pins/1"

#: Settings shared with ``benchmarks/lifting_foil.py``.
PANEL_COUNTS = (32, 64, 128, 256)
ANGLES = (0, 2, 4, 6)
REFERENCE_ALPHA = 4
#: Mesh at which the note reports the boundary-value residual spread.
SPREAD_PANELS = 128

#: Tolerance on the order claim.  The note says "first order"; 0.15 covers
#: platform BLAS differences (measured 0.90 here) and still rejects "second
#: order", which would be 1.1 away from the measurement.
ORDER_TOL = 0.15

#: Kinds of claim.
VALUE = "value"
UPPER_BOUND = "upper_bound"
SETTING = "setting"


# ---------------------------------------------------------------------------
# printed-precision tolerances
# ---------------------------------------------------------------------------
_TOKEN = re.compile(
    r"^(?P<prefix>[<~]?)(?P<num>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)(?P<pct>%?)$"
)


def half_unit_in_last_place(number: str) -> float:
    """Half of one unit in the last printed place: ``"4.80e-3" -> 5e-6``."""
    if "e" in number or "E" in number:
        mantissa, exponent = re.split("[eE]", number)
        exponent = int(exponent)
    else:
        mantissa, exponent = number, 0
    mantissa = mantissa.lstrip("+-")
    decimals = len(mantissa.split(".")[1]) if "." in mantissa else 0
    return 0.5 * 10.0 ** (exponent - decimals)


@dataclass(frozen=True)
class Claim:
    """One number printed in the document, as written."""

    token: str     # exactly as printed, e.g. "4.67%"
    value: float   # numeric value, in the unit printed
    tol: float     # half a unit in the last printed place (0 for bounds/settings)
    kind: str = VALUE

    @property
    def unit(self) -> str:
        return "%" if self.token.endswith("%") else ""


def parse_claim(raw: str) -> Claim:
    """Turn one printed markdown cell into a :class:`Claim`."""
    token = raw.strip().replace("\\|", "|")
    match = _TOKEN.match(token)
    if match is None:
        raise ValueError(f"not a numeric claim: {raw!r}")
    number = match.group("num")
    value = float(number)
    if match.group("prefix") == "<":
        return Claim(token=token, value=value, tol=0.0, kind=UPPER_BOUND)
    return Claim(token=token, value=value, tol=half_unit_in_last_place(number))


# ---------------------------------------------------------------------------
# reading the document back
# ---------------------------------------------------------------------------
_CELL_SPLIT = re.compile(r"(?<!\\)\|")


def _section(text: str, heading: str) -> list[str]:
    """Lines of the markdown section whose heading contains ``heading``."""
    lines = text.splitlines()
    start = next(
        (index for index, line in enumerate(lines)
         if line.startswith("#") and heading in line),
        None,
    )
    if start is None:
        raise ValueError(f"document has no section containing {heading!r}")
    collected: list[str] = []
    for line in lines[start + 1 :]:
        if line.startswith("#"):
            break
        collected.append(line)
    return collected


def _cells(line: str) -> list[str]:
    return [cell.strip() for cell in _CELL_SPLIT.split(line.strip())[1:-1]]


def _rows(block: Iterable[str]) -> list[list[str]]:
    """Markdown table body rows, identified by an integer first column."""
    rows: list[list[str]] = []
    for line in block:
        stripped = line.strip()
        if not stripped.startswith("|") or set(stripped) <= set("|-: "):
            continue
        cells = _cells(stripped)
        if cells and re.fullmatch(r"-?\d+", cells[0]):
            rows.append(cells)
    return rows


def _header(block: Iterable[str]) -> list[str]:
    for line in block:
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = _cells(stripped)
        if cells and not re.fullmatch(r"-?\d+", cells[0]):
            return cells
    raise ValueError("table without a header row")


def _require(text: str, pattern: str, name: str) -> re.Match[str]:
    match = re.search(pattern, text, re.S)
    if match is None:
        raise ValueError(f"document no longer states {name!r} (pattern {pattern!r})")
    return match


ANALYTIC_COLUMNS = ("circulation", "cl_gamma", "cl_pressure", "gap_percent")
REFINEMENT_COLUMNS = (
    "relative_potential_error_percent",
    "potential_rms_error",
    "relative_speed_error_percent",
    "condition_number",
)


def parse_document(text: str) -> dict[str, Claim]:
    """Every numeric claim of ``docs/lifting_foil.md``, keyed by claim name."""
    claims: dict[str, Claim] = {}

    analytic_block = _section(text, "### 3.1")
    analytic_header = _header(analytic_block)
    for keyword in ("alpha", "Gamma", "pressure", "gap"):
        if not any(keyword in cell for cell in analytic_header):
            raise ValueError(f"unexpected 3.1 header: {analytic_header}")
    for row in _rows(analytic_block):
        alpha = int(row[0])
        if alpha not in ANGLES:
            continue
        for column, cell in zip(ANALYTIC_COLUMNS, row[1:]):
            claims[f"analytic.{alpha}.{column}"] = parse_claim(cell)

    refinement_block = _section(text, "### 3.2")
    refinement_header = _header(refinement_block)
    for keyword in ("panels", "dphi", "cond"):
        if not any(keyword in cell for cell in refinement_header):
            raise ValueError(f"unexpected 3.2 header: {refinement_header}")
    for row in _rows(refinement_block):
        panels = int(row[0])
        if panels not in PANEL_COUNTS:
            continue
        for column, cell in zip(REFINEMENT_COLUMNS, row[1:]):
            claims[f"refinement.{panels}.{column}"] = parse_claim(cell)

    claims.update(_parse_prose(text))
    claims.update(_parse_order(text))
    claims.update(_parse_spread(text))
    return claims


def _parse_prose(text: str) -> dict[str, Claim]:
    """Numbers quoted in running prose rather than in the two tables."""
    out: dict[str, Claim] = {}

    thin = _require(
        text,
        r"Thin-aerofoil limit[^:]*:\s*([\d.]+)\s+vs\s+([\d.]+)\s+at\s+4\s*deg\s*\(([\d.]+)%\)",
        "the thin-aerofoil limit",
    )
    out["prose.thin_aerofoil_cl"] = parse_claim(thin.group(1))
    out["prose.thin_aerofoil_section_cl"] = parse_claim(thin.group(2))
    out["prose.thin_aerofoil_gap_percent"] = parse_claim(thin.group(3) + "%")

    closure = _require(
        text,
        r"agree to `([\d.]+)%`.*?`([\d.]+)%`\s+below the exact value",
        "the Kutta-closure agreement",
    )
    out["prose.closure_gap_percent"] = parse_claim(closure.group(1) + "%")
    out["prose.lift_below_exact_percent"] = parse_claim(closure.group(2) + "%")

    singular = _require(
        text,
        r"moving\s+it\s+from\s+([\d.]+)\s*c\s+to\s+([\d.]+)\s*c\s+changes\s+"
        r"`Gamma_ccw`\s+by\s+less\s+than\s+([\d.]+)%",
        "the singular-point sensitivity",
    )
    out["prose.singular_from_fraction"] = Claim(
        token=singular.group(1), value=float(singular.group(1)), tol=0.0, kind=SETTING)
    out["prose.singular_to_fraction"] = Claim(
        token=singular.group(2), value=float(singular.group(2)), tol=0.0, kind=SETTING)
    out["prose.singular_variation_percent"] = Claim(
        token="<" + singular.group(3) + "%",
        value=float(singular.group(3)),
        tol=0.0,
        kind=UPPER_BOUND,
    )

    layer = _require(
        text,
        r"returns\s+a\s+circulation\s+~([\d.]+)%\s+too\s+large\s+at\s+(\d+)\s+panels,\s+"
        r"while\s+the\s+three-point\s+rule\s+gives\s+~([\d.]+)%",
        "the layer-gradient negative control",
    )
    out["prose.layer_route_error_percent"] = parse_claim(layer.group(1) + "%")
    out["prose.layer_route_panels"] = Claim(
        token=layer.group(2), value=float(layer.group(2)), tol=0.0, kind=SETTING)
    out["prose.three_point_error_percent"] = parse_claim(layer.group(3) + "%")

    offset = _require(
        text, r"=\s*([\d.]+)`\s+at\s+infinity", "the decay offset"
    )
    out["prose.decay_offset"] = parse_claim(offset.group(1))
    return out


_ORDER_WORDS = {"first": 1.0, "second": 2.0}


def _parse_order(text: str) -> dict[str, Claim]:
    """The order claimed for the rms potential error, and its three values.

    The note states the order in words and the three observed values as
    numbers.  Both are parsed here: changing "first" back to "second" has to be
    a build failure, not a silent prose edit.
    """
    out: dict[str, Claim] = {}
    word = _require(
        text,
        r"\((first|second)\s+order\s+in\s+the\s+panel\s+size",
        "the observed order of the rms potential error",
    )
    out["prose.rms_order_word"] = Claim(
        token=f"{word.group(1)} order", value=_ORDER_WORDS[word.group(1)],
        tol=ORDER_TOL)
    values = _require(
        text,
        r"observed orders over the three refinements are\s+([\d.]+),\s*([\d.]+)\s+and\s+"
        r"([\d.]+)\s*\(mean\s+([\d.]+)\)",
        "the observed orders of the rms potential error",
    )
    pairs = ("32_64", "64_128", "128_256")
    for pair, token in zip(pairs, values.groups()[:3]):
        out[f"prose.rms_order_{pair}"] = parse_claim(token)
    out["prose.rms_order_mean"] = parse_claim(values.group(4))
    return out


def _parse_spread(text: str) -> dict[str, Claim]:
    """The boundary-value offset and its residual spread, with mesh and statistic."""
    spread = _require(
        text,
        r"`(-?[\d.]+)`\s+at\s+the\s+boundary\s+with\s+a\s+residual\s+spread\s+of\s+"
        r"`\+/-([\d.]+)`\s*\(one\s+standard\s+deviation,\s*(\d+)\s+panels\)",
        "the boundary-value residual spread",
    )
    return {
        "prose.boundary_offset": parse_claim(spread.group(1)),
        "prose.boundary_spread": parse_claim(spread.group(2)),
        "prose.boundary_spread_panels": Claim(
            token=spread.group(3), value=float(spread.group(3)), tol=0.0, kind=SETTING),
    }


# ---------------------------------------------------------------------------
# recomputing the same numbers
# ---------------------------------------------------------------------------
def measure(declared: Mapping[str, Claim] | None = None) -> dict[str, Claim]:
    """Recompute every claimed quantity; keys match :func:`parse_document`."""
    from tensorbem.benchmarks import lifting_solver
    from tensorbem.benchmarks.lifting_foil import analytic_reference
    from tensorbem.lifting import JoukowskiAirfoil, compare_nonlifting_field

    out: dict[str, Claim] = {}

    def pin(name: str, value: float) -> None:
        out[name] = Claim(token=repr(float(value)), value=float(value), tol=0.0)

    foil = JoukowskiAirfoil(0.1, 1.0, float(REFERENCE_ALPHA))
    settings = _parse_settings(DOC.read_text(encoding="utf-8")) if declared is None else declared

    # --- 3.1 analytic self-consistency -----------------------------------
    for alpha in ANGLES:
        row = analytic_reference(JoukowskiAirfoil(0.1, 1.0, float(alpha)))
        from_circ = row["lift_coefficient_from_circulation"]
        from_press = row["lift_coefficient_from_pressure"]
        gap = 0.0 if abs(from_circ) < 1e-12 else abs(from_press - from_circ) / abs(from_circ)
        pin(f"analytic.{alpha}.circulation", row["circulation"])
        pin(f"analytic.{alpha}.cl_gamma", from_circ)
        pin(f"analytic.{alpha}.cl_pressure", from_press)
        pin(f"analytic.{alpha}.gap_percent", 100.0 * gap)

    # --- 3.2 refinement ladder, plus the order it implies -----------------
    rms: dict[int, float] = {}
    for panels in PANEL_COUNTS:
        comparison = compare_nonlifting_field(foil, panels)
        rms[panels] = comparison.potential_rms_error
        pin(f"refinement.{panels}.relative_potential_error_percent",
            100.0 * comparison.relative_potential_error)
        pin(f"refinement.{panels}.potential_rms_error", comparison.potential_rms_error)
        pin(f"refinement.{panels}.relative_speed_error_percent",
            100.0 * comparison.relative_speed_error)
        pin(f"refinement.{panels}.condition_number", comparison.condition_number)
        if panels == SPREAD_PANELS:
            residual = comparison.result.boundary_values - comparison.reference_potential
            # The note quotes the difference against the reference *without*
            # the decay term removed, i.e. bv - (phi_d + U m cos alpha); the
            # spread is shift-invariant, so the same field serves both claims.
            offset = comparison.result.boundary_values - (
                comparison.reference_potential + foil.decay_offset())
            pin("prose.boundary_offset", float(offset.mean()))
            pin("prose.boundary_spread", float(residual.std()))

    orders = {
        pair: math.log2(rms[coarse] / rms[fine])
        for pair, coarse, fine in zip(("32_64", "64_128", "128_256"),
                                      PANEL_COUNTS, PANEL_COUNTS[1:])
    }
    for pair, value in orders.items():
        pin(f"prose.rms_order_{pair}", value)
    mean_order = sum(orders.values()) / len(orders)
    pin("prose.rms_order_mean", mean_order)
    pin("prose.rms_order_word", mean_order)

    # --- sections 4 and 5 -------------------------------------------------
    exact_cl = foil.exact_lift_coefficient()
    thin_cl = foil.exact_thin_aerofoil_lift_coefficient()
    pin("prose.thin_aerofoil_cl", thin_cl)
    pin("prose.thin_aerofoil_section_cl", exact_cl)
    pin("prose.thin_aerofoil_gap_percent", 100.0 * (thin_cl - exact_cl) / thin_cl)

    closure = lifting_solver._row(float(REFERENCE_ALPHA), 128, 0.25)
    pin("prose.closure_gap_percent", 100.0 * closure["circulation_vs_pressure_gap"])
    pin("prose.lift_below_exact_percent", 100.0 * closure["lift_relative_error"])

    near = lifting_solver._row(
        float(REFERENCE_ALPHA), 128,
        settings["prose.singular_from_fraction"].value)["circulation_ccw"]
    far = lifting_solver._row(
        float(REFERENCE_ALPHA), 128,
        settings["prose.singular_to_fraction"].value)["circulation_ccw"]
    pin("prose.singular_variation_percent", 100.0 * abs(far - near) / abs(near))

    layer_panels = int(settings["prose.layer_route_panels"].value)
    probe = lifting_solver._layer_gradient_probe(float(REFERENCE_ALPHA), layer_panels)
    pin("prose.layer_route_error_percent", 100.0 * probe["circulation_relative_error"])
    three_point = lifting_solver._row(float(REFERENCE_ALPHA), layer_panels, 0.25)
    pin("prose.three_point_error_percent", 100.0 * three_point["circulation_relative_error"])

    pin("prose.decay_offset", foil.decay_offset())
    return out


def _parse_settings(text: str) -> dict[str, Claim]:
    """The claims that only configure the measurement (fractions, mesh)."""
    claims = parse_document(text)
    return claims


def settings_of(claims: Mapping[str, Claim]) -> dict[str, Claim]:
    return {name: claim for name, claim in claims.items() if claim.kind == SETTING}


# ---------------------------------------------------------------------------
# verdicts
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Verdict:
    name: str
    declared: str
    measured: float
    tolerance: float
    unit: str
    status: str
    deviation: float
    note: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "claim": self.name,
            "declared": self.declared,
            "measured": self.measured,
            "abs_deviation": self.deviation,
            "tolerance": self.tolerance,
            "unit": self.unit,
            "status": self.status,
        }


def compare(declared: Mapping[str, Claim], measured: Mapping[str, Claim],
            ) -> list[Verdict]:
    """Judge every declared claim against its recomputed value."""
    judgeable = {name: claim for name, claim in declared.items()
                 if claim.kind != SETTING}
    missing = sorted(set(judgeable) - set(measured))
    if missing:
        raise KeyError(f"no measurement for {missing}")
    verdicts: list[Verdict] = []
    for name in sorted(judgeable):
        claim = judgeable[name]
        actual = measured[name].value
        if claim.kind == UPPER_BOUND:
            status = "PASS" if actual <= claim.value else "DRIFT"
            deviation, tolerance = abs(actual), claim.value
            note = f"claim is an upper bound <{claim.token}"
        else:
            deviation, tolerance = abs(actual - claim.value), claim.tol
            status = "PASS" if deviation <= tolerance else "DRIFT"
            note = ""
        verdicts.append(Verdict(name=name, declared=claim.token, measured=actual,
                                tolerance=tolerance, unit=claim.unit,
                                status=status, deviation=deviation, note=note))
    return verdicts


def check(doc_text: str | None = None,
          measurements: Mapping[str, Claim] | None = None) -> list[Verdict]:
    """Parse the document (or ``doc_text``), recompute, and judge every number."""
    text = DOC.read_text(encoding="utf-8") if doc_text is None else doc_text
    declared = parse_document(text)
    actual = measure(declared) if measurements is None else measurements
    return compare(declared, actual)


def summary(verdicts: Iterable[Verdict]) -> dict[str, object]:
    rows = list(verdicts)
    drifted = [row for row in rows if row.status != "PASS"]
    return {
        "schema": PIN_SCHEMA,
        "document": str(DOC.relative_to(REPO_ROOT)) if DOC.is_file()
        and REPO_ROOT in DOC.parents else str(DOC),
        "claims": len(rows),
        "passed": len(rows) - len(drifted),
        "drifted": len(drifted),
        "verdict": "PASS" if not drifted else "DRIFT",
        "rows": [row.as_dict() for row in rows],
        "drift": [row.as_dict() for row in drifted],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Recompute every number quoted in docs/lifting_foil.md."
    )
    parser.add_argument("--check", action="store_true",
                        help="exit non-zero if any documented number moved")
    parser.add_argument("--json", type=Path, help="write the full ledger here")
    parser.add_argument("--doc", type=Path, help="check this document instead")
    args = parser.parse_args(argv)

    global DOC
    if args.doc is not None:
        DOC = args.doc
    ledger = summary(check())
    text = json.dumps(ledger, indent=2)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n")
    print(text)
    return 0 if (not args.check or ledger["drifted"] == 0) else 1


if __name__ == "__main__":  # pragma: no cover - CLI
    sys.exit(main())