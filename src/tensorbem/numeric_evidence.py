"""Numeric evidence layer: the number behind a capability claim, re-derived.

``tensorbem.capability_evidence`` grades a capability from *structure* -- the
module exists, a test node references it, no ``pass`` body or pseudorandom RNG marker is
found -- and publishes one line of prose (the claim) in
``docs/audit_baseline.md``.  That is necessary but not sufficient: a capability
whose acceptance criterion is a tolerance can be structurally ``REAL`` while the
number it produces is outside its own band.  The KP505 propeller is the live
example in this repository: the structural row says ``REAL``, while the
committed benchmark artifact reports ``|dK_T| = 29.1%`` against a 3% criterion.

This module adds the layer the platform's capability ledger asks for -- *判据 +
工件 + 独立复核* per item:

* 判据  the acceptance criterion, its tolerance and its literature/analytic
  source, held in a registry (``docs/numeric_evidence_registry.json``);
* 工件  the committed artifact that carries the raw computed and reference
  values, with a pinned ``sha256`` so an artifact cannot move underneath the
  record without the record going red;
* 独立复核 the error is *recomputed from the raw pair* by this module, never
  copied from an artifact's own ``relative_error`` field.  When an artifact also
  reports an error, the two channels are cross-checked and any disagreement
  fails the item closed.

Grading is fail-closed: an item is ``REAL`` only if the artifact is present, its
bytes match the pinned digest, both numbers resolve, and the recomputed error is
inside the declared tolerance.  Everything else is ``WITHHELD``, ``MISSING`` or
``STALE``, and the reason string carries the numbers that produced the verdict.

Run it with::

    python -m tensorbem.numeric_evidence --check   # gate (CI / tests)
    python -m tensorbem.numeric_evidence --write   # re-pin digests, refresh doc
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from dataclasses import dataclass, field
from typing import Any, Sequence

SCHEMA = "tensorbem-numeric-evidence/1"
REGISTRY = "docs/numeric_evidence_registry.json"
DOCUMENT = "docs/numeric_evidence_matrix.md"

REAL = "REAL"
WITHHELD = "WITHHELD"
MISSING = "MISSING"
STALE = "STALE"

_LEVEL_ORDER = (REAL, WITHHELD, MISSING, STALE)


# ---------------------------------------------------------------------------
# paths, digests, selectors
# ---------------------------------------------------------------------------


def repo_root(start: pathlib.Path | None = None) -> pathlib.Path:
    here = pathlib.Path(start or __file__).resolve()
    for parent in (here, *here.parents):
        if (parent / "pyproject.toml").is_file() and (parent / "src" / "tensorbem").is_dir():
            return parent
    raise RuntimeError(f"cannot locate the tensorbem repository root above {here}")


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pointer(document: Any, pointer: str) -> Any:
    """Resolve an RFC 6901 JSON pointer (``/a/0/b``) inside ``document``."""

    if pointer in ("", "/"):
        return document
    if not pointer.startswith("/"):
        raise KeyError(f"selector {pointer!r} is not a JSON pointer")
    node = document
    for token in pointer.lstrip("/").split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        if isinstance(node, list):
            node = node[int(token)]
        elif isinstance(node, dict):
            node = node[token]
        else:  # pragma: no cover - defensive
            raise KeyError(f"cannot walk {pointer!r} through {type(node).__name__}")
    return node


def resolve_value(document: Any, selector: Any) -> Any:
    """Resolve a registry selector against an artifact document.

    Two forms are accepted:

    * ``"/criteria/0/KT_bem"`` -- a plain JSON pointer;
    * ``{"array": "/points", "match": {"J": 0.9}, "field": "eta0_bem"}`` -- pick
      the single array element whose ``match`` keys equal the given values, then
      take ``field`` from it.  Name-based selection keeps the registry readable
      and survives an array reorder that a bare index would silently mistrack.
    """

    if isinstance(selector, str):
        return _pointer(document, selector)
    if isinstance(selector, dict):
        array = _pointer(document, selector["array"])
        match = selector.get("match", {})
        hits = [
            row
            for row in array
            if all(row.get(key) == value for key, value in match.items())
        ]
        if len(hits) != 1:
            raise KeyError(
                f"selector {selector!r} matched {len(hits)} rows, expected exactly one"
            )
        value: Any = hits[0]
        for token in selector.get("field", "").lstrip("/").split("/"):
            if not token:
                continue
            value = value[token] if isinstance(value, dict) else value[int(token)]
        return value
    raise TypeError(f"unsupported selector type {type(selector).__name__}")


# ---------------------------------------------------------------------------
# grading
# ---------------------------------------------------------------------------


@dataclass
class ItemEvidence:
    item_id: str
    capability: str
    criterion: str
    tolerance_pct: float
    artifact: str
    level: str
    reasons: list[str] = field(default_factory=list)
    computed: float | None = None
    reference: float | None = None
    error_pct: float | None = None
    reported_error_pct: float | None = None
    sha256: str | None = None
    pinned_sha256: str | None = None
    expected_level: str | None = None
    gap_note: str | None = None

    @property
    def passed(self) -> bool:
        return self.level == REAL

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.item_id,
            "capability": self.capability,
            "criterion": self.criterion,
            "tolerance_pct": self.tolerance_pct,
            "artifact": self.artifact,
            "level": self.level,
            "passed": self.passed,
            "computed": self.computed,
            "reference": self.reference,
            "recomputed_error_pct": self.error_pct,
            "reported_error_pct": self.reported_error_pct,
            "sha256": self.sha256,
            "pinned_sha256": self.pinned_sha256,
            "expected_level": self.expected_level,
            "gap_note": self.gap_note,
            "reasons": list(self.reasons),
        }


def _relative_error_pct(computed: float, reference: float) -> float:
    computed = float(computed)
    reference = float(reference)
    if reference == 0.0:
        if computed == 0.0:
            return 0.0
        raise ZeroDivisionError("reference value is zero; declare an absolute scale")
    return abs(computed - reference) / abs(reference) * 100.0


def grade_item(
    item: dict[str, Any],
    root: pathlib.Path | None = None,
    *,
    capabilities: Sequence[str] | None = None,
    pin_check: bool = True,
) -> ItemEvidence:
    """Grade one registry item from the raw numbers in its artifact."""

    root = repo_root(root)
    row = ItemEvidence(
        item_id=str(item.get("id", "<unnamed>")),
        capability=str(item.get("capability", "")),
        criterion=str(item.get("criterion", "")),
        tolerance_pct=float(item.get("tolerance_pct", 0.0)),
        artifact=str(item.get("artifact", "")),
        level=WITHHELD,
        expected_level=item.get("expected_level"),
        gap_note=item.get("gap_note"),
    )

    if capabilities is not None and row.capability not in capabilities:
        row.reasons.append(
            f"capability {row.capability!r} is not published by tensorbem.capability_evidence"
        )
        return row

    artifact_path = root / row.artifact
    if not row.artifact or not artifact_path.is_file():
        row.level = MISSING
        row.reasons.append(f"artifact {row.artifact!r} is not in the tree")
        return row

    row.sha256 = sha256_file(artifact_path)
    row.pinned_sha256 = item.get("sha256")
    if pin_check and row.pinned_sha256 and row.sha256 != row.pinned_sha256:
        row.level = STALE
        row.reasons.append(
            "artifact digest differs from the pinned evidence record "
            f"({row.sha256[:12]} != {row.pinned_sha256[:12]}); re-run --write after an "
            "intentional change, or restore the artifact"
        )
        return row

    try:
        document = json.loads(artifact_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:  # pragma: no cover - defensive
        row.reasons.append(f"artifact is not valid JSON: {error}")
        return row

    try:
        computed = float(resolve_value(document, item["computed"]))
        if "reference" in item:
            reference = float(resolve_value(document, item["reference"]))
        else:
            reference = float(item["reference_value"])
    except (KeyError, IndexError, TypeError, ValueError) as error:
        row.reasons.append(f"selector did not resolve: {error!r}")
        return row

    row.computed = computed
    row.reference = reference
    try:
        row.error_pct = _relative_error_pct(computed, reference)
    except ZeroDivisionError as error:
        row.reasons.append(str(error))
        return row

    reported_key = None
    scale = 1.0
    if "reported_error_pct" in item:
        reported_key, scale = "reported_error_pct", 1.0
    elif "reported_error_ratio" in item:
        # dimensionless artifacts (relative_error = 0.00147) are common; the
        # channel is explicit so a unit mismatch can never masquerade as agreement
        reported_key, scale = "reported_error_ratio", 100.0
    if reported_key is not None:
        try:
            raw_reported = float(resolve_value(document, item[reported_key]))
        except (KeyError, IndexError, TypeError, ValueError) as error:
            row.reasons.append(f"reported-error selector did not resolve: {error!r}")
            return row
        row.reported_error_pct = raw_reported * scale
        slack = max(1e-9, 1e-6 * abs(row.reported_error_pct))
        if abs(row.reported_error_pct - row.error_pct) > slack:
            row.reasons.append(
                f"artifact reports {row.reported_error_pct:.6g}% but the raw pair gives "
                f"{row.error_pct:.6g}%; the artifact's own error field disagrees with "
                "its own numbers and is not trusted"
            )
            return row

    if row.error_pct <= row.tolerance_pct:
        row.level = REAL
    else:
        row.reasons.append(
            f"recomputed error {row.error_pct:.6g}% exceeds the declared tolerance "
            f"{row.tolerance_pct:.6g}%"
        )
    return row


def load_registry(path: pathlib.Path | None = None) -> dict[str, Any]:
    root = repo_root()
    registry_path = pathlib.Path(path) if path else root / REGISTRY
    return json.loads(registry_path.read_text(encoding="utf-8"))


def audit(
    root: pathlib.Path | None = None,
    *,
    registry_path: pathlib.Path | None = None,
    capabilities: Sequence[str] | None = None,
    pin_check: bool = True,
) -> list[ItemEvidence]:
    """Regrade every registry item from its artifact."""

    root = repo_root(root)
    registry = load_registry(registry_path if registry_path else root / REGISTRY)
    return [
        grade_item(item, root, capabilities=capabilities, pin_check=pin_check)
        for item in registry.get("items", [])
    ]


def structural_capabilities() -> list[str]:
    """Capability keys published by the structural layer (lazy import)."""

    from tensorbem import capability_evidence as ce

    return [row.capability.key for row in ce.audit()]


def contradictions(
    rows: Sequence[ItemEvidence],
    structural: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Capabilities graded REAL structurally but not REAL numerically.

    This is the whole point of the layer: a structural ``REAL`` that no number
    backs is exactly the overstated claim the ledger is meant to catch.
    """

    if structural is None:
        from tensorbem import capability_evidence as ce

        structural = {row.capability.key: row.level for row in ce.audit()}
    out: list[dict[str, Any]] = []
    for row in rows:
        if row.level == REAL:
            continue
        declared = structural.get(row.capability)
        if declared is not None and declared != row.level:
            out.append(
                {
                    "capability": row.capability,
                    "item": row.item_id,
                    "structural_level": declared,
                    "numeric_level": row.level,
                    "reasons": list(row.reasons),
                }
            )
    return out


def summary(rows: Sequence[ItemEvidence]) -> dict[str, int]:
    counts = {level: 0 for level in _LEVEL_ORDER}
    for row in rows:
        counts[row.level] = counts.get(row.level, 0) + 1
    counts["n_items"] = len(rows)
    counts["n_capabilities"] = len({row.capability for row in rows})
    return counts


# ---------------------------------------------------------------------------
# published document
# ---------------------------------------------------------------------------


def render_markdown(
    rows: Sequence[ItemEvidence],
    *,
    structural: dict[str, str] | None = None,
    registry: dict[str, Any] | None = None,
) -> str:
    counts = summary(rows)
    lines = [
        "# Numeric evidence matrix (generated by `tensorbem.numeric_evidence`)",
        "",
        "Every row is one acceptance criterion: the tolerance comes from the literature",
        "or analytic reference named in the registry, the error is **recomputed here from",
        "the raw computed/reference pair** in the committed artifact, and the artifact",
        "digest is pinned so the record cannot drift silently.",
        "",
        f"Items: {counts['n_items']} across {counts['n_capabilities']} capabilities "
        f"(REAL {counts[REAL]}, WITHHELD {counts[WITHHELD]}, "
        f"MISSING {counts[MISSING]}, STALE {counts[STALE]}).",
        "",
        "| item | capability | tolerance | recomputed | artifact | level |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        error = "n/a" if row.error_pct is None else f"{row.error_pct:.3g}%"
        lines.append(
            f"| `{row.item_id}` | `{row.capability}` | {row.tolerance_pct:.3g}% | {error} "
            f"| `{row.artifact}` | **{row.level}** |"
        )
    withheld = [row for row in rows if row.level != REAL]
    lines += ["", "## Withheld items and why", ""]
    if not withheld:
        lines.append("None: every declared criterion is met by the committed numbers.")
    for row in withheld:
        lines.append(f"* `{row.item_id}` ({row.capability}) -- {'; '.join(row.reasons)}")
    accepted = [row for row in rows if row.expected_level == row.level and row.level != REAL]
    lines += ["", "## Accepted gaps (honest FAIL, registered)", ""]
    if not accepted:
        lines.append("None.")
    for row in accepted:
        lines.append(f"* `{row.item_id}` ({row.capability}): {row.gap_note}")
    crossing = contradictions(rows, structural)
    lines += [
        "",
        "## Structural/numeric disagreement",
        "",
        "A capability counts as backed only when both layers agree. These rows are",
        "declared `REAL` by the structural audit but are not backed by a number that",
        "meets its own criterion -- they are overstated claims, kept visible on purpose:",
        "",
    ]
    if not crossing:
        lines.append("None.")
    for entry in crossing:
        lines.append(
            f"* `{entry['capability']}` (item `{entry['item']}`): structural "
            f"{entry['structural_level']} vs numeric {entry['numeric_level']} -- "
            f"{'; '.join(entry['reasons'])}"
        )
    if registry is not None:
        lines += [
            "",
            f"Registry schema `{registry.get('schema_version', SCHEMA)}`; "
            f"{len(registry.get('items', []))} items; regenerate with "
            "`python -m tensorbem.numeric_evidence --write`.",
            "",
        ]
    return "\n".join(lines)


def document_drift(
    root: pathlib.Path | None = None, registry_path: pathlib.Path | None = None
) -> str | None:
    root = repo_root(root)
    registry_path = pathlib.Path(registry_path) if registry_path else root / REGISTRY
    path = root / DOCUMENT
    if not path.is_file():
        return f"{DOCUMENT} is missing"
    registry = load_registry(registry_path)
    rows = audit(root, registry_path=registry_path, capabilities=None)
    expected = render_markdown(rows, structural=_structural_levels(), registry=registry)
    actual = path.read_text(encoding="utf-8")
    if actual.strip() != expected.strip():
        return f"{DOCUMENT} does not match the registry-derived table"
    return None


def _structural_levels() -> dict[str, str]:
    try:
        from tensorbem import capability_evidence as ce

        return {row.capability.key: row.level for row in ce.audit()}
    except Exception:  # pragma: no cover - structural layer unavailable
        return {}


# ---------------------------------------------------------------------------
# write / check
# ---------------------------------------------------------------------------


def write(root: pathlib.Path | None = None) -> pathlib.Path:
    """Re-pin artifact digests and refresh the published matrix."""

    root = repo_root(root)
    registry_path = root / REGISTRY
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry["schema_version"] = SCHEMA
    for item in registry.get("items", []):
        artifact = root / item["artifact"]
        if artifact.is_file():
            item["sha256"] = sha256_file(artifact)
    registry_path.write_text(
        json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    rows = audit(root, registry_path=registry_path)
    document = root / DOCUMENT
    document.write_text(
        render_markdown(rows, structural=_structural_levels(), registry=registry) + "\n",
        encoding="utf-8",
    )
    return document


def check(
    root: pathlib.Path | None = None, registry_path: pathlib.Path | None = None
) -> list[str]:
    """Fail-closed gate over the committed numeric evidence.

    Hard failures (always red): a missing or digest-drifted artifact, a REAL row
    whose own numbers overrun its tolerance, a row that moved away from its
    registered expectation, and document drift.

    An honest gap -- a criterion the committed numbers do not meet -- is *not* a
    gate failure, but it must be registered: ``expected_level`` names the level
    the record accepts and ``gap_note`` says why the residual is accepted. An
    unregistered WITHHELD row goes red, so a gap cannot appear (or disappear)
    without someone editing the record.
    """

    root = repo_root(root)
    registry_path = pathlib.Path(registry_path) if registry_path else root / REGISTRY
    problems: list[str] = []
    rows = audit(root, registry_path=registry_path, capabilities=None)
    for row in rows:
        detail = "; ".join(row.reasons)
        if row.level in (STALE, MISSING):
            problems.append(f"{row.item_id}: {row.level} -- {detail}")
            continue
        if row.level == REAL:
            if row.error_pct is not None and row.error_pct > row.tolerance_pct:
                problems.append(
                    f"{row.item_id}: graded REAL although the recomputed error "
                    f"{row.error_pct:.6g}% overruns {row.tolerance_pct:.6g}%"
                )
            if row.expected_level is not None and row.expected_level != REAL:
                problems.append(
                    f"{row.item_id}: registry expects {row.expected_level} but the numbers "
                    "now meet the criterion; re-grade and re-declare the record"
                )
            continue
        if row.expected_level is None:
            problems.append(
                f"{row.item_id}: {row.level} with no registered expectation -- declare "
                f"expected_level/gap_note ({detail})"
            )
        elif row.expected_level != row.level:
            problems.append(
                f"{row.item_id}: registry expects {row.expected_level}, measured {row.level} "
                f"-- {detail}"
            )
        elif not row.gap_note:
            problems.append(
                f"{row.item_id}: accepted gap {row.level} carries no gap_note explaining the "
                "residual"
            )
    drift = document_drift(root, registry_path)
    if drift:
        problems.append(drift)
    return problems


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true", help="gate the committed evidence")
    group.add_argument("--write", action="store_true", help="re-pin digests and refresh the doc")
    parser.add_argument("--json", action="store_true", help="dump the graded rows as JSON")
    args = parser.parse_args(argv)

    if args.write:
        path = write()
        print(f"wrote {path.relative_to(repo_root())}")
        return 0

    rows = audit(capabilities=None)
    if args.json:
        print(json.dumps([row.as_dict() for row in rows], indent=2))
    counts = summary(rows)
    print(json.dumps(counts, sort_keys=True))
    problems = check()
    for problem in problems:
        print(f"FAIL {problem}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())