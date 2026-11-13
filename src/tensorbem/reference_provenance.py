"""Machine-checked provenance for literature and experimental reference constants.

Why this module exists
----------------------
A verification driver is only as honest as its reference numbers.  TensorBEM's
reports quote error percentages against WAMIT added mass, Journee (1992) RAO
peaks, published semisubmersible displacements and so on.  Every one of those
percentages is a *derived claim*: change the memorised constant and the claim
moves, with nothing in the test suite noticing.  That is the failure mode the
repository's "no hand transcription" rule exists to stop, and until now it was
enforced only by review attention -- the capability registry grades *code,
tests and documents*, never the provenance of the constants a driver compares
against.

This module closes that hole mechanically:

* :func:`scan` walks ``src/tensorbem`` and collects every module-level numeric
  constant whose **name** announces a literature origin (``WAMIT``,
  ``JOURNEE``, ``EFD``, ``EXPERIMENT``, ``LITERATURE``, ``PUBLISHED``,
  ``MEASURED``, ``OBSERVED``, ``REFERENCE_DATA``).  The number itself is never
  graded -- provenance is.
* :data:`DECLARED` is the triage registry: every scanned constant must appear
  in it, with a status and, for :data:`CITED`, the in-repository document plus
  the literal digits that document publishes.  The citation is resolved by
  reading the file, so a citation cannot rot into a dangling reference.
* :func:`audit` returns ``undeclared`` (a new un-sourced reference constant --
  the fail-closed part), ``stale`` (a registry entry whose constant is gone),
  ``unverified`` (a citation that no longer resolves) and ``unsourced`` (known
  debt, published honestly in the generated document).
* :func:`readers` adds the dimension the registry used to only promise in prose:
  *who reads a recalled number*.  A recalled constant is harmless only while
  nothing downstream of it publishes a value, so every cross-module reader is
  enumerated over ``src/``, ``tests/`` and ``scripts/`` and compared with
  :data:`ADJUDICATED`, the allow-list of adjudicating drivers.
* :func:`adjudication` reads the verdicts that adjudicator published when it put
  each recalled number against two independent codes, so the debt table states
  what was measured about the value, not only that its source is missing.

Design notes
------------
Statuses are deliberately only two:

``CITED``
    the value is published somewhere this repository ships, and the registry
    carries ``(path, needle)`` evidence pairs that :func:`audit` re-reads.
``UNSOURCED``
    the value has no in-repository source.  It stays in the registry with a
    ``note`` saying what it is and why it exists, so the debt is visible in the
    generated document instead of being silently trusted.  Nothing about this
    status is inferred: a driver marked ``UNSOURCED`` is *not* evidence.

Only module-level assignments are scanned.  Locals inside a function are part
of an algorithm (a tabulated Bessel zero, a Gauss abscissa); the module-level
form is where a driver puts "the reference answer", which is what this gate is
about.  Names are matched case-insensitively so ``_WAMIT_A33`` and
``journee_rao`` are both found.

CLI::

    python -m tensorbem.reference_provenance            # report, exit 1 on debt
    python -m tensorbem.reference_provenance --write    # regenerate the doc block
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import pathlib
import re
import sys
from dataclasses import dataclass, field

__all__ = [
    "CITED",
    "UNSOURCED",
    "ADJUDICATED",
    "ADJUDICATION_ARTEFACT",
    "Adjudication",
    "Declaration",
    "Entry",
    "DECLARED",
    "Reader",
    "adjudication",
    "audit",
    "document_drift",
    "extract_document_block",
    "readers",
    "render_block",
    "scan",
    "write_document",
    "main",
]

CITED = "CITED"
UNSOURCED = "UNSOURCED"
STATUSES = (CITED, UNSOURCED)

#: Name markers that announce a literature or experimental origin.
MARKER = re.compile(
    r"(WAMIT|JOURNEE|EFD|EXPERIMENT|LITERATURE|PUBLISHED|MEASURED|OBSERVED|REFERENCE_DATA)",
    re.IGNORECASE,
)

DOCUMENT = "docs/reference_provenance.md"
BLOCK_START = "<!-- reference-provenance:begin -->"
BLOCK_END = "<!-- reference-provenance:end -->"


# --------------------------------------------------------------------------
# what a scanned constant looks like
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Declaration:
    """One module-level constant that announces a literature origin."""

    module: str
    name: str
    line: int
    kind: str  # "scalar" or "mapping"
    literals: tuple[str, ...]  # source spelling of the numbers it holds

    @property
    def key(self) -> str:
        return f"{self.module}.{self.name}"


@dataclass(frozen=True)
class Entry:
    """The triage verdict for one scanned constant."""

    status: str
    evidence: tuple[tuple[str, str], ...] = ()
    note: str = ""

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"unknown provenance status {self.status!r}")
        if not self.note:
            raise ValueError("every entry must carry a note: provenance debt is never silent")
        if self.status == CITED and not self.evidence:
            raise ValueError("a CITED entry needs at least one (path, needle) evidence pair")


def _numeric_literals(node: ast.AST) -> tuple[str, ...]:
    """Source spelling of the plain numeric literals a value expression holds."""

    def walk(item: ast.AST) -> list[str]:
        if (
            isinstance(item, ast.Constant)
            and isinstance(item.value, (int, float))
            and not isinstance(item.value, bool)
        ):
            return [repr(item.value)]
        if isinstance(item, (ast.Dict, ast.List, ast.Tuple, ast.Set)):
            elts = item.values if isinstance(item, ast.Dict) else item.elts
            out: list[str] = []
            for elt in elts:
                if elt is not None:
                    out.extend(walk(elt))
            return out
        if isinstance(item, ast.UnaryOp) and isinstance(item.op, (ast.USub, ast.UAdd)):
            inner = walk(item.operand)
            sign = "-" if isinstance(item.op, ast.USub) else ""
            return [sign + text for text in inner]
        return []

    return tuple(walk(node))


def _is_mapping(node: ast.AST) -> bool:
    return isinstance(node, (ast.Dict, ast.List, ast.Tuple, ast.Set))


def repo_root(start: pathlib.Path | None = None) -> pathlib.Path:
    """Repository root: the nearest ancestor that owns ``src/tensorbem``."""

    if start is None:
        if "__file__" in globals():
            here = pathlib.Path(__file__).resolve().parent
        else:  # pragma: no cover - dynamic import edge case
            here = pathlib.Path.cwd()
    else:
        here = pathlib.Path(start).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "src" / "tensorbem" / "capability_evidence.py").is_file():
            return candidate
    return here


def _modules_under(root: pathlib.Path) -> list[pathlib.Path]:
    package = root / "src" / "tensorbem"
    return [
        path
        for path in sorted(package.rglob("*.py"))
        if "__pycache__" not in path.parts
    ]


def _module_name(path: pathlib.Path, root: pathlib.Path) -> str:
    rel = path.relative_to(root / "src").with_suffix("")
    parts = list(rel.parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def scan(root: pathlib.Path | None = None) -> list[Declaration]:
    """Every module-level literature-marked numeric constant in ``src/tensorbem``."""

    root = root or repo_root()
    found: list[Declaration] = []
    for path in _modules_under(root):
        module = _module_name(path, root)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            value: ast.AST | None = None
            names: list[str] = []
            if isinstance(node, ast.Assign):
                value = node.value
                names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                value = node.value
                if isinstance(node.target, ast.Name):
                    names = [node.target.id]
            if value is None:
                continue
            literals = _numeric_literals(value)
            if not literals:
                continue
            for name in names:
                if not MARKER.search(name):
                    continue
                found.append(
                    Declaration(
                        module=module,
                        name=name,
                        line=node.lineno,
                        kind="mapping" if _is_mapping(value) else "scalar",
                        literals=literals,
                    )
                )
    return found


# --------------------------------------------------------------------------
# who reads a recalled number
# --------------------------------------------------------------------------

#: Where a consumer of a scanned constant can live.  ``docs/`` is excluded on
#: purpose: a narrative document that spells a constant's name is not a
#: consumer of the number.
READER_ROOTS = ("src", "tests", "scripts")

#: The artefact the adjudicating driver publishes: one record per recalled value,
#: carrying the verdict and the factor between the recalled number and what two
#: independent codes compute for the same quantity.
ADJUDICATION_ARTEFACT = "benchmarks/verified/wigley3_recalled_ref/results.json"


@dataclass(frozen=True)
class Reader:
    """One module that mentions a scanned constant's name outside its definition."""

    consumer: str  # dotted module for src/, repository-relative path otherwise
    line: int
    how: str  # "import" or "attribute"

    def __str__(self) -> str:
        return f"{self.consumer}:{self.line}"


def _consumer_name(path: pathlib.Path, root: pathlib.Path) -> str:
    if (root / "src") in path.parents:
        return _module_name(path, root)
    return path.relative_to(root).as_posix()


def _reader_files(root: pathlib.Path) -> list[pathlib.Path]:
    files: list[pathlib.Path] = []
    for directory in READER_ROOTS:
        base = root / directory
        if base.is_dir():
            files.extend(
                path for path in sorted(base.rglob("*.py")) if "__pycache__" not in path.parts
            )
    return files


def readers(
    root: pathlib.Path | None = None,
    declarations: list[Declaration] | None = None,
) -> dict[str, list[Reader]]:
    """Cross-module readers of every scanned constant, keyed by registry key.

    A constant's own module is not its reader, and a module that *assigns* the
    same name is not a reader either: only ``from ... import NAME`` and attribute
    access (``something.NAME``) count.  Matching is by name and deliberately
    blunt -- a false positive is a conversation, a false negative is a recalled
    number reaching a claim unannounced.

    Scope is :data:`READER_ROOTS`.  The reason the question is asked at all: a
    recalled constant is only harmless while nothing downstream of it publishes
    a number, so the reader graph is what turns that prose promise into a gate.
    """

    root = root or repo_root()
    declarations = scan(root) if declarations is None else declarations
    by_name: dict[str, list[Declaration]] = {}
    for declaration in declarations:
        by_name.setdefault(declaration.name, []).append(declaration)

    found: dict[str, list[Reader]] = {declaration.key: [] for declaration in declarations}
    if not by_name:
        return found

    for path in _reader_files(root):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:  # pragma: no cover - syntax errors are another gate's business
            continue
        consumer = _consumer_name(path, root)
        hits: dict[str, tuple[int, str]] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name in by_name:
                        hits.setdefault(alias.name, (node.lineno, "import"))
            elif isinstance(node, ast.Attribute) and node.attr in by_name:
                hits.setdefault(node.attr, (getattr(node, "lineno", 0), "attribute"))
        for name, (line, how) in hits.items():
            for declaration in by_name[name]:
                if declaration.module == consumer:
                    continue  # the defining module is not a reader of its own constant
                found[declaration.key].append(Reader(consumer=consumer, line=line, how=how))

    for key in found:
        found[key].sort(key=lambda reader: (reader.consumer, reader.line))
    return found


@dataclass(frozen=True)
class Adjudication:
    """What the adjudicating driver published about one recalled constant."""

    artefact: str
    records: tuple[tuple[str, float], ...]  # (verdict, computed/recalled factor)

    @property
    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for verdict, _ in self.records:
            out[verdict] = out.get(verdict, 0) + 1
        return out

    def describe(self) -> str:
        if not self.records:
            return "no records"
        verdicts = ", ".join(
            f"{verdict} {count}/{len(self.records)}"
            for verdict, count in sorted(self.counts.items())
        )
        factors = [factor for _, factor in self.records if math.isfinite(factor)]
        if not factors:
            return verdicts
        return f"{verdicts} (computed/recalled {min(factors):.3g}-{max(factors):.3g}x)"


def adjudication(root: pathlib.Path | None = None) -> dict[str, Adjudication]:
    """Verdicts published in :data:`ADJUDICATION_ARTEFACT`, keyed by registry key."""

    root = root or repo_root()
    path = root / ADJUDICATION_ARTEFACT
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}
    grouped: dict[str, list[tuple[str, float]]] = {}
    for item in payload.get("adjudication", []):
        if not isinstance(item, dict):
            continue
        key, verdict = item.get("constant"), item.get("verdict")
        if not isinstance(key, str) or not isinstance(verdict, str):
            continue
        try:
            factor = float(item.get("factor", "nan"))
        except (TypeError, ValueError):
            factor = float("nan")
        grouped.setdefault(key, []).append((verdict, factor))
    return {
        key: Adjudication(artefact=ADJUDICATION_ARTEFACT, records=tuple(records))
        for key, records in grouped.items()
    }


# --------------------------------------------------------------------------
# the triage registry
# --------------------------------------------------------------------------

#: Every literature-marked constant in ``src/tensorbem``, with its verdict.
#: A new constant that is not listed here fails ``tests/test_reference_provenance.py``:
#: that is the point -- a reference number has to be justified before it is used
#: to produce an error percentage.
DECLARED: dict[str, Entry] = {
    "tensorbem.qtf_drift.SEMISUB_PUBLISHED_DISPLACEMENT": Entry(
        status=CITED,
        evidence=(("docs/qtf_drift_oc4.md", "13473"),),
        note=(
            "DeepCwind/OC4 semisubmersible displacement; the value and the Robertson et al. (2014)"
            " source are published in docs/qtf_drift_oc4.md, which also reports the +1.8% ratio between"
            " the smooth-geometry integral and this number."
        ),
    ),
    # ---- Wigley III frequency-domain drivers ---------------------------------
    # Four drivers (m-era P40/8822 lineage) each re-type a small WAMIT/Journee
    # table and print an error column against it.  No digitised table is shipped
    # in this repository and the machine has no route to one, so the numbers are
    # recalled rather than read: they are recorded here as debt and are NOT
    # evidence.  The graded Wigley III surface is the core-module row, whose
    # anchors are computed (Michell/Kochin, hydrostatics) and never transcribed.
    "tensorbem.benchmarks.wigley3_complete_verification._WAMIT_A33_MASS_RATIO": Entry(
        status=UNSOURCED,
        note=(
            "recalled WAMIT heave added-mass ratio at omega = 1 rad/s, marked 'approximate' in the"
            " driver itself; no digitised WAMIT table is shipped in the repository.  Legacy driver"
            " with no capability row: the number reaches no published claim."
            "  Read only by the adjudicating driver"
            " ``tensorbem.benchmarks.wigley3_recalled_ref``, which measures it against two"
            " independent codes and publishes the verdict in"
            " ``benchmarks/verified/wigley3_recalled_ref/results.json``."
        ),
    ),
    "tensorbem.benchmarks.wigley3_complete_verification._JOURNEE_HEAVE_RAO_PEAK": Entry(
        status=UNSOURCED,
        note=(
            "recalled Journee (1992) heave RAO peak (comment in the driver says 'approximate values"
            " from experimental data'); the report it cites is not shipped.  Legacy driver, ungraded."
        ),
    ),
    "tensorbem.benchmarks.wigley3_complete_verification._JOURNEE_HEAVE_RAO_PEAK_OMEGA": Entry(
        status=UNSOURCED,
        note=(
            "recalled frequency of the Journee (1992) heave RAO peak; source report not shipped."
            "  Legacy driver, ungraded."
        ),
    ),
    "tensorbem.benchmarks.wigley3_freq_domain_verification.WAMIT_REF": Entry(
        status=UNSOURCED,
        note=(
            "recalled WAMIT reference block (added mass and damping) typed into a legacy Wigley III"
            " driver; no shipped artefact carries the table.  Ungraded driver."
        ),
    ),
    "tensorbem.benchmarks.wigley3_freq_domain_verification.JOURNEE1992_RAO": Entry(
        status=UNSOURCED,
        note=(
            "recalled Journee (1992) RAO table; the report is not shipped, so the driver's error"
            " column cannot be reproduced from repository contents.  Ungraded driver."
        ),
    ),
    "tensorbem.benchmarks.wigley3_freq_verification._WAMIT_ADDED_MASS_DIAGONAL": Entry(
        status=UNSOURCED,
        note=(
            "recalled diagonal WAMIT added mass at omega = 1 rad/s (legacy driver); no shipped table."
            "  No capability row reads this constant; its only reader is the adjudicating driver"
            " ``tensorbem.benchmarks.wigley3_recalled_ref``, which measures it against two"
            " independent codes and publishes the verdict in"
            " ``benchmarks/verified/wigley3_recalled_ref/results.json``."
        ),
    ),
    "tensorbem.benchmarks.wigley3_freq_verification._WAMIT_DAMPING_DIAGONAL": Entry(
        status=UNSOURCED,
        note=(
            "recalled diagonal WAMIT damping at omega = 1 rad/s; no shipped table.  Its only reader"
            " is the adjudicating driver ``tensorbem.benchmarks.wigley3_recalled_ref``, which"
            " measures it against two independent codes and publishes the verdict in"
            " ``benchmarks/verified/wigley3_recalled_ref/results.json``."
        ),
    ),
    "tensorbem.benchmarks.wigley3_freq_verification._JOURNEE_EXCITATION": Entry(
        status=UNSOURCED,
        note=(
            "recalled Journee (1992) excitation amplitudes at beta = 0 deg; source report not shipped."
            "  Ungraded driver."
        ),
    ),
    "tensorbem.benchmarks.wigley3_freq_verification._JOURNEE_RAO_PEAKS": Entry(
        status=UNSOURCED,
        note="recalled Journee (1992) RAO peaks (driver comment: 'approximate').  Ungraded driver.",
    ),
    "tensorbem.benchmarks.wigley_diffraction_radiation.WAMIT_A33_REF": Entry(
        status=UNSOURCED,
        note=(
            "recalled WAMIT heave added mass for Wigley III; the WAMIT verification case is not"
            " shipped with the repository.  Its only reader is the adjudicating driver"
            " ``tensorbem.benchmarks.wigley3_recalled_ref``, which measures it against two"
            " independent codes and publishes the verdict in"
            " ``benchmarks/verified/wigley3_recalled_ref/results.json``."
        ),
    ),
    "tensorbem.benchmarks.wigley_diffraction_radiation.WAMIT_B33_REF": Entry(
        status=UNSOURCED,
        note=(
            "recalled WAMIT heave damping for Wigley III; no shipped table.  Its only reader is the"
            " adjudicating driver ``tensorbem.benchmarks.wigley3_recalled_ref``, which measures it"
            " against two independent codes and publishes the verdict in"
            " ``benchmarks/verified/wigley3_recalled_ref/results.json``."
        ),
    ),
    "tensorbem.benchmarks.wigley_diffraction_radiation.WAMIT_F3_BETA0_REF": Entry(
        status=UNSOURCED,
        note="recalled WAMIT heave excitation at beta = 0 deg; no shipped table.  Ungraded driver.",
    ),
    "tensorbem.benchmarks.wigley_diffraction_radiation.JOURNEE_HEAVE_RAO_REF": Entry(
        status=UNSOURCED,
        note="recalled Journee (1992) heave RAO; source report not shipped.  Ungraded driver.",
    ),
    "tensorbem.benchmarks.wigley_diffraction_radiation.JOURNEE_PITCH_RAO_REF": Entry(
        status=UNSOURCED,
        note="recalled Journee (1992) pitch RAO; source report not shipped.  Ungraded driver.",
    ),
}


#: Recalled constants are not evidence, but they are not ignored either.  Each
#: one below is adjudicated by a driver that measures the recalled number against
#: two independent codes and publishes a verdict per degree of freedom in
#: :data:`ADJUDICATION_ARTEFACT`.  This map is therefore the allow-list of
#: cross-module readers: a recalled constant may be read by its adjudicator and
#: by nothing else, because any other reader would put a number with no source
#: behind a claim.  ``audit`` fails if a new reader appears, if a registered
#: adjudicator never actually reads its constant (registry rot), or if the
#: artefact carries no verdict for a registered constant.
ADJUDICATED: dict[str, str] = {
    "tensorbem.benchmarks.wigley3_complete_verification._WAMIT_A33_MASS_RATIO":
        "tensorbem.benchmarks.wigley3_recalled_ref",
    "tensorbem.benchmarks.wigley3_freq_verification._WAMIT_ADDED_MASS_DIAGONAL":
        "tensorbem.benchmarks.wigley3_recalled_ref",
    "tensorbem.benchmarks.wigley3_freq_verification._WAMIT_DAMPING_DIAGONAL":
        "tensorbem.benchmarks.wigley3_recalled_ref",
    "tensorbem.benchmarks.wigley_diffraction_radiation.WAMIT_A33_REF":
        "tensorbem.benchmarks.wigley3_recalled_ref",
    "tensorbem.benchmarks.wigley_diffraction_radiation.WAMIT_B33_REF":
        "tensorbem.benchmarks.wigley3_recalled_ref",
}


# --------------------------------------------------------------------------
# audit
# --------------------------------------------------------------------------


@dataclass
class Audit:
    declarations: list[Declaration] = field(default_factory=list)
    registry: dict[str, Entry] = field(default_factory=dict)
    undeclared: list[Declaration] = field(default_factory=list)
    stale: list[str] = field(default_factory=list)
    unverified: list[tuple[str, str, str]] = field(default_factory=list)
    cited: list[Declaration] = field(default_factory=list)
    unsourced: list[Declaration] = field(default_factory=list)
    readers: dict[str, list[Reader]] = field(default_factory=dict)
    leaked: list[tuple[str, Reader]] = field(default_factory=list)
    adjudication: dict[str, Adjudication] = field(default_factory=dict)
    unadjudicated: list[str] = field(default_factory=list)
    unread_adjudicator: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (
            self.undeclared
            or self.stale
            or self.unverified
            or self.leaked
            or self.unadjudicated
            or self.unread_adjudicator
        )

    def summary(self) -> dict[str, int]:
        return {
            "declarations": len(self.declarations),
            "cited": len(self.cited),
            "unsourced": len(self.unsourced),
            "undeclared": len(self.undeclared),
            "stale": len(self.stale),
            "unverified": len(self.unverified),
            "readers": sum(len(found) for found in self.readers.values()),
            "leaked": len(self.leaked),
            "adjudicated": len(self.adjudication),
            "unadjudicated": len(self.unadjudicated) + len(self.unread_adjudicator),
        }


def audit(
    root: pathlib.Path | None = None,
    declared: dict[str, Entry] | None = None,
    adjudicated: dict[str, str] | None = None,
) -> Audit:
    """Cross-check the scanned constants against the triage registry.

    Three questions are asked of every scanned constant: is it triaged (with a
    citation that still resolves), who reads it, and -- for a recalled value --
    was it measured against an independent code?
    """

    root = root or repo_root()
    registry = DECLARED if declared is None else declared
    adjudicators = ADJUDICATED if adjudicated is None else adjudicated
    result = Audit(declarations=scan(root), registry=registry)
    keys = {declaration.key for declaration in result.declarations}

    for declaration in result.declarations:
        entry = registry.get(declaration.key)
        if entry is None:
            result.undeclared.append(declaration)
            continue
        if entry.status == CITED:
            missing = False
            for path, needle in entry.evidence:
                target = root / path
                if not target.is_file() or needle not in target.read_text(encoding="utf-8"):
                    result.unverified.append((declaration.key, path, needle))
                    missing = True
            if not missing:
                result.cited.append(declaration)
        else:
            result.unsourced.append(declaration)

    result.stale = sorted(key for key in registry if key not in keys)
    result.readers = readers(root, result.declarations)

    verdicts = adjudication(root)
    for declaration in result.unsourced:
        owner = adjudicators.get(declaration.key)
        consumers = result.readers.get(declaration.key, [])
        for reader in consumers:
            if reader.consumer != owner:
                result.leaked.append((declaration.key, reader))
        if owner is None:
            continue
        if not any(reader.consumer == owner for reader in consumers):
            result.unread_adjudicator.append(declaration.key)
        record = verdicts.get(declaration.key)
        if record is None:
            result.unadjudicated.append(declaration.key)
        else:
            result.adjudication[declaration.key] = record

    return result


# --------------------------------------------------------------------------
# the published document
# --------------------------------------------------------------------------


def _readers_cell(result: "Audit", key: str) -> str:
    found = result.readers.get(key, [])
    if not found:
        return "none"
    return ", ".join(f"`{reader.consumer}`" for reader in found)


def render_block(result: Audit | None = None, root: pathlib.Path | None = None) -> str:
    """The machine-generated block published in :data:`DOCUMENT`."""

    result = result or audit(root)
    totals = result.summary()
    registry = result.registry or DECLARED
    lines = [BLOCK_START, ""]
    lines.append(
        f"Scanned `src/tensorbem` for module-level constants that announce a literature origin:"
        f" **{totals['declarations']}** found, **{totals['cited']}** CITED,"
        f" **{totals['unsourced']}** UNSOURCED."
    )
    lines.append("")

    if result.cited:
        lines.append("### CITED -- the value is published in this repository")
        lines.append("")
        lines.append("| constant | location | evidence (file :: digits) | read by |")
        lines.append("|---|---|---|---|")
        for declaration in result.cited:
            entry = registry[declaration.key]
            evidence = "; ".join(f"`{path}` :: `{needle}`" for path, needle in entry.evidence)
            lines.append(
                f"| `{declaration.name}` | `{declaration.module}` | {evidence}"
                f" | {_readers_cell(result, declaration.key)} |"
            )
        lines.append("")

    if result.unsourced:
        lines.append("### UNSOURCED -- open provenance debt, not evidence")
        lines.append("")
        lines.append(
            "These constants are recalled rather than read. They are kept because the drivers that"
            " carry them are legacy, but **no published claim may rest on them**: the CITED surface"
            " above and the computed anchors in `docs/audit_baseline.md` are what the verification"
            " claims use."
        )
        lines.append("")
        lines.append(
            "Two columns below are machine-checked rather than asserted. **read by** is the"
            " cross-module reader graph over `src/`, `tests/` and `scripts/`: a recalled constant is"
            " allowed exactly one reader, the adjudicator registered in `ADJUDICATED`, and any other"
            " reader turns the suite red. **adjudicated** says what that adjudicator measured when it"
            " put the recalled number against two independent codes"
            f" (`{ADJUDICATION_ARTEFACT}`): a `REFUTED` verdict is the registry's expectation"
            " confirmed -- the number was wrong, which is why it is not evidence."
        )
        lines.append("")
        lines.append("| constant | location | line | read by | adjudicated | why it is still here |")
        lines.append("|---|---|---|---|---|---|")
        for declaration in sorted(result.unsourced, key=lambda d: (d.module, d.name)):
            entry = registry[declaration.key]
            note = " ".join(entry.note.split())
            record = result.adjudication.get(declaration.key)
            verdict = "not adjudicated" if record is None else record.describe()
            lines.append(
                f"| `{declaration.name}` | `{declaration.module}` | {declaration.line}"
                f" | {_readers_cell(result, declaration.key)} | {verdict} | {note} |"
            )
        lines.append("")

    lines.append(
        f"Fail-closed checks: undeclared **{totals['undeclared']}**, stale registry entries"
        f" **{totals['stale']}**, unresolvable citations **{totals['unverified']}**, unregistered"
        f" readers of a recalled constant **{totals['leaked']}**, adjudicator/artefact mismatches"
        f" **{totals['unadjudicated']}**.  A new literature-marked constant that is not triaged turns"
        " `tests/test_reference_provenance.py` red, so an un-sourced reference number cannot enter a"
        " driver silently."
    )
    lines.append("")
    lines.append(BLOCK_END)
    return "\n".join(lines)


def extract_document_block(text: str) -> str | None:
    start = text.find(BLOCK_START)
    end = text.find(BLOCK_END)
    if start < 0 or end < 0 or end < start:
        return None
    return text[start : end + len(BLOCK_END)]


def document_drift(root: pathlib.Path | None = None) -> str | None:
    """``None`` when the published block equals the generated one, else a unified diff."""

    import difflib

    root = root or repo_root()
    path = root / DOCUMENT
    if not path.is_file():
        return f"{DOCUMENT} is missing"
    published = extract_document_block(path.read_text(encoding="utf-8"))
    generated = render_block(root=root)
    if published is None:
        return f"{DOCUMENT} carries no reference-provenance block"
    if published == generated:
        return None
    return "\n".join(
        difflib.unified_diff(
            generated.splitlines(),
            published.splitlines(),
            fromfile="generated",
            tofile="published",
            lineterm="",
        )
    )


HEADER = """# Reference-constant provenance: which numbers a driver is allowed to trust

TensorBEM's verification claims are error percentages: *this* added mass against
WAMIT, *this* RAO peak against Journee (1992), *this* displacement against the
published OC4 value.  A percentage is only worth what its denominator is worth,
and the repository's rule is that no number is transcribed by hand into a claim.

The capability registry (`docs/audit_baseline.md`) grades code, tests and
documents; it does not look at the constants those drivers compare against.
This document is the missing half.  `tensorbem.reference_provenance` scans the
package for module-level constants whose **name** announces a literature origin
(`WAMIT`, `JOURNEE`, `EFD`, `EXPERIMENT`, `LITERATURE`, `PUBLISHED`, `MEASURED`,
`OBSERVED`, `REFERENCE_DATA`) and requires every one of them to be triaged:

* **CITED** -- the value is published by a document this repository ships, and
  the registry names that document plus the literal digits, which are re-read on
  every run (a citation that rots is a red build, not a dangling footnote);
* **UNSOURCED** -- a recalled value.  It is recorded, published here, and
  explicitly *not* evidence.

Being "not evidence" is a property of the *graph*, not of the constant, so two
further columns are computed rather than promised.  **read by** is the
cross-module reader set over `src/`, `tests/` and `scripts/`: a recalled
constant may be read by its registered adjudicator and by nothing else, because
any other reader would let an unsourced number reach a claim.  **adjudicated**
is that adjudicator's verdict, read back from
`benchmarks/verified/wigley3_recalled_ref/results.json`, where each recalled
number was put against two independent codes on three panel counts.  A
`REFUTED` verdict is the registry working as designed -- the recalled value was
wrong, which is precisely why no claim may be built on it.

The gate is fail-closed in the direction that matters: an un-triaged constant is
`undeclared` and turns `tests/test_reference_provenance.py` red, so a recalled
reference table cannot be introduced into a driver silently -- which is exactly
how the four legacy Wigley III drivers below accumulated theirs.

Regenerate this block with::

    python -m tensorbem.reference_provenance --write

"""


def write_document(root: pathlib.Path | None = None) -> pathlib.Path:
    root = root or repo_root()
    path = root / DOCUMENT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(HEADER + "\n" + render_block(root=root) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true", help=f"regenerate {DOCUMENT}")
    parser.add_argument("--root", type=pathlib.Path, default=None)
    args = parser.parse_args(argv)

    result = audit(args.root)
    totals = result.summary()

    if args.write:
        path = write_document(args.root)
        print(f"wrote {path}")
    else:
        print(render_block(result, args.root))

    print("---")
    print(
        f"declarations={totals['declarations']} cited={totals['cited']}"
        f" unsourced={totals['unsourced']} undeclared={totals['undeclared']}"
        f" stale={totals['stale']} unverified={totals['unverified']}"
        f" readers={totals['readers']} leaked={totals['leaked']}"
        f" adjudicated={totals['adjudicated']} unadjudicated={totals['unadjudicated']}"
    )
    for declaration in result.undeclared:
        print(f"UNDECLARED {declaration.key} (line {declaration.line})")
    for key in result.stale:
        print(f"STALE {key}")
    for key, path, needle in result.unverified:
        print(f"UNVERIFIED {key}: {path} does not publish {needle!r}")
    for key, reader in result.leaked:
        print(f"LEAKED READER {key}: {reader} reads a recalled constant that is not its adjudicator")
    for key in result.unread_adjudicator:
        print(f"STALE ADJUDICATOR {key}: the registered adjudicator does not read it")
    for key in result.unadjudicated:
        print(f"UNADJUDICATED {key}: no verdict in {ADJUDICATION_ARTEFACT}")

    drift = document_drift(args.root) if (args.root or repo_root()) / DOCUMENT else None
    if drift and not args.write:
        print("document drift:")
        print(drift)
    return 0 if (result.ok and not drift) else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())