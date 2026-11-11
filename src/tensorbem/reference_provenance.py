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
import pathlib
import re
import sys
from dataclasses import dataclass, field

__all__ = [
    "CITED",
    "UNSOURCED",
    "Declaration",
    "Entry",
    "DECLARED",
    "audit",
    "document_drift",
    "extract_document_block",
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
            " driver itself; no digitised WAMIT table is shipped in the repository.  Legacy driver:"
            " no test reference and no capability row, so the number reaches no published claim."
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
            "  Ungraded: no test and no capability row reads this constant."
        ),
    ),
    "tensorbem.benchmarks.wigley3_freq_verification._WAMIT_DAMPING_DIAGONAL": Entry(
        status=UNSOURCED,
        note="recalled diagonal WAMIT damping at omega = 1 rad/s; no shipped table.  Ungraded driver.",
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
            " shipped with the repository.  Ungraded driver."
        ),
    ),
    "tensorbem.benchmarks.wigley_diffraction_radiation.WAMIT_B33_REF": Entry(
        status=UNSOURCED,
        note="recalled WAMIT heave damping for Wigley III; no shipped table.  Ungraded driver.",
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


# --------------------------------------------------------------------------
# audit
# --------------------------------------------------------------------------


@dataclass
class Audit:
    declarations: list[Declaration] = field(default_factory=list)
    undeclared: list[Declaration] = field(default_factory=list)
    stale: list[str] = field(default_factory=list)
    unverified: list[tuple[str, str, str]] = field(default_factory=list)
    cited: list[Declaration] = field(default_factory=list)
    unsourced: list[Declaration] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.undeclared and not self.stale and not self.unverified

    def summary(self) -> dict[str, int]:
        return {
            "declarations": len(self.declarations),
            "cited": len(self.cited),
            "unsourced": len(self.unsourced),
            "undeclared": len(self.undeclared),
            "stale": len(self.stale),
            "unverified": len(self.unverified),
        }


def audit(
    root: pathlib.Path | None = None,
    declared: dict[str, Entry] | None = None,
) -> Audit:
    """Cross-check the scanned constants against the triage registry."""

    root = root or repo_root()
    registry = DECLARED if declared is None else declared
    result = Audit(declarations=scan(root))
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
    return result


# --------------------------------------------------------------------------
# the published document
# --------------------------------------------------------------------------


def render_block(result: Audit | None = None, root: pathlib.Path | None = None) -> str:
    """The machine-generated block published in :data:`DOCUMENT`."""

    result = result or audit(root)
    totals = result.summary()
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
        lines.append("| constant | location | evidence (file :: digits) |")
        lines.append("|---|---|---|")
        for declaration in result.cited:
            entry = DECLARED[declaration.key]
            evidence = "; ".join(f"`{path}` :: `{needle}`" for path, needle in entry.evidence)
            lines.append(
                f"| `{declaration.name}` | `{declaration.module}` | {evidence} |"
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
        lines.append("| constant | location | line | why it is still here |")
        lines.append("|---|---|---|---|")
        for declaration in sorted(result.unsourced, key=lambda d: (d.module, d.name)):
            entry = DECLARED[declaration.key]
            note = " ".join(entry.note.split())
            lines.append(
                f"| `{declaration.name}` | `{declaration.module}` | {declaration.line} | {note} |"
            )
        lines.append("")

    lines.append(
        f"Fail-closed checks: undeclared **{totals['undeclared']}**, stale registry entries"
        f" **{totals['stale']}**, unresolvable citations **{totals['unverified']}**."
        "  A new literature-marked constant that is not triaged turns `tests/test_reference_provenance.py`"
        " red, so an un-sourced reference number cannot enter a driver silently."
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
    )
    for declaration in result.undeclared:
        print(f"UNDECLARED {declaration.key} (line {declaration.line})")
    for key in result.stale:
        print(f"STALE {key}")
    for key, path, needle in result.unverified:
        print(f"UNVERIFIED {key}: {path} does not publish {needle!r}")

    drift = document_drift(args.root) if (args.root or repo_root()) / DOCUMENT else None
    if drift and not args.write:
        print("document drift:")
        print(drift)
    return 0 if (result.ok and not drift) else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())