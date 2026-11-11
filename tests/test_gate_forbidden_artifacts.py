"""No test may assert that a push-gate-filtered artifact exists in the tree.

Background: the platform push gate drops ``*.png|*.db|*.sqlite3?|*.log`` outside
``docs/`` (``tp_pusher._EXCLUDE2`` plus the v5.8 F3 ``docs/`` exemption).  Such
files are legal in a local commit but **never** arrive on GitHub ``main``, so a
test that asserts their presence is red on every fresh clone -- a bug in the
test, not in the code.  This class of defect produced five reds on ``main``
(m13806: finite-depth / thin-ship / w20 figure gate; m15295: MacCamy-Fuchs).

The durable invariant the repository *can* keep is reproducibility: regenerate
the artifact from the committed numbers into ``tmp_path`` and assert on the
render (see ``tests/_figure_integrity.py::assert_reproducible_figures``).

This module is the static guard against the defect coming back:

* walk every ``tests/**/*.py`` module as an AST and collect ``assert``
  statements that actually probe the filesystem (``exists`` / ``is_file`` /
  ``stat`` / ``glob`` / ...) -- pointer checks such as
  ``results["artifacts"]["figure_png"].endswith(".png")`` are never flagged;
* a probed path counts as *gate-filtered* if it carries a filtered suffix
  literal directly or through local variables (a two-line dataflow, which is
  what makes the ``for figure in ("x.png", ...): assert path.exists()`` shape
  visible);
* such an assertion is still **allowed** when the same dataflow shows the path
  is generated into ``tmp_path`` (not committed) or lives under ``docs/`` (the
  gate's own exemption);
* a self-test plants both bad shapes and three legitimate ones, so the scanner
  cannot silently degrade into a no-op.

Limitation (declared, not hidden): the check is syntactic.  A path assembled
entirely from non-literal data without a filtered literal anywhere in the
module (e.g. a suffix glued on at runtime) is not flagged; the runtime
reproducibility probes are what keep those in line.
"""

from __future__ import annotations

import ast
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[1]
TESTS = REPO / "tests"

#: Suffixes the platform push gate filters out unless they live under ``docs/``.
GATE_FILTERED_SUFFIXES = (".png", ".db", ".sqlite", ".sqlite3", ".log")

#: Names whose presence in the dataflow means the file is generated, not committed.
GENERATED_MARKERS = frozenset({"tmp_path", "tmpdir", "tmp_path_factory", "mkdtemp",
                               "TemporaryDirectory"})

#: Filesystem probes -- only an assertion that touches the filesystem can pin a
#: committed artifact.
PROBE_NAMES = frozenset({
    "exists", "is_file", "is_dir", "stat", "glob", "rglob", "iterdir",
    "read_bytes", "read_text", "open", "size", "walk",
})

FILTERED = "filtered"
GENERATED = "generated"
EXEMPT = "exempt"


def _string_literals(node: ast.AST) -> list[str]:
    return [n.value for n in ast.walk(node)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)]


def _identifiers(node: ast.AST) -> set[str]:
    names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)}
    names |= {getattr(n.value, "id", "") for n in ast.walk(node)
              if isinstance(n, ast.Attribute)}
    return {name for name in names if name}


def _target_names(node: ast.AST) -> set[str]:
    targets: list[ast.AST] = []
    if isinstance(node, ast.Assign):
        targets = list(node.targets)
    elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
        targets = [node.target]
    elif isinstance(node, (ast.For, ast.AsyncFor)):
        targets = [node.target]
    names: set[str] = set()
    for target in targets:
        names |= {n.id for n in ast.walk(target) if isinstance(n, ast.Name)}
    return names


def _value_of(node: ast.AST) -> ast.AST | None:
    if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        return node.value
    if isinstance(node, (ast.For, ast.AsyncFor)):
        return node.iter
    return None


def _tags_of(value: ast.AST, known: dict[str, set[str]]) -> set[str]:
    """Tags a value expression carries, given the tags of the names it uses."""
    literals = _string_literals(value)
    identifiers = _identifiers(value)
    tags: set[str] = set()
    for name in identifiers:
        tags |= known.get(name, set())
    if any(lit.lower().endswith(GATE_FILTERED_SUFFIXES) for lit in literals):
        tags.add(FILTERED)
    if identifiers & GENERATED_MARKERS:
        tags.add(GENERATED)
    if any(lit.startswith("docs") or "/docs/" in lit or lit.endswith("docs")
           for lit in literals):
        tags.add(EXEMPT)
    return tags


def _name_tags(scope: ast.AST, base: dict[str, set[str]] | None = None) -> dict[str, set[str]]:
    """Fixpoint dataflow: which names carry a filtered / generated / docs path.

    ``base`` seeds the analysis with the enclosing (module) names, so a
    module-level ``DOCS = REPO / "docs" / ...`` constant keeps its exemption
    inside the functions that use it.
    """
    assignments = [n for n in ast.walk(scope) if _value_of(n) is not None]
    known: dict[str, set[str]] = {name: set(tags) for name, tags in (base or {}).items()}
    changed = True
    while changed:
        changed = False
        for node in assignments:
            value = _value_of(node)
            targets = _target_names(node)
            if not targets:
                continue
            tags = _tags_of(value, known)
            for target in targets:
                merged = known.get(target, set()) | tags
                if merged != known.get(target, set()):
                    known[target] = merged
                    changed = True
    return known


def _scope_tags(tree: ast.Module) -> list[tuple[int, tuple[int, int], dict[str, set[str]]]]:
    """``(span size, span, name tags)`` for the module and for every function."""
    module_tags = _name_tags(tree)
    scopes: list[tuple[ast.AST, dict[str, set[str]]]] = [(tree, module_tags)]
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            scopes.append((node, _name_tags(node, base=module_tags)))
    result = []
    for scope, tags in scopes:
        start = getattr(scope, "lineno", 0)
        end = getattr(scope, "end_lineno", 10 ** 9)
        result.append((end - start, (start, end), tags))
    return result


def find_gate_filtered_assertions(source: str, filename: str = "<string>") -> list[str]:
    """Return one message per ``assert`` that pins a gate-filtered artifact."""
    tree = ast.parse(source, filename=filename)
    scope_data = _scope_tags(tree)

    def tags_for(assert_node: ast.Assert) -> set[str]:
        line = getattr(assert_node, "lineno", 0)
        best = None
        for size, (start, end), known in scope_data:
            if start <= line <= end and (best is None or size < best[0]):
                best = (size, known)
        known = best[1] if best else {}
        tags = {tag for name, name_tags in known.items() if name in _identifiers(assert_node)
                for tag in name_tags}
        return tags | _tags_of(assert_node, known)

    offenders: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assert):
            continue
        identifiers = _identifiers(node)
        if not (identifiers & PROBE_NAMES):
            continue
        tags = tags_for(node)
        if FILTERED not in tags:
            continue
        if tags & {GENERATED, EXEMPT}:
            continue
        offenders.append(f"{filename}:{getattr(node, 'lineno', '?')}: "
                         "asserts on a gate-filtered artifact the push gate drops")
    return offenders


def test_no_test_pins_a_push_gate_filtered_artifact() -> None:
    offenders: list[str] = []
    for path in sorted(TESTS.rglob("*.py")):
        offenders.extend(find_gate_filtered_assertions(
            path.read_text(encoding="utf-8"), str(path.relative_to(REPO))))
    assert not offenders, (
        "push-gate-filtered artifacts (*.png/*.db/*.sqlite*/*.log outside docs/) "
        "never reach main; assert reproducibility into tmp_path instead:\n  "
        + "\n  ".join(offenders)
    )


def test_scanner_flags_planted_offenders_and_spares_legitimate_shapes() -> None:
    """Negative control: the detector must be able to fail, and must stay quiet."""
    direct = """
def test_direct(case_dir):
    assert (case_dir / "figures" / "convergence.png").is_file()
"""
    assert find_gate_filtered_assertions(direct, "planted_direct.py")

    indirect = """
def test_for_loop_shape(repo):
    for figure in ("a.png", "b.png"):
        path = repo / "figures" / figure
        assert path.exists()
"""
    assert find_gate_filtered_assertions(indirect, "planted_indirect.py")

    generated = """
def test_generated(tmp_path):
    figure = tmp_path / "convergence.png"
    render(figure)
    assert figure.stat().st_size > 10_000
"""
    assert not find_gate_filtered_assertions(generated, "planted_generated.py")

    docs = """
DOCS = repo / "docs" / "case"
def test_docs():
    for name in ("fig1.png",):
        assert (DOCS / name).exists()
"""
    assert not find_gate_filtered_assertions(docs, "planted_docs.py")

    pointer = """
def test_pointer(results):
    assert results["artifacts"]["figure_png"].endswith("figures/case.png")
"""
    assert not find_gate_filtered_assertions(pointer, "planted_pointer.py")