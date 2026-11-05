"""No verification test may be gated on an artifact that the repository lacks.

Why this file exists
--------------------
Several tests guard themselves with::

    if not ARTIFACT.exists():
        pytest.skip("... not generated")

That pattern is legitimate for genuinely expensive artifacts, but it fails
*silently* in the worst way: if the path is wrong (or the artifact was never
committed, or it lives in a git-ignored directory), the test skips forever and
the claim it was written to protect is unguarded.  This actually happened here:
:mod:`tests.test_ursell_reference_pinning` pointed at an untracked
``outputs/ursell_reproducibility_study.json``, so its two "the committed study
must still describe the committed code" tests had never executed.  The defect is
recorded in ``docs/ci_red_triage_m3314.md`` §3 and closed in
``docs/m3487/dead_skip_gate_audit.md``.

This module closes the *class* of defect, not just the instance:

* :func:`test_every_registered_artifact_gate_is_live` checks every artifact
  named below: it must exist **and** be tracked by git, so an artifact living
  only in a developer's tree (or in a git-ignored ``artifacts/`` / ``outputs/``
  directory) fails here instead of silently disabling a test;
* :func:`test_gate_registry_matches_the_test_suite` walks ``tests/*.py`` with the
  AST, collects every ``if not <expr>.exists(): ... skip()`` gate, and requires
  the gate label to appear below.  A *new* skip gate fails until it is audited
  and registered (and the reverse direction is checked, so the registry cannot
  rot into a list of gates that no longer exist);
* :func:`test_every_registered_artifact_is_parseable` rejects empty or truncated
  artifacts -- a gate on an empty file is a gate on nothing;
* :func:`test_the_old_dead_gate_would_be_caught` is the negative control: the
  pre-fix path is untracked and absent, so it could never have fired.

Cost: filesystem + git plumbing + JSON parsing only, no solver runs.
"""

from __future__ import annotations

import ast
import json
import pathlib
import subprocess

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
TESTS_DIR = REPO_ROOT / "tests"

#: ``(test file, label of the guarded path expression) -> committed artifact``.
#: Every entry was audited by hand: the artifact is committed, so the gate is
#: live and a missing file is a defect.
ARTIFACT_GATES: dict[tuple[str, str], str] = {
    ("test_finite_depth_energy_convergence.py", "ARTIFACT"): (
        "docs/m2498/outputs/finite_depth_energy_identity/convergence.json"
    ),
    ("test_finite_depth_envelope.py", "ARTIFACT"): "docs/finite_depth_envelope/benchmark.json",
    ("test_finite_depth_refinement.py", "ARTEFACT"): (
        "docs/m2952/outputs/finite_depth_refinement/benchmark.json"
    ),
    ("test_mixed_boundary.py", "ARTIFACT"): "docs/mixed_boundary/outputs/study.json",
    ("test_near_surface_band.py", "STUDY"): "docs/m3383/near_surface_band_study.json",
    ("test_three_dimensional_doc_pins.py", "pins.PINS_FILE"): "docs/three_dimensional_pins.json",
    ("test_ursell_quadrature_convergence.py", "summary_path"): "docs/ursell_quadrature_study/summary.json",
    ("test_ursell_quadrature_convergence.py", "raw_path"): "docs/ursell_quadrature_study/raw",
}

#: Artifacts that a test asserts on directly (a missing file fails the test instead
#: of skipping it).  They are listed here so the same existence/tracking/parseable
#: checks cover them.
DIRECT_ARTIFACTS: dict[str, str] = {
    "docs/ursell_reproducibility/ursell_reproducibility_study.json": (
        "tests/test_ursell_reference_pinning.py"
    ),
}

_SKIP_NAMES = {"skip", "skipif", "xfail"}
_EXISTENCE_PROBES = {"exists", "is_file"}


def _is_skip_call(node: ast.AST) -> bool:
    """True for ``pytest.skip(...)`` / ``skip(...)`` / ``pytest.xfail(...)``."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
    return name in _SKIP_NAMES


def _label_of(condition: ast.AST) -> str | None:
    """``not X.exists()`` -> ``X``; anything else -> None."""
    if not isinstance(condition, ast.UnaryOp) or not isinstance(condition.op, ast.Not):
        return None
    probe = condition.operand
    if not (
        isinstance(probe, ast.Call)
        and isinstance(probe.func, ast.Attribute)
        and probe.func.attr in _EXISTENCE_PROBES
    ):
        return None
    # ``pins.PINS_FILE`` matters, not just ``PINS_FILE``: the label must be
    # specific enough that nobody can register a different ``PINS_FILE``.
    return ast.unparse(probe.func.value)


def _gate_labels(module_path: pathlib.Path) -> set[str]:
    """Labels of every artifact skip gate, whether a statement or a ``skipif`` mark.

    Both spellings occur in this suite: ``if not ARTIFACT.exists(): pytest.skip``
    inside a test, and ``@pytest.mark.skipif(not ARTIFACT.exists(), reason=...)``
    on a module or a function.
    """
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    labels: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "skipif":
            if node.args:
                label = _label_of(node.args[0])
                if label:
                    labels.add(label)
            continue
        if not isinstance(node, ast.If) or not isinstance(node.test, ast.UnaryOp):
            continue
        if not isinstance(node.test.op, ast.Not):
            continue
        probe = node.test.operand
        if not (
            isinstance(probe, ast.Call)
            and isinstance(probe.func, ast.Attribute)
            and probe.func.attr in _EXISTENCE_PROBES
        ):
            continue
        if not any(_is_skip_call(inner) for stmt in node.body for inner in ast.walk(stmt)):
            continue
        label = _label_of(node.test)
        if label:
            labels.add(label)
    return labels


def test_gate_registry_matches_the_test_suite() -> None:
    """The AST scan and the registry above must name exactly the same gates."""
    actual: set[tuple[str, str]] = set()
    for module_path in sorted(TESTS_DIR.glob("*.py")):
        if module_path.name == pathlib.Path(__file__).name:
            continue
        actual |= {(module_path.name, label) for label in _gate_labels(module_path)}

    registered = set(ARTIFACT_GATES)
    unregistered = actual - registered
    stale = registered - actual
    assert not unregistered, (
        f"unregistered artifact skip gate(s): {sorted(unregistered)}. Add them to "
        "ARTIFACT_GATES after checking that the artifact is committed and that the "
        "gate can actually fire -- otherwise the guard may be dead."
    )
    assert not stale, f"ARTIFACT_GATES lists gate(s) that no longer exist: {sorted(stale)}."


def _tracked(rel_paths: list[str]) -> set[str]:
    out = subprocess.run(
        ["git", "ls-files", "--"] + rel_paths,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.returncode == 0, f"git ls-files failed: {out.stderr}"
    return {line.strip() for line in out.stdout.splitlines() if line.strip()}


def test_every_registered_artifact_gate_is_live() -> None:
    """Every gate resolves to a committed artifact, so it can actually run."""
    entries = [(test_file, label, rel) for (test_file, label), rel in ARTIFACT_GATES.items()]
    entries += [(owner, "direct assertion", rel) for rel, owner in DIRECT_ARTIFACTS.items()]
    tracked = _tracked(sorted(rel for _, _, rel in entries))
    for test_file, label, rel in sorted(entries):
        path = REPO_ROOT / rel
        assert path.exists(), (
            f"{test_file} skips unless {rel} exists, but it does not. Either commit the "
            "artifact (together with the command that regenerates it) or drop the gate."
        )
        committed = rel in tracked or any(
            entry.startswith(rel.rstrip("/") + "/") for entry in tracked
        )
        assert committed, (
            f"{rel} exists in the working tree but is not tracked by git, so the gate on "
            f"{label} in {test_file} is dead in CI. Move it out of the git-ignored directory."
        )


@pytest.mark.parametrize("rel", sorted({*ARTIFACT_GATES.values(), *DIRECT_ARTIFACTS}))
def test_every_registered_artifact_is_parseable(rel: str) -> None:
    """A gate on an empty or truncated artifact is a gate on nothing."""
    path = REPO_ROOT / rel
    if path.is_dir():
        assert any(p.is_file() for p in path.rglob("*")), f"{rel} is an empty directory"
        return
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload not in ({}, [], None), f"{rel} is empty"


def test_the_old_dead_gate_would_be_caught() -> None:
    """Negative control: the pre-fix gate could never have fired.

    ``tests/test_ursell_reference_pinning.py`` used to resolve its study to
    ``<repo>/outputs/ursell_reproducibility_study.json``; ``outputs/`` is not
    committed and not produced by CI, so both guarded tests skipped forever.
    """
    stale = REPO_ROOT / "outputs" / "ursell_reproducibility_study.json"
    assert not stale.exists(), "the untracked outputs/ directory reappeared"
    assert "outputs/ursell_reproducibility_study.json" not in _tracked(
        ["outputs/ursell_reproducibility_study.json"]
    )
    assert not any(
        rel.startswith("outputs/") for rel in [*ARTIFACT_GATES.values(), *DIRECT_ARTIFACTS]
    ), (
        "no registered gate may point into the git-ignored outputs/ directory"
    )


def test_committed_study_is_regenerated_by_the_example() -> None:
    """The committed artifact must name the script that regenerates it."""
    (rel,) = DIRECT_ARTIFACTS
    assert rel.startswith("docs/"), "the study must live under docs/ to be tracked"
    example = REPO_ROOT / "examples" / "ursell_reproducibility_study.py"
    assert example.is_file()
    assert 'ROOT / "docs" / "ursell_reproducibility"' in example.read_text(encoding="utf-8"), (
        "the example must keep writing into the tracked docs/ directory"
    )