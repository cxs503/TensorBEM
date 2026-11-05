"""Mutation audit: can the TensorBEM verification suite actually detect a defect?

The package ships several hundred assertions.  Passing them shows the code agrees
with its references *on the current commit*; it says nothing about whether the
assertions would notice a wrong kernel, a sign flip or a dropped factor.  A test
that cannot fail is not evidence.  This module answers that question directly by
seeding small, syntactically valid defects ("mutants") into the numerical core
and running the mapped unit tests against each mutant in an isolated source tree:

* a mutant that makes a unit test fail is *killed* -- the suite discriminates;
* a mutant that leaves every mapped test green is a *survivor* -- a blind spot,
  reported with its enclosing function so it can be triaged.

Mutants are generated from the AST rather than by string substitution, so every
mutant is a well-formed edit of an exactly located node, and the edit is written
back as a minimal textual diff (the rest of the file keeps its formatting).

Isolation matters here: this project is often checked out in several worktrees at
once and ``PYTHONPATH`` may already point at another copy of ``tensorbem``.  The
runner therefore copies ``src/tensorbem`` into a private tree, puts *only* that
tree on ``PYTHONPATH``, and refuses to run if the imported package does not come
from the shadow tree.

Run as ``python -m tensorbem.benchmarks.mutation_audit --out DIR``.
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "m2582" / "outputs" / "mutation_audit"

#: Comparison flips keep the operator valid for the same operand types.
COMPARE_FLIP = {
    ast.Lt: ast.LtE,
    ast.LtE: ast.Lt,
    ast.Gt: ast.GtE,
    ast.GtE: ast.Gt,
    ast.In: ast.NotIn,
    ast.NotIn: ast.In,
}

#: Arithmetic swaps.  Integer division mutants usually raise instead of drifting,
#: which still counts as a kill (the suite noticed), but they are reported apart.
BINOP_FLIP = {
    ast.Add: ast.Sub,
    ast.Sub: ast.Add,
    ast.Mult: ast.Div,
    ast.Div: ast.Mult,
    ast.Pow: ast.Mult,
}


@dataclass(frozen=True)
class Target:
    """A mutated module and the unit tests that are supposed to police it."""

    module: str
    source: Path
    unit_tests: tuple[str, ...]


TARGETS: tuple[Target, ...] = (
    Target(
        "tensorbem.kernels",
        SRC_ROOT / "tensorbem" / "kernels.py",
        ("tests/test_kernels.py",),
    ),
    Target(
        "tensorbem.free_surface",
        SRC_ROOT / "tensorbem" / "free_surface.py",
        ("tests/test_free_surface_kernel.py",),
    ),
    Target(
        "tensorbem.hydrostatic",
        SRC_ROOT / "tensorbem" / "hydrostatic.py",
        ("tests/test_hydrostatic_analytic_benchmark.py",),
    ),
    Target(
        "tensorbem.solver",
        SRC_ROOT / "tensorbem" / "solver.py",
        ("tests/test_solver.py",),
    ),
    Target(
        "tensorbem.finite_depth",
        SRC_ROOT / "tensorbem" / "finite_depth.py",
        ("tests/test_finite_depth.py",),
    ),
    # --- wave-load chain, added by m3108 -------------------------------------
    # The five modules above are the Laplace/free-surface *kernels*.  The modules
    # below are what actually turns those kernels into the numbers the package
    # reports: the 3-D Kelvin single-layer integrals, the wave-exciting pressure,
    # the Helmholtz/MacCamy-Fuchs diffraction solve and the RAO assembly.  They
    # were never mutation-audited, so "does the suite notice a dropped factor
    # here?" was an open question.
    Target(
        "tensorbem.kernels3d",
        SRC_ROOT / "tensorbem" / "kernels3d.py",
        ("tests/test_kernels3d.py", "tests/test_kernels3d_gradients.py"),
    ),
    Target(
        "tensorbem.pressure",
        SRC_ROOT / "tensorbem" / "pressure.py",
        ("tests/test_pressure.py",),
    ),
    Target(
        "tensorbem.helmholtz",
        SRC_ROOT / "tensorbem" / "helmholtz.py",
        ("tests/test_helmholtz_maccamy_fuchs.py",),
    ),
    Target(
        "tensorbem.rao",
        SRC_ROOT / "tensorbem" / "rao.py",
        ("tests/test_rao_verification.py",),
    ),
)


@dataclass(frozen=True)
class Mutant:
    """One syntactically valid defect, located by exact source span."""

    module: str
    function: str
    lineno: int
    col: int
    operator: str
    original: str
    replacement: str

    @property
    def key(self) -> str:
        return f"{self.module}:{self.lineno}:{self.col}:{self.operator}"

    @property
    def summary(self) -> str:
        return f"{self.operator} in {self.function}: {self.original} -> {self.replacement}"


def _enclosing_functions(tree: ast.AST) -> dict[int, str]:
    """Map id(node) -> dotted function name for every node inside a def."""
    names: dict[int, str] = {}

    def visit(node: ast.AST, qualname: str) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            qualname = f"{qualname}.{node.name}" if qualname else node.name
        names[id(node)] = qualname or "<module>"
        for child in ast.iter_child_nodes(node):
            visit(child, qualname)

    visit(tree, "")
    return names


def _span_source(lines: list[str], node: ast.AST) -> str | None:
    """Return the exact source text of ``node`` if it lies on a single line."""
    start_line = getattr(node, "lineno", None)
    end_line = getattr(node, "end_lineno", None)
    if start_line is None or end_line is None or start_line != end_line:
        return None
    start_col = node.col_offset
    end_col = node.end_col_offset
    if end_col is None:
        return None
    return lines[start_line - 1][start_col:end_col]


def _skip_constant(node: ast.AST, parents: dict[int, ast.AST]) -> bool:
    """Constants inside subscripts/slices are index wiring, not physics."""
    child = node
    parent = parents.get(id(node))
    while parent is not None:
        if isinstance(parent, ast.Subscript):
            return True
        if isinstance(parent, ast.Index):  # pragma: no cover - Python <3.9
            return True
        child, parent = parent, parents.get(id(parent))
    return False


def _mutate_node(node: ast.AST, operator: str) -> ast.AST | None:
    """Return a mutated deep copy of ``node``, or None if the operator is n/a."""
    mutant = copy.deepcopy(node)
    if operator == "flip_compare":
        if not isinstance(mutant, ast.Compare) or len(mutant.ops) != 1:
            return None
        new_op = COMPARE_FLIP.get(type(mutant.ops[0]))
        if new_op is None:
            return None
        mutant.ops = [new_op()]
        return mutant
    if operator == "swap_binop":
        if not isinstance(mutant, ast.BinOp):
            return None
        new_op = BINOP_FLIP.get(type(mutant.op))
        if new_op is None:
            return None
        mutant.op = new_op()
        return mutant
    if operator in {"shift_constant", "negate_constant"}:
        if not isinstance(mutant, ast.Constant):
            return None
        if isinstance(mutant.value, bool) or not isinstance(mutant.value, (int, float)):
            return None
        if operator == "negate_constant":
            mutant.value = -mutant.value
        elif isinstance(mutant.value, int):
            mutant.value = mutant.value + 1
        else:
            mutant.value = 1.0 if abs(mutant.value) < 1e-12 else mutant.value * 2.0
        return mutant
    raise ValueError(f"unknown mutation operator {operator!r}")


def enumerate_mutants(module: str, source: str, path: Path) -> list[Mutant]:
    """All single-node mutants of ``source`` that survive a round-trip check."""
    tree = ast.parse(source)
    names = _enclosing_functions(tree)
    parents = {id(child): node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    lines = source.splitlines(keepends=True)

    out: list[Mutant] = []
    seen: set[tuple[int, int, str]] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                continue
            if _skip_constant(node, parents):
                continue
            operators = ("shift_constant", "negate_constant")
        elif isinstance(node, ast.BinOp):
            operators = ("swap_binop",)
        elif isinstance(node, ast.Compare):
            operators = ("flip_compare",)
        else:
            continue

        for operator in operators:
            mutated = _mutate_node(node, operator)
            if mutated is None:
                continue
            original = _span_source(lines, node)
            if original is None:
                continue
            try:
                if ast.dump(ast.parse(original, mode="eval").body) != ast.dump(node):
                    # The slice is not a faithful parse of this node; skip
                    # rather than risk a mismatched textual edit.
                    continue
                replacement = ast.unparse(ast.fix_missing_locations(mutated))
            except SyntaxError:
                continue
            if replacement == original:
                continue
            spot = (node.lineno, node.col_offset, operator)
            if spot in seen:
                continue
            seen.add(spot)
            out.append(
                Mutant(
                    module=module,
                    function=names.get(id(node), "<module>"),
                    lineno=node.lineno,
                    col=node.col_offset,
                    operator=operator,
                    original=original,
                    replacement=replacement,
                )
            )
    out.sort(key=lambda m: (m.lineno, m.col, m.operator))
    return out


def apply_mutant(source: str, mutant: Mutant) -> str:
    """Write ``mutant`` back into ``source`` as a minimal single-line edit."""
    lines = source.splitlines(keepends=True)
    line = lines[mutant.lineno - 1]
    start = mutant.col
    end = start + len(mutant.original)
    if line[start:end] != mutant.original:
        raise ValueError(f"stale mutant {mutant.key}: source slice moved")
    lines[mutant.lineno - 1] = line[:start] + mutant.replacement + line[end:]
    mutated = "".join(lines)
    ast.parse(mutated)  # a mutant that does not parse is not a valid experiment
    return mutated


#: Test files too slow to sit in an automatic re-check family (>100 s each).
SLOW_TESTS = frozenset({"tests/test_analytic_references.py"})

_FAMILY_CACHE: dict[str, list[str]] = {}


def sibling_tests(function: str, limit: int = 8) -> tuple[str, ...]:
    """Test files that name ``function`` -- a cheap, automatic re-check family.

    A survivor against the mapped unit tests is only a blind spot if the rest of
    the suite that *talks about* the mutated function also misses it.  Matching
    is textual on the function name, which is deliberately conservative: it can
    over-select (extra tests only make the verdict harsher), never under-select
    the closest tests.
    """
    if function in _FAMILY_CACHE:
        return tuple(_FAMILY_CACHE[function])
    name = function.split(".")[-1]
    hits: list[str] = []
    if name and name != "<module>" and len(name) > 2:
        for path in sorted((REPO_ROOT / "tests").glob("test_*.py")):
            rel = f"tests/{path.name}"
            if rel in SLOW_TESTS:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if name in text:
                hits.append(rel)
    _FAMILY_CACHE[function] = hits[:limit]
    return tuple(hits[:limit])


def _failing_test(output: str) -> str | None:
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith("FAILED ") or stripped.startswith("ERROR "):
            return stripped.split(" ", 1)[1].split(" - ")[0]
    return None


def run_tests(shadow_src: Path, tests: tuple[str, ...], timeout: float) -> tuple[int, str]:
    env = {
        "PYTHONPATH": str(shadow_src),
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "HOME": os.environ.get("HOME", "/root"),
        # A sweep runs many short solves concurrently; without this each one
        # would spin up its own BLAS pool and swamp the machine.
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        *tests,
        "-q",
        "--no-header",
        "--tb=no",
        "-x",
        "-rf",
        "-p",
        "no:cacheprovider",
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return 124, "TIMEOUT"
    return proc.returncode, proc.stdout + proc.stderr


def _classify(returncode: int, output: str) -> tuple[str, str | None]:
    """Map a pytest exit code onto a mutation verdict."""
    if returncode == 0:
        return "survived", None
    if returncode in {1, 124}:
        if returncode == 124:
            return "timeout", None
        return "killed", _failing_test(output)
    # 2 = interrupted, 3 = internal error, 4 = usage error, 5 = nothing collected.
    if "ImportError" in output or "SyntaxError" in output or returncode in {3, 4, 5}:
        return "invalid", None
    return "killed", _failing_test(output)


def _shadow_package(staging: Path) -> Path:
    shadow_src = staging / "src"
    shutil.copytree(
        SRC_ROOT / "tensorbem",
        shadow_src / "tensorbem",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    return shadow_src


def _check_isolation(shadow_src: Path) -> str:
    """Fail fast if ``tensorbem`` would still be imported from another worktree."""
    env = dict(os.environ, PYTHONPATH=str(shadow_src), PYTHONDONTWRITEBYTECODE="1")
    proc = subprocess.run(
        [sys.executable, "-c", "import tensorbem, pathlib; print(tensorbem.__file__)"],
        env=env,
        capture_output=True,
        text=True,
    )
    resolved = Path(proc.stdout.strip()).resolve()
    if shadow_src.resolve() not in resolved.parents:
        raise RuntimeError(
            f"mutation audit would run against {resolved}, not the shadow tree {shadow_src}"
        )
    return str(resolved)


def _run_one(
    target: Target,
    mutant: Mutant,
    staging: Path,
    timeout: float,
    broaden: bool,
) -> dict:
    started = time.time()
    # A private tree per mutant: the sweep runs mutants concurrently, so a
    # shared work directory would race.
    work = Path(tempfile.mkdtemp(prefix="work-", dir=str(staging)))
    shadow_src = work / "src"
    shutil.copytree(staging / "src", shadow_src)

    original = target.source.read_text(encoding="utf-8")
    relative = target.source.relative_to(SRC_ROOT / "tensorbem")
    (shadow_src / "tensorbem" / relative).write_text(
        apply_mutant(original, mutant), encoding="utf-8"
    )

    stage1_rc, stage1_out = run_tests(shadow_src, target.unit_tests, timeout)
    verdict, failing = _classify(stage1_rc, stage1_out)
    stage = 1
    family: tuple[str, ...] = ()
    if verdict == "survived" and broaden:
        # A survival against the mapped unit tests only means the *closest*
        # tests miss it; re-check against every test file that names the
        # mutated function before calling it a blind spot.
        family = tuple(
            t for t in sibling_tests(mutant.function) if t not in target.unit_tests
        )
        if family:
            stage_rc, stage_out = run_tests(shadow_src, family, timeout)
            verdict, failing = _classify(stage_rc, stage_out)
            stage = 2
        else:
            stage = 3

    shutil.rmtree(work, ignore_errors=True)
    return {
        **asdict(mutant),
        "key": mutant.key,
        "summary": mutant.summary,
        "verdict": verdict,
        "killed_by": failing,
        "stage": stage,
        "family_tests": list(family),
        "seconds": round(time.time() - started, 2),
    }


def sweep(
    targets: tuple[Target, ...],
    max_per_module: int,
    seed: int,
    timeout: float,
    broaden: bool,
    jobs: int,
    out_dir: Path,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    plan: list[tuple[Target, list[Mutant]]] = []
    for target in targets:
        source = target.source.read_text(encoding="utf-8")
        mutants = enumerate_mutants(target.module, source, target.source)
        if max_per_module and len(mutants) > max_per_module:
            mutants = sorted(rng.sample(mutants, max_per_module), key=lambda m: (m.lineno, m.col))
        plan.append((target, mutants))
        print(f"{target.module}: {len(mutants)} mutants selected", flush=True)

    staging = Path(tempfile.mkdtemp(prefix="tensorbem-mutation-"))
    results: list[dict] = []
    try:
        shadow_src = _shadow_package(staging)
        isolated = _check_isolation(shadow_src)
        rounds = [(target, mutant) for target, mutants in plan for mutant in mutants]
        with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
            futures = [
                pool.submit(_run_one, target, mutant, staging, timeout, broaden)
                for target, mutant in rounds
            ]
            for index, future in enumerate(futures, start=1):
                row = future.result()
                results.append(row)
                print(
                    f"[{index}/{len(futures)}] {row['verdict']:<8} {row['key']} "
                    f"{row['summary'][:88]}",
                    flush=True,
                )
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    modules: dict[str, dict] = {}
    for target, mutants in plan:
        rows = [r for r in results if r["module"] == target.module]
        modules[target.module] = _score(rows, target.unit_tests)

    payload = {
        "generated_by": "tensorbem.benchmarks.mutation_audit",
        "seed": seed,
        "max_per_module": max_per_module,
        "jobs": jobs,
        "timeout_seconds": timeout,
        "isolated_package": isolated,
        "broaden": broaden,
        "modules": modules,
        "mutants": results,
        "overall": _score(results, ()),
    }
    (out_dir / "mutation_report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    (out_dir / "mutation_summary.md").write_text(render_markdown(payload), encoding="utf-8")
    return payload


def _score(rows: list[dict], tests: tuple[str, ...]) -> dict:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["verdict"]] = counts.get(row["verdict"], 0) + 1
    scored = counts.get("killed", 0) + counts.get("survived", 0) + counts.get("timeout", 0)
    killed = counts.get("killed", 0)
    return {
        "mutants": len(rows),
        "killed": killed,
        "survived": counts.get("survived", 0),
        "invalid": counts.get("invalid", 0),
        "timeout": counts.get("timeout", 0),
        "mutation_score": (killed / scored) if scored else None,
        "tests": list(tests),
        "survivors": [r["summary"] for r in rows if r["verdict"] == "survived"],
    }


def render_markdown(payload: dict) -> str:
    lines = ["# Mutation audit summary", ""]
    lines.append(
        "| module | mutants | killed | survived | invalid | mutation score |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    for module, stats in payload["modules"].items():
        score = stats["mutation_score"]
        shown = "n/a" if score is None else f"{score:.3f}"
        lines.append(
            f"| `{module}` | {stats['mutants']} | {stats['killed']} | "
            f"{stats['survived']} | {stats['invalid']} | {shown} |"
        )
    overall = payload["overall"]
    score = overall["mutation_score"]
    lines.append("")
    lines.append(
        f"Overall: {overall['killed']}/{overall['mutants']} killed, "
        f"{overall['survived']} survived, {overall['invalid']} invalid, "
        f"score = {'n/a' if score is None else f'{score:.3f}'}."
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-per-module", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument(
        "--module",
        action="append",
        default=None,
        help="restrict to these modules (repeatable)",
    )
    parser.add_argument("--no-broaden", action="store_true")
    parser.add_argument(
        "--extra-test",
        action="append",
        default=None,
        help="extra test file(s) every mutant is also judged against (repeatable); "
        "used to measure a back-filled suite",
    )
    args = parser.parse_args(argv)

    selected = tuple(
        t for t in TARGETS if args.module is None or t.module in set(args.module)
    )
    if args.extra_test:
        extra = tuple(args.extra_test)
        selected = tuple(
            Target(t.module, t.source, t.unit_tests + extra) for t in selected
        )
    if not selected:
        parser.error("no matching targets")
    broaden = not args.no_broaden
    payload = sweep(
        selected,
        max_per_module=args.max_per_module,
        seed=args.seed,
        timeout=args.timeout,
        broaden=broaden,
        jobs=args.jobs,
        out_dir=args.out,
    )
    print(render_markdown(payload))
    return payload


if __name__ == "__main__":  # pragma: no cover
    main()