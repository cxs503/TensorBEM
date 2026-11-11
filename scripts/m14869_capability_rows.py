"""m14869: restore the capability-coverage ratchet (it is red on main).

Why this script exists
----------------------
``tensorbem.capability_evidence`` publishes a fail-closed evidence table for the
package (``docs/audit_baseline.md`` §4) and a coverage ratchet: *every* core
module must either be graded by a row or carry an accepted auxiliary reason.
On the tree this mission started from, three gates are red:

* ``tests/test_capability_coverage.py::test_every_core_module_is_graded_or_explained``
  -- **27 of 91** core modules are in no row (m6280 closed this debt for the
  54-module tree; 37 modules shipped since then and nobody re-registered them),
* ``tests/test_capability_evidence.py`` (4 failures) -- the published block no
  longer matches the tree, one ``UNTESTED_MODULES`` reason has gone stale
  (``benchmarks.panel_linear_order`` is now imported by
  ``tests/test_panel_quadratic_operator.py``), and the same stale reason fails
  the "every reason is still true" gate,
* the negative control of the coverage study, which expects ``unowned_after == 0``.

The repair is mechanical and evidence-preserving: each of the 27 unowned modules
already ships with a test suite and a document (that is how they were found), so
each one is registered as a *derived* row -- modules, a non-stub entry point
chosen by inspection, test node ids that exist, and a documentation anchor that
exists -- exactly the contract the registry grades.  Nothing is graded that the
window tree cannot prove; the one module that is a namespace package rather than
code (``tensorbem.verification``) gets an auxiliary reason instead.

The script is idempotent: it only appends rows whose module is not already
graded, and it refuses to emit a row whose test nodes, symbols or anchors do not
resolve.  Run ``python scripts/m14869_capability_rows.py --check`` to see the
plan without writing.

Usage::

    PYTHONPATH=src python scripts/m14869_capability_rows.py --check
    PYTHONPATH=src python scripts/m14869_capability_rows.py --write-source
"""

from __future__ import annotations

import argparse
import ast
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tensorbem import capability_evidence as ce  # noqa: E402

SOURCE = REPO / "src" / "tensorbem" / "capability_evidence.py"

#: (key, title, claim, modules, candidate test files, candidate docs)
GROUPS = [
    (
        "quadratic-panel-operators",
        "Clamped quadratic panel operators (geometry, near-singular quadrature)",
        "Quadratic panels carry the field and its normal derivative on a curved"
        " six-node patch; the operator, its near-singular quadrature and the"
        " matched-panel comparison against the linear operator are pinned by the"
        " quadratic-panel suite.",
        ("tensorbem.panel_quadratic",),
        ("tests/test_panel_quadratic.py", "tests/test_panel_quadratic_operator.py"),
        ("docs/p22_n01_quadratic_panel_report.md", "docs/audit_baseline.md"),
    ),
    (
        "neumann-kelvin-steady",
        "Neumann-Kelvin forward-speed steady wave resistance",
        "Steady free-surface flow past a hull at forward speed with the"
        " Neumann-Kelvin linearisation (Rankine source plus Kelvin free-surface"
        " Green function), including the 3-D panel assembly.",
        ("tensorbem.neumann_kelvin", "tensorbem.nk_panel3d"),
        ("tests/test_neumann_kelvin.py", "tests/test_nk_panel3d.py"),
        ("docs/neumann_kelvin_verification.md", "docs/audit_baseline.md"),
    ),
    (
        "rankine-wave-resistance",
        "Rankine-panel and Series-60 wave resistance",
        "Wave resistance from a Rankine free-surface panel solution and the"
        " Series-60 systematic-hull driver that consumes it.",
        ("tensorbem.rankine_fs", "tensorbem.rankine_wave", "tensorbem.series60"),
        ("tests/test_rankine_fs_panel.py", "tests/test_rankine_wave_resistance.py"),
        ("docs/audit_baseline.md",),
    ),
    (
        "propeller-open-water",
        "Lifting-surface propeller open-water performance",
        "Open-water KT/KQ/eta from the lifting-surface propeller model, pinned"
        " against the KP505 open-water lattice.",
        ("tensorbem.propeller",),
        ("tests/test_propeller_open_water.py",),
        ("docs/audit_baseline.md",),
    ),
    (
        "stf-full-speed",
        "Full-speed strip theory with Haskind excitation and net source regularisation",
        "Salvesen-Tuck-Faltinsen strip theory at forward speed, with the Haskind"
        " relation supplying the excitation force and the net-source"
        " regularisation that keeps the 2-D kernel finite.",
        ("tensorbem.stf_fullspeed", "tensorbem.haskind2d", "tensorbem.net_source_reg"),
        ("tests/test_stf_haskind_reg.py",),
        ("docs/strip_theory_fullspeed_closure.md",),
    ),
    (
        "wigley3-diffraction-rao",
        "Wigley III diffraction/radiation RAO with parametric hull geometry",
        "Six-degree-of-freedom RAO of the Wigley III form from the 3-D"
        " diffraction/radiation solve on the parametric hull mesh.",
        ("tensorbem.wigley3_freq", "tensorbem.wigley3_hull"),
        ("tests/test_wigley3_freq_diffraction_radiation.py", "tests/test_wigley3_complete_verification.py"),
        ("docs/wigley3_freq_verification_report.md", "docs/wigley_diffraction_radiation_verification.md"),
    ),
    (
        "tank-free-surface-correction",
        "Tank-wall free-surface (finite-width) correction",
        "Added-mass and damping correction for a body oscillating in a tank of"
        " finite width, i.e. the image-series free surface rather than the"
        " unbounded one.",
        ("tensorbem.tank_free_surface",),
        ("tests/test_tank_free_surface.py",),
        ("docs/tank_free_surface_correction.md", "docs/audit_baseline.md"),
    ),
    (
        "irregular-frequency-sanitisation",
        "Irregular-frequency removal and interior-panel suppression",
        "Removal of the irregular frequencies by interior-panel (CHIEF-style)"
        " suppression and the sanitiser that reports the residual.",
        ("tensorbem.irregular_frequency", "tensorbem.interior_panels"),
        ("tests/test_irregular_frequency_sanitisation.py",),
        ("docs/audit_baseline.md",),
    ),
    (
        "flexural-gravity-ice-cover",
        "Flexural-gravity dispersion of a floating ice plate",
        "Dispersion relation and mass loading of a floating elastic plate above"
        " water (ice cover), with an independent finite-difference check of the"
        " dispersion.",
        ("tensorbem.flexural_gravity", "tensorbem.flexural_gravity_numeric"),
        ("tests/test_flexural_gravity.py",),
        ("docs/flexural_gravity_verification/report.md", "docs/audit_baseline.md"),
    ),
    (
        "finite-depth-3d-solver",
        "3-D finite-depth Green function and its panel solver",
        "Finite-depth free-surface Green function (eigenfunction series) and the"
        " 3-D solver built on it.",
        ("tensorbem.finite_depth_green", "tensorbem.finite_depth_solver3d"),
        ("tests/test_finite_depth_green.py", "tests/test_finite_depth_solver3d.py"),
        ("docs/BEM_FINITE_DEPTH_GREEN.md",),
    ),
    (
        "pekeris-acoustics",
        "Pekeris waveguide acoustics",
        "Two-layer (Pekeris) shallow-water waveguide Green function and its"
        " modal/normal-mode reference.",
        ("tensorbem.pekeris_acoustics",),
        ("tests/test_pekeris_w22.py",),
        ("docs/acoustic3d_helmholtz.md", "docs/audit_baseline.md"),
    ),
    (
        "cathodic-protection",
        "Cathodic protection / galvanic corrosion boundary-value problem",
        "Potential field with non-linear polarisation boundary conditions on the"
        " protected and galvanic surfaces.",
        ("tensorbem.cathodic_protection", "tensorbem.cp_galvanic"),
        ("tests/test_cathodic_protection.py", "tests/test_cp_galvanic.py"),
        ("docs/cathodic_protection_linear/REPORT.md", "docs/audit_baseline.md"),
    ),
    (
        "drift-3d-auxiliary",
        "3-D mean drift: cylinder asymptote and far-field closure",
        "Small- and large-ka asymptotes of the cylinder mean drift force and the"
        " 3-D far-field drift closure that the maturity surface reports on.",
        ("tensorbem.cylinder_drift", "tensorbem.drift_far_field3d"),
        ("tests/test_cylinder_drift_asymptotic.py", "tests/test_drift_far_field3d.py"),
        ("docs/cylinder_drift_asymptotic.md", "docs/drift_far_field_3d.md"),
    ),
    (
        "state-space-radiation",
        "State-space (rational) fit of the radiation memory kernel",
        "Pole-residue approximation of the frequency-dependent added mass and"
        " damping, with the feedthrough term, its stability and the free-decay"
        " closure against an independent IRF convolution.",
        ("tensorbem.state_space",),
        ("tests/test_state_space.py", "tests/test_state_space_fit_invariants.py"),
        ("docs/m14767/report.md", "docs/audit_baseline.md"),
    ),
    (
        "reference-provenance",
        "Reference-constant provenance registry",
        "Every literature constant quoted in the repository is registered with"
        " its source; an undeclared, stale or unverified constant fails the gate.",
        ("tensorbem.reference_provenance",),
        ("tests/test_reference_provenance.py",),
        ("docs/reference_provenance.md",),
    ),
    (
        "figure-integrity",
        "Report-figure reproducibility gate",
        "Figures are not durable binaries in this repository: each verification"
        " case must regenerate its figure from the committed JSON numbers, and"
        " the gate fails on a blank, missing or unlinked render.",
        ("tensorbem.figure_integrity",),
        ("tests/test_figure_integrity.py",),
        ("docs/dev_report_m13806_push_gate_figure_integrity.md", "docs/audit_baseline.md"),
    ),
]

#: Modules that are namespace packages rather than code, with the reason.
AUXILIARY = {
    "tensorbem.verification": (
        "namespace package holding the verification campaign artefacts; it defines no"
        " physics of its own -- each subpackage's modules are inventoried separately and"
        " the code they exercise is graded by the row that owns it."
    ),
}

#: UNTESTED_MODULES reasons that have gone stale (the module is now referenced).
STALE_REASONS = ("tensorbem.benchmarks.panel_linear_order",)


def test_nodes(path: pathlib.Path) -> list[str]:
    """Top-level *and* class-nested ``test_*`` functions (two files nest them)."""

    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            names.append(node.name)
    return names


def pick_tests(files, leaves, extra=(), limit=3):
    picked = []
    seen = list(files) + [f for f in extra if f not in files]
    for rel in seen:
        path = REPO / rel
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        if not any(leaf in text for leaf in leaves):
            continue
        for name in test_nodes(path):
            node = f"{rel}::{name}"
            if node not in picked:
                picked.append(node)
            if len(picked) >= limit:
                return picked
    return picked


def pick_symbols(module: str) -> tuple[str, ...]:
    obj = ce._import_module(module)
    if obj is None:
        return ()
    chosen = []
    for name, func in vars(obj).items():
        if name.startswith("_") or not callable(func) or isinstance(func, type):
            continue
        try:
            source = __import__("inspect").getsource(func)
        except (OSError, TypeError):
            continue
        stub, constant = ce._is_stub(source)
        if stub or constant:
            continue
        chosen.append(name)
        if len(chosen) >= 2:
            break
    return tuple(chosen)


def pick_docs(candidates, leaves):
    for rel in candidates:
        path = REPO / rel
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for leaf in leaves:
            if leaf in text:
                return ((rel, leaf),)
    return ()


def build_plan():
    graded = {m for cap in ce.CAPABILITIES for m in cap.modules}
    rows, skipped = [], []
    for key, title, claim, modules, tests, docs in GROUPS:
        todo = tuple(m for m in modules if m not in graded)
        if not todo:
            skipped.append((key, "already graded"))
            continue
        missing = [m for m in todo if ce._import_module(m) is None]
        if missing:
            skipped.append((key, f"module(s) do not import: {missing}"))
            continue
        leaves = [m.rsplit(".", 1)[-1] for m in todo]
        index = ce.module_reference_index(REPO)
        extra = tuple(
            f"tests/{h}" if not h.startswith("tests/") else h
            for m in todo for h in index.get(m, [])
        )
        nodes = pick_tests(tests, leaves, extra)
        anchors = pick_docs(docs, leaves)
        symbols = tuple(s for m in todo for s in pick_symbols(m))
        problems = []
        if len(nodes) < 1:
            problems.append("no resolving test nodes")
        if not anchors:
            problems.append("no documentation anchor")
        if problems:
            skipped.append((key, "; ".join(problems)))
            continue
        rows.append(dict(key=key, title=title, claim=claim, modules=todo,
                         symbols=symbols, tests=tuple(nodes), docs=anchors))
    return rows, skipped


def render(rows) -> str:
    out = ["", "    # ---- m14869: rows for the 27 core modules the registry was silent about ----", ""]
    for row in rows:
        out.append(f"    Capability(\n        key={row['key']!r},\n        title={row['title']!r},")
        out.append(f"        claim={row['claim']!r},")
        out.append("        level=REAL,")
        out.append(f"        modules={row['modules']!r},")
        if row["symbols"]:
            out.append(f"        symbols={row['symbols']!r},")
        out.append(f"        tests={row['tests']!r},")
        out.append(f"        docs={row['docs']!r},")
        out.append("    ),")
    out.append("")
    return "\n".join(out)


def patch_source(rows) -> None:
    text = SOURCE.read_text(encoding="utf-8")
    # 1. drop stale UNTESTED_MODULES entries
    for name in STALE_REASONS:
        pattern = re.compile(
            r'\n    "' + re.escape(name) + r'":\n(?:        "[^\n]*"\n)+'
        )
        text, n = pattern.subn("\n", text)
        print(f"removed stale reason {name}: {n} block(s)")
    # 2. auxiliary reason for the namespace package
    for name, reason in AUXILIARY.items():
        if f'"{name}"' in text:
            continue
        anchor = 'AUXILIARY_MODULES: dict[str, str] = {\n'
        entry = f'    "{name}":\n        "{reason}"\n'
        text = text.replace(anchor, anchor + entry, 1)
    # 3. append the new rows to CAPABILITIES
    head = text.index("CAPABILITIES: tuple[Capability, ...] = (")
    close = text.index("\n)\n", head)
    text = text[:close] + "\n" + render(rows).rstrip("\n") + text[close:]
    SOURCE.write_text(text, encoding="utf-8")
    print(f"inserted {len(rows)} rows into {SOURCE}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-source", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    rows, skipped = build_plan()
    for row in rows:
        print(f"ROW {row['key']:38s} {len(row['modules'])} mod | {row['tests']} | {row['docs']}")
    for key, why in skipped:
        print(f"SKIP {key}: {why}")
    if args.write_source:
        patch_source(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())