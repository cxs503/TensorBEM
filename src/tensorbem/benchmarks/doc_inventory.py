"""Executable inventory of the capability claims made in ``README.md``.

Why this module exists
----------------------
``README.md`` is the file every new reader -- and every automated research agent
in this ecosystem -- reads before deciding what to work on.  Its *Roadmap* and
*Numerical conventions* sections therefore make two kinds of statement:

* **shipped claims** ("X is implemented", a milestone marked *Done*), and
* **absence claims** ("Y remains the next increment", "Z is not implemented").

Both kinds rot in opposite directions, and both rot silently:

* a shipped claim rots when code is deleted, renamed, or never merged -- the
  README keeps promising a capability nobody can import;
* an absence claim rots when the capability *is* merged and the README is not
  edited -- and this is the expensive direction, because a fleet of agents
  reading the stale roadmap will redo work that is already shipped, or scope a
  task around a gap that no longer exists.

The second failure actually happened here.  On 2026-11-04 this audit found three
contradictions of the second kind inside ``README.md`` itself:

1. ``Numerical conventions`` said *"near-boundary near-singular quadrature is
   not implemented"* while ``solver3d.evaluate_surface_field_near_surface`` /
   ``evaluate_surface_velocity_near_surface`` and ``tensorbem.near_singular``
   had been shipped and documented (``docs/near_surface_field.md``);
2. roadmap item 5 still listed *"near-singular field quadrature"* among the
   remaining gaps;
3. roadmap item 6 listed the *3-D finite-depth John eigenseries Green function*
   under "Next, needs new physics" while item 6's own introductory sentence 24
   lines above already announced that very kernel as *"now implemented and
   verified"* (``tensorbem.finite_depth3d``, 26 test gates).

Nothing executable objected to any of this, because a sentence is not a test.

What this module is
-------------------
A curated table of the README's capability claims (:data:`CLAIMS`), each row
carrying the *evidence* that decides it:

* ``modules`` -- importable module paths that must exist for a shipped claim
  (and must **not** exist for an absence claim);
* ``test_files`` -- test files that must exist and contain at least one
  ``def test_`` for a shipped claim, so "shipped" means "guarded";
* ``docs`` -- report files that must exist;
* ``probes`` -- optional *numerical* proofs: a callable returning a measured
  error plus the limit it must beat, so the shipped claim carries a number and
  not just a filename;
* ``phrase`` -- for absence claims, the exact sentence fragment the README must
  still contain (the gap must really be stated, not merely implied);
* ``stale_phrases`` / ``stale_sources`` -- fragments that must **not** appear in
  the given documents once a claim is shipped.  These are the regression pins
  for the three defects above: re-introducing any of those sentences turns the
  gate red.

:func:`check` runs the table and returns one row per claim with ``PASS`` or
``DRIFT`` plus the reasons; ``python -m tensorbem.benchmarks.doc_inventory``
prints strict JSON, ``--markdown`` prints a table suitable for programmatic
injection into a report, and ``--strict`` turns any drift into a non-zero exit
status so a CI step or a reviewer can gate on it.

Provenance note (2026-11-04).  The README sentences were corrected in this same
change; the original wording is quoted in
``dev_report_m3203_readme_claim_inventory.md`` rather than silently dropped, and
the ``stale_phrases`` pins keep the old wording from coming back.

Scope and honesty
-----------------
The table is *curated*, not scraped: it covers every capability statement in
``README.md`` sections *Milestones/Roadmap* and *Numerical conventions* whose
status is decidable by evidence, but it does not attempt to parse arbitrary
prose.  A sentence that no row mentions is unchecked; the row count is reported
so the gap between "what the README says" and "what is pinned" stays visible.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

__all__ = [
    "CLAIMS",
    "Claim",
    "InventoryRow",
    "Probe",
    "check",
    "main",
    "render_markdown",
]

#: Repository root, resolved from this file (``src/tensorbem/benchmarks/``).
REPO = Path(__file__).resolve().parents[3]

#: Verdict strings.
PASS = "PASS"
DRIFT = "DRIFT"

#: Claim status strings.
SHIPPED = "shipped"
ABSENT = "absent"


@dataclass(frozen=True)
class Probe:
    """A numerical proof attached to a shipped claim.

    ``measure`` returns a scalar error; the probe passes when the comparison
    ``value <mode> limit`` holds (``mode`` is ``"<"`` or ``">"``).
    """

    name: str
    description: str
    measure: Callable[[], float]
    limit: float
    mode: str = "<"
    unit: str = ""

    def run(self) -> dict:
        try:
            value = float(self.measure())
        except Exception as exc:  # pragma: no cover - defensive, reported as DRIFT
            return {
                "name": self.name,
                "description": self.description,
                "unit": self.unit,
                "limit": self.limit,
                "mode": self.mode,
                "value": None,
                "passed": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        finite = math.isfinite(value)
        passed = finite and (
            value < self.limit if self.mode == "<" else value > self.limit
        )
        return {
            "name": self.name,
            "description": self.description,
            "unit": self.unit,
            "limit": self.limit,
            "mode": self.mode,
            "value": value if finite else None,
            "passed": bool(passed),
            "error": None if finite else "non-finite measurement",
        }


@dataclass(frozen=True)
class Claim:
    """One capability statement from the README, with its evidence."""

    key: str
    statement: str
    source: str
    status: str
    modules: tuple[str, ...] = ()
    test_files: tuple[str, ...] = ()
    docs: tuple[str, ...] = ()
    phrase: tuple[str, ...] = ()
    stale_phrases: tuple[str, ...] = ()
    stale_sources: tuple[str, ...] = ("README.md",)
    probes: tuple[Probe, ...] = ()
    note: str = ""


@dataclass
class InventoryRow:
    """Result of checking one claim."""

    key: str
    statement: str
    source: str
    status: str
    verdict: str
    evidence: list = field(default_factory=list)
    problems: list = field(default_factory=list)
    probes: list = field(default_factory=list)


# --------------------------------------------------------------------------
# probes -- numerical proofs for the claims that deserve more than a filename
# --------------------------------------------------------------------------


def _probe_finite_depth_deep_water_limit() -> float:
    """Max relative deviation of the finite-depth 3-D kernel from the Kelvin one.

    This is the documented point of ``docs/finite_depth3d.md`` section 3.3:
    ``nu = 0.6``, ``h = 128`` (``k0*h = 76.8``) and the five field points of the
    report script, which quote ``2.908e-7``.  Comparing at a *fixed physical*
    geometry matters: at ``k0*R >> 1`` the source and its mirror image nearly
    cancel, so the same comparison made at arbitrary rescaled points is
    cancellation-limited and meaningless as a kernel check.
    """
    import numpy as np

    from tensorbem.finite_depth3d import green_function_3d
    from tensorbem.free_surface3d import green_function_3d as deep_green_function

    nu, depth = 0.6, 128.0
    points = np.array(
        [
            [0.5, 0.0, -1.0],
            [2.0, 0.0, 0.0],
            [3.0, 1.0, -1.2],
            [6.0, 0.0, -0.5],
            [9.0, 0.0, -1.0],
        ]
    )
    source = np.array([0.0, 0.0, -2.0])
    finite = green_function_3d(points, source, nu, depth)
    deep = deep_green_function(points, source, nu)
    return float(np.max(np.abs(finite - deep)) / np.max(np.abs(deep)))


def _probe_finite_depth_free_surface_condition() -> float:
    """Scaled residual ``|dG/dz - nu*G| / (nu*|G|)`` on ``z = 0``.

    Documented setting of ``docs/finite_depth3d.md`` section 3.2: ``nu = 0.6``,
    ``h = 40`` (quoted residual ``2.430e-12``), source at ``z = -h/2``, field
    points along ``z = 0`` with ``0.3 <= x <= 9``.
    """
    import numpy as np

    from tensorbem.finite_depth3d import green_function_3d, green_gradient_z_3d

    nu, depth = 0.6, 40.0
    source = np.array([0.0, 0.0, -0.5 * depth])
    points = np.stack([np.linspace(0.3, 9.0, 14), np.zeros(14), np.zeros(14)], axis=-1)
    value = green_function_3d(points, source, nu, depth)
    derivative = green_gradient_z_3d(points, source, nu, depth)
    return float(np.max(np.abs(derivative - nu * value)) / np.max(np.abs(nu * value)))


def _probe_near_singular_clearance() -> float:
    """Adaptive single-layer quadrature error at ``1e-3`` panel sizes.

    The comparison is against a converged instance of the same rule (order 14,
    ``split_ratio = 2``, ``max_level = 32``), so the number isolates quadrature
    error from any discretisation error.  ``docs/near_surface_field.md`` quotes
    machine-level agreement for this setting, i.e. the error stops depending on
    the clearance; the fixed-order Duffy rule is ``8.3e-2`` here.
    """
    import numpy as np

    from tensorbem.near_singular import single_layer_integrals

    v0 = np.array([[0.0, 0.0, 0.0]])
    v1 = np.array([[0.1, 0.0, 0.0]])
    v2 = np.array([[0.0, 0.1, 0.0]])
    target = np.array([0.03, 0.03, 1.0e-4])  # 1e-4 = 1e-3 of the panel size
    reference = single_layer_integrals(
        target, v0, v1, v2, order=14, split_ratio=2.0, max_level=32
    )
    measured = single_layer_integrals(
        target, v0, v1, v2, order=8, split_ratio=1.0, max_level=24
    )
    return float(
        abs(np.squeeze(measured) - np.squeeze(reference)) / abs(np.squeeze(reference))
    )


def _lifting3d_lift_slope_error() -> float:
    """Relative error of the vortex-lattice lift slope at ``AR = 40``, 128 span panels.

    The reference is Prandtl's elliptic lifting-line slope ``2*pi*AR/(AR+2)``; this
    is the headline gate of ``docs/lifting3d.md`` (0.33% there, recomputed here so
    the claim carries a number and not only a filename).
    """
    import math

    from tensorbem.lifting3d import LiftingWing, build_lattice, lifting_line_lift_slope, loads, solve_circulation

    span = 8.0
    aspect_ratio = 40.0
    alpha = math.radians(4.0)
    wing = LiftingWing(
        span=span,
        planform="elliptic",
        chord_root=4.0 * span / (math.pi * aspect_ratio),
        alpha=alpha,
    )
    lattice = build_lattice(wing, 128)
    result = loads(lattice, solve_circulation(lattice, 1.0))
    measured = result["CL"] / alpha
    return float(abs(measured / lifting_line_lift_slope(aspect_ratio) - 1.0))


def _registered_number_count() -> float:
    """How many documentation numbers the declared-number registry guards."""
    from tensorbem.benchmarks.declared_numbers import DECLARED_NUMBERS

    return float(len(DECLARED_NUMBERS))


# --------------------------------------------------------------------------
# the curated claim table
# --------------------------------------------------------------------------

CLAIMS: tuple[Claim, ...] = (
    # ---------------------------------------------------------------- shipped
    Claim(
        key="exterior_neumann_2d",
        statement="2-D exterior Neumann rigid-body added masses (3x3) via the "
        "second-kind operator, against Lamb's closed forms",
        source="README.md (Milestone 5 item 1)",
        status=SHIPPED,
        modules=("tensorbem.solver",),
        test_files=(
            "tests/test_exterior_neumann_added_mass.py",
            "tests/test_exterior_solver.py",
        ),
    ),
    Claim(
        key="rigid_body_added_mass_3d",
        statement="3-D (6x6) exterior Neumann added mass / added moment of inertia",
        source="README.md (Milestone 5 item 1)",
        status=SHIPPED,
        modules=("tensorbem.solver3d",),
        test_files=("tests/test_exterior_neumann_added_mass.py",),
    ),
    Claim(
        key="ursell_2d_radiation",
        statement="2-D deep-water submerged-cylinder radiation against Ursell's "
        "multipole solution, with the near/far-field energy identity",
        source="README.md (Milestone 4); docs/marine_bem_benchmarks.md section 4.6",
        status=SHIPPED,
        modules=("tensorbem.free_surface_solver", "tensorbem.benchmarks.ursell_cylinder"),
        test_files=(
            "tests/test_ursell_benchmark.py",
            "tests/test_ursell_reference_pinning.py",
        ),
        docs=("docs/marine_bem_benchmarks.md",),
    ),
    Claim(
        key="dean_diffraction_haskind",
        statement="2-D diffraction, Dean's zero-reflection theorem and Haskind "
        "radiation-diffraction reciprocity",
        source="README.md (Milestone 5 item 2)",
        status=SHIPPED,
        modules=("tensorbem.benchmarks.dean_diffraction",),
        test_files=("tests/test_dean_diffraction.py",),
    ),
    Claim(
        key="helmholtz_maccamy_fuchs",
        statement="MacCamy-Fuchs pile diffraction with Schenck CHIEF "
        "irregular-frequency suppression",
        source="README.md (Milestone 5 item 3)",
        status=SHIPPED,
        modules=("tensorbem.helmholtz", "tensorbem.benchmarks.maccamy_fuchs"),
        test_files=(
            "tests/test_helmholtz_maccamy_fuchs.py",
            "tests/test_doc_numbers_maccamy_fuchs.py",
        ),
    ),
    Claim(
        key="kelvin_kernel_3d",
        statement="3-D deep-water Kelvin/Havelock/Delhommeau free-surface Green "
        "function with a free-surface residual below 1e-14",
        source="README.md (Milestone 6 item 1)",
        status=SHIPPED,
        modules=("tensorbem.free_surface3d",),
        test_files=(
            "tests/test_free_surface3d.py",
            "tests/test_free_surface_kernel.py",
        ),
    ),
    Claim(
        key="solver3d_6dof",
        statement="3-D 6-DOF radiation/diffraction with Kochin damping, Haskind "
        "excitation and lid-free interior CHIEF",
        source="README.md (Milestone 6 item 2)",
        status=SHIPPED,
        modules=("tensorbem.free_surface_solver3d",),
        test_files=(
            "tests/test_floating_hemisphere_3d.py",
            "tests/test_submerged_sphere_wave_3d.py",
        ),
    ),
    Claim(
        key="near_singular_field",
        statement="Target-adaptive near-singular field quadrature accepting "
        "arbitrary clearance (near-wall potential, velocity, pressure)",
        source="README.md (Numerical conventions, 3-D paragraph; roadmap item 5)",
        status=SHIPPED,
        modules=("tensorbem.near_singular", "tensorbem.solver3d"),
        test_files=(
            "tests/test_near_singular.py",
            "tests/test_near_surface_error_budget.py",
        ),
        docs=("docs/near_surface_field.md", "docs/near_surface_error_budget.md"),
        stale_phrases=(
            "near-boundary near-singular quadrature is not implemented",
            "near-singular field quadrature remain",
            "as the only remaining field-evaluation gap",
        ),
        stale_sources=("README.md", "docs/near_surface_field.md"),
        probes=(
            Probe(
                name="near_singular_clearance_1e3",
                description="adaptive single-layer error at 1e-3 panel sizes "
                "versus a converged instance of the same rule",
                measure=_probe_near_singular_clearance,
                limit=1.0e-10,
            ),
        ),
    ),
    Claim(
        key="finite_depth3d_kernel",
        statement="3-D finite-depth John eigenseries Green function (kernel) with "
        "deep-water limit and free-surface condition verified",
        source="README.md (Milestone 6 closing paragraph)",
        status=SHIPPED,
        modules=("tensorbem.finite_depth3d",),
        test_files=("tests/test_finite_depth3d.py",),
        docs=("docs/finite_depth3d.md",),
        stale_phrases=(
            "needs new physics:** finite-depth 3-D",
            "finite-depth 3-D John eigenseries Green function (Garrett",
        ),
        probes=(
            Probe(
                name="deep_water_limit_k0h77",
                description="max relative deviation from the Kelvin kernel at "
                "k0*h = 77",
                measure=_probe_finite_depth_deep_water_limit,
                limit=1.0e-5,
            ),
            Probe(
                name="free_surface_residual",
                description="scaled free-surface residual |dG/dz - nu*G|/(nu|G|) "
                "on z = 0",
                measure=_probe_finite_depth_free_surface_condition,
                limit=1.0e-10,
            ),
        ),
    ),
    Claim(
        key="finite_depth_2d",
        statement="2-D finite-depth free-surface radiation with the dispersion "
        "relation and energy identity verified",
        source="README.md (Milestone 4/6); docs/finite_depth_radiation.md",
        status=SHIPPED,
        modules=("tensorbem.finite_depth", "tensorbem.finite_depth_solver"),
        test_files=(
            "tests/test_finite_depth.py",
            "tests/test_finite_depth_solver.py",
            "tests/test_finite_depth_energy_convergence.py",
        ),
    ),
    Claim(
        key="symmetry3d",
        statement="Symmetry-plane reduction for y = 0 symmetric hulls",
        source="README.md (roadmap item 5)",
        status=SHIPPED,
        modules=("tensorbem.symmetry3d",),
        test_files=("tests/test_symmetry3d.py",),
        docs=("docs/symmetry3d.md",),
    ),
    Claim(
        key="hydrostatics_restoring",
        statement="Hydrostatic restoring matrix from the waterplane traced over "
        "every closed waterline loop",
        source="README.md (roadmap item 6)",
        status=SHIPPED,
        modules=("tensorbem.hydrostatic", "tensorbem.benchmarks.hydrostatic_benchmarks"),
        test_files=(
            "tests/test_hydrostatic_verification.py",
            "tests/test_waterplane_multi_loop.py",
        ),
        docs=("docs/hydrostatic_verification.md",),
    ),
    Claim(
        key="rao_6dof",
        statement="Frequency-domain 6-DOF RAO coupling with hydrostatic restoring, "
        "added mass/damping and excitation",
        source="README.md (roadmap item 6)",
        status=SHIPPED,
        modules=("tensorbem.rao",),
        test_files=(
            "tests/test_rao_verification.py",
            "tests/test_hydrostatic_rao.py",
        ),
    ),
    Claim(
        key="lifting_kutta_2d",
        statement="Steady 2-D lifting flow with Kutta closure",
        source="README.md (Milestone 6 closing paragraph; roadmap item 6)",
        status=SHIPPED,
        modules=("tensorbem.lifting",),
        test_files=(
            "tests/test_lifting_flow.py",
            "tests/test_lifting_foil.py",
        ),
        docs=("docs/lifting_kutta_closure.md",),
    ),
    Claim(
        key="causality_kkr",
        statement="Kramers-Kronig causality checks on the radiation coefficients",
        source="README.md / docs/marine_bem_benchmarks.md",
        status=SHIPPED,
        modules=("tensorbem.causality",),
        test_files=("tests/test_causality.py",),
    ),
    Claim(
        key="declared_number_registry",
        statement="Every headline number quoted in the documentation is "
        "recomputable and judged PASS/DRIFT",
        source="docs/classical_benchmarks.md sections 2, 3, 5.2; "
        "docs/marine_bem_benchmarks.md section 4.6",
        status=SHIPPED,
        modules=("tensorbem.benchmarks.declared_numbers",),
        test_files=("tests/test_declared_numbers.py",),
        probes=(
            Probe(
                name="registry_size",
                description="number of documentation numbers under registry guard",
                measure=_registered_number_count,
                limit=20.0,
                mode=">",
            ),
        ),
    ),
    # ---------------------------------------------------------------- absent
    Claim(
        key="finite_depth3d_bem_solver",
        statement="3-D finite-depth boundary-element solver (panel integration, "
        "layer jumps, Garrett's truncated dock) is NOT shipped",
        source="README.md (Milestone 6 closing paragraph)",
        status=ABSENT,
        modules=("tensorbem.finite_depth3d_solver",),
        phrase=("3-D finite-depth *boundary element solver*",),
        note="Absence claim: if the module ever appears, the README sentence must "
        "be rewritten in the same change.",
    ),
    Claim(
        key="forward_speed_rankine",
        statement="Forward-speed Rankine-source wave resistance is outside the "
        "current scope",
        source="README.md (Milestone 6 closing paragraph)",
        status=ABSENT,
        modules=("tensorbem.rankine", "tensorbem.wave_resistance"),
        phrase=("Forward-speed Rankine-source wave resistance",),
    ),
    Claim(
        key="lifting_3d",
        statement="Three-dimensional steady lifting surfaces: horseshoe vortex "
        "lattice verified against Prandtl/Munk lifting-line theory",
        source="README.md (Milestone 6 closing paragraph; roadmap item 6); "
        "docs/lifting3d.md",
        status=SHIPPED,
        modules=("tensorbem.lifting3d",),
        test_files=("tests/test_lifting3d.py",),
        docs=("docs/lifting3d.md",),
        probes=(
            Probe(
                name="lifting3d_lift_slope_ar40",
                description="vortex-lattice elliptic lift slope at AR=40, 128 span "
                "panels, relative error against Prandtl's 2*pi*AR/(AR+2)",
                measure=_lifting3d_lift_slope_error,
                limit=0.03,
                mode="<",
            ),
        ),
        stale_phrases=(
            "three-dimensional lifting wakes remain outside current scope",
            "three-dimensional lifting flows (the steady 2-D Kutta closure is shipped",
        ),
        note="Was an absence claim until 2026-11-05 (m4062); the module, the report "
        "and the tests arrived together with the README rewrite.",
    ),
)


# --------------------------------------------------------------------------
# the checker
# --------------------------------------------------------------------------


def _normalise(text: str) -> str:
    """Collapse whitespace so line wrapping cannot hide a phrase."""
    return re.sub(r"\s+", " ", text)


def _read_raw(
    root: Path, relative: str, overrides: dict | None = None
) -> str | None:
    """Raw file text (line structure preserved), or ``None`` when missing.

    ``overrides`` lets a caller substitute the content of a file without
    touching the working tree; the negative controls of the test suite use it to
    prove that a re-introduced stale sentence is actually detected.
    """
    if overrides is not None and relative in overrides:
        return overrides[relative]
    path = root / relative
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8")


def _read(
    root: Path, relative: str, overrides: dict | None = None
) -> str | None:
    """Whitespace-normalised file text, or ``None`` when the file is missing."""
    text = _read_raw(root, relative, overrides)
    return None if text is None else _normalise(text)


def _module_exists(dotted: str) -> bool:
    try:
        importlib.import_module(dotted)
    except Exception:
        return False
    return True


def _count_tests(root: Path, relative: str, overrides: dict | None = None) -> int:
    """Number of ``def test_`` definitions in a test file (0 when missing)."""
    text = _read_raw(root, relative, overrides)
    if text is None:
        return 0
    return len(re.findall(r"^\s*def test_", text, flags=re.MULTILINE))


def check(
    claims: Sequence[Claim] = CLAIMS,
    *,
    root: Path | None = None,
    overrides: dict | None = None,
) -> list[InventoryRow]:
    """Evaluate every claim against the working tree; return one row per claim.

    ``overrides`` maps repository-relative paths to replacement text, so a caller
    (the test suite, or a reviewer trying a reworded README) can evaluate a
    hypothetical tree without writing to it.
    """
    root = Path(root) if root is not None else REPO
    rows: list[InventoryRow] = []

    for claim in claims:
        row = InventoryRow(
            key=claim.key,
            statement=claim.statement,
            source=claim.source,
            status=claim.status,
            verdict=PASS,
        )
        if claim.status == SHIPPED:
            for dotted in claim.modules:
                if _module_exists(dotted):
                    row.evidence.append(f"module:{dotted}")
                else:
                    row.problems.append(f"module {dotted!r} is not importable")
            for relative in claim.test_files:
                count = _count_tests(root, relative, overrides)
                if count:
                    row.evidence.append(f"tests:{relative}({count})")
                else:
                    row.problems.append(
                        f"test file {relative!r} is missing or has no tests"
                    )
            for relative in claim.docs:
                if _read_raw(root, relative, overrides) is not None:
                    row.evidence.append(f"doc:{relative}")
                else:
                    row.problems.append(f"report {relative!r} is missing")
            for probe in claim.probes:
                result = probe.run()
                row.probes.append(result)
                if result["passed"]:
                    row.evidence.append(
                        f"probe:{probe.name}={result['value']:.3e}{probe.unit} "
                        f"{probe.mode} {probe.limit:.1e}"
                    )
                else:
                    detail = (
                        result["error"]
                        if result.get("error")
                        else f"{result['value']:.3e} !{probe.mode} {probe.limit:.1e}"
                    )
                    row.problems.append(f"probe {probe.name!r} failed: {detail}")
            for relative in claim.stale_sources:
                text = _read(root, relative, overrides)
                if text is None:
                    row.problems.append(f"stale-phrase source {relative!r} is missing")
                    continue
                normalised = _normalise(text).lower()
                for phrase in claim.stale_phrases:
                    if _normalise(phrase).lower() in normalised:
                        row.problems.append(
                            f"stale sentence still in {relative}: "
                            f"'{_normalise(phrase)}'"
                        )
        elif claim.status == ABSENT:
            readme = _read(root, "README.md", overrides)
            if readme is None:
                row.problems.append("README.md is missing")
            else:
                lowered = readme.lower()
                for phrase in claim.phrase:
                    if _normalise(phrase).lower() in lowered:
                        row.evidence.append(f"stated-gap:{_normalise(phrase)}")
                    else:
                        row.problems.append(
                            f"absence statement '{_normalise(phrase)}' not found in "
                            "README.md"
                        )
            for dotted in claim.modules:
                if _module_exists(dotted):
                    row.problems.append(
                        f"module {dotted!r} exists, so the documented gap is closed"
                    )
                else:
                    row.evidence.append(f"absent-module:{dotted}")
        else:  # pragma: no cover - guarded by the dataclass contract
            row.problems.append(f"unknown status {claim.status!r}")

        if row.problems:
            row.verdict = DRIFT
        rows.append(row)
    return rows


def summary_rows(rows: Iterable[InventoryRow]) -> list[dict]:
    """Compact JSON rows for a report or a ledger."""
    out = []
    for row in rows:
        out.append(
            {
                "key": row.key,
                "status": row.status,
                "verdict": row.verdict,
                "source": row.source,
                "evidence": row.evidence,
                "problems": row.problems,
                "probes": row.probes,
            }
        )
    return out


def render_markdown(rows: Sequence[InventoryRow]) -> str:
    """Markdown table of the inventory, for programmatic doc injection."""
    lines = [
        "| # | claim | status | verdict | evidence |",
        "|---:|---|---|---|---|",
    ]
    for index, row in enumerate(rows, start=1):
        evidence = "; ".join(row.evidence) if row.evidence else "—"
        if row.problems:
            evidence = f"**{'; '.join(row.problems)}**"
        lines.append(
            f"| {index} | `{row.key}` | {row.status} | {row.verdict} | {evidence} |"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tensorbem.benchmarks.doc_inventory",
        description="Check the capability claims of README.md against the tree.",
    )
    parser.add_argument("--output", type=Path, default=None, help="write JSON here")
    parser.add_argument(
        "--markdown", action="store_true", help="print a markdown table instead of JSON"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit 1 when any claim drifts (default: always exit 0)",
    )
    parser.add_argument(
        "--root", type=Path, default=None, help="repository root (default: detected)"
    )
    args = parser.parse_args(argv)

    rows = check(root=args.root)
    drifted = [row.key for row in rows if row.verdict == DRIFT]

    if args.markdown:
        print(render_markdown(rows))
    else:
        payload = {
            "claim_count": len(rows),
            "shipped": sum(1 for row in rows if row.status == SHIPPED),
            "absent": sum(1 for row in rows if row.status == ABSENT),
            "passed": len(rows) - len(drifted),
            "drifted": drifted,
            "rows": summary_rows(rows),
        }
        text = json.dumps(payload, indent=2, sort_keys=False)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text + "\n", encoding="utf-8")
        print(text)

    return 1 if (args.strict and drifted) else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())