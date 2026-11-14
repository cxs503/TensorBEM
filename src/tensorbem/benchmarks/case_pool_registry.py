"""Machine registry of case-pool anchors that own an in-repo one-shot entry.

The platform case pool (``tensor_case_pool``) names anchors by ``case_code``;
each anchor is answered by a driver script that states the predeclared criteria,
computes the comparison and leaves committed artefacts.  That mapping used to
live only in the mission reports, so nothing in the repository could answer
"which anchors can this checkout actually run, with which entry, and what did
they last verdict?" without a human.

This module is the machine-readable answer.  It is deliberately data-only: the
entries execute the drivers, this registry only resolves them, and
``scripts/case_pool_run.py`` is the command-line front end.

Every registered path is checked by ``tests/test_case_pool_registry.py`` (entry,
artefact and document must exist on disk), so a moved or deleted artefact turns
the registry red instead of silently degrading to prose.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

#: repository root (``src/tensorbem/benchmarks/case_pool_registry.py`` -> 3 up)
REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class CaseEntry:
    """One case-pool anchor as it is wired inside this repository."""

    case_code: str
    pool_code: str
    repo: str
    tolerance: str
    entry: str
    """repo-relative driver script; run with ``PYTHONPATH=src python3 <entry>``"""
    entry_args: tuple[str, ...]
    """default arguments of the driver (``--stage all`` etc.)"""
    library: str
    """importable module that carries the case cycle / reference solution"""
    artifacts: tuple[str, ...]
    """committed machine-readable artefacts (JSON/CSV), repo-relative"""
    doc: str
    """generated case document, repo-relative"""
    figures: tuple[str, ...] = ()
    repro_cmd: str = ""
    notes: str = ""
    extra: Mapping[str, Any] = field(default_factory=dict)

    def path(self, rel: str) -> Path:
        return REPO_ROOT / rel

    def missing(self) -> list[str]:
        """Registered paths that do not exist (empty list == registry is sound)."""
        out = [rel for rel in (self.entry, self.doc, *self.artifacts) if not self.path(rel).is_file()]
        out += [rel for rel in self.figures if not self.path(rel).is_file()]
        return out

    def artifact_payload(self, rel: str | None = None) -> dict[str, Any]:
        target = self.artifacts[0] if rel is None else rel
        return json.loads(self.path(target).read_text())


#: keys a case artefact may use for a boolean pass/fail flag
PASS_KEYS: tuple[str, ...] = ("passed", "pass", "ok", "all_pass", "overall_passed")

#: keys a case artefact may use for a scalar verdict string
VERDICT_KEYS: tuple[str, ...] = ("overall", "verdict", "status", "level")


def _as_bool(value: Any) -> bool | None:
    """Return ``value`` when it is a real boolean, else ``None``."""
    return value if isinstance(value, bool) else None


def _verdict_from_criteria(criteria: Mapping[str, Any]) -> dict[str, Any] | None:
    """Fold a ``{name: {...}}`` criteria mapping into one pass/fail record.

    The pool's drivers write their per-item gates in several shapes
    (``{"pass": bool}``, ``{"passed": bool}``, a bare boolean).  Only entries
    that carry an explicit boolean are counted; a criteria block with no
    boolean at all yields ``None`` instead of a fabricated verdict.
    """
    failed: list[str] = []
    count = 0
    for name, item in criteria.items():
        if isinstance(item, Mapping):
            flag = next((_as_bool(item.get(key)) for key in PASS_KEYS
                         if _as_bool(item.get(key)) is not None), None)
        else:
            flag = _as_bool(item)
        if flag is None:
            continue
        count += 1
        if not flag:
            failed.append(str(name))
    if not count:
        return None
    return {"passed": not failed, "failed_checks": failed, "check_count": count}


def _verdict_of(payload: Mapping[str, Any]) -> Any:
    """Best-effort verdict extraction from the assorted artefact schemas.

    Every branch of the pool writes a verdict, but in four different shapes:
    a scalar string at the top level, ``{"verdict": "PASS"}`` inside a nested
    block, a boolean flag (``{"passed": true}``), or a ``{"criteria": {...}}``
    mapping whose per-item booleans have to be folded.  The function answers
    "what did this case last verdict?" for all four, and never returns ``None``
    (a missing verdict is reported as a string, not as silence).
    """
    if not isinstance(payload, Mapping):
        return "unreadable artifact"

    # (1) scalar verdict written straight into the artefact
    for key in ("overall", "status", "level"):
        value = payload.get(key)
        if value is not None and not isinstance(value, (dict, list)):
            return value

    # (2) the nested verdict block
    verdict = payload.get("verdict")
    if isinstance(verdict, str) and verdict:
        return verdict
    if isinstance(verdict, Mapping):
        # a verdict the artefact states in words wins over a derived boolean
        for key in VERDICT_KEYS:
            value = verdict.get(key)
            if value is not None and not isinstance(value, (dict, list)):
                return value
        for key in PASS_KEYS:
            flag = _as_bool(verdict.get(key))
            if flag is not None:
                return {"passed": flag,
                        "failed_checks": list(verdict.get("failed_checks") or
                                              verdict.get("failures") or [])}
        folded = _verdict_from_criteria(verdict.get("criteria") or {})
        if folded is not None:
            return folded
        acceptance = verdict.get("acceptance")
        if isinstance(acceptance, Mapping):
            for key in PASS_KEYS:
                flag = _as_bool(acceptance.get(key))
                if flag is not None:
                    return {"passed": flag,
                            "failed_checks": list(acceptance.get("failed_checks") or [])}

    # (3) acceptance block / (4) criteria block at the top level
    for key in ("acceptance", "criteria"):
        block = payload.get(key)
        if isinstance(block, Mapping):
            for name in PASS_KEYS:
                flag = _as_bool(block.get(name))
                if flag is not None:
                    return {"passed": flag,
                            "failed_checks": list(block.get("failed_checks") or [])}
            folded = _verdict_from_criteria(block)
            if folded is not None:
                return folded

    # (5) tiered artefacts: {tier: {"verdict": ...}}
    tiers = payload.get("tiers")
    if isinstance(tiers, Mapping) and tiers:
        return {name: tier.get("verdict") for name, tier in tiers.items()
                if isinstance(tier, Mapping)}
    return "see artifact"


CASES: dict[str, CaseEntry] = {
    "CASE-BEM-SUBMERGED-SPHERE-HAVELOCK": CaseEntry(
        case_code="CASE-BEM-SUBMERGED-SPHERE-HAVELOCK",
        pool_code="BEM-SUBMERGED-SPHERE-HAVELOCK",
        repo="TensorBEM",
        tolerance="<5%",
        entry="benchmarks/verified/casepool_bem_submerged_sphere_havelock/run_case.py",
        entry_args=("--stage", "all"),
        library="tensorbem.benchmarks.case_havelock_submerged_sphere",
        artifacts=(
            "docs/case_bem_submerged_sphere_havelock/numbers.json",
            "docs/case_bem_submerged_sphere_havelock/verdict.json",
            "docs/case_bem_submerged_sphere_havelock/tiers.json",
            "docs/case_bem_submerged_sphere_havelock/table_band_case.csv",
            "docs/case_bem_submerged_sphere_havelock/table_band_economy.csv",
            "docs/case_bem_submerged_sphere_havelock/table_ladder.csv",
            "docs/case_bem_submerged_sphere_havelock/table_low_frequency.csv",
            "docs/case_bem_submerged_sphere_havelock/table_submergence.csv",
            "docs/case_bem_submerged_sphere_havelock/run.log",
        ),
        doc="docs/case_bem_submerged_sphere_havelock/CASE.md",
        figures=(
            "docs/case_bem_submerged_sphere_havelock/fig1_band.png",
            "docs/case_bem_submerged_sphere_havelock/fig2_error.png",
            "docs/case_bem_submerged_sphere_havelock/fig3_ladder.png",
            "docs/case_bem_submerged_sphere_havelock/fig4_submergence.png",
        ),
        repro_cmd=(
            "PYTHONPATH=src python3 "
            "benchmarks/verified/casepool_bem_submerged_sphere_havelock/run_case.py "
            "--stage all --outdir docs/case_bem_submerged_sphere_havelock"
        ),
        notes="submerged sphere, deep water, f/a=2, ka in 0.1..5, economy 320 / case 1280 panels",
    ),
    "CASE-BEM-HULME-HEMISPHERE": CaseEntry(
        case_code="CASE-BEM-HULME-HEMISPHERE",
        pool_code="BEM-HULME-HEMISPHERE",
        repo="TensorBEM",
        tolerance="<5%",
        entry="benchmarks/verified/casepool_bem_hulme_hemisphere/run_case.py",
        entry_args=("--shard", "0", "--shards", "1", "--out", "runs/casepool_bem_hulme_hemisphere"),
        library="tensorbem.analytic.hulme_hemisphere_radiation",
        artifacts=("benchmarks/verified/casepool_bem_hulme_hemisphere/results.json",),
        doc="docs/bem_hulme_hemisphere_case.md",
        figures=(),
        notes="floating hemisphere radiation/diffraction (Hulme 1982); six quantities",
    ),
    "CASE-BEM-ANNULUS-LAPLACE": CaseEntry(
        case_code="CASE-BEM-ANNULUS-LAPLACE",
        pool_code="BEM-ANNULUS-LAPLACE",
        repo="TensorBEM",
        tolerance="<3%",
        entry="src/tensorbem/benchmarks/case_annulus_laplace.py",
        entry_args=(),
        library="tensorbem.benchmarks.case_annulus_laplace",
        artifacts=(
            "benchmarks/verified/bem_annulus_laplace/results.json",
            "benchmarks/verified/bem_annulus_laplace_eccentric/results.json",
        ),
        doc="benchmarks/verified/bem_annulus_laplace/report.md",
        figures=(
            "benchmarks/verified/bem_annulus_laplace_eccentric/figures/convergence.png",
            "benchmarks/verified/bem_annulus_laplace_eccentric/figures/clearance_sweep.png",
            "benchmarks/verified/bem_annulus_laplace_eccentric/figures/field.png",
        ),
        notes="multiply connected 2-D Laplace, closed-form harmonic reference; round 2 "
        "adds the eccentric (bipolar) annulus with the classical log(rho+/rho-) solution",
    ),
    "CASE-BEM-LAPLACE-CYLINDER": CaseEntry(
        case_code="CASE-BEM-LAPLACE-CYLINDER",
        pool_code="BEM-LAPLACE-CYLINDER",
        repo="TensorBEM",
        tolerance="<3%",
        entry="src/tensorbem/benchmarks/case_cylinder_stream.py",
        entry_args=(
            "--outdir",
            "benchmarks/verified/case_bem_laplace_cylinder",
            "--doc",
            "docs/CASE-BEM-LAPLACE-CYLINDER.md",
        ),
        library="tensorbem.benchmarks.case_cylinder_stream",
        artifacts=(
            "benchmarks/verified/case_bem_laplace_cylinder/results.json",
            "benchmarks/verified/case_bem_laplace_cylinder/case.json",
        ),
        doc="docs/CASE-BEM-LAPLACE-CYLINDER.md",
        figures=(
            "benchmarks/verified/case_bem_laplace_cylinder/figures/field.png",
            "benchmarks/verified/case_bem_laplace_cylinder/figures/convergence.png",
            "benchmarks/verified/case_bem_laplace_cylinder/figures/near_wall.png",
        ),
        repro_cmd=(
            "PYTHONPATH=src python3 -m tensorbem.benchmarks.case_cylinder_stream "
            "--outdir benchmarks/verified/case_bem_laplace_cylinder "
            "--doc docs/CASE-BEM-LAPLACE-CYLINDER.md"
        ),
        notes=(
            "2-D exterior Laplace operator on the cylinder: uniform stream vs the "
            "complex potential W = U(z + a^2/z). Route D (Dirichlet perturbation "
            "potential) drives the new exterior field evaluator (potential AND "
            "velocity from closed-form panel-integral gradients); route N (Neumann "
            "zero normal flow) checks the fluid-side surface speed 2U|sin theta| and "
            "Cp = 1 - 4 sin^2(theta). Controls: mirrored image sign, far-field decay, "
            "polygon geometry gate."
        ),
    ),
    "CASE-BEM-NEUMANN-KELVIN": CaseEntry(
        case_code="CASE-BEM-NEUMANN-KELVIN",
        pool_code="BEM-NEUMANN-KELVIN",
        repo="TensorBEM",
        tolerance="<5%",
        entry="scripts/case_bem_neumann_kelvin_michell.py",
        entry_args=("--stage", "cheap"),
        library="tensorbem.nk_panel3d",
        artifacts=("docs/case_bem_neumann_kelvin/numbers.json",),
        doc="docs/case_bem_neumann_kelvin/REPORT.md",
        figures=(),
        notes="steady Neumann-Kelvin thin-ship route vs Michell (1898)",
    ),
    "CASE-BEM-ELLIPSE-2D-ADDED-MASS": CaseEntry(
        case_code="CASE-BEM-ELLIPSE-2D-ADDED-MASS",
        pool_code="BEM-ELLIPSE-2D-ADDED-MASS",
        repo="TensorBEM",
        tolerance="<3%",
        entry="src/tensorbem/benchmarks/case_ellipse_2d_added_mass.py",
        entry_args=(
            "--output",
            "benchmarks/verified/case_bem_ellipse_2d_added_mass",
            "--doc",
            "docs/CASE-BEM-ELLIPSE-2D-ADDED-MASS.md",
        ),
        library="tensorbem.benchmarks.case_ellipse_2d_added_mass",
        artifacts=("benchmarks/verified/case_bem_ellipse_2d_added_mass/results.json",),
        doc="docs/CASE-BEM-ELLIPSE-2D-ADDED-MASS.md",
        figures=(
            "benchmarks/verified/case_bem_ellipse_2d_added_mass/figures/convergence.png",
            "benchmarks/verified/case_bem_ellipse_2d_added_mass/figures/aspect_ratio.png",
            "benchmarks/verified/case_bem_ellipse_2d_added_mass/figures/"
            "boundary_flux_error.png",
        ),
        repro_cmd=(
            "PYTHONPATH=src python3 -m tensorbem.benchmarks.case_ellipse_2d_added_mass "
            "--output benchmarks/verified/case_bem_ellipse_2d_added_mass "
            "--doc docs/CASE-BEM-ELLIPSE-2D-ADDED-MASS.md"
        ),
        notes="2-D exterior operator on a non-circular section: added mass of an "
        "elliptic cylinder vs rho*pi*b^2 / rho*pi*a^2; two routes (first-kind S and "
        "second-kind 1/2 I - H) plus an interior-extension negative control",
    ),
    "CASE-BEM-RIGID-BODY-MASS": CaseEntry(
        case_code="CASE-BEM-RIGID-BODY-MASS",
        pool_code="BEM-RIGID-BODY-MASS",
        repo="TensorBEM",
        tolerance="<3%",
        entry="src/tensorbem/benchmarks/casepool_bem_rigid_body_mass.py",
        entry_args=("--outdir", "docs/case_bem_rigid_body_mass"),
        library="tensorbem.benchmarks.casepool_bem_rigid_body_mass",
        artifacts=(
            "docs/case_bem_rigid_body_mass/results.json",
            "docs/case_bem_rigid_body_mass/verdict.json",
            "docs/case_bem_rigid_body_mass/adversarial_audit.json",
        ),
        doc="docs/case_bem_rigid_body_mass/CASE.md",
        figures=(
            "docs/case_bem_rigid_body_mass/fig_momentum_route.png",
            "docs/case_bem_rigid_body_mass/fig_adversarial_audit.png",
        ),
        repro_cmd=(
            "PYTHONPATH=src python3 -m tensorbem.benchmarks.casepool_bem_rigid_body_mass "
            "--outdir docs/case_bem_rigid_body_mass"
        ),
        notes="6x6 rigid-body mass matrix M_RB: closed block form (Fossen 2011 eq. 3.44) "
        "vs signed-tetrahedron quadrature of the panelised body, rotated body with a full "
        "inertia tensor, capytaine external assembly, and the momentum-definition route "
        "(route P) that pins the off-diagonal sign convention; pre-fix coupling control "
        "and an antisymmetric control show the discriminating power of each check; the independent "
        "adversarial review (docs/case_bem_rigid_body_mass/ADVERSARIAL_REVIEW.md, built by "
        "scripts/audit_m19935_rigid_body_mass.py) re-derives M_RB as a symbolic kinetic-energy Hessian, "
        "fits the quadrature convergence order and records the validity domain of the quadrature route "
        "(mesh_topology_report guard)",
    ),
}


def normalise(code: str) -> str:
    """Accept an anchor slug, a pool code or a bare suffix."""
    key = code.strip().upper()
    if key in CASES:
        return key
    for case_code, entry in CASES.items():
        if key in (entry.case_code.upper(), entry.pool_code.upper()):
            return case_code
    prefixed = f"CASE-{key}" if not key.startswith("CASE-") else key
    if prefixed in CASES:
        return prefixed
    raise KeyError(f"unknown case code {code!r}; known: {', '.join(sorted(CASES))}")


def resolve(code: str) -> CaseEntry:
    return CASES[normalise(code)]


def list_cases() -> list[CaseEntry]:
    return [CASES[key] for key in sorted(CASES)]


def artifact_verdict(code: str) -> Any:
    entry = resolve(code)
    if not entry.artifacts:
        return "no artifact registered"
    return _verdict_of(entry.artifact_payload())


def missing_paths() -> dict[str, list[str]]:
    return {key: entry.missing() for key, entry in CASES.items() if entry.missing()}


def main(argv: Iterable[str] | None = None) -> int:  # pragma: no cover - CLI
    rows = [("case_code", "pool_code", "tolerance", "verdict", "missing")]
    for entry in list_cases():
        try:
            verdict = artifact_verdict(entry.case_code)
        except Exception as exc:  # noqa: BLE001 - registry must never crash the listing
            verdict = f"unreadable: {exc}"
        rows.append((
            entry.case_code,
            entry.pool_code,
            entry.tolerance,
            verdict if isinstance(verdict, str) else json.dumps(verdict),
            ",".join(entry.missing()) or "-",
        ))
    widths = [max(len(str(row[i])) for row in rows) for i in range(len(rows[0]))]
    for row in rows:
        print("  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())