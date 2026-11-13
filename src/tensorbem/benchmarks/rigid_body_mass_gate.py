"""Machine gate for the committed ``CASE-BEM-RIGID-BODY-MASS`` artefact.

Why this module exists
----------------------
``docs/cases/case_bem_rigid_body_mass/run.json`` *states* its own verdict
("worst accepted level 1.412 % < 3 %"), but a JSON file that states its own
verdict is not evidence: nothing in the repository re-derived those numbers
from the stored matrices, and nothing would go red if the artefact went stale
after the assembler changed.  The case-pool anchor ``CASE-BEM-RIGID-BODY-MASS``
therefore could not be answered mechanically ("which entry, what verdict") by
:mod:`tensorbem.benchmarks.case_pool_registry`.

This gate closes that hole.  It never trusts the stored verdict; it re-derives
it from the stored matrices:

1. **Reference regeneration.**  The closed-form reference block is rebuilt from
   the *recorded geometry* (``parameters.semi_axes_m``, ``density_kg_m3``,
   ``cg_offset_m``, ``reference_point_m``) with
   :func:`tensorbem.rigid_body_mass.solid_ellipsoid_mass_properties` and
   :func:`tensorbem.rigid_body_mass.rigid_body_mass_matrix`.  A committed
   ``reference`` / ``expected`` block that no longer matches the geometry (hand
   edit, stale run after a code change) fails here.
2. **Criterion re-evaluation.**  Per-entry relative errors are recomputed from
   the stored ``levels[i].matrix`` against the regenerated reference, compared
   both with the stored ``max_entry_rel_error`` (reproducibility) and with the
   declared ``parameters.tolerance`` on the accepted levels (the actual <3 %
   judgement of the case goal).
3. **Invariant re-checks on the stored matrices.**  Symmetry, positive
   definiteness and the frame-independent kinetic-energy identity
   (:func:`tensorbem.rigid_body_mass.energy_residual`) evaluated with each
   level's *own* mass/centroid/inertia, so the energy identity is exact for a
   genuine block-form assembly and O(1) wrong for the legacy (pre-fix) form.

Scope and honesty
-----------------
The reference regeneration re-uses the same closed form the case used, so it is
a **freshness / tamper** check, *not* a second physics route: the independent
route (surface-mesh tetrahedron quadrature) lives in the case itself.  What the
gate does add over the case is that the <3 % number quoted in prose is now
recomputed from committed data by a separate code path, and that a stale or
edited artefact cannot keep a PASS label.

Every number written by :func:`evaluate_payload` is evaluated in machine
arithmetic from the artifact; nothing is transcribed from the report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from tensorbem.rigid_body_mass import (
    energy_residual,
    rigid_body_mass_matrix,
    solid_ellipsoid_mass_properties,
)

#: repository root (``src/tensorbem/benchmarks/rigid_body_mass_gate.py`` -> 3 up)
REPO_ROOT = Path(__file__).resolve().parents[3]

CASE_CODE = "CASE-BEM-RIGID-BODY-MASS"
DEFAULT_ARTIFACT = "docs/cases/case_bem_rigid_body_mass/run.json"
DEFAULT_VERDICT = "docs/cases/case_bem_rigid_body_mass/verdict.json"

#: A regenerated reference must agree with the committed one to this level; the
#: value is far below any discretisation error but far above a legitimate
#: float-reordering difference, so it separates "same arithmetic" from "edited".
FRESHNESS_TOL = 1e-10
#: Stored vs recomputed per-entry error must agree to this absolute tolerance.
REPRODUCIBILITY_TOL = 1e-12
#: Kinetic-energy identity on a genuine block-form matrix is round-off.
ENERGY_TOL = 1e-10
#: Symmetry, normalised by the matrix scale.
SYMMETRY_TOL = 1e-12
#: Threshold applied to the invariants the artefact *reports* without storing
#: the underlying matrix (assembly / platform-assembly residual blocks).
REPORTED_INVARIANT_TOL = 1e-9


def _rel(delta: float, reference: float) -> float:
    """Case convention: ``|got-ref|/|ref|``, absolute when ``ref`` is zero.

    This is exactly the convention of
    :func:`tensorbem.benchmarks.rigid_body_mass_case._matrix_entry_rows`, so the
    gate and the case cannot disagree about what a "3 % entry" means.
    """
    scale = abs(float(reference))
    return abs(float(delta) - float(reference)) / scale if scale > 0.0 else abs(
        float(delta) - float(reference)
    )


def _entry_rows(got: Sequence[Sequence[float]], ref: Sequence[Sequence[float]]) -> list[dict[str, Any]]:
    got_arr = np.asarray(got, dtype=float)
    ref_arr = np.asarray(ref, dtype=float)
    if got_arr.shape != (6, 6) or ref_arr.shape != (6, 6):
        raise ValueError(f"both matrices must be 6x6, got {got_arr.shape} and {ref_arr.shape}")
    return [
        {"i": i, "j": j, "ref": float(ref_arr[i, j]), "got": float(got_arr[i, j]),
         "rel_error": _rel(got_arr[i, j], ref_arr[i, j])}
        for i in range(6) for j in range(6)
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _regenerate_reference(parameters: Mapping[str, Any]) -> dict[str, Any]:
    """Closed-form reference block rebuilt from the artefact's own geometry."""
    semi_axes = [float(v) for v in parameters["semi_axes_m"]]
    density = float(parameters["density_kg_m3"])
    cg = [float(v) for v in parameters["cg_offset_m"]]
    ref_point = [float(v) for v in parameters.get("reference_point_m", (0.0, 0.0, 0.0))]
    props = solid_ellipsoid_mass_properties(semi_axes, density)
    matrix = rigid_body_mass_matrix(
        float(props["mass"]), cg, props["inertia_cg"], inertia_about="cg",
        reference_point=ref_point,
    )
    return {
        "volume": float(props["volume"]),
        "mass": float(props["mass"]),
        "centroid": [float(v) for v in props["centroid"]],
        "inertia_cg": np.asarray(props["inertia_cg"], dtype=float),
        "radii_of_gyration": [float(v) for v in props["radii_of_gyration"]],
        "matrix": matrix,
    }


def evaluate_payload(
    payload: Mapping[str, Any],
    *,
    artifact_rel: str = DEFAULT_ARTIFACT,
    artifact_sha256: str | None = None,
) -> dict[str, Any]:
    """Re-derive the case verdict from a committed ``run.json`` payload."""
    parameters = payload["parameters"]
    committed = payload["reference"]
    expected = payload.get("expected", {})
    tolerance = float(parameters["tolerance"])
    accepted_from = int(parameters["accepted_from_subdivision"])
    regen = _regenerate_reference(parameters)
    ref_matrix = regen["matrix"]

    checks: dict[str, bool] = {}
    detail: dict[str, Any] = {}

    # ---- 1. reference / expected fresh against the recorded geometry --------
    ref_committed = np.asarray(committed["matrix"], dtype=float)
    ref_diff = float(np.max(np.abs(ref_committed - ref_matrix)) / np.max(np.abs(ref_matrix)))
    inertia_committed = np.asarray(committed["inertia_cg"], dtype=float)
    inertia_diff = float(
        np.max(np.abs(inertia_committed - regen["inertia_cg"]))
        / np.max(np.abs(regen["inertia_cg"]))
    )
    mass_diff = float(
        abs(float(committed["mass"]) - regen["mass"]) / regen["mass"]
    )
    checks["reference_matches_recorded_geometry"] = bool(
        max(ref_diff, inertia_diff, mass_diff) <= FRESHNESS_TOL
    )
    detail["reference_regeneration"] = {
        "max_rel_diff_matrix": ref_diff,
        "max_rel_diff_inertia_cg": inertia_diff,
        "rel_diff_mass": mass_diff,
        "tolerance": FRESHNESS_TOL,
    }

    if expected:
        expected_diag = np.asarray(expected["diagonal"], dtype=float)
        diag_diff = float(
            np.max(np.abs(expected_diag - np.diag(ref_matrix))) / np.max(np.abs(expected_diag))
        )
        expected_mass_diff = float(
            abs(float(expected["mass_kg"]) - regen["mass"]) / regen["mass"]
        )
        checks["expected_block_matches_recorded_geometry"] = bool(
            max(diag_diff, expected_mass_diff) <= FRESHNESS_TOL
        )
        detail["expected_regeneration"] = {
            "max_rel_diff_diagonal": diag_diff,
            "rel_diff_mass": expected_mass_diff,
        }

    # ---- 2. criterion re-evaluation on the stored level matrices -------------
    levels = payload["levels"]
    level_rows: list[dict[str, Any]] = []
    errors_reproducible = True
    flags_reproducible = True
    symmetry_ok = True
    positive_definite = True
    energy_ok = True
    for level in levels:
        sub = int(level["subdivisions"])
        rows = _entry_rows(level["matrix"], ref_matrix)
        worst = max(rows, key=lambda row: row["rel_error"])
        stored = float(level["max_entry_rel_error"])
        stored_flag = bool(level["within_tolerance"])
        recomputed_flag = bool(worst["rel_error"] < tolerance)
        errors_reproducible &= bool(abs(stored - worst["rel_error"]) <= REPRODUCIBILITY_TOL)
        flags_reproducible &= bool(stored_flag == recomputed_flag)
        matrix = np.asarray(level["matrix"], dtype=float)
        scale = float(np.max(np.abs(matrix)))
        sym = float(np.max(np.abs(matrix - matrix.T)))
        symmetry_ok &= bool(sym <= SYMMETRY_TOL * max(scale, 1.0))
        eig_min = float(np.min(np.linalg.eigvalsh(0.5 * (matrix + matrix.T))))
        positive_definite &= bool(eig_min > 0.0)
        # energy identity uses the level's OWN properties -> exact for a block form
        e_res = energy_residual(
            matrix, float(level["mass"]), np.asarray(level["inertia_cg"], dtype=float),
            np.asarray(level["centroid"], dtype=float),
        )
        energy_ok &= bool(e_res <= ENERGY_TOL)
        level_rows.append({
            "subdivisions": sub,
            "panels": int(level["panels"]),
            "stored_max_entry_rel_error": stored,
            "recomputed_max_entry_rel_error": float(worst["rel_error"]),
            "worst_entry": {"i": worst["i"], "j": worst["j"],
                            "ref": worst["ref"], "got": worst["got"]},
            "accepted": bool(sub >= accepted_from),
            "within_tolerance": recomputed_flag,
            "energy_identity_residual": e_res,
            "smallest_eigenvalue": eig_min,
            "symmetry_error": sym,
        })

    accepted = [row for row in level_rows if row["accepted"]]
    if not accepted:
        raise ValueError("artifact declares no accepted level; cannot evaluate the case goal")
    worst_accepted = max(accepted, key=lambda row: row["recomputed_max_entry_rel_error"])

    checks["stored_errors_reproducible"] = bool(errors_reproducible)
    checks["within_tolerance_flags_reproducible"] = bool(flags_reproducible)
    checks["tolerance_respected_on_accepted_levels"] = bool(
        worst_accepted["recomputed_max_entry_rel_error"] < tolerance
    )
    checks["stored_matrices_symmetric"] = bool(symmetry_ok)
    checks["stored_matrices_positive_definite"] = bool(positive_definite)
    checks["kinetic_energy_identity_all_levels"] = bool(energy_ok)

    # ---- 3. invariants the artefact reports without storing the matrix ------
    reported: dict[str, float] = {}
    for block_name in ("assembly", "platform_assembly"):
        block = payload.get(block_name)
        if not isinstance(block, Mapping):
            continue
        for key in ("energy_identity_error", "symmetry_error", "block_form_residual",
                    "block_relative_error", "max_entry_rel_error",
                    "max_entry_rel_error_vs_reference"):
            if key in block and isinstance(block[key], (int, float)):
                reported[f"{block_name}.{key}"] = float(block[key])
    worst_reported = max((abs(v) for v in reported.values()), default=0.0)
    checks["reported_invariants_within_tolerance"] = bool(
        worst_reported <= REPORTED_INVARIANT_TOL
    )
    detail["reported_invariants"] = {
        "values": dict(sorted(reported.items())),
        "worst_abs": worst_reported,
        "tolerance": REPORTED_INVARIANT_TOL,
        "note": "reported by the case without storing the underlying matrix; "
                "gated, not re-derived",
    }

    # ---- 4. verdict ---------------------------------------------------------
    failed = [name for name, ok in checks.items() if not ok]
    out: dict[str, Any] = {
        "case_code": str(payload.get("benchmark", CASE_CODE)),
        "subject": payload.get("subject"),
        "artifact": artifact_rel,
        "artifact_sha256": artifact_sha256,
        "criterion": f"|ref-got|/|ref| < {tolerance:.1%} per matrix entry on accepted levels",
        "tolerance": tolerance,
        "accepted_from_subdivision": accepted_from,
        "regenerated_reference_diagonal": [float(v) for v in np.diag(ref_matrix)],
        "levels": level_rows,
        "worst_accepted_level": {
            "subdivisions": worst_accepted["subdivisions"],
            "panels": worst_accepted["panels"],
            "max_entry_rel_error": worst_accepted["recomputed_max_entry_rel_error"],
            "worst_entry": worst_accepted["worst_entry"],
            "within_tolerance": bool(
                worst_accepted["recomputed_max_entry_rel_error"] < tolerance
            ),
        },
        "checks": checks,
        "details": detail,
        "failed_checks": failed,
        "verdict": "PASS" if not failed else "FAIL",
    }
    # Deterministic serialisation: no timestamps, no platform strings.
    json.dumps(out, allow_nan=False)
    return out


def evaluate_file(path: str | Path = DEFAULT_ARTIFACT) -> dict[str, Any]:
    """Gate the committed artefact at ``path`` (repo-relative or absolute)."""
    file_path = Path(path)
    if not file_path.is_absolute():
        file_path = REPO_ROOT / file_path
    payload = json.loads(file_path.read_text(encoding="utf-8"))
    try:
        rel = str(file_path.resolve().relative_to(REPO_ROOT))
    except ValueError:  # outside the checkout (tests use tmp copies)
        rel = str(file_path)
    return evaluate_payload(payload, artifact_rel=rel, artifact_sha256=_sha256(file_path))


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifact", default=DEFAULT_ARTIFACT,
                        help="committed run.json to gate")
    parser.add_argument("--out", default=DEFAULT_VERDICT,
                        help="where to write verdict.json")
    parser.add_argument("--refresh", action="store_true",
                        help="re-run the case module before gating")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.refresh:
        subprocess.run(
            [sys.executable, "-m", "tensorbem.benchmarks.rigid_body_mass_case",
             "--out", args.artifact],
            check=True, cwd=REPO_ROOT,
        )
    verdict = evaluate_file(args.artifact)
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(verdict, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if not args.quiet:
        worst = verdict["worst_accepted_level"]
        print(json.dumps({
            "case_code": verdict["case_code"],
            "verdict": verdict["verdict"],
            "failed_checks": verdict["failed_checks"],
            "worst_accepted_level": worst,
            "artifact_sha256": verdict["artifact_sha256"],
            "wrote": str(out_path),
        }, indent=2, ensure_ascii=False))
    return 0 if verdict["verdict"] == "PASS" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())