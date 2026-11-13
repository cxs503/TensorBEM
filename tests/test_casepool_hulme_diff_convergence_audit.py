"""Regression gate for the third-party-route convergence audit of
``CASE-BEM-HULME-DIFF-VERIF``.

The anchor's gate ``G4`` compares the TensorBEM Kelvin-BIE hemisphere against
the third-party Rankine-BIE code capytaine.  An audit of that artefact found
that the producer's Richardson helper took the refinement ratio the wrong way
round (``r = sqrt(N_coarse/N_fine) < 1``), so the published
``worst_richardson2 = 7.48 %`` was not an extrapolation at all: the sign of the
correction was inverted and its size divided by ``1 - r**p`` instead of
``r**p - 1``, which *amplified* the discretisation error (the extrapolated
error exceeded the finest-mesh error in 12 of 12 cells).

These tests pin

* the defect's signature (pre-fix error > finest error, corrected error <
  finest error, in every cell), so a reintroduced inversion is caught;
* the corrected numbers (worst 2.51 % at assumed ``p = 2``) and their
  consistency with the regenerated ``results.json``;
* the extended ladders (capytaine lat-lon 1024 -> 2304 panels and the
  **geometry-matched** repo mesh 256 -> 4096 panels), which show the residual
  is O(h) discretisation of the third-party code rather than a formulation
  disagreement;
* that the reference values in the audit are re-derived from the library, not
  hand-copied.

capytaine is never imported here: the audit artefact is frozen evidence and the
cheap algebra is re-derived from it.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

from tensorbem.analytic import hulme_hemisphere_radiation

CASE = Path("benchmarks/verified/casepool_bem_hulme_diff_verif")
AUDIT = CASE / "capytaine_convergence_audit.json"
RESULTS = CASE / "results.json"
FROZEN = CASE / "capytaine_scattered.json"
PRODUCER = Path("scripts/capytaine_hulme_diff_convergence_audit.py")

DENSITY = 1025.0
GRAVITY = 9.80665
LEDGER_TOLERANCE = 0.05


def _audit() -> dict:
    return json.loads(AUDIT.read_text())


def _module():
    spec = importlib.util.spec_from_file_location("hulme_conv_audit", PRODUCER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_artefact_present_and_sectioned():
    assert AUDIT.exists() and AUDIT.stat().st_size > 5000
    data = _audit()
    assert data["anchor"] == "CASE-BEM-HULME-DIFF-VERIF"
    for key in ("frozen_reproduction", "frozen_ladder_reanalysis", "uv_family", "octa_family"):
        assert key in data, f"audit artefact lost its {key} section"
    assert data["gate_total"] == pytest.approx(LEDGER_TOLERANCE)


def test_frozen_third_party_route_is_reproducible():
    """The audit must re-run the frozen resolution bit-for-bit before it says anything."""
    repro = _audit()["frozen_reproduction"]
    assert repro["verdict"] == "REPRODUCED"
    assert repro["panels"] == repro["frozen_ladder_panels"]
    assert repro["worst_relative_deviation"] < 1.0e-12


def test_richardson_helper_converges_a_synthetic_family():
    """Unit-level guard against the ratio inversion that the audit found."""
    module = _module()
    limit = complex(2.0, -3.0)
    panels = [64, 256, 1024]
    # F(N) = limit + C h**p with h ~ N**-0.5, p = 1
    values = [limit + complex(0.4, -0.2) / math.sqrt(n) for n in panels]
    fixed = module._richardson(values, panels, 1.0)
    assert abs(fixed - limit) / abs(limit) < 1.0e-12
    assert abs(fixed - limit) < abs(values[-1] - limit)

    # the pre-fix form (r < 1) moves the estimate *away* from the limit
    inverted = module._richardson(values, panels, 1.0, inverted=True)
    assert abs(inverted - limit) > abs(values[-1] - limit)

    with pytest.raises(ValueError):
        module._richardson(values[::-1], panels[::-1], 1.0)


def test_defect_signature_in_every_cell():
    """Pre-fix error > finest error, corrected error < finest error, 12/12 cells."""
    frozen = _audit()["frozen_ladder_reanalysis"]
    cells = frozen["cells"]
    assert len(cells) == 12
    for name, cell in cells.items():
        finest = cell["finest_relative_error"]
        assert cell["prefix_inverted_ratio_relative_error"] > finest, name
        assert cell["richardson_p2_relative_error"] < finest, name
        assert cell["stored_extrapolation_matches_fix"], name


def test_corrected_gate_passes_with_margin():
    frozen = _audit()["frozen_ladder_reanalysis"]
    worst_finest = frozen["worst_finest_relative"]
    worst_fixed = frozen["worst_richardson_p2_relative"]
    worst_prefix = frozen["worst_prefix_inverted_ratio_relative"]
    assert frozen["verdict_richardson_p2"] == "PASS"
    assert worst_fixed < LEDGER_TOLERANCE
    assert worst_fixed < worst_finest  # extrapolation must help, not hurt
    # the published number was the inverted-ratio one and inflated the gap ~3x
    assert worst_prefix == pytest.approx(0.0748, abs=5.0e-4)
    assert worst_prefix / worst_fixed == pytest.approx(3.0, abs=0.2)


def test_regenerated_results_json_carries_the_fixed_number():
    gate = json.loads(RESULTS.read_text())["gates"]["G4_independent_route"]
    audit = _audit()["frozen_ladder_reanalysis"]
    assert gate["verdict"] == "PASS"
    assert gate["worst_richardson2"] == pytest.approx(
        audit["worst_richardson_p2_relative"], rel=1.0e-9
    )
    assert gate["worst_richardson2"] < gate["worst_finest"]


def test_extended_ladders_converge_at_first_order():
    data = _audit()
    for family in ("uv_family", "octa_family"):
        block = data[family]
        assert block["all_monotone"], family
        assert block["verdict_finest"] == "PASS"
        assert block["worst_finest_relative"] <= LEDGER_TOLERANCE
        assert block["worst_richardson_p2"] < block["worst_finest_relative"]
        for name, cell in block["cells"].items():
            order = cell["observed_order"]
            assert order is not None and 0.3 < order < 1.6, (family, name, order)


def test_geometry_matched_ladder_is_the_repo_mesh():
    """The octahedral ladder must be the same mesh family TensorBEM solves on."""
    octa = _audit()["octa_family"]["ladder"]
    assert [row["subdivisions"] for row in octa] == [3, 4, 5]
    for row in octa:
        assert row["n_faces"] == 4 * 4 ** row["subdivisions"]
        assert row["family"] == "tensorbem.geometry3d.hemisphere_hull_mesh"
    exact = 2.0 * math.pi / 3.0
    deficits = [abs(row["volume"] / exact - 1.0) for row in octa]
    assert deficits == sorted(deficits, reverse=True)  # volume deficit shrinks


def test_geometry_matched_residual_is_small_at_the_finest_mesh():
    """isolating formulation from geometry: same panels, different code."""
    block = _audit()["octa_family"]
    assert block["worst_finest_relative"] < 0.025  # 4096 panels, ka = 2 heave
    heave_2 = block["cells"]["2.000/heave"]
    errors = [level["relative_error"] for level in heave_2["levels"]]
    assert errors[0] > 0.08 and errors[-1] < 0.025
    assert heave_2["richardson_p2_relative_error"] < 0.02


def test_reference_values_are_re_derived_not_copied():
    """No hand-copied digits: the audit's reference row comes from analytic.py."""
    cell = _audit()["uv_family"]["cells"]["1.000/surge"]
    fresh = hulme_hemisphere_radiation(
        1.0, radius=1.0, density=DENSITY, gravity=GRAVITY, n_terms=144, quad_order=288
    )
    want = complex(fresh.excitation_force_x)
    stored = complex(cell["reference"]["re"], cell["reference"]["im"])
    assert abs(stored - want) / abs(want) < 1.0e-9


def test_producer_imports_capytaine_lazily():
    """capytaine is an optional dependency: it must not be imported at module level."""
    import ast

    tree = ast.parse(PRODUCER.read_text())
    for node in tree.body:  # module level only
        if isinstance(node, ast.Import):
            assert all(alias.name != "capytaine" for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.module != "capytaine"
    # ... but the producer does use it, inside main()
    main = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"
    )
    imports = [
        alias.name
        for node in ast.walk(main)
        if isinstance(node, ast.Import)
        for alias in node.names
    ]
    assert "capytaine" in imports
