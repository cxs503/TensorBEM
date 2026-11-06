"""External verification of the 3-D Kelvin wave function against its own integral.

The kernel routine in :mod:`tensorbem.free_surface3d` used to be checked only by
internal identities (which hold by construction) and by seam continuity between
its own branches.  ``tensorbem.benchmarks.kelvin_kernel_reference`` supplies the
missing external reference -- the defining principal-value integral evaluated
with mpmath -- and pins the comparison in
``docs/kelvin_kernel_reference/reference.json``.  These tests keep the pin
honest and keep the published accuracy envelope from drifting.
"""

from __future__ import annotations

import copy
import pathlib

import pytest

from tensorbem.benchmarks import kelvin_kernel_reference as kk
from tensorbem.free_surface3d import region_label

DOC = pathlib.Path(__file__).resolve().parents[1] / "docs" / "kelvin_kernel_reference.md"
BEGIN = "<!-- BEGIN GENERATED: kelvin-kernel-envelope -->"
END = "<!-- END GENERATED: kelvin-kernel-envelope -->"


@pytest.fixture(scope="module")
def payload() -> dict:
    return kk.load()


def test_pin_covers_every_region_and_is_well_formed(payload: dict) -> None:
    assert payload["schema"] == kk.SCHEMA
    regions = {point["region"] for point in payload["points"]}
    assert regions == {"cone", "near", "middle", "far"}
    assert len(payload["points"]) >= 30
    # every point in the pin is classified by the evaluator's own rule
    for point in payload["points"]:
        assert region_label(point["x"], point["y"]) == point["region"]
    assert {seam["seam"].split(" (")[0] for seam in payload["seams"]} >= {
        "cone / near",
        "near / middle",
        "middle / far",
    }


def test_production_reproduces_the_pinned_reference(payload: dict) -> None:
    rows = kk.compare(payload)
    bad = [row for row in rows if not row["ok"]]
    assert not bad, f"{len(bad)} point(s) drifted from the pinned reference: {bad[:3]}"


def test_documented_envelope_matches_the_pin() -> None:
    text = DOC.read_text(encoding="utf-8")
    start = text.index(BEGIN) + len(BEGIN)
    stop = text.index(END)
    generated = text[start:stop].strip()
    assert generated == kk.render_document_block(kk.load()).strip()


def test_accuracy_envelope_is_stated_and_not_optimistic(payload: dict) -> None:
    by_region = payload["by_region"]
    # cone and near branches are exact to the double-precision floor
    for name in ("cone", "near"):
        assert by_region[name]["max_dev_f"] < 1e-12
        assert by_region[name]["max_dev_fx"] < 1e-12
    # middle: the Laguerre tail, again at the floor for F
    assert by_region["middle"]["max_dev_f"] < 1e-12
    # far: the 17-term asymptotic series -- the weakest point of the kernel, and
    # the number the module docstring and docs/kelvin_kernel_reference.md publish
    assert by_region["far"]["max_dev_f"] < 5e-9
    assert by_region["far"]["max_dev_fx"] < 5e-6
    # the reference itself must be converged well below those numbers
    assert payload["envelope"]["max_reference_self_convergence_f"] < 1e-10


def test_seam_bounds_are_consistent_with_the_published_claim(payload: dict) -> None:
    bounds = {seam["seam"].split(" (")[0]: seam for seam in payload["seams"]}
    for name in ("cone / near", "near / middle"):
        assert bounds[name]["bound_f"] < 1e-12
        assert bounds[name]["bound_fx"] < 1e-12
    far = [seam for seam in payload["seams"] if seam["seam"].startswith("middle / far")]
    assert far, "the weakest seam must stay audited"
    assert max(seam["bound_f"] for seam in far) < 5e-9
    assert max(seam["bound_fx"] for seam in far) < 5e-6


def test_zero_x_tail_matches_its_closed_form(payload: dict) -> None:
    for check in payload["tail_identity"]:
        assert check["rel_dev"] < 1e-12, check


def test_a_mutated_reference_value_is_caught(payload: dict) -> None:
    """Negative control: the gate must reject a 1e-6 relative corruption."""
    mutated = copy.deepcopy(payload)
    mutated["points"][0]["f_reference"] = str(
        float(mutated["points"][0]["f_reference"]) * (1.0 + 1e-6)
    )
    rows = kk.compare(mutated)
    assert not rows[0]["ok"], "a corrupted reference value must fail the comparison"


def test_region_boundaries_are_the_documented_ones() -> None:
    assert region_label(0.35, 1.0) == "cone"
    assert region_label(0.3500001, 1.0) == "near"
    assert region_label(2.0, 4.5) == "near"
    assert region_label(2.0, 4.5000001) == "middle"
    assert region_label(0.25, 26.0) == "middle"
    assert region_label(0.2500001, 26.0) == "far"


def test_reference_regularisation_identity() -> None:
    """The reference itself must satisfy ``dF/dY = -F - 1/hypot(X, Y)``."""
    mpmath = pytest.importorskip("mpmath")
    x, y, dps = 0.8, 2.5, 22
    with mpmath.workdps(dps + 6):
        f, _ = kk.reference_wave_function(x, y, dps)
        h = mpmath.mpf("1e-6")
        f_up, _ = kk.reference_wave_function(x, y + float(h), dps)
        f_dn, _ = kk.reference_wave_function(x, y - float(h), dps)
        dFdY = (f_up - f_dn) / (2 * h)
        residual = dFdY + f + 1 / mpmath.sqrt(mpmath.mpf(x) ** 2 + mpmath.mpf(y) ** 2)
    # Coarse on purpose: the residual is dominated by the finite-difference
    # derivative of a reference carrying ~1e-22 absolute quadrature noise, so
    # this guards sign/factor errors, not the last digits.
    assert abs(residual) < mpmath.mpf("1e-6"), f"residual={mpmath.nstr(abs(residual), 5)}"
