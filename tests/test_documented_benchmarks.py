"""Guard the numbers quoted in the README and in ``docs/``.

``docs/benchmark_pins.json`` holds every number that the documentation quotes
from a benchmark, together with the settings that produced it. These tests
re-run the cheap cases and fail if the code no longer delivers them, which is
the only way to keep handwritten prose honest: the 128-panel Ursell added-mass
error quoted in the README silently went stale by 10% when the indirect solver
switched to an analytic Rankine integral (see
``docs/ursell_quadrature_provenance.md``).

The 256-panel Ursell case and the 320-panel ellipsoid case are marked slow in
the pin file and are only checked by
``python -m tensorbem.benchmarks.doc_pins --check --include-slow``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tensorbem.benchmarks import doc_pins
from tensorbem.benchmarks.ursell_cylinder import submerged_circle_mesh
from tensorbem.free_surface import green_function

REPO_ROOT = Path(__file__).resolve().parents[1]
PIN_PATH = REPO_ROOT / "docs" / "benchmark_pins.json"
DOCS = (REPO_ROOT / "docs" / "marine_bem_benchmarks.md", REPO_ROOT / "README.md")


def test_pin_file_exists_and_is_well_formed():
    document = json.loads(PIN_PATH.read_text(encoding="utf-8"))
    assert document["schema"] == doc_pins.PIN_SCHEMA
    pins = document["pins"]
    names = {case.name for case in doc_pins.CASES}
    fast = {case.name for case in doc_pins.CASES if not case.slow}
    assert set(pins) <= names, sorted(set(pins) - names)
    assert fast <= set(pins), "the fast cases must all be pinned"
    for case in doc_pins.CASES:
        if case.name not in pins:
            assert case.slow, f"{case.name} is missing from the pin file"
            continue
        entry = pins[case.name]
        assert entry["runner"] == case.runner
        assert set(entry["metrics"]) == set(case.metrics)
        for metric, value in entry["metrics"].items():
            assert isinstance(value, (int, float)) and math.isfinite(value), metric


def test_documentation_pins_match_current_code():
    """Every fast documented number must still be reproduced by the code."""
    problems, measured = doc_pins.check(PIN_PATH)
    assert measured, "no pinned case ran"
    assert not problems, "\n".join(problems)


def test_pinned_cases_reference_existing_docs():
    document = json.loads(PIN_PATH.read_text(encoding="utf-8"))
    for name, entry in document["pins"].items():
        doc = entry["doc"]
        target = doc.split(" ")[0]
        path = REPO_ROOT / target
        assert path.is_file(), f"{name}: pinned doc {target} does not exist"
    text = "\n".join(path.read_text(encoding="utf-8") for path in DOCS)
    assert "8.51\\times10^{-3}" in text or "8.51e-3" in text, (
        "the documentation must quote the reproduced 128-panel Ursell error "
        "(8.51e-3); see docs/ursell_quadrature_provenance.md"
    )
    assert "benchmark_pins.json" in text, "docs must point at the pin file"


def test_analytic_rankine_panel_integral_is_exact():
    """The close-form ``log`` panel integral must beat a uniform Gauss rule.

    ``_evaluate_source_potential`` integrates ``log|P-Q|`` over a straight
    panel analytically. This checks the closed form against a composite
    Gauss rule refined until it is converged, including the self panel where
    a uniform rule is exactly what the old code got wrong.
    """
    from tensorbem.free_surface_solver import _log_panel_primitive  # noqa: PLC0415

    def analytic(xi: float, h: float, length: float) -> float:
        upper = _log_panel_primitive(np.array([[length - xi]]), np.array([[h]]))
        lower = _log_panel_primitive(np.array([[-xi]]), np.array([[h]]))
        return float((upper - lower)[0, 0])

    def composite_gauss(xi: float, h: float, length: float, pieces: int, order: int) -> float:
        nodes, weights = np.polynomial.legendre.leggauss(order)
        edges = np.linspace(0.0, length, pieces + 1)
        total = 0.0
        for start, stop in zip(edges[:-1], edges[1:]):
            half = 0.5 * (stop - start)
            mid = 0.5 * (stop + start)
            s = mid + half * nodes
            total += half * np.sum(weights * 0.5 * np.log((s - xi) ** 2 + h**2))
        return total

    def elementary_log_integral(xi: float, length: float) -> float:
        """``int_0^L log|s-xi| ds`` for an interior collocation point, in closed form.

        Independent of the panel primitive under test: split the panel at
        ``xi``, where ``int_0^a log u du = a log a - a``.  The composite Gauss
        rule cannot serve as the reference for this case: with the singularity
        *inside* a subinterval its error decays only like ``1/pieces`` (measured
        7.5e-6 at 400 pieces, 7.5e-7 at 4000), so it is the reference that is
        inexact, not the analytic primitive.
        """
        left, right = xi, length - xi
        assert left > 0.0 and right > 0.0, "interior point required"
        return (left * math.log(left) - left) + (right * math.log(right) - right)

    for xi, h, length in ((0.5, 0.0, 1.0), (0.0, 1.0e-3, 1.0), (0.9, 0.3, 1.0)):
        closed = analytic(xi, h, length)
        if h == 0.0:
            # Self panel: the logarithmic singularity sits at an interior
            # point, so the reference is the elementary integral and the
            # composite rule is the *defect* being demonstrated.
            fine = elementary_log_integral(xi, length)
            coarse = composite_gauss(xi, h, length, pieces=1, order=24)
            # A single uniform Gauss panel straddling the logarithmic
            # singularity of the self panel is the historical defect: it is
            # biased by orders of magnitude more than the closed form.
            assert abs(coarse - fine) > 1.0e-4, (coarse, fine)
        else:
            fine = composite_gauss(xi, h, length, pieces=400, order=20)
        assert abs(closed - fine) < 1.0e-8, (xi, h, closed, fine)


def test_legacy_log_singular_quadrature_bias_is_quantified():
    """Quantify the 2026-10 potential-evaluation change that moved the docs.

    The README used to quote a 128-panel Ursell added-mass error of
    ``7.72e-3``. Reproducing that number requires evaluating the single-layer
    potential with one uniform 24-point Gauss rule over every panel, i.e. with
    the ``log r`` self-panel singularity unresolved. This test pins the size of
    the effect (a few times ``1e-4`` in the potential) so the provenance note
    stays checkable rather than anecdotal.
    """
    panels = 64
    mesh = submerged_circle_mesh(panels, 1.0, 2.0)
    nu = 0.6
    sources = np.exp(1.0j * np.linspace(0.0, 2.0 * np.pi, panels))

    nodes, weights = np.polynomial.legendre.leggauss(24)
    fraction = 0.5 * (nodes + 1.0)
    source_x = (mesh.vertices[:, 0][:, None] + fraction * mesh.vectors[:, 0][:, None]).ravel()
    source_z = (mesh.vertices[:, 1][:, None] + fraction * mesh.vectors[:, 1][:, None]).ravel()
    jacobian = np.tile(weights, panels) * np.repeat(0.5 * mesh.lengths, nodes.size)

    from tensorbem.free_surface_solver import _evaluate_source_potential  # noqa: PLC0415

    current = _evaluate_source_potential(mesh, sources, nu, 24)

    legacy = np.zeros(panels, dtype=complex)
    for row in range(panels):
        values = green_function(
            mesh.centers[row, 0], mesh.centers[row, 1], source_x, source_z, nu
        )
        legacy[row] = np.sum(values * jacobian * np.repeat(sources, nodes.size))

    relative_difference = np.max(np.abs(legacy - current)) / np.max(np.abs(current))
    # A few 1e-4 in the potential, which is the 0.08% shift in added mass that
    # moved the headline error from 7.72e-3 to 8.51e-3.
    assert 1.0e-4 < relative_difference < 1.0e-2, relative_difference