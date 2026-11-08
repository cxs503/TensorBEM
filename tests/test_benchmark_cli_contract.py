"""Entry-point contract for the benchmark drivers that no test used to touch.

A 2026-11-04 audit of the package (`tensorbem.capability_evidence`) found that
seven shipped benchmark drivers had no reference from any test file at all:
``finite_depth_kernel``, ``hulme_diffraction_verification``,
``hydrostatic_verification``, ``kkr_causality``, ``lifting_solver``,
``symmetry_reduction`` and ``lifting_foil``.  They are the modules a reader is
told to run to reproduce a report table, so a broken ``--help`` or a renamed
``main`` silently invalidates the reproduction path.

The checks here are deliberately shallow and cheap: they pin the *contract*
(command line parses, ``main`` exists, the analytic reference is wired), not
the numbers -- the numbers are pinned by the capability-specific tests cited in
``docs/audit_baseline.md`` section 4.  ``--help`` is used because it must exit
without running any solve, so the whole module costs about a second.
"""

from __future__ import annotations

import importlib
import math
import os
import pathlib
import subprocess
import sys

import pytest

#: Drivers whose ``main`` is argparse-based, so ``--help`` is a safe, fast probe.
ARGPARSE_DRIVERS = (
    "tensorbem.benchmarks.drift_second_order",
    "tensorbem.benchmarks.finite_depth_kernel",
    "tensorbem.benchmarks.hulme_diffraction_verification",
    "tensorbem.benchmarks.hydrostatic_verification",
    "tensorbem.benchmarks.kkr_causality",
    "tensorbem.benchmarks.lifting_solver",
    "tensorbem.benchmarks.symmetry_reduction",
    "tensorbem.benchmarks.wigley_kcs_kochin_energy",
)

#: ``lifting_foil`` has no argparse: running it with an unknown flag executes the
#: whole benchmark, so it is probed in-process instead.
NO_ARGPARSE_DRIVER = "tensorbem.benchmarks.lifting_foil"


def _pythonpath() -> str:
    entries = [str(pathlib.Path(__file__).resolve().parents[1] / "src")]
    existing = os.environ.get("PYTHONPATH")
    if existing:
        entries.append(existing)
    return os.pathsep.join(entries)


@pytest.mark.parametrize("module_name", ARGPARSE_DRIVERS)
def test_driver_help_exits_zero(module_name: str) -> None:
    env = dict(os.environ, PYTHONPATH=_pythonpath())
    completed = subprocess.run(
        [sys.executable, "-m", module_name, "--help"],
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stderr
    output = completed.stdout + completed.stderr
    assert "usage:" in output.lower()
    assert module_name.rsplit(".", 1)[-1] in output, output[:400]


@pytest.mark.parametrize("module_name", (*ARGPARSE_DRIVERS, NO_ARGPARSE_DRIVER))
def test_driver_is_importable_and_exposes_main(module_name: str) -> None:
    module = importlib.import_module(module_name)
    assert callable(getattr(module, "main", None)), f"{module_name} lost its main()"


def test_lifting_foil_analytic_reference_is_wired() -> None:
    """The Joukowski reference must stay attached to the benchmark driver.

    The two analytic lift routes (pressure integral and circulation) share no
    code path, so agreement between them is a real check that the reference was
    not replaced by a constant while the driver kept running.
    """

    module = importlib.import_module(NO_ARGPARSE_DRIVER)
    foil = module.JoukowskiAirfoil()
    reference = module.analytic_reference(foil)
    assert isinstance(reference, dict) and reference
    for key, value in reference.items():
        assert isinstance(value, (int, float)) and math.isfinite(value), (key, value)

    from_circulation = reference["lift_coefficient_from_circulation"]
    from_pressure = reference["lift_coefficient_from_pressure"]
    thin = reference["thin_aerofoil_lift_coefficient"]
    assert from_circulation > 0.0 and from_pressure > 0.0
    # two independent analytic routes must agree to well under a percent
    assert abs(from_pressure - from_circulation) / from_circulation < 1.0e-2
    # and both stay within a few percent of the thin-aerofoil estimate
    assert abs(from_circulation - thin) / thin < 5.0e-2
