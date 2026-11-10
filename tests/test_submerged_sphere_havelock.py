"""P190-N03: Havelock (1955) submerged sphere radiation verification tests."""
from __future__ import annotations
import json, math, subprocess, sys
import numpy as np
import pytest
from tensorbem.benchmarks.submerged_sphere_havelock import (
    run_convergence_study, run_frequency_sweep, run_low_frequency_limit
)
from tensorbem.geometry3d import submerged_sphere_mesh


def test_icosphere_panel_counts():
    """Verify icosphere subdivisions produce expected panel counts."""
    expected = {0: 20, 1: 80, 2: 320, 3: 1280}
    for sub, exp in expected.items():
        m = submerged_sphere_mesh(sub, radius=1.0, submergence=2.0)
        assert m.panel_count == exp


def test_convergence_study_order():
    """A_33 convergence order >= 1.5."""
    result = run_convergence_study(ka=0.8, radius=1.0, submergence=2.0)
    order = result["observed_order_a33"]
    assert order is not None, "Could not compute convergence order"
    assert order >= 1.5, f"Convergence order {order} < 1.5"


def test_frequency_sweep_errors():
    """All errors < 5% for ka=0.1~5.0 at 320 panels."""
    rows = run_frequency_sweep(ka_values=(0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0),
                               radius=1.0, submergence=2.0, subdivision=2)
    max_a33_err = max(r["a33_rel_error"] for r in rows)
    max_b33_err = max(r["b33_rel_error"] for r in rows)
    max_a11_err = max(r["a11_rel_error"] for r in rows)
    max_b11_err = max(r["b11_rel_error"] for r in rows)
    assert max_a33_err < 0.05, f"A33 max error {max_a33_err:.4f} >= 5%"
    assert max_b33_err < 0.05, f"B33 max error {max_b33_err:.4f} >= 5%"
    assert max_a11_err < 0.05, f"A11 max error {max_a11_err:.4f} >= 5%"
    assert max_b11_err < 0.05, f"B11 max error {max_b11_err:.4f} >= 5%"


def test_low_frequency_limit():
    """Low-frequency A_33 limit error < 3%."""
    result = run_low_frequency_limit(ka_values=(0.01, 0.02, 0.05, 0.1, 0.2),
                                      radius=1.0, submergence=2.0, subdivision=2)
    assert result["limit_relative_error"] < 0.03, \
        f"Low-frequency limit error {result['limit_relative_error']:.4f} >= 3%"


def test_cli_quick_smoke():
    """CLI smoke test."""
    proc = subprocess.run(
        [sys.executable, "-m", "tensorbem.benchmarks.submerged_sphere_havelock", "--quick"],
        check=True, capture_output=True, text=True, timeout=300,
    )
    payload = json.loads(proc.stdout)
    assert payload["benchmark"] == "submerged_sphere_havelock_radiation"
    assert len(payload["frequency_sweep"]) >= 2
    assert payload["convergence_study"]["observed_order_a33"] is not None
