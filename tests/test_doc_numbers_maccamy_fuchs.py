"""Drift guard for the declared MacCamy-Fuchs numbers in the benchmark docs.

``docs/marine_bem_benchmarks.md`` Table 8 (force/moment convergence and the
effective inertia coefficient sweep) and Table 9 (irregular-frequency
suppression by the Schenck CHIEF augmentation) quote specific numbers.  This
test recomputes them from the release code and fails on any drift, so the
documentation cannot silently fall behind the solver a second time.

The tolerance is deliberately tight for quantities with an analytic reference
(force error: 0.5 % of the quoted value) and looser for the near-resonant
condition numbers, which are quoted to four significant digits (0.5 %).
"""

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
VERIFIER = REPO / "src" / "tensorbem" / "benchmarks" / "doc_number_ledger.py"


@pytest.fixture(scope="module")
def ledger():
    from tensorbem.benchmarks import doc_number_ledger as verify_doc_numbers
    payload = verify_doc_numbers.run_benchmark("16 32 64 128")
    return verify_doc_numbers.check(payload)


def test_every_declared_number_reproduces(ledger):
    drift = [row for row in ledger["checks"] if row["verdict"] != "PASS"]
    message = json.dumps(drift, indent=2)
    assert not drift, f"documentation tables 8/9 drifted from the solver:\n{message}"


def test_ledger_covers_both_tables(ledger):
    tables = {row["table"] for row in ledger["checks"]}
    assert tables == {"table8", "sweep", "irregular"}
    assert ledger["summary"]["registered"] == 28


def test_observed_order_is_second_order(ledger):
    orders = [row for row in ledger["checks"] if row["quantity"] == "order_force_x_error"]
    assert len(orders) == 3
    for row in orders:
        assert 1.95 < row["recomputed"] < 2.10


def test_chief_suppresses_the_dipole_resonance():
    """The documented modal selectivity must hold, not just the numbers."""
    from tensorbem.benchmarks import doc_number_ledger as verify_doc_numbers
    payload = verify_doc_numbers.run_benchmark("64")
    rows = {row["resonance"]: row for row in payload["irregular_frequency_study"]}

    axisymmetric = rows["j01_axisymmetric_resonance"]
    dipole = rows["j11_dipole_resonance"]

    # Both resonances blow the unaugmented condition number up by orders of
    # magnitude; only the dipole one pollutes the cos(theta) force.
    assert axisymmetric["unaugmented_condition_number"] > 100.0
    assert dipole["unaugmented_condition_number"] > 100.0
    assert axisymmetric["unaugmented_force_relative_error"] < 1.0e-2
    assert dipole["unaugmented_force_relative_error"] > 5.0e-2

    # CHIEF restores an O(1) condition number and the correct force.
    for row in (axisymmetric, dipole):
        assert row["chief_condition_number"] < 10.0
        assert row["chief_force_relative_error"] < 5.0e-3


def test_cli_is_reproducible_across_processes():
    """Two independent processes must agree to machine precision."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("OMP_NUM_THREADS", "1")
    command = [
        sys.executable,
        "-m",
        "tensorbem.benchmarks.maccamy_fuchs",
        "--panels",
        "16",
    ]
    runs = [
        json.loads(subprocess.run(command, cwd=REPO, env=env, capture_output=True, text=True, check=True).stdout)
        for _ in range(2)
    ]
    assert runs[0]["refinement"]["levels"][0] == runs[1]["refinement"]["levels"][0]
    error = runs[0]["refinement"]["levels"][0]["force_x_relative_error"]
    assert math.isclose(error, 8.7138e-3, rel_tol=5.0e-3)