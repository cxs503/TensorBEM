"""Drift guard for the declared rigid-body Neumann numbers in the benchmark docs.

``docs/marine_bem_benchmarks.md`` Tables 5 and 6 (Section 4.7) quote specific
numbers for the 2-D elliptic-cylinder and 3-D triaxial-ellipsoid rigid-body
Neumann added-mass tensors.  This test recomputes them from the release code
and fails on any drift, so the documentation cannot silently fall behind the
solver a second time.

The tolerance is deliberately tight for quantities with an analytic reference
(error: 0.5 % of the quoted value) and looser for the near-resonant condition
numbers, which are quoted to four significant digits (0.5 %).
"""

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
VERIFIER = REPO / "src" / "tensorbem" / "benchmarks" / "rigid_body_doc_ledger.py"


@pytest.fixture(scope="module")
def ledger():
    from tensorbem.benchmarks import rigid_body_doc_ledger as verify_doc_numbers
    payload = verify_doc_numbers.run_benchmark("16 32 64 128", "0 1 2")
    return verify_doc_numbers.check(payload)


def test_every_declared_number_reproduces(ledger):
    drift = [row for row in ledger["checks"] if row["verdict"] != "PASS"]
    message = json.dumps(drift, indent=2)
    assert not drift, f"documentation tables 5/6 drifted from the solver:\n{message}"


def test_ledger_covers_both_tables(ledger):
    tables = {row["table"] for row in ledger["checks"]}
    assert tables == {"table5", "table6"}
    assert ledger["summary"]["registered"] == 30


def test_observed_order_is_second_order(ledger):
    orders = [row for row in ledger["checks"] if row["quantity"] == "roll_xy_order"]
    assert len(orders) == 3
    for row in orders:
        assert 1.90 < row["recomputed"] < 2.10


def test_cli_is_reproducible_across_processes():
    """Two independent processes must agree to machine precision."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("OMP_NUM_THREADS", "1")
    command = [
        sys.executable,
        "-m",
        "tensorbem.benchmarks.rigid_body_added_mass",
        "--panels",
        "16",
        "--subdivisions",
        "0",
    ]
    runs = [
        json.loads(subprocess.run(command, cwd=REPO, env=env, capture_output=True, text=True, check=True).stdout)
        for _ in range(2)
    ]
    assert runs[0]["ellipse_2d"]["levels"][0] == runs[1]["ellipse_2d"]["levels"][0]
    error = runs[0]["ellipse_2d"]["levels"][0]["diagonal_relative_errors"]["roll_xy"]
    assert math.isclose(error, 3.23e-2, rel_tol=5.0e-3)
