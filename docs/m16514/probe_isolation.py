"""Probe: does the mutation harness's shadow tree actually control the pytest session?

Reproduces one surviving mutant from the wave-load sweep (``tensorbem.rao``:
``m + added_mass`` -> ``m - added_mass``) exactly the way
``tensorbem.benchmarks.mutation_audit.run_tests`` does, and reports
(a) the pytest exit code and (b) which ``tensorbem`` the session imported.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]  # docs/m16514/ -> repo root


def main() -> int:
    work = Path(tempfile.mkdtemp(prefix="probe-iso-"))
    shadow = work / "src"
    shutil.copytree(REPO / "src" / "tensorbem", shadow / "tensorbem")
    target = shadow / "tensorbem" / "rao.py"
    text = target.read_text(encoding="utf-8")
    needle = "m + added_mass"
    if needle not in text:
        print("needle missing; rao.py changed")
        return 2
    target.write_text(text.replace(needle, "m - added_mass", 1), encoding="utf-8")

    env = {
        "PYTHONPATH": str(shadow),
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", "/root"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
    session = subprocess.run(
        [sys.executable, "-c", "import tensorbem, sys; print(tensorbem.__file__)"],
        cwd=str(REPO),
        env=env,
        capture_output=True,
        text=True,
    )
    print("bare import          ->", session.stdout.strip())

    probe = work / "which_tensorbem.py"
    probe.write_text(
        "import pytest\n\n"
        "def test_probe():\n"
        "    import tensorbem\n"
        "    print('SESSION_IMPORT', tensorbem.__file__)\n"
        "    assert True\n",
        encoding="utf-8",
    )
    tests = subprocess.run(
        [sys.executable, "-m", "pytest", str(probe), "-q", "--no-header",
         "-p", "no:cacheprovider", "-s"],
        cwd=str(REPO),
        env=env,
        capture_output=True,
        text=True,
    )
    for line in (tests.stdout + tests.stderr).splitlines():
        if "SESSION_IMPORT" in line:
            print("pytest session import->", line.split("SESSION_IMPORT", 1)[1].strip())

    run = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_rao_verification.py", "-q",
         "--no-header", "--tb=no", "-p", "no:cacheprovider"],
        cwd=str(REPO),
        env=env,
        capture_output=True,
        text=True,
    )
    tail = (run.stdout + run.stderr).strip().splitlines()
    print("mutated test run rc =", run.returncode, "|", tail[-1] if tail else "")
    shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())