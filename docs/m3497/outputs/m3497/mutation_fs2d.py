"""Mutation-gap audit of the 2-D free-surface / finite-depth kernel family.

Why this family
---------------
``tests/test_mutation_gaps.py`` already nails down 2-D *Laplace* and hydrostatic
internals, and m3471 audited the 3-D Laplace kernels.  The kernels that carry
the free-surface physics -- the Cauchy principal value integrals in
:mod:`tensorbem.free_surface`, the finite-depth dispersion/mode machinery in
:mod:`tensorbem.finite_depth` and its continuation into
:mod:`tensorbem.finite_depth_solver` -- had no discriminating-power measurement:
the tests pass, but nobody had asked whether they *can* fail.

Method
------
Each mutant is a semantic defect injected at run time by rebinding a module
attribute (never by rewriting source files).  The guard set is the existing
kernel test files; a mutant is *killed* when at least one guard test fails, and
*survives* when the whole guard set stays green -- i.e. the suite would ship the
bug.  Everything runs in a fresh subprocess per mutant, so no state leaks.

Usage::

    PYTHONPATH=src python3 outputs/m3497/mutation_fs2d.py \
        --json outputs/m3497/mutation_fs2d.json

Cost: each mutant re-runs the 2-D guard set (seconds per solve, 64-128 panels);
the whole sweep is minutes on one core.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

GUARD_FILES = [
    "tests/test_mutation_gaps.py",
    "tests/test_free_surface_kernel.py",
    "tests/test_finite_depth.py",
    "tests/test_finite_depth_solver.py",
]

# Each mutant: (id, human description, patch source, expected-to-be-detected?)
MUTANTS = [
    (
        "M1_pv_cosine_regulariser_sign",
        "PV cosine integrand: subtract the removable reference with the wrong "
        "sign (exp(-kd)cos(kx) + exp(-nu d)cos(nu x))",
        """
import tensorbem.free_surface as fs
_orig = fs._pv_cosine
def _mut(nu, depth, offset):
    import numpy as np
    saved = fs._LEGENDRE_W
    # flip the regulariser sign by negating the difference term
    def patched(nu_, depth_, offset_):
        d = np.ravel(np.asarray(depth_, dtype=float))[:, None]
        x = np.ravel(np.asarray(offset_, dtype=float))[:, None]
        nu_ = float(nu_)
        ref = np.exp(-nu_ * d) * np.cos(nu_ * x)
        k = nu_ * (1.0 + fs._LEGENDRE_X)[None, :]
        diff = np.exp(-k * d) * np.cos(k * x) + ref      # <-- wrong sign
        head = np.sum(fs._LEGENDRE_W[None, :] * diff / fs._LEGENDRE_X[None, :], axis=1)
        sigma = nu_ * d
        t = fs._LAGUERRE_X[None, :] / sigma
        integ = np.cos(nu_ * (2.0 + t) * x) / (1.0 + t)
        tail = np.exp(-2.0 * sigma[:, 0]) / sigma[:, 0] * np.sum(
            fs._LAGUERRE_W[None, :] * integ, axis=1)
        out = head + tail
        return out.reshape(np.broadcast(np.asarray(depth_), np.asarray(offset_)).shape)
    return patched(nu, depth, offset)
fs._pv_cosine = _mut
""",
    ),
    (
        "M2_pv_sine_regulariser_dropped",
        "PV sine integrand: drop the removable-point regulariser entirely "
        "(integrate exp(-kd)sin(kx)/(k-nu) through k = nu)",
        """
import numpy as np
import tensorbem.free_surface as fs
_orig = fs._pv_sine
def _mut(nu, depth, offset):
    d = np.ravel(np.asarray(depth, dtype=float))[:, None]
    x = np.ravel(np.asarray(offset, dtype=float))[:, None]
    nu_ = float(nu)
    k = nu_ * (1.0 + fs._LEGENDRE_X)[None, :]
    diff = np.exp(-k * d) * np.sin(k * x)          # no subtraction, singular
    head = np.sum(fs._LEGENDRE_W[None, :] * diff / fs._LEGENDRE_X[None, :], axis=1)
    sigma = nu_ * d
    t = fs._LAGUERRE_X[None, :] / sigma
    integ = np.sin(nu_ * (2.0 + t) * x) / (1.0 + t)
    tail = np.exp(-2.0 * sigma[:, 0]) / sigma[:, 0] * np.sum(
        fs._LAGUERRE_W[None, :] * integ, axis=1)
    out = head + tail
    return out.reshape(np.broadcast(np.asarray(depth), np.asarray(offset)).shape)
fs._pv_sine = _mut
""",
    ),
    (
        "M3_green_function_wave_scale",
        "Free-surface Green function: scale the whole kernel by 1+1e-3 "
        "(a 0.1% kernel error a converged solver would carry into A and B)",
        """
import tensorbem.free_surface as fs
_orig = fs.green_function
def _mut(*a, **k):
    return 1.001 * _orig(*a, **k)
fs.green_function = _mut
""",
    ),
    (
        "M4_green_gradient_components_swapped",
        "Free-surface Green gradient: return (dg/dz, dg/dx) instead of "
        "(dg/dx, dg/dz)",
        """
import tensorbem.free_surface as fs
_orig = fs.green_gradient
def _mut(*a, **k):
    gx, gz = _orig(*a, **k)
    return gz, gx
fs.green_gradient = _mut
""",
    ),
    (
        "M5_scaled_depth_guard_disabled",
        "Free-surface kernel: accept nu*depth below the documented validity "
        "floor instead of raising",
        """
import tensorbem.free_surface as fs
fs._MIN_SCALED_DEPTH = 0.0
""",
    ),
    (
        "M6_finite_depth_wavenumber_is_deep_water",
        "Finite-depth dispersion: return the deep-water root nu instead of "
        "solving k tanh(k h) = nu (i.e. silently ignore the bottom)",
        """
import tensorbem.finite_depth as fd
_orig = fd.finite_depth_wavenumber
def _mut(nu, depth):
    return float(nu)
fd.finite_depth_wavenumber = _mut
""",
    ),
    (
        "M7_mode_norm_factor",
        "Propagating mode norm: drop the sinh(2 k0 h)/(2 k0 h) term "
        "(N0 = h instead of (h/2)(1 + sinh/(2k0 h)))",
        """
import tensorbem.finite_depth as fd
_orig = fd.propagating_mode_norm
def _mut(k0, depth):
    return 2.0 * _orig(k0, depth)
fd.propagating_mode_norm = _mut
""",
    ),
    (
        "M8_group_velocity_is_half_phase",
        "Finite-depth group velocity: return cp/2 (the deep-water shortcut) "
        "at every depth",
        """
import tensorbem.finite_depth as fd
_orig = fd.group_velocity
def _mut(nu, depth, *, gravity=9.80665):
    return 0.5 * fd.phase_velocity(nu, depth, gravity=gravity)
fd.group_velocity = _mut
""",
    ),
    (
        "M9_evanescent_roots_unpolished",
        "Evanescent roots: return the bracket midpoints without Newton polish",
        """
import numpy as np
import tensorbem.finite_depth as fd
_orig = fd.evanescent_wavenumbers
def _mut(nu, depth, count):
    roots = np.asarray(_orig(nu, depth, count), dtype=float)
    return roots * (1.0 + 1.0e-2)      # 1% root error, no polish
fd.evanescent_wavenumbers = _mut
""",
    ),
    (
        "M10_wave_amplitude_coefficient_scale",
        "Wave amplitude coefficient: scale by 1+1e-2 (a 1% error in the "
        "far-field amplitude = 2% in radiated damping)",
        """
import tensorbem.finite_depth as fd
_orig = fd.wave_amplitude_coefficient
def _mut(z, nu, depth):
    return 1.01 * _orig(z, nu, depth)
fd.wave_amplitude_coefficient = _mut
""",
    ),
]

RUNNER = """
import sys, pytest
sys.path.insert(0, {src!r})
{src_root_setup}
{patch}
sys.exit(pytest.main([{files}, "-q", "--tb=no", "-rf", "-p", "no:cacheprovider"]))
"""


def run_mutant(mutant_id: str, patch: str, files: list[str], root: Path) -> dict:
    src = str(root / "src")
    code = RUNNER.format(
        src=src,
        src_root_setup=f"import os; os.chdir({str(root)!r})",
        patch=patch,
        files=", ".join(repr(f) for f in files),
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = src
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=1800
    )
    out = proc.stdout + proc.stderr
    failing = sorted(set(re.findall(r"^FAILED (\S+)", out, flags=re.M)))
    skipped_all = re.search(r"(\d+) skipped", out)
    return {
        "mutant": mutant_id,
        "returncode": proc.returncode,
        "killed": proc.returncode != 0,
        "killing_tests": failing,
        "tail": out.strip().splitlines()[-1] if out.strip() else "",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", default="outputs/m3497/mutation_fs2d.json")
    ap.add_argument("--only", default=None, help="comma-separated mutant ids")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    only = set(args.only.split(",")) if args.only else None
    results = []
    for mid, desc, patch in MUTANTS:
        if only and mid not in only:
            continue
        res = run_mutant(mid, patch, GUARD_FILES, root)
        res["description"] = desc
        results.append(res)
        print(f"{mid:42s} {'KILLED' if res['killed'] else 'SURVIVED'}  {res['tail']}", flush=True)

    n = len(results)
    killed = sum(r["killed"] for r in results)
    summary = {
        "guard_files": GUARD_FILES,
        "mutants": n,
        "killed": killed,
        "survived": n - killed,
        "kill_ratio": killed / n if n else 0.0,
        "results": results,
    }
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nkill ratio {killed}/{n} = {summary['kill_ratio']:.3f} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())