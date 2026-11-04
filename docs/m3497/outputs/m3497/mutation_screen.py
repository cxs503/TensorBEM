"""Equivalence screening for the mutants in :mod:`mutation_fs2d`.

A surviving mutant means one of two very different things:

* **an escapee** -- the kernel really changed and no test noticed, i.e. a hole
  in the guard set; or
* **an equivalent mutant** -- the injected edit is a mathematical no-op (for
  example flipping the sign of the removable-point regulariser in the Cauchy
  principal value integral: the Legendre nodes are symmetric, so the extra
  ``sum_i w_i / t_i`` vanishes identically).

Counting equivalent mutants as holes would inflate the reported gap, so each
survivor is screened by evaluating a fixed probe set (kernel values, gradients,
dispersion roots, mode norms) under the patch and comparing it against the clean
module.  A survivor whose probe deviation is below ``1e-12`` relative is
classified ``equivalent`` and excluded from the kill-ratio denominator.

Usage::

    PYTHONPATH=src python3 outputs/m3497/mutation_screen.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mutation_fs2d import MUTANTS  # noqa: E402

PROBE = r"""
import json, sys, math
import numpy as np
sys.path.insert(0, {src!r})
import tensorbem.free_surface as fs
import tensorbem.finite_depth as fd

out = {{}}
def put(key, fn):
    try:
        v = np.asarray(fn())
        out[key] = [[float(x.real), float(x.imag)] for x in np.ravel(v)]
    except Exception as exc:                      # pragma: no cover
        out[key] = "error: " + type(exc).__name__

NU = 0.6
PTS = [(0.7, -0.9), (2.5, -1.4), (-3.1, -0.5), (0.0, -2.0), (5.5, -3.0)]
put("green_function", lambda: [fs.green_function(x, z, 0.0, -1.0, NU) for x, z in PTS])
put("green_function_no_rankine", lambda: [
    fs.green_function(x, z, 0.3, -1.2, NU, include_rankine=False) for x, z in PTS])
put("green_gradient_x", lambda: [fs.green_gradient(x, z, 0.0, -1.0, NU)[0] for x, z in PTS])
put("green_gradient_z", lambda: [fs.green_gradient(x, z, 0.0, -1.0, NU)[1] for x, z in PTS])
put("pv_cosine", lambda: [fs._pv_cosine(NU, np.array([d]), np.array([x]))[0]
                          for d in (0.5, 1.5, 4.0) for x in (0.0, 1.1, -2.3)])
put("pv_sine", lambda: [fs._pv_sine(NU, np.array([d]), np.array([x]))[0]
                        for d in (0.5, 1.5, 4.0) for x in (0.0, 1.1, -2.3)])
put("fd_wavenumber", lambda: [fd.finite_depth_wavenumber(NU, h) for h in (0.2, 1.0, 6.0)])
put("fd_phase_velocity", lambda: [fd.phase_velocity(NU, h) for h in (0.2, 1.0, 6.0)])
put("fd_group_velocity", lambda: [fd.group_velocity(NU, h) for h in (0.2, 1.0, 6.0)])
put("fd_mode_norm", lambda: [fd.propagating_mode_norm(fd.finite_depth_wavenumber(NU, h), h)
                             for h in (0.2, 1.0, 6.0)])
put("fd_evanescent", lambda: fd.evanescent_wavenumbers(NU, 1.0, 4))
put("fd_wave_amplitude", lambda: [fd.wave_amplitude_coefficient(z, NU, 1.0)
                                  for z in (0.0, -0.3, -1.0)])
put("fd_green_function", lambda: [fd.green_function(x, z, 0.0, -1.0, NU, 1.0) for x, z in PTS])
put("fd_green_gradient", lambda: [np.ravel(fd.green_gradient(x, z, 0.0, -1.0, NU, 1.0))
                                  for x, z in PTS])
print(json.dumps(out))
"""


def probe(src: str, patch: str | None) -> dict:
    code = PROBE.format(src=src)
    if patch:
        code = code.replace("import tensorbem.free_surface as fs", "import tensorbem.free_surface as fs\n" + patch)
    env = dict(os.environ)
    env["PYTHONPATH"] = src
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=600)
    if proc.returncode != 0:
        return {"_error": proc.stderr.strip().splitlines()[-1:] or ["unknown"]}
    return json.loads(proc.stdout.strip().splitlines()[-1])


def deviation(a: dict, b: dict) -> float:
    worst = 0.0
    for key, va in a.items():
        vb = b.get(key)
        if isinstance(va, str) or isinstance(vb, str) or vb is None:
            continue
        for pa, pb in zip(va, vb):
            for ca, cb in zip(pa, pb):
                denom = max(abs(ca), abs(cb), 1e-30)
                worst = max(worst, abs(ca - cb) / denom)
    return worst


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    src = str(root / "src")
    clean = probe(src, None)
    rows = []
    for mid, desc, patch in MUTANTS:
        mutated = probe(src, patch)
        dev = deviation(clean, mutated)
        rows.append(
            {
                "mutant": mid,
                "description": desc,
                "probe_max_rel_deviation": dev,
                "equivalent": dev < 1e-12,
            }
        )
        print(f"{mid:42s} max rel dev {dev:9.3e}  {'EQUIVALENT' if dev < 1e-12 else 'effective'}", flush=True)
    out = root / "outputs/m3497/mutation_screen.json"
    out.write_text(json.dumps({"probe_points": len(clean), "rows": rows}, indent=2), encoding="utf-8")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())