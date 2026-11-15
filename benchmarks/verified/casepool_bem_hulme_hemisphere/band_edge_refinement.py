#!/usr/bin/env python3
"""CASE-BEM-HULME-HEMISPHERE: where does the ka > 2 band actually end?

Why this file exists
--------------------
``band_current_code/results.json`` re-runs the predeclared band-extension cell
list with the current solver (stationarity-audited linear algebra).  Seven of
its nine cells come back inside the 5 % gate; the extension cell at
``ka = 4.0`` on the 1024-panel case mesh does not (8.94 %), and its 256-panel
rung is far worse (34.49 %).  Two readings of that pair are possible and the
artifact cannot tell them apart:

* **discretisation limit** -- constant panels are too coarse at ``ka = 4`` and
  the error falls under refinement (a convergent route, band edge at a mesh
  the case does not own);
* **method limit** -- the error saturates, so no mesh makes ``ka = 4``
  creditable and the band edge is a property of the Kelvin-BIE route itself.

The pair ``256 -> 1024`` cannot decide: two rungs admit no convergence order.
This driver adds the missing rung, ``4096`` panels (subdivisions 5), on the
two band-edge frequencies ``ka = 3.5`` (last credited point) and ``ka = 4.0``
(first failing point), with CHIEF on -- i.e. the *same* cell definition as
``band_extension.py`` (``chief = True``, ``with_diffraction = True``), so the
numbers are directly comparable with ``band_current_code/``.

Predeclared experiment
----------------------
Decision
    Is the ``ka = 4.0`` miss a discretisation limit (band edge movable by
    mesh) or a method limit (band edge fixed)?

Hypothesis
    If the route is merely mesh-limited, the band-max error at ``ka = 4.0``
    must fall by at least a factor 2 from 1024 to 4096 panels and the
    observed convergence order must be positive and consistent with the
    256 -> 1024 pair.  If it saturates (ratio < 1.2), the route's band edge is
    a method property and ``ka = 4.0`` stays NOT creditable on any mesh the
    case can afford.

Metrics
    ``band_max`` = max |relative error| over the six magnitudes
    (``A11, B11, A33, B33, Fx, Fz``), non-dimensionalised exactly as in
    ``band_extension.py``; per cell also the solver guard output
    (``radiation_relative_residual``, ``diffraction_relative_residual``,
    ``radiation_solve_method``, ``condition_number``) and the geometry
    pre-flight (mesh volume vs ``2*pi*a^3/3``, waterplane area vs ``pi*a^2``).

Gates (predeclared here, this driver does not touch E1..E5 of the case)
    R1 ``ladder_is_self_consistent``
        The 1024-panel rungs reproduced here match ``band_current_code``
        bit for bit (same solver, same cell definition, deterministic run).
    R2 ``ka4_error_falls_under_refinement``
        band_max(ka=4, 4096) < band_max(ka=4, 1024); a rising error would mean
        the refinement is not converging and no statement about the edge is
        possible.
    R3 ``credited_band_edge_localised``
        The band edge is reported as the largest ``ka`` of the predeclared
        grid whose cells pass on the case mesh, plus the mesh at which the
        first failing ``ka`` would pass (or ``None`` if it does not pass at
        any mesh computed here).  This is a *reporting* gate: it never widens
        the case band by itself.

Usage (one cell per invocation keeps a cell cheap and resumable; the parent
call without ``--only`` fills in whatever is missing)::

    PYTHONPATH=src python benchmarks/verified/casepool_bem_hulme_hemisphere/\\
        band_edge_refinement.py --only ka4_sub5 --out-dir <dir>

Cell ids: ``ka3.5_sub4``, ``ka3.5_sub5``, ``ka4_sub4``, ``ka4_sub5``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))

import band_extension as be  # noqa: E402  (same cell definition, no fork)

#: Band-edge frequencies: the last credited and the first failing point of the
#: predeclared grid in ``band_current_code/results.json``.
KA_EDGE = (3.5, 4.0)
#: Panel ladder.  subdivisions 4 = 1024 panels (case mesh), 5 = 4096 panels.
SUBDIVISIONS = (4, 5)
GATE = be.E_GATE
QUANTITIES = be.QUANTITIES
OUT_DIR = HERE / "band_edge_refinement"

SOURCE_FILES = be.SOURCE_FILES


def cells() -> list[dict[str, Any]]:
    """Deterministic cell list; ``kind`` reuses the band_extension cell code."""
    out: list[dict[str, Any]] = []
    for ka in KA_EDGE:
        for sub in SUBDIVISIONS:
            out.append({"kind": "extension", "ka": float(ka),
                        "subdivisions": int(sub), "chief": True})
    return out


def cell_id(spec: dict[str, Any]) -> str:
    return f"ka{spec['ka']:g}_sub{spec['subdivisions']}"


def panels(spec: dict[str, Any]) -> int:
    return 4 * 4 ** spec["subdivisions"]


def run_cell(spec: dict[str, Any]) -> dict[str, Any]:
    cell = be.run_one(spec)
    cell["id"] = cell_id(spec)
    cell["panels"] = panels(spec)
    return cell


def run_missing(out_dir: Path, parallel: int, only: set[str] | None) -> None:
    cdir = out_dir / "cells"
    cdir.mkdir(parents=True, exist_ok=True)
    specs = cells()
    todo = [s for s in specs
            if (only is None or cell_id(s) in only)
            and not (cdir / f"{cell_id(s)}.json").exists()]
    env = dict(os.environ)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[var] = "1"
    running: list[tuple[dict[str, Any], subprocess.Popen]] = []
    queue = list(todo)
    failed: list[str] = []
    while queue or running:
        while queue and len(running) < max(1, parallel):
            spec = queue.pop(0)
            spec_path = out_dir / "specs" / f"{cell_id(spec)}.json"
            spec_path.parent.mkdir(parents=True, exist_ok=True)
            spec_path.write_text(json.dumps(spec))
            cmd = [sys.executable, str(Path(__file__).resolve()),
                   "--worker", str(spec_path), "--out-dir", str(out_dir)]
            running.append((spec, subprocess.Popen(cmd, cwd=str(REPO), env=env)))
        still = []
        for spec, proc in running:
            rc = proc.poll()
            if rc is None:
                still.append((spec, proc))
            elif rc != 0:
                failed.append(cell_id(spec))
        running = still
        if running:
            import time
            time.sleep(1.0)
    if failed:
        raise SystemExit(f"cells failed: {failed}")


def converge(errors: list[tuple[int, float]]) -> dict[str, Any]:
    """Observed order + Richardson limit from a (panels, band_max) ladder.

    Constant panels on a curved body have ``h ~ panels**-0.5``; with the
    documented ``O(h**2)`` behaviour of this collocation scheme the expected
    panel exponent is ``-1``, which is what ``p`` estimates.  Reported, never
    used to *pass* a gate.
    """
    pts = sorted(errors)
    out: dict[str, Any] = {"ladder": [{"panels": n, "band_max": e} for n, e in pts]}
    if len(pts) >= 2:
        (n1, e1), (n2, e2) = pts[-2], pts[-1]
        if e1 > 0 and e2 > 0 and n2 > n1:
            p = math.log(e1 / e2) / math.log(n2 / n1)
            out["observed_panel_exponent"] = p
            out["ratio"] = e1 / e2
            # Richardson: e(n) = C n^-p -> limit at n2 from the last two rungs
            out["richardson_limit_band_max"] = abs(e2 - (e1 - e2) / ((n2 / n1) ** p - 1))
    return out


def build(out_dir: Path) -> dict[str, Any]:
    cdir = out_dir / "cells"
    specs = cells()
    have = {s["id"] if "id" in s else cell_id(s): s for s in specs
            if (cdir / f"{cell_id(s)}.json").exists()}
    records: list[dict[str, Any]] = []
    for s in specs:
        p = cdir / f"{cell_id(s)}.json"
        if p.exists():
            records.append(json.loads(p.read_text()))
    present = {r["id"] for r in records}
    missing = [cell_id(s) for s in specs if cell_id(s) not in present]

    ladders: dict[str, Any] = {}
    for ka in KA_EDGE:
        rows = [(r["panels"], be.band_max(r)) for r in records
                if abs(r["ka"] - ka) < 1e-9]
        if rows:
            ladders[f"{ka:g}"] = converge(rows)

    crit: dict[str, Any] = {}
    # R1: the 1024-panel rung must reproduce the band_extension run exactly.
    ref_path = HERE / "band_current_code" / "results.json"
    r1_meas: dict[str, Any] = {}
    if ref_path.exists():
        ref = {c["ka"]: c for c in json.loads(ref_path.read_text())["cells"]
               if c["kind"] == "extension" and abs(c["panels"] - 1024) < 1e-9}
        for r in records:
            if r["panels"] != 1024 or r["ka"] not in ref:
                continue
            d = max(abs(r[f"rel_error_{q}"] - ref[r["ka"]][f"rel_error_{q}"])
                    for q in QUANTITIES)
            r1_meas[f"ka{r['ka']:g}"] = d
    crit["R1_ladder_is_self_consistent"] = {
        "criterion": "1024-panel rungs reproduce band_current_code bit for bit "
                     "(max |delta rel_error| == 0)",
        "measured": r1_meas or None,
        "pass": bool(r1_meas) and all(v == 0.0 for v in r1_meas.values()),
    }
    # R2: ka = 4 error must fall under refinement.
    k4 = {r["panels"]: be.band_max(r) for r in records if abs(r["ka"] - 4.0) < 1e-9}
    crit["R2_ka4_error_falls_under_refinement"] = {
        "criterion": "band_max(ka=4, 4096) < band_max(ka=4, 1024)",
        "measured": k4 or None,
        "pass": (len(k4) >= 2 and k4[max(k4)] < k4[min(k4)]),
    }
    # R3: report the credited band edge; never widen it.
    ext = sorted([r for r in records if r["panels"] == 1024], key=lambda r: r["ka"])
    credited = [r["ka"] for r in ext if be.band_max(r) < GATE]
    first_fail = next((r["ka"] for r in ext if be.band_max(r) >= GATE), None)
    passes_at = None
    if first_fail is not None:
        hi = {r["panels"]: be.band_max(r) for r in records
              if abs(r["ka"] - first_fail) < 1e-9 and be.band_max(r) < GATE}
        passes_at = min(hi) if hi else None
    crit["R3_credited_band_edge_localised"] = {
        "criterion": "the largest predeclared ka that passes on the case mesh is "
                     "reported, together with the mesh (if any) at which the first "
                     "failing ka would pass",
        "measured": {"credited_ka_on_case_mesh": credited,
                     "first_failing_ka": first_fail,
                     "first_failing_ka_passes_at_panels": passes_at},
        "pass": first_fail is not None and bool(credited),
    }

    values = [c["pass"] for c in crit.values()]
    verdict = "FAIL" if any(v is False for v in values) else (
        "PASS" if all(v is True for v in values) else "INCOMPLETE")

    payload = {
        "driver": "band_edge_refinement",
        "case_code": "BEM-HULME-HEMISPHERE",
        "gate": GATE,
        "ka_edge": list(KA_EDGE),
        "subdivisions": list(SUBDIVISIONS),
        "cells": records,
        "ladders": ladders,
        "criteria": crit,
        "missing_cells": missing,
        "absence_reason": (None if not missing else
                           "cell(s) not returned in this session; every number "
                           "here comes from a committed cell artifact, nothing is "
                           "interpolated"),
        "verdict": verdict,
        "metadata": {
            "generated_utc": __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ",
                                                         __import__("time").gmtime()),
            "git_commit": be._git_head(),
            "git_dirty": be._git_dirty(),
            "host": __import__("platform").node(),
            "numpy": __import__("numpy").__version__,
            "python": __import__("platform").python_version(),
            "source_sha256": be._source_sha256(),
            "thread_env": {k: os.environ.get(k) for k in
                           ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                            "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                            "VECLIB_MAXIMUM_THREADS")},
        },
        "repro_cmd": "PYTHONPATH=src python benchmarks/verified/"
                     "casepool_bem_hulme_hemisphere/band_edge_refinement.py "
                     f"--out-dir {out_dir} --parallel 2",
    }
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--only", type=str, default=None,
                    help="comma separated cell ids (ka3.5_sub4,ka4_sub5,...)")
    ap.add_argument("--worker", default=None)
    ap.add_argument("--spec", action="store_true", help="print the cell list")
    args = ap.parse_args(argv)

    if args.spec:
        print(json.dumps([{"id": cell_id(s), "panels": panels(s), **s}
                          for s in cells()], indent=1))
        return 0
    if args.worker:
        spec = json.loads(Path(args.worker).read_text())
        cell = run_cell(spec)
        cdir = Path(args.out_dir) / "cells"
        cdir.mkdir(parents=True, exist_ok=True)
        (cdir / f"{cell['id']}.json").write_text(
            json.dumps(cell, indent=1, sort_keys=True))
        print(json.dumps({"id": cell["id"], "panels": cell["panels"],
                          "band_max": be.band_max(cell),
                          "radiation_relative_residual":
                              cell.get("radiation_relative_residual"),
                          "radiation_solve_method":
                              cell.get("radiation_solve_method")}))
        return 0

    only = ({s.strip() for s in args.only.split(",") if s.strip()}
            if args.only else None)
    run_missing(args.out_dir, args.parallel, only)
    payload = build(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "results.json").write_text(
        json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: v["pass"] for k, v in payload["criteria"].items()}, indent=1))
    print("VERDICT", payload["verdict"], "missing", payload["missing_cells"])
    return 0 if payload["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())