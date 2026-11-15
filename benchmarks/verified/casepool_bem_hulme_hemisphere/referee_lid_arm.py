#!/usr/bin/env python3
"""Lid-closed referee arm for CASE-BEM-HULME-HEMISPHERE (irregular-frequency test).

The companion producer ``referee_robustness.py`` shows that capytaine's error
versus the Hulme (1982) series *sign-crosses* between ``ka = 2.54`` and
``ka = 2.58`` (B33: ``-68 %`` -> ``+108 %`` on 4096 panels) while staying below
8 % a quarter of a wave number away.  A smooth analytic solution cannot produce
that; a *discrete* interior-Dirichlet pole (an irregular frequency of the
referee's panel system) can.

This driver tests that mechanism directly instead of inferring it: capytaine
3.1 accepts ``lid_mesh``, i.e. a closing disk on the waterplane that removes the
interior Dirichlet eigenproblem the pole comes from.  If the blow-up is an
irregular frequency, closing the lid must remove it *without* changing anything
far from the pole.  If it is a genuine under-resolution of the referee, the lid
changes nothing.

Pre-registered gate
-------------------
``L1_lid_removes_spike`` -- with the lid closed, the referee's worst error among
``ka in {2.54, 2.58, 2.62}`` on B33 must drop below 10 % on the 4096-panel mesh
(it is 108 % without the lid), and the ``ka = 2.30 / 2.86`` control cells must
move by less than 2 % absolute, i.e. the lid fixes the pole and nothing else.

Usage::

    PYTHONPATH=src python benchmarks/verified/casepool_bem_hulme_hemisphere/\\
        referee_lid_arm.py --out-dir <dir>
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

RHO = 1025.0
GRAVITY = 9.80665
RADIUS = 1.0
NABLA = 2.0 / 3.0 * math.pi * RADIUS**3
MAGNITUDES = ("A11", "B11", "A33", "B33")

#: Pole band plus two control cells a quarter wave number away.
KA_POLE = (2.54, 2.58, 2.62)
KA_CONTROL = (2.30, 2.86)
#: Full case band + the extended band: the lid arm is also the *referee accuracy*
#: arm, because a lid-closed capytaine is a usable third-party referee where the
#: lidless one is not (see ``L1a``/``L1b`` in the merge step).
KA_BAND = (1.0, 2.0, 2.2, 2.54, 2.58, 2.62, 3.0, 3.5, 4.0)
KA_GRID = KA_CONTROL[:1] + KA_POLE + KA_CONTROL[1:]
MESHES = (4, 5)  # 1024 / 4096 wetted panels
REF_N_TERMS = 144
LID_SPIKE_GATE = 0.10
LID_CONTROL_DRIFT = 0.02


def hull_mesh(subdivisions: int):
    from tensorbem.geometry3d import hemisphere_hull_mesh

    return hemisphere_hull_mesh(int(subdivisions), radius=RADIUS)


def disk_lid(radius: float = RADIUS, rings: int = 8, sectors: int = 64):
    """Triangulated disk on ``z = 0`` closing the waterplane (the Nemoh lid).

    Rings are geometrically graded towards the rim so the lid panels match the
    wetted-panel size at the waterline, which is where the lid meets the hull.
    """
    verts: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0)]
    ring_index: list[list[int]] = []
    for k in range(1, rings + 1):
        r = radius * (k / rings) ** 1.5
        idx = []
        for j in range(sectors):
            ang = 2.0 * math.pi * j / sectors
            idx.append(len(verts))
            verts.append((r * math.cos(ang), r * math.sin(ang), 0.0))
        ring_index.append(idx)
    faces: list[list[int]] = []
    first = ring_index[0]
    for j in range(sectors):
        faces.append([0, first[j], first[(j + 1) % sectors]])
    for a, b in zip(ring_index[:-1], ring_index[1:]):
        for j in range(sectors):
            jn = (j + 1) % sectors
            faces.append([a[j], b[j], b[jn]])
            faces.append([a[j], b[jn], a[jn]])
    return np.asarray(verts, dtype=float), np.asarray(faces, dtype=np.int64)


def capytaine_radiation(subdivisions: int, ka: float, lid: bool) -> dict[str, Any]:
    import capytaine as cpt

    mesh = hull_mesh(subdivisions)
    kwargs: dict[str, Any] = {}
    if lid:
        lv, lf = disk_lid()
        kwargs["lid_mesh"] = cpt.Mesh(vertices=lv, faces=lf)
    body = cpt.FloatingBody(
        mesh=cpt.Mesh(vertices=np.asarray(mesh.vertices, dtype=float),
                      faces=np.asarray(mesh.faces, dtype=np.int64)),
        dofs=cpt.rigid_body_dofs(rotation_center=(0.0, 0.0, 0.0)),
        **kwargs,
    )
    solver = cpt.BEMSolver()
    omega = math.sqrt(GRAVITY * ka / RADIUS)
    out: dict[str, Any] = {
        "subdivisions": int(subdivisions),
        "panels": int(np.asarray(mesh.faces).shape[0]),
        "lid": bool(lid),
        "lid_panels": int(sum(len(f) - 2 for f in disk_lid()[1])) if lid else 0,
    }
    t0 = time.time()
    for dof, ka_key, kb_key in (("Surge", "A11", "B11"), ("Heave", "A33", "B33")):
        problem = cpt.RadiationProblem(
            body=body, wavenumber=ka / RADIUS, water_depth=math.inf,
            radiating_dof=dof, rho=RHO, g=GRAVITY,
        )
        result = solver.solve(problem)
        out[ka_key] = float(result.added_mass[dof]) / (RHO * NABLA)
        out[kb_key] = float(result.radiation_damping[dof]) / (RHO * omega * NABLA)
    out["seconds"] = time.time() - t0
    out["first_irregular_wavenumber"] = float(body.first_irregular_frequency_estimate())
    return out


def hulme_reference(ka: float, n_terms: int = REF_N_TERMS) -> dict[str, float]:
    from tensorbem.benchmarks.hulme_hemisphere_case import reference as _ref

    got = _ref(float(ka), int(n_terms))
    return {m: float(got[f"{m}_dimless"]) for m in MAGNITUDES}


def run(ka_list, meshes=MESHES) -> dict[str, Any]:
    rows = []
    for ka in ka_list:
        ref = hulme_reference(ka)
        row: dict[str, Any] = {"ka": ka, "reference": ref}
        for sub in meshes:
            for lid in (False, True):
                cell = capytaine_radiation(sub, ka, lid)
                cell["rel_error"] = {m: (cell[m] - ref[m]) / ref[m] for m in MAGNITUDES}
                row[f"sub{sub}_{'lid' if lid else 'open'}"] = cell
        rows.append(row)
        print(json.dumps({"ka": ka,
                          "open4096": {m: round(row['sub5_open']['rel_error'][m], 4) for m in MAGNITUDES},
                          "lid4096": {m: round(row['sub5_lid']['rel_error'][m], 4) for m in MAGNITUDES}}),
              flush=True)
    return {"probe": "lid_arm", "ka_grid": list(ka_list), "meshes": list(meshes), "rows": rows}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=Path("."))
    ap.add_argument("--band", action="store_true", help="run the full ka band instead of the pole probe")
    ap.add_argument("--ka-start", type=float, default=None)
    ap.add_argument("--ka-stop", type=float, default=None)
    args = ap.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    kas = list(KA_BAND if args.band else KA_GRID)
    if args.ka_start is not None:
        kas = [k for k in kas if k >= args.ka_start - 1e-12]
    if args.ka_stop is not None:
        kas = [k for k in kas if k <= args.ka_stop + 1e-12]
    payload = run(kas)
    payload["generated_at"] = _dt.datetime.now(_dt.timezone.utc).isoformat()
    payload["environment"] = {"python": platform.python_version(),
                              "numpy": np.__version__,
                              "machine": platform.machine()}
    tag = "band" if args.band else "pole"
    path = args.out_dir / (f"referee_robustness_part_lid_{tag}_{kas[0]:g}_{kas[-1]:g}.json"
                           if kas else f"referee_robustness_part_lid_{tag}.json")
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())