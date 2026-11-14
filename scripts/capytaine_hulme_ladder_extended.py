#!/usr/bin/env python3
"""Extended capytaine refinement ladder for the anchor ``BEM-HULME-HEMISPHERE``.

Why this file exists
--------------------
``benchmarks/verified/casepool_bem_hulme_hemisphere/capytaine_excitation.json``
freezes capytaine -- the *only* third-party code that adjudicates the anchor's
reference series -- at 1600 panels, and claims convergence through the gate
``I3_capytaine_mesh_converged`` = "worst drift of the last ladder step
(1024 -> 1600 panels) below 1 %".

A *step* below 1 % is not the same statement as *converged to* 1 %: a sequence
that still moves 0.67 % between its last two rungs can sit several percent away
from its own continuum limit.  Since the anchor's three-way comparison quotes a
worst code-to-code spread of 3.36 %, that distinction decides how much of the
spread is evidence about the *reference* and how much is the independent code's
own discretisation error.

This producer therefore extends the ladder to seven rungs (256 -> 4096 panels,
i.e. 4x the anchor's case mesh) at the full case band, storing every quantity at
every level so that the convergence order and the continuum limit can be
extrapolated from the artifact alone (``tensorbem.benchmarks.
hemisphere_independent_convergence`` does the extrapolation; this file only
measures).

Provenance and the excitation-force convention (diffraction part + Froude-Krylov
part summed by hand, capytaine >= 3.0) are inherited from
``scripts/capytaine_hulme_excitation_ref.py`` -- imported, not re-implemented,
so the two artifacts cannot drift apart in convention.

Usage
-----
    python scripts/capytaine_hulme_ladder_extended.py \
        --out benchmarks/verified/casepool_bem_hulme_hemisphere/\
capytaine_ladder_extended.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
_SPEC = importlib.util.spec_from_file_location(
    "_capytaine_hulme_excitation_ref", HERE / "capytaine_hulme_excitation_ref.py"
)
assert _SPEC and _SPEC.loader
_BASE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _BASE
_SPEC.loader.exec_module(_BASE)

#: seven rungs; the first four reproduce the frozen artifact exactly (256, 576,
#: 1024, 1600 panels) so the extension is anchored to already-published numbers
LEVELS = ((16, 32), (24, 48), (32, 64), (40, 80), (48, 96), (56, 112), (64, 128))

DEFAULT_OUT = Path(
    "benchmarks/verified/casepool_bem_hulme_hemisphere/capytaine_ladder_extended.json"
)


def build(out_path: Path, levels: tuple[tuple[int, int], ...] = LEVELS) -> dict[str, Any]:
    import capytaine as cpt

    solver = cpt.BEMSolver()
    rows: list[dict[str, Any]] = []
    for resolution in levels:
        started = time.time()
        mesh = cpt.mesh_sphere(
            radius=_BASE.RADIUS, center=(0.0, 0.0, 0.0), resolution=resolution
        ).immersed_part()
        level = {"resolution": list(resolution), **_BASE._sweep(cpt, solver, mesh)}
        level["wall_seconds"] = time.time() - started
        rows.append(level)
        print(
            f"level {level['n_faces']:5d} panels  "
            f"vol_err {100 * level['volume_rel_error']:+.3f}%  "
            f"{level['wall_seconds']:.1f} s",
            flush=True,
        )
    payload = {
        "metadata": _BASE._provenance(cpt),
        "case_code": "BEM-HULME-HEMISPHERE",
        "purpose": (
            "extended refinement ladder of the anchor's independent code, so "
            "that its convergence order and continuum limit are extrapolable "
            "from the artifact (the frozen capytaine_excitation.json stops at "
            "1600 panels and only evidences a *step*, not convergence)"
        ),
        "geometry": (
            "surface-piercing hemisphere, radius 1.0 m, flat face on the mean "
            "free surface (capytaine mesh_sphere(...).immersed_part())"
        ),
        "nabla_exact": _BASE._norm_volume(),
        "levels": rows,
        "convergence": _BASE._convergence(rows),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--max-levels",
        type=int,
        default=len(LEVELS),
        help="truncate the ladder (development aid; the artifact must use all)",
    )
    args = parser.parse_args(argv)
    payload = build(args.out, LEVELS[: args.max_levels])
    print(
        json.dumps(
            {
                "out": str(args.out),
                "levels": [lvl["n_faces"] for lvl in payload["levels"]],
                "convergence": payload["convergence"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())