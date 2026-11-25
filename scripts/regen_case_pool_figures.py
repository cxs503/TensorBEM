"""Regenerate the figures of a registered case-pool anchor from its committed artefact.

Why this exists
---------------
``case_pool_registry`` declares, for every anchor, the figures its case document
promises.  The figure files are *derived* artifacts, and derived artifacts get
lost (deleted by a repository gate, dropped from a squash, never added at all);
when that happens the registry goes red and the case document shows broken image
links, even though the numbers in ``results.json`` are intact.

Re-running the driver is the canonical fix, but it re-solves the whole case.  This
script re-renders the figures from the committed JSON instead: the plotting
functions of each driver are pure functions of the artefact dict, so the PNG is
reproduced from exactly the numbers the case was accepted on -- the figure cannot
drift away from the verdict it illustrates.

Usage::

    PYTHONPATH=src python3 scripts/regen_case_pool_figures.py CASE-BEM-LAPLACE-CYLINDER
    PYTHONPATH=src python3 scripts/regen_case_pool_figures.py --all

Cases whose driver exposes no separable plotting entry point are reported as
``skipped`` (this is a maintenance tool, not a second implementation of the
benchmarks), and the exit status is non-zero only when a declared figure is still
missing afterwards.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tensorbem.benchmarks import case_pool_registry as registry


def _render_cylinder(entry, artifact: dict) -> list[str]:
    from tensorbem.benchmarks import case_cylinder_stream as driver

    out = entry.path(entry.figures[0]).parent
    driver.plot_field(artifact, out / "field.png")
    driver.plot_convergence(artifact, out / "convergence.png")
    driver.plot_near_wall(artifact, out / "near_wall.png")
    driver.plot_multipole(artifact, out / "multipole.png")
    driver.plot_circulation(artifact, out / "circulation.png")
    return [Path(rel).name for rel in entry.figures]


def _render_annulus(entry, artifact: dict) -> list[str]:
    from tensorbem.benchmarks import eccentric_annulus as driver

    written = driver.write_figures(artifact, entry.path(entry.figures[0]).parent)
    return [Path(p).name for p in written]


def _render_ellipse(entry, artifact: dict) -> list[str]:
    from tensorbem.benchmarks import case_ellipse_2d_added_mass as driver

    return [Path(p).name for p in driver._render_figures(artifact, entry.path(entry.figures[0]).parent)]


def _render_cummins(entry, artifact: dict) -> list[str]:
    from tensorbem.benchmarks import case_bem_cummins_wigley as driver

    out = entry.path(entry.figures[0]).parent
    label = list(artifact["settings"]["meshes"])[-1]
    metadata = artifact["sweeps"][label]
    mesh_keys = ("label", "n_sections", "n_levels", "panel_count", "max_panel_scale")
    sweep = driver.load_sweep(out / f"sweep_{label}.npz",
                              **{key: metadata[key] for key in mesh_keys})
    name = driver._make_figure({label: sweep}, artifact, out)
    return [name] if name else []


RENDERERS = {
    "CASE-BEM-LAPLACE-CYLINDER": _render_cylinder,
    "CASE-BEM-CUMMINS-WIGLEY": _render_cummins,
    "CASE-BEM-ANNULUS-LAPLACE": _render_annulus,
    "CASE-BEM-ELLIPSE-2D-ADDED-MASS": _render_ellipse,
}


def regenerate(code: str) -> dict:
    entry = registry.resolve(code)
    if not entry.figures:
        return {"case_code": entry.case_code, "status": "skipped", "reason": "no figures declared"}
    renderer = RENDERERS.get(entry.case_code)
    if renderer is None:
        return {"case_code": entry.case_code, "status": "skipped", "reason": "no renderer"}
    # The annulus registry entry covers two studies. Its figures accompany
    # the eccentric study, whose record sits alongside the figures directory.
    if entry.case_code == "CASE-BEM-ANNULUS-LAPLACE":
        record = Path(entry.figures[0]).parents[1] / "results.json"
        artifact = entry.artifact_payload(str(record))
    else:
        artifact = entry.artifact_payload()
    entry.path(entry.figures[0]).parent.mkdir(parents=True, exist_ok=True)
    names = renderer(entry, artifact)
    missing = [rel for rel in entry.figures if not entry.path(rel).is_file()]
    return {
        "case_code": entry.case_code,
        "status": "restored" if not missing else "incomplete",
        "rendered": names,
        "missing": missing,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", nargs="*", help="anchor slug / pool code")
    parser.add_argument("--all", action="store_true", help="every registered case with figures")
    args = parser.parse_args(argv)

    codes = [e.case_code for e in registry.list_cases() if e.figures] if args.all else args.cases
    if not codes:
        parser.error("give at least one case code or --all")

    results = [regenerate(code) for code in codes]
    for row in results:
        print(json.dumps(row, sort_keys=True))
    return 1 if any(row["status"] == "incomplete" for row in results) else 0


if __name__ == "__main__":
    sys.exit(main())