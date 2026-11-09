"""Driver for the interior (lid) panel suppression verification study.

Phases are separated so the expensive level can be re-run or sharded without
recomputing the cheap ones.  Every phase writes strict JSON under
``runs/interior_panel_suppression/`` and prints a progress line to stdout, so a
cluster shard can be parsed from its log.

    PYTHONPATH=src python tools/run_interior_panel_study.py --phase cheap
    PYTHONPATH=src python tools/run_interior_panel_study.py --phase fine
    PYTHONPATH=src python tools/run_interior_panel_study.py --phase anchors
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from tensorbem.benchmarks import interior_panel_suppression as study

OUT = Path("runs/interior_panel_suppression")


def _dump(name: str, payload: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[wrote] {path} ({path.stat().st_size} B)", flush=True)


def phase_cheap() -> None:
    t0 = time.time()
    payload = study.run(levels=(64, 256), configs=study.CONFIGS)
    payload["wall_seconds"] = time.time() - t0
    _dump("sweep_cheap.json", payload)
    print(
        "[phase cheap]",
        {k: v["rao"]["max_curvature"] for k, v in payload["summary"].items()},
        flush=True,
    )


def phase_fine() -> None:
    t0 = time.time()
    grid = (0.5, 1.0, 1.5, 2.0, 2.45, 2.5, 2.55, 2.6, 2.65, 2.7, 2.9)
    payload = study.run(levels=(1024,), configs=("none", "lid"), ka_filter=grid)
    payload["wall_seconds"] = time.time() - t0
    _dump("sweep_fine_1024.json", payload)


def phase_anchors() -> None:
    t0 = time.time()
    payload = {
        "hemisphere": study.anchor_hemisphere_accuracy(level=1024, config="lid"),
        "sphere": study.anchor_sphere(panel_levels=(320, 1280), ka=0.6),
        "sphere_with_interior_rows": study.anchor_sphere(
            panel_levels=(320,), ka=0.6, interior_rows=True
        ),
        "ursell": study.anchor_ursell(),
        "box": study.anchor_box(),
        "box_low_frequency_rao": study.anchor_box_low_frequency_rao(),
        "wall_seconds": time.time() - t0,
    }
    _dump("anchors.json", payload)
    print(
        "[phase anchors]",
        {
            "sphere_max_a33_err": payload["sphere"]["max_added_mass_rel_error"],
            "ursell_a_err": payload["ursell"]["added_mass_rel_error"],
            "ursell_b_err": payload["ursell"]["damping_rel_error"],
            "box_c33_err": payload["box"]["c33_rel_error"],
        },
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("cheap", "fine", "anchors"), required=True)
    args = parser.parse_args()
    {"cheap": phase_cheap, "fine": phase_fine, "anchors": phase_anchors}[args.phase]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())