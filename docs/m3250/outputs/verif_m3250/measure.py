"""Measure the three 3-D free-surface verification tables (docs tables 10/11/12).

Writes one JSON artefact per table plus a timing summary; nothing is transcribed
by hand downstream (the pins, the docs and the report read these files).
"""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import numpy as np

from tensorbem.benchmarks import floating_hemisphere, submerged_sphere_wave

OUT = Path(__file__).resolve().parent


def _dump(name: str, payload: object) -> None:
    path = OUT / name
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    print(f"[written] {path}", flush=True)


def main() -> None:
    timings: dict[str, float] = {}

    t0 = time.time()
    hemi = floating_hemisphere.run_refinement_study(
        subdivisions=(1, 2, 3), ka=1.0, radius=1.0, density=1025.0
    )
    timings["hemisphere_refinement_s"] = time.time() - t0
    _dump("hemisphere_refinement.json", hemi)

    t0 = time.time()
    sphere = submerged_sphere_wave.run_refinement_study(
        subdivisions=(0, 1, 2), ka=0.8, radius=1.0, submergence=1.5, density=1025.0
    )
    timings["submerged_sphere_refinement_s"] = time.time() - t0
    _dump("submerged_sphere_refinement.json", sphere)

    t0 = time.time()
    chief = floating_hemisphere.run_irregular_frequency_study(
        subdivision=3, ka_values=(2.58,), radius=1.0, density=1025.0
    )
    timings["hemisphere_chief_s"] = time.time() - t0
    _dump("hemisphere_chief.json", chief)

    _dump("timings.json", timings)
    print(json.dumps(timings, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001 - the artefact must record the failure
        (OUT / "measure_failure.txt").write_text(traceback.format_exc())
        raise