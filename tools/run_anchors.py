"""Analytical anchors of the study (each anchor writes its own JSON shard)."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
from tensorbem.benchmarks import interior_panel_suppression as study

RUNS = Path("runs/interior_panel_suppression")
p = argparse.ArgumentParser()
p.add_argument("--which", required=True,
               choices=("ursell", "box", "hemisphere", "sphere320", "sphere1280",
                        "sphere320_interior"))
a = p.parse_args()
t0 = time.time()
if a.which == "ursell":
    payload = study.anchor_ursell()
elif a.which == "box":
    payload = {"box": study.anchor_box(),
               "box_low_frequency_rao": study.anchor_box_low_frequency_rao()}
elif a.which == "hemisphere":
    payload = {"hemisphere": study.anchor_hemisphere_accuracy(level=256, config="lid"),
               "hemisphere_none": study.anchor_hemisphere_accuracy(level=256, config="none")}
elif a.which == "sphere320":
    payload = study.anchor_sphere(panel_levels=(320,), ka=0.6)
elif a.which == "sphere1280":
    payload = study.anchor_sphere(panel_levels=(1280,), ka=0.6)
else:
    payload = study.anchor_sphere(panel_levels=(320,), ka=0.6, interior_rows=True)
payload["wall_seconds"] = time.time() - t0
out = RUNS / f"anchor_{a.which}.json"
RUNS.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print("[ok]", out, round(time.time() - t0, 1), "s", flush=True)
