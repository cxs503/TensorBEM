"""Frequency sweep for one 3-D free-surface body (kept under the 120 s shell cap)."""
import json
import sys
import time

from tensorbem.benchmarks import floating_hemisphere as fh
from tensorbem.benchmarks import submerged_sphere_wave as sw

body = sys.argv[1]
t = time.time()
if body == "hulme":
    rows = fh.run_frequency_sweep(
        subdivision=3, ka_values=(0.3, 0.6, 1.0, 1.5, 2.0, 3.0), density=1025.0, gravity=9.80665
    )
elif body == "sphere":
    rows = sw.run_frequency_sweep(
        subdivision=2, ka_values=(0.2, 0.5, 0.8, 1.5), submergence=1.5,
        density=1025.0, gravity=9.80665,
    )
else:
    raise SystemExit(f"unknown body: {body}")
with open(f"outputs/verif_m3264/sweep_{body}.json", "w") as handle:
    json.dump({"body": body, "wall_s": time.time() - t, "rows": rows}, handle, indent=1, default=float)
print(body, "done in %.1f s" % (time.time() - t))
