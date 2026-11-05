"""Fresh recomputation of the 3-D free-surface numbers quoted in
docs/marine_bem_benchmarks.md tables 10-12."""
import json, time, platform
import numpy as np
from tensorbem.benchmarks import floating_hemisphere as fh, submerged_sphere_wave as sw

out = {"environment": {"python": platform.python_version(), "numpy": np.__version__}}
t = time.time()
hem = fh.run_refinement_study(subdivisions=(1, 2, 3), ka=1.0, radius=1.0, density=1025.0, gravity=9.80665)
out["hulme_refinement"] = hem
out["timing_hulme_refinement_s"] = time.time() - t

t = time.time()
chief = fh.run_irregular_frequency_study(subdivision=3, ka_values=(2.58,), radius=1.0, density=1025.0, gravity=9.80665)
out["hulme_chief"] = chief
out["timing_hulme_chief_s"] = time.time() - t

t = time.time()
sph = sw.run_refinement_study(subdivisions=(0, 1, 2), ka=0.8, radius=1.0, submergence=1.5, density=1025.0, gravity=9.80665)
out["submerged_sphere_refinement"] = sph
out["timing_sphere_refinement_s"] = time.time() - t

with open("outputs/verif_m3264/measured.json", "w") as f:
    json.dump(out, f, indent=1, default=float)
print(json.dumps({k: v for k, v in out.items() if k.startswith("timing")}, indent=1))
