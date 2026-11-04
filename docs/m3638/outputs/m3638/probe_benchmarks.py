"""Probe runtime + quantity keys of the two 3-D free-surface benchmarks (m3638).

Read-only: runs the existing benchmark entry points and prints the keys of the
row dictionaries, so the registry entries can be written against real names.
"""

import json
import time

from tensorbem.benchmarks import floating_hemisphere, submerged_sphere_wave

t0 = time.perf_counter()
fh = floating_hemisphere.run_refinement_study(subdivisions=(3,), ka=1.0)
t1 = time.perf_counter()
print(f"hulme subdivisions=3 (256 panels): {t1 - t0:.2f} s")
row = fh["levels"][0]
print("hulme keys:", sorted(row))
print(json.dumps({k: v for k, v in row.items() if isinstance(v, (int, float))}, indent=1))

ss = submerged_sphere_wave.run_refinement_study(subdivisions=(2,), ka=0.8)
t2 = time.perf_counter()
print(f"wang-linton subdivisions=2 (320 panels): {t2 - t1:.2f} s")
row2 = ss["levels"][0]
print("wls keys:", sorted(row2))
print(json.dumps({k: v for k, v in row2.items() if isinstance(v, (int, float))}, indent=1))

irr = floating_hemisphere.run_irregular_frequency_study(
    subdivision=3, ka_values=(2.58,)
)
t3 = time.perf_counter()
print(f"chief study sub=3 ka=2.58: {t3 - t2:.2f} s")
print(json.dumps(irr[0], indent=1))