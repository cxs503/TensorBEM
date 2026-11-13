"""Interleaved A/B of the O(p^3) vs O(p^2) harmonic ladder inside the FMM matvec.

A = old implementation (runs/old_harmonics.py), B = shipped implementation.
Order alternates A,B,B,A to cancel drift; CPU time is reported with min and median.
"""
import sys, time, json, statistics
import numpy as np
sys.path.insert(0, 'src')
import tensorbem.fmm as F
from tensorbem.geometry3d import icosphere
from tensorbem.fmm import FMMOperator

ns = dict(vars(F)); exec(open('runs/old_harmonics.py').read(), ns)
old = ns['all_real_spherical_harmonics']
new = F.all_real_spherical_harmonics
import types

m = {3: icosphere(subdivisions=3, radius=1.0), 4: icosphere(subdivisions=4, radius=1.0)}
out = {}
for level in (3, 4):
    c = np.asarray(m[level].centroids, float); q = np.asarray(m[level].areas, float)
    for p in (2, 4):
        op = FMMOperator(p=p, max_leaf=128); op.setup(c, q)
        runs = {'A': [], 'B': []}
        for rep in range(4):
            for tag in ('A', 'B', 'B', 'A'):
                saved = F.all_real_spherical_harmonics
                F.all_real_spherical_harmonics = old if tag == 'A' else new
                s = time.process_time(); op.matvec(c, c, q); runs[tag].append(time.process_time() - s)
                F.all_real_spherical_harmonics = saved
        a, b = runs['A'], runs['B']
        out[f"L{level}_p{p}"] = dict(A_min=min(a), B_min=min(b), A_med=statistics.median(a),
                                     B_med=statistics.median(b), speedup_med=statistics.median(a)/statistics.median(b))
        print(f"L{level} p={p}: A min={min(a):.4f} med={statistics.median(a):.4f} | "
              f"B min={min(b):.4f} med={statistics.median(b):.4f} | speedup={statistics.median(a)/statistics.median(b):.2f}x", flush=True)
json.dump(out, open('runs/ab_harmonics.json', 'w'), indent=1)
