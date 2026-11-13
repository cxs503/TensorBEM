"""Repeatability of the FMM/dense wall-time ratio on the shared 99 host."""
import sys, json, statistics, time
import numpy as np
sys.path.insert(0, 'src')
from tensorbem.benchmarks.fmm_cost_audit import dense_matvec, surface_panels
from tensorbem.fmm import FMMOperator

out = {}
for level, ml in ((3, 128), (4, 128)):
    c, a = surface_panels(level)
    op = FMMOperator(p=2, max_leaf=ml); op.setup(c, a)
    ratios, tf, td = [], [], []
    for rep in range(5):
        s = time.process_time(); op.matvec(c, c, a); t1 = time.process_time() - s
        s = time.process_time(); dense_matvec(c, c, a); t2 = time.process_time() - s
        ratios.append(t1 / t2); tf.append(t1); td.append(t2)
    out[f"L{level}_ml{ml}_p2"] = dict(ratios=ratios, median=statistics.median(ratios),
                                      lo=min(ratios), hi=max(ratios), t_fmm=tf, t_dense=td)
    print(level, ml, "ratios:", [f"{r:.2f}" for r in ratios],
          f"median={statistics.median(ratios):.2f} lo={min(ratios):.2f} hi={max(ratios):.2f}", flush=True)
json.dump(out, open('runs/noise_ratio.json', 'w'), indent=1)
