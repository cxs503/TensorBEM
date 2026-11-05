"""Feasibility + attribution probe: who is wrong at f/a = 1.05?"""
import math, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from tensorbem.analytic import ursell_cylinder_coefficients
from tensorbem.free_surface_solver import radiation_reference, solve_radiation_indirect
from tensorbem.geometry import PanelMesh

G, RHO, A = 9.80665, 1025.0, 1.0

def circle(n, r, d):
    th = 2*np.pi*np.arange(n)/n
    return PanelMesh(np.column_stack((r*np.cos(th), -d+r*np.sin(th))))

def comp(v, d):
    a = np.asarray(v, float).ravel()
    return float(a[0] if d == "sway" else a[1])

def exact(ka, depth_ratio, direction):
    w = ka/A
    mu, nu = ursell_cylinder_coefficients(ka, A/(depth_ratio*A))
    Aa = mu * RHO*math.pi*A**2
    Bb = nu * RHO*math.pi*A**2*w
    return (Aa, Bb)

for ka in (1.0,):
    for dr in (1.05,):
        w = ka/A
        # 1) analytic truncation convergence
        t0 = time.time()
        ref = None
        for terms in (20, 40, 60, 80, 120, 160, 200):
            mu, nu = ursell_cylinder_coefficients(ka, 1.0/dr, terms)
            Af, Bf = mu*RHO*math.pi, nu*RHO*math.pi*w
            print(f"terms={terms:4d} A={Af:.10f} B={Bf:.10f} dt={time.time()-t0:.2f}s", flush=True)
        # 2) panel ladder
        for n in (32, 64, 128, 256):
            t1 = time.time()
            r = solve_radiation_indirect(circle(n, A, dr*A), w, "heave", density=RHO, panel_nodes=24)
            print(f"panel {n:4d} A={comp(r.added_mass,'heave'):.8f} B={comp(r.damping,'heave'):.8f} t={time.time()-t1:.1f}s", flush=True)
        # 3) MFS ladder
        for s in (48, 96, 192, 384):
            t1 = time.time()
            r = radiation_reference(circle(128, A, dr*A), w, "heave", source_count=s, density=RHO)
            print(f"mfs   {s:4d} A={comp(r.added_mass,'heave'):.8f} B={comp(r.damping,'heave'):.8f} t={time.time()-t1:.1f}s", flush=True)
