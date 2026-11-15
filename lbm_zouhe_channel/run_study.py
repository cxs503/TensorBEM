"""Cross-validation study: Zou-He channel inlet/outlet reconstruction
(anchor XVAL-LBM-ZOU-HE-CHANNEL-INLET-OUTLET-RECONSTRUCTI).

Quantifies, against the analytic plane-Poiseuille solution, the error of

  V1  the literal anchor recipe   (inlet parabola with ``yp = i - 1.5``,
      reconstruction applied to the whole inlet/outlet column, ``rho_out = 1``)
  V2  same inlet parabola, reconstruction restricted to the fluid rows
  V3  corrected inlet parabola    (``yp = i - 0.5``, vanishes on both wall
      mid-planes), reconstruction restricted to the fluid rows
  V4  periodic body-force-driven channel (no inlet/outlet at all) -- the
      lattice-discretisation floor that no boundary condition can beat

plus a Mach sweep at fixed resolution, where the compressibility error of a
velocity-inlet/pressure-outlet pair is expected to grow with the imposed
velocity.

Usage (resumable, results are merged into results/zouhe_study.json):
    python3 run_study.py --res 16 32          # main grid study
    python3 run_study.py --mach               # Mach sweep at ny=32

Outputs (all numbers programmatic, nothing typed by hand):
    results/zouhe_study.json, results/table_main.md, results/table_mach.md,
    results/fig1..fig4*.png
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import channel
from zou_he import poiseuille_profile

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
JSON_PATH = os.path.join(RES, "zouhe_study.json")
os.makedirs(RES, exist_ok=True)

NU = 0.05
U_PEAK = 0.08
MACH_U = [0.02, 0.04, 0.08]
MACH_NY = 32
TAG = ["V1_anchor_literal", "V2_anchor_profile_interior_rows",
       "V3_corrected_profile", "V4_periodic_forced_baseline"]
STYLE = {TAG[0]: dict(color="crimson", ls="-", marker="o"),
         TAG[1]: dict(color="darkorange", ls="--", marker="s"),
         TAG[2]: dict(color="navy", ls="-", marker="^"),
         TAG[3]: dict(color="green", ls=":", marker="v")}


def _clean(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, (np.floating, np.integer)):
            out[k] = float(v)
        elif isinstance(v, float) and not np.isfinite(v):
            out[k] = None
        else:
            out[k] = v
    return out


def _load():
    if os.path.exists(JSON_PATH):
        with open(JSON_PATH) as fh:
            return json.load(fh)
    return dict(nu=NU, u_peak=U_PEAK, runs=[], mach_runs=[])


def _save(b):
    with open(JSON_PATH, "w") as fh:
        json.dump(b, fh, indent=2)


def _upsert(lst, m):
    key = (m["variant"], m["config"]["ny"], m["config"].get("u_peak"))
    lst[:] = [r for r in lst
              if (r["variant"], r["config"]["ny"], r["config"].get("u_peak")) != key]
    lst.append(m)


def run_grid(bundle, resolutions):
    for ny in resolutions:
        nx = 2 * ny
        cfgs = [
            (TAG[0], dict(profile="reference", rows="all")),
            (TAG[1], dict(profile="reference", rows="interior")),
            (TAG[2], dict(profile="corrected", rows="interior")),
        ]
        for name, kw in cfgs:
            t = time.time()
            m = _clean(channel.run_zouhe_channel(
                ny=ny, nx=nx, u_peak=U_PEAK, nu=NU,
                max_steps=60000, tol=1e-8, check_every=500, **kw))
            m["variant"] = name
            m["wall_seconds"] = round(time.time() - t, 2)
            _upsert(bundle["runs"], m)
            _save(bundle)
            print(f"{name:32s} ny={ny:3d} steps={m['steps']:6d} "
                  f"err_l2={m['err_l2_profile']:.5f} ({m['wall_seconds']}s)", flush=True)
        t = time.time()
        m = _clean(channel.run_forced_channel(
            ny=ny, nx=nx, u_peak=U_PEAK, nu=NU,
            max_steps=60000, tol=1e-8, check_every=500))
        m["variant"] = TAG[3]
        m["wall_seconds"] = round(time.time() - t, 2)
        _upsert(bundle["runs"], m)
        _save(bundle)
        print(f"{TAG[3]:32s} ny={ny:3d} steps={m['steps']:6d} "
              f"err_l2={m['err_l2_profile']:.5f} ({m['wall_seconds']}s)", flush=True)


def run_mach(bundle):
    for u in MACH_U:
        for name, kw in ((TAG[0], dict(profile="reference", rows="all")),
                         (TAG[2], dict(profile="corrected", rows="interior"))):
            m = _clean(channel.run_zouhe_channel(
                ny=MACH_NY, nx=2 * MACH_NY, u_peak=u, nu=NU,
                max_steps=60000, tol=1e-8, check_every=500, **kw))
            m["variant"] = name
            _upsert(bundle["mach_runs"], m)
            _save(bundle)
            print(f"mach {name:28s} u={u:.2f} err_l2={m['err_l2_profile']:.5f}", flush=True)


def _md_table(rows, cols):
    head = "| " + " | ".join(cols) + " |\n|" + "|".join(["---"] * len(cols)) + "|\n"
    return head + "".join("| " + " | ".join(str(r[c]) for c in cols) + " |\n" for r in rows)


def tables(bundle):
    order = {t: i for i, t in enumerate(TAG)}
    runs = sorted(bundle["runs"], key=lambda r: (order[r["variant"]], r["config"]["ny"]))
    main_rows = [dict(
        variant=r["variant"].split("_")[0], ny=r["config"]["ny"], nx=r["config"]["nx"],
        steps=r["steps"], peak=f"{r['u_peak_measured']:.5f}",
        err_peak_pct=f"{100 * r['err_peak']:.3f}",
        err_l2_pct=f"{100 * r['err_l2_profile']:.3f}",
        err_shape_pct=f"{100 * r['err_shape']:.3f}",
        wallslip_pct=f"{100 * r['u_wall_slip_rel']:.2f}",
        Qdrift=f"{r['Q_rel_drift']:.2e}",
        rho_int=f"{r['rho_mean_interior']:.5f}") for r in runs]
    cols = ["variant", "ny", "nx", "steps", "peak", "err_peak_pct", "err_l2_pct",
            "err_shape_pct", "wallslip_pct", "Qdrift", "rho_int"]
    with open(os.path.join(RES, "table_main.md"), "w") as fh:
        fh.write(_md_table(main_rows, cols))

    mach = sorted(bundle["mach_runs"], key=lambda r: (order[r["variant"]], r["config"]["u_peak"]))
    mach_rows = [dict(variant=r["variant"].split("_")[0],
                      u=f"{r['config']['u_peak']:.2f}",
                      Ma=f"{r['config']['u_peak'] * np.sqrt(3):.3f}",
                      err_peak_pct=f"{100 * r['err_peak']:.3f}",
                      err_l2_pct=f"{100 * r['err_l2_profile']:.3f}",
                      err_shape_pct=f"{100 * r['err_shape']:.3f}",
                      uy_inlet=f"{r['spurious_uy_inlet']:.2e}") for r in mach]
    with open(os.path.join(RES, "table_mach.md"), "w") as fh:
        fh.write(_md_table(mach_rows, ["variant", "u", "Ma", "err_peak_pct",
                                       "err_l2_pct", "err_shape_pct", "uy_inlet"]))
    return main_rows, mach_rows


def figures(bundle):
    runs = {(r["variant"], r["config"]["ny"]): r for r in bundle["runs"]}
    ns = sorted({ny for (_, ny) in runs})

    # fig 1: imposed inlet parabola, anchor offset vs corrected
    ny = 32
    y = np.arange(ny)
    u_ref = poiseuille_profile(ny, U_PEAK, offset="reference")
    u_cor = poiseuille_profile(ny, U_PEAK, offset="corrected")
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.plot(u_cor, y, "o-", ms=3, color="navy", label=r"corrected: $y_p=i-0.5$")
    ax.plot(u_ref, y, "s--", ms=3, color="crimson", label=r"anchor: $y_p=i-1.5$")
    ax.axvline(0.0, color="k", lw=0.8)
    for yy in (0.5, ny - 1.5):
        ax.axhline(yy, color="grey", lw=0.8, ls=":")
    ax.annotate("", xy=(-0.0168, 0), xytext=(0.0, 0),
                arrowprops=dict(arrowstyle="->", color="crimson"))
    ax.text(-0.0175, 2.0, r"$u_x(0)=-0.0168$" + "\n" + r"$=-21\%\,U_{peak}$",
            color="crimson", fontsize=8)
    ax.set_xlabel(r"imposed $u_x$ on the inlet column")
    ax.set_ylabel(r"row index $i$")
    ax.set_title("Inlet velocity profile imposed by the anchor reconstruction\n"
                 "(dotted lines: wall mid-planes)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "fig1_inlet_profiles.png"), dpi=150)
    plt.close(fig)

    # fig 2: developed profile at x=L/2 vs analytic (largest resolved ny)
    ny = max(ns)
    u_an = np.array(runs[(TAG[2], ny)]["u_analytic_profile"])
    idx = np.arange(1, 1 + len(u_an))
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    for name in TAG:
        r = runs[(name, ny)]
        axes[0].plot(r["u_mid_profile"], idx, ms=3, label=name.split("_")[0], **STYLE[name])
        axes[1].plot(np.array(r["u_mid_profile"]) - u_an, idx, ms=3,
                     label=name.split("_")[0], **STYLE[name])
    axes[0].plot(u_an, idx, "k-", lw=2.4, alpha=0.45, label="analytic Poiseuille")
    axes[0].set_xlabel(r"$u_x$ at $x=L/2$  (lattice units)")
    axes[0].set_ylabel(r"row index $i$")
    axes[0].set_title(f"(a) developed profile, $n_y={ny}$")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[1].axvline(0.0, color="k", lw=0.8)
    axes[1].set_xlabel(r"$u_x-u_x^{\rm analytic}$ at $x=L/2$  (lattice units)")
    axes[1].set_ylabel(r"row index $i$")
    axes[1].set_title("(b) pointwise deviation from the analytic solution")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "fig2_developed_profiles.png"), dpi=150)
    plt.close(fig)

    # fig 3: grid convergence
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    orders = {}
    for name in TAG:
        pts = sorted([(ny, runs[(name, ny)]["err_l2_profile"]) for ny in ns
                      if (name, ny) in runs])
        if not pts:
            continue
        xs = np.array([p[0] for p in pts], float)
        es = np.array([p[1] for p in pts], float)
        ax.plot(xs, es, ms=5, label=name.split("_")[0], **STYLE[name])
        if len(xs) >= 2:
            orders[name] = float(-np.polyfit(np.log(xs), np.log(es), 1)[0])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$n_y$ (channel height, lattice units)")
    ax.set_ylabel(r"$L_2$ developed-profile error")
    ax.set_title("Grid convergence of the developed-profile error")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "fig3_convergence.png"), dpi=150)
    plt.close(fig)

    # fig 4: Mach sweep
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    for name in (TAG[0], TAG[2]):
        pts = sorted([(r["config"]["u_peak"], r["err_l2_profile"])
                      for r in bundle["mach_runs"] if r["variant"] == name])
        if not pts:
            continue
        ax.plot([p[0] for p in pts], [p[1] for p in pts], ms=5,
                label=name.split("_")[0], **STYLE[name])
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "k--", lw=0.6, alpha=0.4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"imposed $U_{peak}$  (lattice units)")
    ax.set_ylabel(r"$L_2$ developed-profile error")
    ax.set_title(r"Mach sweep at $n_y=32$")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "fig4_mach_sweep.png"), dpi=150)
    plt.close(fig)
    return orders


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", nargs="*", type=int, default=None)
    ap.add_argument("--mach", action="store_true")
    args = ap.parse_args()
    b = _load()
    t0 = time.time()
    if args.res:
        run_grid(b, args.res)
    if args.mach:
        run_mach(b)
    tables(b)
    b["convergence_orders"] = figures(b)
    b["last_wall_seconds"] = round(time.time() - t0, 1)
    with open(JSON_PATH, "w") as fh:
        json.dump(b, fh, indent=2)
    print("orders:", {k: round(v, 3) for k, v in b["convergence_orders"].items()})
    print("done")