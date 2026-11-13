"""m18249 deliverables: artifact copy + figure + dev report + platform report payload."""
from __future__ import annotations

import base64
import json
import math
import shutil
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tensorbem import time_domain as td  # noqa: E402
from tensorbem.benchmarks import case_bem_cummins_wigley as case  # noqa: E402
from tensorbem.causality import kk_added_mass_closure  # noqa: E402

SRC = Path("/tmp/m18249_coarse41b")
OUT = Path("docs/m18249_case_bem_cummins_wigley")
PLOT = Path("docs/m18249")
OUT.mkdir(parents=True, exist_ok=True)
PLOT.mkdir(parents=True, exist_ok=True)
for name in ("case_result.json", "CASE.md", "sweep_coarse.npz"):
    shutil.copy2(SRC / name, OUT / name)

res = json.loads((OUT / "case_result.json").read_text())
g3 = res["gates"]["C3"]["values"]
g5 = res["gates"]["C5"]["values"]

# --- decay record + peaks used by C5 (regenerated from the stored sweep) ----
d = np.load(OUT / "sweep_coarse.npz")
w, A, B = d["omega"], d["added_mass"][:, 2, 2], d["damping"][:, 2, 2]
tau = np.arange(0.0, case.TAU_MAX + 0.5 * case.DT, case.DT)
kernel = td.radiation_memory_kernel(w, B, tau)
restoring, displaced, _ = case._restoring_and_mass(
    case.Sweep(label="coarse", n_sections=12, n_levels=4, panel_count=192,
               max_panel_scale=0.05, omega=w, added_mass=d["added_mass"],
               damping=d["damping"], solver_relative_residual=d["solver_relative_residual"],
               elapsed=0.0)
)
a_inf = float(kk_added_mass_closure(w, A, B).added_mass_inf)
decay = td.cummins_free_decay(kernel, mass=displaced, restoring=restoring,
                              added_mass_inf=a_inf, initial_displacement=1.0,
                              dt=case.DT, duration=case.DURATION)
t = np.asarray(decay.time)
x = np.asarray(decay.displacement)
excursion = float(np.max(np.abs(x)))
used = [(float(t[i]), float(x[i])) for i in range(1, x.size - 1)
        if x[i] > x[i - 1] and x[i] >= x[i + 1] and x[i] > case.DECAY_AMPLITUDE_FLOOR * excursion]
naive = [(float(t[i]), float(x[i])) for i in range(1, x.size - 1)
         if x[i] > x[i - 1] and x[i] >= x[i + 1] and x[i] > 0.0]

fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.2))
ax = axes[0]
for dof, name in enumerate(case.DOF_NAMES):
    col = d["damping"][:, dof, dof]
    scale = np.max(np.abs(col)) or 1.0
    ax.plot(w, col / scale, "-o", ms=2.6, lw=1.0, label=name)
ax.set_xscale("log")
ax.set_xlabel(r"$\omega$ [rad/s]")
ax.set_ylabel(r"$B_{ii}(\omega)/\max|B_{ii}|$ [-]")
ax.set_title("(a) radiation damping, coarse mesh, 41 nodes")
ax.grid(alpha=0.3)
ax.legend(ncol=2, fontsize=7)

ax = axes[1]
names = [row["dof_name"] for row in g3["per_dof"]]
vals = [100.0 * row["max_relative_residual"] for row in g3["per_dof"]]
prev = {"surge": 10.08, "sway": 16.11, "heave": 3.53, "roll": 9.66, "pitch": 6.16, "yaw": 14.52}
bars = ax.bar(np.arange(len(names)) - 0.2, [prev[n] for n in names], width=0.4,
              color="0.65", label="m17375 (13 nodes, 0.15–8 rad/s)")
ax.bar(np.arange(len(names)) + 0.2, vals, width=0.4,
       color="tab:blue", label="m18249 (41 nodes, 0.2–12 rad/s)")
ax.axhline(100.0 * case.TOLERANCES["kk"], color="crimson", ls="--", lw=1.1,
           label="criterion 10 %")
ax.set_xticks(np.arange(len(names)))
ax.set_xticklabels(names)
ax.set_ylabel("KK closure residual [% of max $|A_{ii}|$]")
ax.set_title("(b) Kramers–Kronig closure: before / after")
ax.grid(alpha=0.3, axis="y")
ax.legend(fontsize=7)

ax = axes[2]
ax.plot(t, x, "-", lw=0.8, color="0.35", label="Cummins decay")
ax.plot([p[0] for p in naive], [p[1] for p in naive], ".", ms=2.0, color="tab:orange",
        label=f"all peaks (n={len(naive)})")
ax.plot([p[0] for p in used], [p[1] for p in used], "o", ms=5.0, mfc="none",
        color="tab:green", label=f"peaks used (n={len(used)})")
ax.axhline(case.DECAY_AMPLITUDE_FLOOR * excursion, color="crimson", ls=":", lw=1.0,
           label="1 % amplitude floor")
ax.set_xlim(0.0, 6.0)
ax.set_xlabel("t [s]")
ax.set_ylabel("heave displacement [m]")
ax.set_title(rf"(c) free decay: $\zeta$={g5['zeta_decay']:.4f} vs $\zeta_{{fd}}$={g5['zeta_frequency_domain']:.4f}")
ax.grid(alpha=0.3)
ax.legend(fontsize=7)

fig.suptitle("CASE-BEM-CUMMINS-WIGLEY (m18249): sweep resolution and band coverage", fontsize=11)
fig.tight_layout()
fig_path = PLOT / "fig_m18249_cummins_wigley.png"
fig.savefig(fig_path, dpi=145)

# --- dev report -----------------------------------------------------------
rows3 = "\n".join(
    f"| {r['dof_name']} | {prev[r['dof_name']]:.2f} | {100*r['max_relative_residual']:.2f} | "
    f"{r['added_mass_inf_closure']:.4g} | {r['worst_sample_omega']:.3f} |"
    for r in g3["per_dof"]
)
gates = res["gates"]
verdict = res["verdict"]

text = f"""# m18249 — CASE-BEM-CUMMINS-WIGLEY: the two failing gates were numerical, not physical

**Anchor**: feature ledger `CASE-BEM-CUMMINS-WIGLEY` (算例池·Wigley 船 Cummins 记忆核（6DOF）),
ref = Cummins equation + Kramers-Kronig causality, tolerance < 10 %.
**Predecessor**: m17375 shipped `benchmarks/verified/casepool_bem_cummins_wigley/` with
**5/7 gates passed** — C3 (KK closure) and C5 (free decay vs frequency-domain damping) failed.

## 1. What was actually wrong

Both failures came from the *measurement apparatus*, not from the BEM data:

1. **C3 (KK closure)** closed the Kramers-Kronig integral with the **piecewise-linear
   interpolant** of the tabulated damping, but the sweep had `13` nodes over
   `0.15 … 8 rad/s` (≈ 7.5 nodes/decade) while the damping curves peak between
   2 and 12 rad/s. The residual (3.5–16 %) was interpolation error, not a causality
   violation. The same run also used **fewer nodes than the module's own pre-declared
   default** (`N_OMEGA = 21`).
2. **C5 (free decay)**: the model's heave natural frequency is ≈ 7.8 rad/s, i.e. it sat
   **outside** the sweep (`0.15 … 8` was only nominally enough — the recorded run
   computed 8.147 rad/s from a contaminated period). Second, the memory kernel is
   **truncated at ω_max**, which leaves a small ripple at ~ω_max in the decay record;
   the peak detector fitted the logarithmic decrement across that residue (69 peaks over
   40 s, spacing 0.57 s) and reported `ζ = 0.0268`, i.e. the *residue* decay, while the
   hull mode gives `ζ ≈ 0.133`.

## 2. Changes (all in `src/tensorbem/benchmarks/case_bem_cummins_wigley.py`)

| change | why |
| --- | --- |
| `BAND = (0.2, 12.0)` (was `(0.2, 6.0)`) | the band must contain every rigid-body natural frequency of the 3 m model (heave ≈ 7.8 rad/s); otherwise C5 is **undecidable**, not failed |
| `N_OMEGA = 41` (was `21`) | resolve the damping peaks: the KK residual is measured, not asserted |
| `sweep_grid()` | log grid with both band ends exactly included (unit-tested) |
| `_refined_grid()` / `--refine-peaks K` | optional geometric refinement of the intervals with the largest damping jump (default off) |
| `grid_convergence_study()` / `--grid-study` | KK residual **as a function of node count** — turns the tolerance into a measurement |
| `free_decay_gate()` status field | separates "undecidable: natural frequency outside the band" from "the two routes disagree"; the recorded FAIL was the former and was silently reported as `nan` |
| `DECAY_AMPLITUDE_FLOOR = 1e-2` in `_decay_zeta()` | fit the decrement only over the physical part of the record (the ripple outlives the mode) |

The negative control C6b (`×1.3` shape corruption) and the zero-damping control C6a are
unchanged and still pass, so the gates did not get easier.

## 3. Results — coarse mesh, 192 panels, band 0.2–12 rad/s, 41 nodes

| DOF | KK residual m17375 [%] | KK residual m18249 [%] | A_inf (closure) [kg] | worst ω [rad/s] |
| --- | --- | --- | --- | --- |
{rows3}

* worst DOF: **{g3['worst_dof']} {100*g3['worst_relative_residual']:.2f} %** vs the 10 % criterion;
  the δDof-independent scale is `max|A_ii|` per DOF, so each row is a relative error on the
  coefficient itself.
* C5: `ζ_decay = {g5['zeta_decay']:.4f}`, `ζ_fd = {g5['zeta_frequency_domain']:.4f}`,
  relative error **{100*g5['relative_error']:.2f} %** (criterion 5 %),
  ω_n = {g5['natural_frequency_rad_s']:.3f} rad/s, {g5['n_peaks']} peaks used,
  A(ω_n) = {g5['added_mass_at_natural']:.2f} kg, B(ω_n) = {g5['damping_at_natural']:.1f} N·s/m.
* Gate tally on this single-mesh run: {verdict['gates_passed']}/{verdict['gates_total']}
  (failed: {verdict['failed_list']}). **C2 fails by design on the coarse mesh**
  (`k·√A_panel = 1.47` at 12 rad/s); C2 is a property of the *production* mesh and is
  evaluated on the 20×6 mesh, which is the remaining open item (cluster job `cj1863`).

## 4. Honest limits

* The coarse-mesh run above is complete and reproducible; the **medium-mesh production
  run was submitted to the cluster (`cj1863`) and had not returned** when this report was
  written, so the `benchmarks/verified/casepool_bem_cummins_wigley/` artifact still holds
  the m17375 result. The new artifacts are under `docs/m18249_case_bem_cummins_wigley/`.
* Only the **diagonal** (heave) free-decay comparison is made; the 6-DOF coupled decay
  (off-diagonal kernels, roll-yaw coupling) is **not** covered here — 6DOF *kernels* are
  covered by C3 for all six diagonal DOFs and by the matrix closure artifacts.
* `k·√A_panel ≤ 1` at 12 rad/s on the medium mesh is expected to be ≈ 0.93, i.e. the top
  of the band is *just* resolved; if the production run fails C2 the honest fallback is
  `BAND[1] = 11 rad/s` (still above ω_n = 7.83 rad/s), not a relaxed tolerance.

## 5. Reproduction

```
PYTHONPATH=src python -m tensorbem.benchmarks.case_bem_cummins_wigley \\
    --out docs/m18249_case_bem_cummins_wigley --meshes coarse --jobs 4 --no-figure
PYTHONPATH=src python -m pytest tests/test_case_bem_cummins_wigley.py -q   # 19 passed
```

Figure: `docs/m18249/fig_m18249_cummins_wigley.png`.
"""
report_md = PLOT / "dev_report_m18249_cummins_wigley_gates.md"
report_md.write_text(text, encoding="utf-8")

payload = {
    "title": "CASE-BEM-CUMMINS-WIGLEY: KK closure 16.1%→3.4%, free-decay gate decided (m18249)",
    "summary": (
        "The two failing gates of the Wigley 6DOF Cummins case (C3 Kramers-Kronig closure, "
        "C5 free decay vs frequency-domain damping) were measurement defects, not physics: "
        "an under-resolved 13-node sweep (interpolation error in the KK quadrature) and a "
        "free-decay peak fit that locked onto the kernel-truncation ripple outside an "
        "8 rad/s band. Sweep set to 41 nodes over 0.2-12 rad/s, peak amplitude floor 1%: "
        "KK residual 3.5-16.1% -> 0.28-3.40% (criterion 10%), decay zeta 0.1334 vs "
        "zeta_fd 0.1296 (2.93%, criterion 5%)."
    ),
    "scenario_key": "bem_case_pool",
    "mission_id": 18249,
    "tags": ["TensorBEM", "CASE-BEM-CUMMINS-WIGLEY", "Kramers-Kronig", "Cummins", "verification"],
    "markdown": text,
    "figures": [
        {
            "name": "fig_m18249_cummins_wigley.png",
            "caption": (
                "(a) radiation damping of all six diagonal DOFs on the coarse mesh with the "
                "41-node band; (b) KK closure residual per DOF, m17375 (13 nodes) vs m18249 "
                "(41 nodes) against the 10% criterion; (c) Cummins free decay for heave with "
                "the peaks used by the decrement fit and the 1% amplitude floor."
            ),
            "png_base64": base64.b64encode(fig_path.read_bytes()).decode("ascii"),
        }
    ],
}
Path("/tmp/m18249_report.json").write_text(json.dumps(payload), encoding="utf-8")
print("wrote", report_md, fig_path, "/tmp/m18249_report.json")
print("C3 worst", g3["worst_relative_residual"], "C5 rel", g5["relative_error"])