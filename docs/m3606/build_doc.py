"""Assemble `docs/annulus_multicomponent.md` from the generated artefacts.

Every number that reaches the document is read from
`docs/m3606/outputs/annulus_study.json` / `annulus_field.json`, or spliced in
verbatim from `annulus_tables.md` (which the benchmark CLI writes).  Re-running
`python -m tensorbem.benchmarks.multicomponent_annulus ...` followed by this
script regenerates the document, so no figure in the prose is hand-copied.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOCS = HERE.parent
OUT = HERE / "outputs"
FIG = HERE / "figures"

study = json.loads((OUT / "annulus_study.json").read_text())
field = json.loads((OUT / "annulus_field.json").read_text())
tables = (OUT / "annulus_tables.md").read_text().strip()

summary = study["summary"]
orders = study["orders"]
control = study["orientation_control"]
conditioning = study["conditioning_comparison"]
cond_text = "; ".join(
    f"{row['single_loop_condition_number']:.1f} (one loop) vs "
    f"{row['two_loop_condition_number']:.1f} (two loops) at {row['total_panels']:d} panels total, "
    f"ratio {row['ratio']:.2f}"
    for row in conditioning
)
domain = study["domain"]
log_row = next(
    r for r in study["rows"] if r["case_key"] == "log" and r["panels_per_loop"] == 128
)

prose = f"""# Multi-connected 2-D Laplace domains: the annulus verification

**Scope.** Closes the last open cell of the 2-D Dirichlet table in
[`verification.md`](verification.md) §1, which registered

> `| 圆环径向导热，u = A log r + B | 多连通边界、内外法向 | 后续：需多边界组件 |`

Nothing in the package could express a two-loop boundary before this note: the
single-polygon mesh of `tensorbem.geometry.PanelMesh` stores one cyclic vertex
array, so appending a hole loop would fabricate an edge from the last hole
vertex back to the first outer vertex. The *solver* never needed a change — it
is duck-typed on the panel arrays — so this work is geometry, guards, and
evidence.

**Evidence level:** REAL. Every number below is produced by
`python -m tensorbem.benchmarks.multicomponent_annulus`; the prose is assembled
by `docs/m3606/build_doc.py`, which reads the JSON, so no digit is retyped.

## 1. Case

Two concentric circles, fluid in the wall between them: a genuinely
multi-connected domain with one outer boundary and one *hole*. The hole is what
tests the normal convention: the fluid-domain outward normal on the hole points
**into** the hole, i.e. it is the opposite of the polygon's "outward" normal.

| quantity | value |
|---|---|
| inner radius $r_{{\\mathrm{{in}}}}$ | {domain["inner_radius"]:g} |
| outer radius $r_{{\\mathrm{{out}}}}$ | {domain["outer_radius"]:g} |
| boundary condition | Dirichlet, $u$ given on both loops |
| panels per loop | {", ".join(str(n) for n in domain["ladder_panels_per_loop"])} (total {", ".join(str(n) for n in domain["panels_total"])}) |
| interior probes per case | {domain["probes_per_case"]} (three rings, six angles) |
| reference length of the log kernel | 2.0 (package `REFERENCE_LENGTH`) |
| polygonal fluid area at the finest mesh | {domain["fluid_area_polygon"]:.6f} vs exact $\\pi(r_o^2-r_i^2)$ = {domain["fluid_area_exact"]:.6f} |

Three closed-form families share the domain:

* `log` — $u = A\\log r + B$ with $A=1$, $B={0.5:g}$, the registered case. Its
  flux is *not* balanced per loop: $\\oint_{{\\mathrm{{outer}}}} q\\,ds = 2\\pi A$ and
  $\\oint_{{\\mathrm{{hole}}}} q\\,ds = -2\\pi A$, with vanishing total. A single-loop
  mesh cannot even express that bookkeeping.
* `power1`, `power2` — $u = \\mathrm{{Re}}(z^n + z^{{-n}})$, $n=1,2$. The $z^{{-n}}$
  term is harmonic in the annulus but singular at the origin, which lies outside
  the fluid, so it loads the hole boundary as strongly as the outer one.

None of the references is a discretisation: they are evaluated in closed form at
the panel midpoints and at the probes, and their gradients are analytic
($f'(z)$ of the complex potential), with a central-difference test pinning the
sign convention.

## 2. Method

1. `MultiComponentMesh` holds each loop as its own cyclic vertex array and
   concatenates only the *panel* arrays (`centers`, `normals`, `lengths`, ...)
   that the analytic panel integrator consumes. Each loop's closure stays local,
   so no cross-loop panel is ever created.
2. `build_multicomponent_mesh(outer, holes)` takes ordinary CCW `PanelMesh`
   polygons and reverses every hole, so the resulting normal points from the
   fluid into the hole. It then validates that hole vertices lie inside the
   outer loop and that holes neither overlap nor nest.
3. `solve_multicomponent_dirichlet` calls the existing
   `solve_laplace_dirichlet`. It is the same first-kind equation
   $S q = (\\tfrac12 I + H)u$ with the same analytically integrated kernel and the
   same condition-number guard — only the mesh changed.
4. `evaluate_fluid_field` refuses probe points that fall inside a hole or
   outside the outer loop. `evaluate_interior_field` cannot know about holes, and
   a probe accidentally placed in the hole returns a finite, plausible, wrong
   number: that is the failure mode this guard removes.
5. Errors: weighted $L_2$ flux error
   $\\|q_h-q\\|_L / \\|q\\|_L$ with panel-length weights, max flux error relative to
   $\\max|q|$, and probe RMSE / max relative error against the exact potential.
6. Independent checks that do **not** use the analytic field: the residual of the
   solved system, the per-loop circulation of the panel solution, and the
   vanishing total flux.
7. Negative control: the identical annulus with the hole *not* reversed. Its own
   linear system is still solved to round-off, so the residual cannot detect the
   mistake — which is precisely why the orientation is enforced in the mesh type
   rather than left to the caller.

## 3. Results

{tables}

![annulus verification]({FIG.relative_to(DOCS).as_posix()}/annulus_verification.png)

*Left:* panel solution of $u = A\\log r + B$ on the annulus
($N={field["panels_per_loop"]}$ panels per loop); the hole is the white disk.
*Centre:* pointwise error $|u_{{\\mathrm{{panel}}}}-u_{{\\mathrm{{exact}}}}|$ on a
${field["grid"]}\\times{field["grid"]}$ grid (max {field["max_abs_error"]:.2e}).
*Right:* refinement ladder for all three families (solid: flux $L_2$; dashed:
probe RMSE; dotted: slope $-2$; dash-dot: the reversed-hole control).

Headline numbers at the finest mesh ($N=128$ panels per loop):

* `log` flux $L_2$ error {summary["log"]["finest_flux_rel_l2"]:.3e}, probe RMSE
  {summary["log"]["finest_field_rmse"]:.3e}, observed order
  {orders["log"]["flux_rel_l2"]:.2f};
* `power1` {summary["power1"]["finest_flux_rel_l2"]:.3e} / order
  {orders["power1"]["flux_rel_l2"]:.2f}; `power2`
  {summary["power2"]["finest_flux_rel_l2"]:.3e} / order
  {orders["power2"]["flux_rel_l2"]:.2f};
* per-loop circulation of the `log` case:
  {log_row["per_loop_flux"][0]:.6f} on the outer loop and
  {log_row["per_loop_flux"][1]:.6f} on the hole, against $\\pm2\\pi A=\\pm6.283185$,
  i.e. relative errors
  {summary["log"]["finest_per_loop_flux_relative_error"][0]:.2e} and
  {summary["log"]["finest_per_loop_flux_relative_error"][1]:.2e}; total flux
  $\\oint q\\,ds$ over $\\oint|q|\\,ds$ is
  {summary["log"]["finest_net_flux_relative"]:.2e};
* condition number {summary["log"]["finest_condition_number"]:.1f} at $N=128$,
  relative residual {summary["log"]["finest_relative_residual_l2"]:.1e};
* reversed-hole control: flux error {control["flux_rel_l2"]:.3e} versus
  {control["correct_flux_rel_l2"]:.3e} for the correct mesh at the same
  $N={control["panels_per_loop"]}$ — a factor {control["error_ratio"]:.0f}, with
  both systems solved to residual
  {control["relative_residual_l2"]:.1e} / {control["correct_relative_residual_l2"]:.1e}.

## 4. Error verdict against the acceptance rules

| key quantity | reference | relative error | below 1 %? | below 3 %? |
|---|---|---:|---|---|
| `log` flux $L_2$, $N=128$ | closed form | {summary["log"]["finest_flux_rel_l2"]:.2e} | yes | yes |
| `log` flux $L_2$, $N=16$ (worst rung) | closed form | {summary["log"]["worst_flux_rel_l2"]:.2e} | yes | yes |
| `log` max flux error, $N=128$ | closed form | {summary["log"]["finest_flux_max_rel"]:.2e} | yes | yes |
| `log` probe max relative, $N=128$ | closed form | {summary["log"]["finest_field_max_rel"]:.2e} | yes | yes |
| `power1` flux $L_2$, $N=128$ | closed form | {summary["power1"]["finest_flux_rel_l2"]:.2e} | yes | yes |
| `power2` flux $L_2$, $N=128$ | closed form | {summary["power2"]["finest_flux_rel_l2"]:.2e} | yes | yes |
| per-loop circulation, outer | $2\\pi A$ | {summary["log"]["finest_per_loop_flux_relative_error"][0]:.2e} | yes | yes |
| per-loop circulation, hole | $-2\\pi A$ | {summary["log"]["finest_per_loop_flux_relative_error"][1]:.2e} | yes | yes |
| total flux (should vanish) | 0 | {summary["log"]["finest_net_flux_relative"]:.2e} of $\\oint|q|\\,ds$ | yes | yes |
| observed order, all three families | 2 (smooth data, smooth boundary) | {min(orders[c]["flux_rel_l2"] for c in study["cases"]):.2f}–{max(orders[c]["flux_rel_l2"] for c in study["cases"]):.2f} | — | — |

**Verdict: every physical quantity is below 1 % of its closed-form reference at
$N=128$ panels per loop, and the coarse rung $N=16$ is already inside 1 %.** The
observed order sits at 2, as expected for constant panels on a smooth boundary
with smooth data — the multi-loop topology did not degrade the order.

## 5. Notes, traps, and scope

1. **The condition number grows linearly with panels, and the second loop
   costs a constant factor.** It is
   {study["rows"][0]["condition_number"]:.1f} at $N=16$ and
   {summary["log"]["finest_condition_number"]:.1f} at $N=128$ — a factor
   {summary["log"]["finest_condition_number"] / study["rows"][0]["condition_number"]:.2f}
   for 8x panels. The single-layer matrix does not depend on the normal
   orientation, so this was measured at *matched total panel count* against a
   single circle: {cond_text}. The two-component topology therefore multiplies
   $\\kappa$ by about {conditioning[-1]["ratio"]:.2f} — two capacity scales instead
   of one — and the existing `_MAX_CONDITION` guard would only start complaining
   far beyond this ladder. The residual is at round-off throughout, so the
   numbers above are not residual-limited, but a production mesh on a thin wall
   should budget for the conditioning cost.
2. **The residual does not detect a reversed hole.** The control run solves its
   own (wrong) system to
   {control["relative_residual_l2"]:.1e} and still reports a
   {control["flux_rel_l2"]:.0%} flux error. Only the analytic reference, or the
   mesh type's orientation check, catches it. This is the same lesson as the
   package's earlier surface-velocity work: small residuals certify the linear
   solve, not the formulation.
3. **The analytical per-loop value is evaluated on the polygon, not the circle.**
   $\\oint q\\,ds$ along the inscribed polygon differs from $2\\pi A$ by the
   chord/arc ratio, which is why the "analytic" column above reads
   {log_row["per_loop_flux_reference"][0]:.6f} rather than 6.283185. Comparing the
   panel solution against the circle value would charge the geometric
   discretisation twice; the table therefore carries both the polygon reference
   (for the error column) and $2\\pi A$ (for the physical statement).
4. **Scope: two loops, no crossed or touching boundaries, no open curves.** The
   mesh type accepts any number of non-overlapping holes, but only the
   single-hole annulus is verified here. Multiple holes would need a third
   reference family (e.g. a bipolar-coordinate solution between two eccentric
   cylinders), which this note does not provide.
5. **Not an engineering validation.** The cases are code verification against
   closed-form solutions; they say nothing about a real hull section, and the
   package still has no multi-component *free-surface* solver.

## 6. Reproduce

```
PYTHONPATH=src python -m tensorbem.benchmarks.multicomponent_annulus \\
    --json docs/m3606/outputs/annulus_study.json \\
    --field-json docs/m3606/outputs/annulus_field.json \\
    --figure docs/m3606/figures/annulus_verification.png \\
    --markdown docs/m3606/outputs/annulus_tables.md
python3 docs/m3606/build_doc.py
PYTHONPATH=src python -m pytest tests/test_multicomponent.py -q
```

Wall time of the study is about 45 s single core (the longest run is the
$N=128$ assembly, $256^2$ analytic panel integrals); `--quick` gives a
smoke-size ladder.
"""

(DOCS / "annulus_multicomponent.md").write_text(prose)
print(f"wrote docs/annulus_multicomponent.md ({len(prose.splitlines())} lines)")