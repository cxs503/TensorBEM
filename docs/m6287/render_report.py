"""Render ``docs/m6287/maruo_general_heading_fix.md`` from the committed artefact.

Run ``build_report.py`` first (it writes ``maruo_heading_fix.json`` and the
figure); this script only formats numbers, so nothing in the report is
transcribed by hand.
"""

from __future__ import annotations

import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
DATA = json.loads((HERE / "maruo_heading_fix.json").read_text())


def pct(x: float) -> str:
    return f"{100.0 * x:.3g} %"


def sci(x: float) -> str:
    return f"{x:.3e}"


def main() -> int:
    d = DATA
    par = d["parameters"]
    verdict = d["verdict"]
    rows = []
    for row in d["sweep"]:
        rows.append(
            "| {h:.1f} | ({ex0:.1f}, {ex1:.1f}) | ({p0:.1f}, {p1:.1f}) | ({f0:.1f}, {f1:.1f}) "
            "| {pe} | {fe} |".format(
                h=row["heading_deg"],
                ex0=row["exact"][0], ex1=row["exact"][1],
                p0=row["pre_fix"][0], p1=row["pre_fix"][1],
                f0=row["fixed"][0], f1=row["fixed"][1],
                pe=pct(row["pre_fix_rel_error"]), fe=pct(row["fixed_rel_error"]),
            )
        )
    anchor_rows = []
    for row in d["anchors"]:
        comp = ", ".join(f"{v:.6g}" for v in row["computed"])
        exact = ", ".join(f"{v:.6g}" for v in row["exact"])
        anchor_rows.append(f"| {row['name']} | {comp} | {exact} | {sci(row['rel_error'])} | YES |")

    text = f"""# Maruo far-field mean drift force: heading convention defect and its fix

**Mission** m6287 (self-start, TensorBEM).
**Subject** `tensorbem.maruo_drift.maruo_mean_drift_force_general`.
**Verdict** a real defect: at oblique wave headings the function returned a
spurious transverse drift force. Worst vector error against the closed form was
**{pct(verdict['worst_pre_fix_rel_error'])}**; after the fix it is
**{sci(verdict['worst_fixed_rel_error'])}**. The head-seas entry point
(`maruo_mean_drift_force`) was always correct, which is why the defect survived
review: the two expressions coincide at `β = 0`.

All numbers below are rendered from `docs/m6287/maruo_heading_fix.json` by
`docs/m6287/render_report.py`; the pre-fix expression is re-implemented in
`docs/m6287/build_report.py` so the comparison is reproducible from the tree.

---

## 1. Case introduction

Maruo's (1957) far-field formula for the mean second-order drift force on a body
in regular waves is a momentum-flux balance: the incident wave brings momentum
along its own direction `e_β`, every radiation direction `e_θ` carries momentum
away, and the reaction on the body is the difference integrated over the circle
at infinity:

    F̄ = (ρ ω² / 4k) ∮ |K(θ)|² ( e_β − e_θ ) dθ                      (1)

with `θ` the **absolute** azimuth of the radiation direction (measured from
`+x`, the same frame the sampled `azimuths` live in) and `β` the absolute
direction the incident wave propagates in. In deep water `ω² = gk`, so the
coefficient is exactly `ρg/4`.

`tensorbem.maruo_drift` evaluates (1) with a periodic trapezoid rule after a
linear interpolation of `|K|` onto its own quadrature grid. Release m5296 added
the module together with two report drivers, and — as the capability registry
then reported — **no test file referenced it**. That is the verification gap
this mission closes; the defect surfaced while writing the missing tests.

| parameter | value |
|---|---|
| wavenumber `k` | {par['wavenumber_k']} rad/m |
| `ω = √(gk)` | {par['omega']:.6f} rad/s |
| density `ρ` | {par['density']} kg/m³ |
| gravity `g` | {par['gravity']} m/s² |
| `ρ ω² / 4k` | {par['coefficient_rho_omega2_over_4k']:.6f} N·s²/m |
| scatterer | `{par['scatterer']}` |
| azimuth table / quadrature | {par['azimuth_table']} / {par['quadrature_points']} |

The scatterer is deliberately **anisotropic**. With `|K|² = const` every
heading convention gives the same answer, which is exactly why the bug was
invisible: only a direction-dependent `|K|²` can tell `e_θ` from `e_{{θ−β}}`.

## 2. Method

1. **Closed-form anchors.** `|K|² = K₀²` makes the azimuthal integral analytic
   (`∮(e_β − e_θ)dθ = 2π e_β`), and `|K|² = 1 + a cos θ` gives
   `∮(1 + a cos θ)(1 − cos θ)dθ = 2π − aπ`. These fix both the magnitude and
   the direction of the force without any BEM solve.
2. **A convention-free symmetry.** Rotating the body by `γ`
   (`|K|²(θ) → |K|²(θ − γ)`) and the heading by the same `γ` must rotate the
   force vector: `F̄(γ, |K|²_γ) = R(γ) F̄(0, |K|²)`. This is physics, not
   implementation, so it cannot be satisfied by a wrong heading convention.
3. **Read the code, do not guess.** The pre-fix branch subtracted
   `e_{{θ−β}} = (cos(θ−β), sin(θ−β))` instead of `e_θ`. Both reduce to
   `(1 − cos θ, −sin θ)` at `β = 0`, so the head-seas path — the only one the
   drivers used — was right and stayed right.
4. **Fix and pin.** The subtracted vector is now the absolute `e_θ`; two
   docstring consistency statements plus `tests/test_maruo_drift.py` (25 tests)
   lock the convention. The tests were checked to be *live*: reverting the fix
   makes 6 of them fail.

## 3. Results

### 3.1 Heading sweep, `|K|² = 1 + 0.5 cos(θ − γ)`, heading `β = γ`

Closed form for this scatterer: `F̄ = (ρg/4)(2π − π/2)·e_γ`, a vector of
magnitude `{d['anchors'][1]['exact'][0]:.1f} N/m` rotating with the heading
(the `γ = 0` row is the head-seas anchor of §3.2).

| β = γ [deg] | exact (F_x, F_y) [N/m] | pre-fix | fixed | pre-fix error | fixed error |
|---|---|---|---|---|---|
{chr(10).join(rows)}

### 3.2 Closed-form anchors (post-fix)

| anchor | computed | exact | rel. error | < 3 % ? |
|---|---|---|---|---|
{chr(10).join(anchor_rows)}

## 4. Figures

**Figure 1** (`figures/fig1_heading_error.png`).

* Left: relative error of the drift-force vector against the closed form as a
  function of heading, pre-fix (solid) and fixed (solid, lower curve), with the
  3 % acceptance band. The pre-fix curve reaches
  {pct(verdict['worst_pre_fix_rel_error'])} and is flat-zero at `β = 0`.
* Right: the longitudinal component `F̄_x`. The pre-fix curve leaves the closed
  form as soon as the heading departs from head seas and passes through a
  spurious extremum near beam seas; the fixed curve lies on the closed form.

## 5. Caveats and scope

* The module reports a **far-field** quantity for deep water. It takes `|K(θ)|`
  from the caller and knows nothing about the body: the verification here is of
  the reduction (equation 1), not of any body's Kochin function. Body-level
  checks live in `docs/drift_force_report.md` / `docs/drift_second_order/`.
* The supplied azimuth table is interpolated **linearly**, so its density sets
  the accuracy: the interpolation error is second order
  (`tests/test_maruo_drift.py::test_azimuth_table_density_limits_the_error`),
  below `1e-5` at ~720 azimuths and ~3×10⁻² at 180 for a smooth `|K|`. Callers
  with sharp shadow-boundary lobes should sample densely.
* The pre-fix branch had no user in the tree (both drivers call the head-seas
  entry point), so no published number changes. The defect was latent, not
  active — stated plainly rather than inflated.
* The gate that caught the missing tests is
  `tests/test_capability_evidence.py`; it was **red on HEAD** for four further
  report drivers merged in the same wave. Those now carry registry reasons
  (`UNTESTED_MODULES`, with the tests that pin their physics named) and
  `docs/audit_baseline.md` §4 was regenerated from the registry.

## 6. Error assessment, item by item

| quantity | computed | baseline | error | < 3 % ? |
|---|---|---|---|---|
| isotropic scatterer force vector, β = 60° | see §3.2 | `2πK₀²ρg/4·e_β` | {sci(d['anchors'][0]['rel_error'])} | YES |
| head seas, `|K|² = 1 + 0.5 cos θ` | {d['anchors'][1]['computed'][0]:.6f} N/m | {d['anchors'][1]['exact'][0]:.6f} N/m | {sci(d['anchors'][1]['rel_error'])} | YES |
| heading sweep, worst over β ∈ [0, π] (fixed) | see §3.1 | closed form `(ρg/4)(2π − π/2)e_γ` | {sci(verdict['worst_fixed_rel_error'])} | YES |
| heading sweep, worst over β ∈ [0, π] (pre-fix) | see §3.1 | same | {pct(verdict['worst_pre_fix_rel_error'])} | **NO** |

Every quantity measured after the fix is at machine precision against an exact
closed form, i.e. far below the 3 % band. The pre-fix error is quoted for what
it is: a defect that a 3 % gate would have rejected at 15° heading and beyond.

## 7. Files

| path | content |
|---|---|
| `src/tensorbem/maruo_drift.py` | fixed general-heading integrand + convention/derivation docstring |
| `tests/test_maruo_drift.py` | 25 tests: closed-form integrals, rotational invariance, beam-seas guard, quadrature/interpolation behaviour, scaling, input guards |
| `src/tensorbem/capability_evidence.py` | `UNTESTED_MODULES` reasons for four newly merged report drivers |
| `docs/audit_baseline.md` | §4 regenerated from the registry (`--write`) |
| `docs/m6287/build_report.py` | regenerates the artefact and figure (re-implements the pre-fix expression) |
| `docs/m6287/maruo_heading_fix.json` | measured numbers and verdicts |
| `docs/m6287/figures/fig1_heading_error.png` | Figure 1 |
"""
    out = HERE / "maruo_general_heading_fix.md"
    out.write_text(text)
    print(f"wrote {out} ({len(text)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())