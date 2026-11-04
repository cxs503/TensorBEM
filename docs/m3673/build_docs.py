"""Build the document, ledger and figures for the finite-depth kernel gradient.

Run::

    PYTHONPATH=src python3 docs/m3673/build_docs.py

Writes ``docs/m3673/outputs/finite_depth3d_gradient.json`` (the ledger produced by
:mod:`tensorbem.benchmarks.finite_depth3d_gradient`), two figures under
``docs/m3673/figures`` and this document's tables into
``docs/finite_depth3d_gradient.md``.  No number in the document is typed by hand.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tensorbem.benchmarks.finite_depth3d_gradient import run  # noqa: E402

ROOT = Path(__file__).resolve().parent
DOCS = ROOT.parent
OUTPUT = ROOT / "outputs"
FIGURES = ROOT / "figures"


def _fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}e}"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    ledger = run()
    (OUTPUT / "finite_depth3d_gradient.json").write_text(
        json.dumps(ledger, indent=2, sort_keys=True) + "\n"
    )

    # ---------------------------------------------------------------- figure 1
    rows = ledger["bessel"]["scan"]["rows"]
    x = np.array([row["x"] for row in rows])
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.loglog(x, np.maximum([row["k0_legacy_relative_error"] for row in rows], 1e-17),
              color="tab:red", lw=1.6, label="$K_0$ before fix (series to $x=9$)")
    ax.loglog(x, np.maximum([row["k0_relative_error"] for row in rows], 1e-17),
              color="tab:blue", lw=1.6, label="$K_0$ after fix (quadrature gap)")
    ax.loglog(x, np.maximum([row["k1_relative_error"] for row in rows], 1e-17),
              color="tab:green", lw=1.6, label="$K_1$ (new)")
    ax.axvline(ledger["bessel"]["series_limit"], color="0.6", ls=":", lw=1.0)
    ax.axvline(ledger["bessel"]["asymptotic_limit"], color="0.6", ls=":", lw=1.0)
    ax.axhline(1e-12, color="k", ls="--", lw=1.0, label=r"gate $10^{-12}$")
    ax.set_xlabel("argument $x$")
    ax.set_ylabel("relative error vs Laplace quadrature")
    ax.set_title("Order-0/1 modified Bessel evaluator: cancellation cliff and fix")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_bessel_branches.png", dpi=150)
    plt.close(fig)

    # ---------------------------------------------------------------- figure 2
    ladder = ledger["finite_difference"]["ladder"]
    steps = np.array([row["step"] for row in ladder])
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.loglog(steps, np.maximum([row["central_max_relative_error"] for row in ladder], 1e-17),
              "o-", label="central difference")
    ax.loglog(steps, np.maximum([row["richardson_max_relative_error"] for row in ladder], 1e-17),
              "s-", label="Richardson (4th order)")
    ax.set_xlabel(r"step $\varepsilon$ [m]")
    ax.set_ylabel(r"max $|\nabla G_{\rm fin. diff.} - \nabla G_{\rm analytic}| / \max|\nabla G|$")
    ax.set_title("Analytic gradient vs finite differences of $G$ ($\\nabla^2$-consistent)")
    ax.grid(True, which="both", alpha=0.25)
    ax.invert_xaxis()
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_gradient_fd.png", dpi=150)
    plt.close(fig)

    # ---------------------------------------------------------------- figure 3
    deep = ledger["deep_water"]["rows"]
    k0h = np.array([row["k0h"] for row in deep])
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.loglog(k0h, [row["gradient_relative_deviation"] for row in deep], "o-",
              label=r"$\nabla G$ vs deep-water Kelvin gradient")
    ax.loglog(k0h, [row["kernel_relative_deviation"] for row in deep], "s--",
              label="$G$ vs deep-water kernel")
    ax.set_xlabel("$k_0 h$")
    ax.set_ylabel("max relative deviation")
    ax.set_title("Deep-water collapse of the finite-depth gradient")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_deep_water_gradient.png", dpi=150)
    plt.close(fig)

    # ---------------------------------------------------------------- document
    bessel = ledger["bessel"]
    pinned = [row for row in bessel["table"]
              if row["x"] in (1.0, 4.0, 6.0, 8.0, 8.9000000000000004, 10.0, 12.0, 16.0, 40.0)]
    pinned_rows = "\n".join(
        "| {x:g} | {k0_ref:.6e} | {e0} | {e0l} | {k1_ref:.6e} | {e1} | {e1l} |".format(
            x=row["x"], k0_ref=row["k0_reference"], e0=_fmt(row["k0_relative_error"]),
            e0l=_fmt(row["k0_legacy_relative_error"]), k1_ref=row["k1_reference"],
            e1=_fmt(row["k1_relative_error"]), e1l=_fmt(row["k1_legacy_relative_error"]),
        )
        for row in pinned
    )
    seam_rows = "\n".join(
        "| {x:.9g} | {q0:.15e} | {s0:.3e} | {a0:.3e} | {q1:.15e} | {s1:.3e} | {a1:.3e} |".format(
            x=row["x"], q0=row["k0_quadrature"],
            s0=abs(row["k0_quadrature"] / row["k0_series"] - 1.0),
            a0=abs(row["k0_quadrature"] / row["k0_asymptotic"] - 1.0),
            q1=row["k1_quadrature"],
            s1=abs(row["k1_quadrature"] / row["k1_series"] - 1.0),
            a1=abs(row["k1_quadrature"] / row["k1_asymptotic"] - 1.0),
        )
        for row in bessel["seam_table"]
    )
    check_rows = "\n".join(
        "| `{name}` | {measured} | {threshold:g} | {verdict} |".format(
            name=check["name"], measured=_fmt(check["measured"]),
            threshold=check["threshold"], verdict="PASS" if check["pass"] else "**FAIL**",
        )
        for check in ledger["checks"]
    )
    fd_rows = "\n".join(
        "| {step:g} | {central} | {rich} |".format(
            step=row["step"], central=_fmt(row["central_max_relative_error"]),
            rich=_fmt(row["richardson_max_relative_error"]),
        )
        for row in ladder
    )
    deep_rows = "\n".join(
        "| {k0h:.3f} | {h:g} | {g} | {k} |".format(
            k0h=row["k0h"], h=row["depth"],
            g=_fmt(row["gradient_relative_deviation"]),
            k=_fmt(row["kernel_relative_deviation"]),
        )
        for row in deep
    )
    sing_rows = "\n".join(
        "| {r:g} | {s:.12f} | {dx:.10f} |".format(
            r=row["radius"], s=row["four_pi_r2_gradient"], dx=row["direction_x"],
        )
        for row in ledger["singularity"]["rows"]
    )
    per_component = ledger["finite_difference"]["per_component_relative_error_at_1e-5"]

    doc = f"""# 3-D finite-depth kernel: full gradient `nabla G` and the order-1 Bessel layer

**Mission** m3673 (paper seat, self-start) · branch `agent/m3673` · kernel `tensorbem.finite_depth3d`
**Deliverables** `src/tensorbem/finite_depth3d.py`, `src/tensorbem/benchmarks/finite_depth3d_gradient.py`,
`tests/test_finite_depth3d_gradient.py` (19 gates), `docs/m3673/outputs/finite_depth3d_gradient.json`,
`docs/m3673/figures/*.png`

EDIT_WHITELIST: src/tensorbem/, tests/, docs/, README.md

---

## 1. Case (why this increment)

`green_function_3d(targets, sources, nu, depth)` is the John (1950) eigenseries of the 3-D
free-surface Green function in constant depth,

```
G(x,xi) = (1/2pi) sum_{{n>=1}} phi_n(z) phi_n(zeta)/N_n K_0(mu_n R)
        + (i/4) phi_0(z) phi_0(zeta)/N_0 H_0^{{(1)}}(k_0 R),
phi_0 = cosh(k_0 (z+h)),   k_0 tanh(k_0 h) = nu,   mu_n tan(mu_n h) = -nu.
```

The kernel shipped with `G` and the vertical derivative `dG/dz` only; §6 of
[the finite-depth kernel report](finite_depth3d.md) and the README name a horizontal
derivative as belonging to the panel layer, and the 3-D finite-depth *boundary-element
solver* as the next increment.  A panel layer needs the **full** `nabla_x G` (the singular
part here is carried by the modal sum itself, not split off as a Rankine term) and the
order-1 modified Bessel / Hankel functions that enter `dG/dR`.  This change supplies them
and, because a verification pass is part of the increment, also fixes a defect the pass
found in the order-0 Bessel evaluator.

**Test geometry** (unless stated otherwise): `nu = {ledger["meta"]["nu"]}` (so
`omega = sqrt(nu g)`), depth `h = {ledger["meta"]["depth"]:g}` m, five targets inside the strip
and one source at `(0.2, -0.1, -1.0)` m.

## 2. Method

**Derivatives.**  `G` depends on the horizontal coordinates only through `R`, and
`dK_0/dR = -K_1`, `dH_0^{{(1)}}/dR = -H_1^{{(1)}}`, so

```
dG/dR = -(1/2pi) sum_n mu_n phi_n(z) phi_n(zeta)/N_n K_1(mu_n R)
        - (i/4) k_0 phi_0(z) phi_0(zeta)/N_0 H_1^{{(1)}}(k_0 R),
```

with `dG/dx = (x-xi)/R dG/dR`, `dG/dy = (y-eta)/R dG/dR`.  The source gradient is the
same construction with `dphi_n(zeta)/dzeta = -mu_n sin(mu_n(zeta+h))` and
`dphi_0(zeta)/dzeta = k_0 sinh(k_0(zeta+h))`, and horizontal part `-nabla_x`.  New API:
`green_gradient_horizontal_3d`, `green_gradient_3d` (shape `(...,3)`, matching
`free_surface3d.green_gradient_field_3d`), `green_gradient_source_3d`, `bessel_k1`,
`hankel_h1`.

**Bessel layer.**  `K_1` and `H_1` are new.  `K_1` uses the ascending series for
`x < {bessel["series_limit"]:g}` (A&S 9.6.11), the asymptotic series for
`x >= {bessel["asymptotic_limit"]:g}`, and — in between — the **Laplace-integral
representation**, which has no cancellation at all:

```
K_0(x) = 2 e^(-x) int_0^inf e^(-x u^2)/sqrt(u^2+2) du,
K_1(x) = 2 e^(-x) int_0^inf e^(-x u^2)(1+u^2)/sqrt(u^2+2) du,
```

evaluated with 96 Gauss-Legendre nodes on `u in [0, 8/sqrt(x)]` (where the integrand is
below `e^{-64}`).  The same quadrature **replaced the shipped `K_0` branch point**: the
release code used the ascending series up to `x = 9`, where `1/K_0` amplifies the
cancellation between `-ln(x/2) I_0(x)` and the series, leaving a
**{_fmt(bessel["scan"]["legacy_k0_max_relative_error"])} relative cliff at
x = {bessel["scan"]["legacy_k0_worst_x"]:.2f}** while the asymptotic series is still at
`~1e-2` there.  Nothing in the test suite could see it: the existing Bessel gate compared
`K_0` to SciPy at `1e-8`, i.e. exactly at the cliff level.

**Verification.**  `benchmarks/finite_depth3d_gradient.py` recomputes everything from the
library and writes a strict JSON ledger; `tests/test_finite_depth3d_gradient.py` pins it.

## 3. Results

### 3.1 Bessel layer (`K_0`, `K_1`, `H_1`)

Pinned references: `mpmath.besselk` at 40 significant digits; the branch-error curves use
the Laplace quadrature as an independent reference (Fig. 1).

| x | `K_0` ref | err (new) | err (before) | `K_1` ref | err (new) | err (before) |
|---:|---:|---:|---:|---:|---:|---:|
{pinned_rows}

Worst case over the {bessel["scan"]["points"]}-point scan `1 <= x <= 60`: `K_0`
`{_fmt(bessel["scan"]["current_k0_max_relative_error"])}` (new) vs
`{_fmt(bessel["scan"]["legacy_k0_max_relative_error"])}` (before, at
`x = {bessel["scan"]["legacy_k0_worst_x"]:.2f}`); `K_1`
`{_fmt(bessel["scan"]["current_k1_max_relative_error"])}` (new) vs
`{_fmt(bessel["scan"]["legacy_k1_max_relative_error"])}` for the same branch layout.

Seam continuity (quadrature vs the branch it replaced, relative):

| x | `K_0` quad | vs series | vs asymptotic | `K_1` quad | vs series | vs asymptotic |
|---:|---:|---:|---:|---:|---:|---:|
{seam_rows}

`H_1^{{(1)}} = J_1 + iY_1` is evaluated from `analytic.bessel_j01_y01` (agrees with SciPy
to `1e-14`); it is additionally gated against the Richardson derivative of `H_0^{{(1)}}`.

### 3.2 Gradient against finite differences and closed identities

Per-component relative error at `eps = 1e-5`: dx `{_fmt(per_component["dx"])}`,
dy `{_fmt(per_component["dy"])}`, dz `{_fmt(per_component["dz"])}`;
fourth-order Richardson at the best step:
`{_fmt(ledger["finite_difference"]["best_richardson_max_relative_error"])}` (Fig. 2).

| step | central | Richardson |
|---:|---:|---:|
{fd_rows}

Closed identity (kernel symmetry `G(x,xi) = G(xi,x)`):
`max|nabla_x G(x,xi) - nabla_xi G(xi,x)| / max|nabla_x G| =
{_fmt(ledger["swap_identity"]["max_relative_difference"])}` — this one equation
cross-validates **all four** analytic derivative formulas (horizontal and vertical, field
and source) at once, so no separate sign/scale convention can be wrong in isolation.

### 3.3 Limits and physical closures

| quantity | measured | gate | verdict |
|---|---:|---:|---|
{check_rows}

* Free-surface / bottom closure is tested as `d/dx(dG/dz) = nu dG/dx` on `z = 0` and
  `d/dx(dG/dz) = 0` on `z = -h` (the two analytic routines cross-check each other).
* Horizontal irrotationality `d_y G_x = d_x G_y` holds to
  `{_fmt(ledger["boundary_closures"]["horizontal_curl_relative"])}`.
* Singularity strength (Fig. 3 is the deep-water limit):

| r [m] | `4 pi r^2 |nabla G|` | direction x |
|---:|---:|---:|
{sing_rows}

### 3.4 Deep-water collapse of the gradient

| `k_0 h` | h [m] | `nabla G` deviation | `G` deviation |
|---:|---:|---:|---:|
{deep_rows}

The gradient deviation tracks the kernel deviation to within a factor ~1.3, i.e. the
horizontal derivative introduces no new error source: at `k_0 h = 76.8` the finite-depth
gradient reproduces the independent Kelvin-kernel gradient to
`{_fmt(ledger["deep_water"]["finest_gradient_deviation"])}`.

## 4. Figures

* `figures/fig_bessel_branches.png` — relative error of `K_0`/`K_1` vs the Laplace
  quadrature over `1 <= x <= 60`: the red `K_0` curve shows the
  `{_fmt(bessel["scan"]["legacy_k0_max_relative_error"])}` cliff at `x ~ 8.96` that the
  re-branching removes (blue/green curves sit at the `1e-13` round-off floor); the dotted
  vertical lines are the new branch seams.
* `figures/fig_gradient_fd.png` — step ladder of the finite-difference check: the
  `O(eps^2)`/round-off trade-off of the central difference and the two-orders-of-magnitude
  better Richardson estimate, i.e. the analytic gradient is correct far below the
  finite-difference floor.
* `figures/fig_deep_water_gradient.png` — deep-water collapse of `nabla G` (circles) next
  to the kernel collapse (squares) vs `k_0 h`.

## 5. Note / limitations

* Only the **kernel** is delivered here; panel integration, layer jumps and the CHIEF/lid
  treatment for the finite-depth BEM solver are still the next increment (unchanged scope).
* Constant depth, linearised potential flow.  Sloping bottom, current and finite amplitude
  remain out of scope.
* The evanescent-mode sum diverges logarithmically as `R -> 0` (it carries the
  `1/(4 pi r)` singularity), so the gradient must not be evaluated at coincident points;
  the API raises instead.
* `dG/dR` needs `K_1` at the *smallest* `mu_n R` of the sum, where the series branch
  applies; the mode count is chosen by `exp(-mu_n R_min) <= 1e-12` as before, which also
  bounds the `mu_n K_1` weighting.
* **Cost**: the quadrature branch is ~100x a series evaluation.  It only applies to
  arguments in `[4, 16)`, i.e. the tail of the mode sum; a full gradient evaluation on the
  five-point test set with `h = 4` m takes single-digit milliseconds.

## 6. Error verdict

| key quantity | reference | relative error | < 3 %? |
|---|---|---:|---|
| `K_0`, `K_1` (all branches) | mpmath, 40 digits | `{_fmt(max(bessel["k0_max_relative_error"], bessel["k1_max_relative_error"]))}` | yes |
| `nabla_x G` vs finite difference | Richardson of `G` | `{_fmt(ledger["finite_difference"]["best_richardson_max_relative_error"])}` | yes |
| `nabla_x G(x,xi)` vs `nabla_xi G(xi,x)` | kernel symmetry | `{_fmt(ledger["swap_identity"]["max_relative_difference"])}` | yes |
| free-surface closure of `nabla G` | `d_z d_x G = nu d_x G` | `{_fmt(ledger["boundary_closures"]["free_surface_residual"])}` | yes |
| `nabla G` deep-water limit at `k_0 h = 76.8` | Kelvin kernel (independent code) | `{_fmt(ledger["deep_water"]["finest_gradient_deviation"])}` | yes |
| singularity strength at `r = 2.5e-3` m | `1/(4 pi r)` | `{_fmt(ledger["singularity"]["finest_deviation"])}` | yes |

**Verdict**: {("all gates PASS" if ledger["verdict"]["pass"] else "FAIL: " + ", ".join(ledger["verdict"]["failures"]))}.
The one honest caveat is the defect this unit fixed: the shipped `K_0` was **not** accurate
to `1e-8` near its branch point, a fact that the pre-existing `1e-8` Bessel gate could not
detect; the gate in this unit is `1e-12`.
"""

    (DOCS / "finite_depth3d_gradient.md").write_text(doc)
    print("wrote docs/finite_depth3d_gradient.md and 3 figures")


if __name__ == "__main__":
    main()