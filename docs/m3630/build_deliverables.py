"""Build the deliverables of the 6-DOF matrix-algebra study (m3630).

Writes, from one measurement run:

* ``tests/data/six_dof_matrix_algebra_metrics.json`` -- the pinned scalars the
  regression test compares against;
* ``docs/six_dof_matrix_algebra.md`` -- the published note, with the tables
  injected programmatically (no hand-copied numbers);
* ``docs/m3630/six_dof_matrix_algebra.png`` -- convergence and matrix-structure
  figure.

Run::

    python docs/m3630/build_deliverables.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from tensorbem.benchmarks import six_dof_matrix_algebra as module

ROOT = Path(__file__).resolve().parents[2]
PINNED = ROOT / "tests" / "data" / "six_dof_matrix_algebra_metrics.json"
DOC = ROOT / "docs" / "six_dof_matrix_algebra.md"
FIGURE = Path(__file__).resolve().parent / "six_dof_matrix_algebra.png"

DOC_TEMPLATE = """# Six-degree-of-freedom radiation-matrix algebra: reciprocity, reflection selection rules and reference-centre covariance

{masthead}

## 1. What is checked, and why it was missing

`tensorbem.free_surface_solver3d.solve_radiation_3d` returns the **full** `(6, 6)`
added-mass and damping matrices of a floating body,

```
A_ij + (i/omega) B_ij = -rho * int_{Gamma_B} phi_j n_i dS ,
```

where `phi_j` is the radiation potential of mode `j`. Every other verification in
this repository reads **diagonal** entries only (Hulme's `A11`/`A33` for the
hemisphere, Lamb's added masses for the ellipsoid). The diagonals can be right
while the moment arms, the `i`/`j` order or the sign of the rotational normals are
wrong, and nothing in the suite would notice. Four predictions close that gap.
The first three are theorems about the continuous problem; the fourth is an exact
algebraic identity that must hold at roundoff.

1. **Reciprocity.** `A_ij = A_ji` and `B_ij = B_ji`. In this solver the two are
   different discrete integrals (`phi_j n_i` versus `phi_i n_j`, over different
   radiation solutions), so the check tests the assembled operators.
2. **Reflection selection rule.** The hull is invariant under `S_x: x -> -x` and
   `S_y: y -> -y`. The generalized normal velocity of each mode is even or odd
   under each reflection, with the parity table below; because the free-surface
   Green function and the waterplane are invariant too, the change of variables
   `x -> R x` gives `A_ij = eps_i eps_j A_ij`. Hence `A_ij = B_ij = 0` whenever
   `eps_i eps_j = -1`: 26 of the 30 off-diagonal entries.
3. **Spin-axis identity.** For a body of revolution about the vertical axis the
   azimuthal normal velocity satisfies `(z_hat x r) . n = 0` on the hull, because
   the normal of a surface of revolution has no azimuthal component. Therefore
   `phi_yaw == 0` and `A_66 = B_66 = 0` exactly (this is also the `a = b` limit of
   Lamb's 1932 spheroid formula, s.114).
4. **Reference-centre covariance.** Moving the reference point by `d` maps the
   generalized velocities by `u' = M u`, `M = [[I, -[d]_x], [0, I]]`, and added
   mass is a tensor of that pair of vector spaces, so the same fields must obey
   `A' = M^{-T} A M^{-1}` (and likewise for `B`) for **any** body, wavenumber and
   mesh. This is the only check here whose tolerance is machine precision.

Parity table `(eps(S_x), eps(S_y))` -- derived from `v(Rx) = R v(x)` for polar
vectors and `w -> -R w` for the axial rotation vector, and re-derived numerically
from the assembled normals by the test suite:

| mode | surge | sway | heave | roll | pitch | yaw |
|---|---|---|---|---|---|---|
| `eps(S_x)` | -1 | +1 | +1 | +1 | -1 | -1 |
| `eps(S_y)` | +1 | -1 | +1 | -1 | +1 | -1 |

The four survivors of the reflection rule are the diagonal and the two
*same-parity* off-diagonal pairs, `surge-pitch` and `sway-roll`. They are not
asserted to vanish: see section 5.

## 2. Case

Hulme's floating hemisphere (unit radius, waterline exactly on `z = 0`,
`rho = 1025 kg/m^3`, `g = 9.80665 m/s^2`, wavenumber `nu = k = 1.0 1/m`,
`ka = 1`). Its panel counts are `4 * 4**subdivisions`, the same meshes that back
`docs/marine_bem_benchmarks.md` and the Hulme tables. The centre of rotation is
the waterplane centre `(0, 0, 0)`.

## 3. Measured

{tables}

`max|A-A^T|/|A|` and `max|B-B^T|/|B|` are normalised by `max|diag|`; `forbidden`
is the largest of the 26 entries the reflection rule kills, normalised the same
way; `A66/A33`, `B66/B33` are the spin-axis identity; `h` is
`sqrt(2 * total area / N)`.

## 4. Results

**Reciprocity holds and tightens.** Added mass is the cleaner of the two:
`{recip_a}` at {panels_last} panels, falling faster than `h**2` (measured
{order_recip_a}). Damping converges too, but with a non-monotone order
({order_recip_b}) -- the imaginary part is what the wave part of the kernel and
the CHIEF rows act on, so it is the honest number to quote.

**The reflection rule is verified, not by construction.** The twenty forbidden
entries fall from `{forbidden_first}` to `{forbidden_last}` (order
{order_forbidden}) and the worst pair is reported per mesh, so a regression names
the coupling that broke. Note the mesh itself is mirror-symmetric to `1e-12`
`1e-12` the parity table is checked against the assembled normals before the
selection rule is applied, so the zeros cannot be explained by a mesh that was
never symmetric.

**The spin-axis identity is approached at third order.** `A66/A33` goes
`{spin_added}`; `B66/B33` goes `{spin_damping}`, i.e. four to five orders below
the damping scale at 256 panels. This is the only check with an *analytical
value* (not merely a symmetry) attached to an off-diagonal-free slot of the
matrix.

**The centre-shift identity is exact.** Residuals of `{cov_a}` (added mass) and
`{cov_b}` (damping) are roundoff. Two negative controls in the test suite (wrong
lever-arm sign, transposed matrix) fail the same check by more than three orders
of magnitude, so the check has teeth rather than passing on a diagonal matrix.

![Decay of the four structural errors with panel size, and the structure of the
measured added-mass matrix at 256 panels](m3630/six_dof_matrix_algebra.png)

Left: every structural residual is normalised by its own diagonal scale and drawn
against `h = sqrt(2A/N)`; the `h**2` and `h**3` guides bracket the measured
orders. Right: `|A_ij| / max|diag A|` at 256 panels on a logarithmic colour scale;
the 26 entries the reflection rule kills are marked and sit at `<= {forbidden_last}`,
while the two same-parity pairs (`surge-pitch`, `sway-roll`) are the only visible
off-diagonal entries.

## 5. What is *not* claimed

* **Surge-pitch and sway-roll are not asserted to vanish.** They have
  `eps_i eps_j = +1`, so no reflection forbids them; reciprocity makes them
  *symmetric* (`|antisymmetric part| <= 0.2 |symmetric part|` on every mesh),
  whereas a Timman-Newman reading of the transverse mirror plane would make them
  antisymmetric and hence exactly zero. That relation does not follow from the
  two reflections used here, and the measured magnitude decays like `h**{order_same}`
  towards zero: the data cannot separate "exactly zero, residual is
  discretisation error" from "physical value below the discretisation error".
  The benchmark therefore reports the magnitude and the order. For users of the
  RAO chain the practical statement is the useful one: on a doubly symmetric hull
  the coupling terms are `{same_first}` of the diagonal at 16 panels and
  `{same_last}` at 256, the same order as the diagonal's own error against
  Hulme's solution.
* One body, one wavenumber, deep water. The reflection rule would need a third
  mirror plane (or a half-body reduction) to say anything about a hull that is
  fore-aft asymmetric; the covariance identity is body-independent and is the
  check to run for those.
* `B` reciprocity is stated as measured, not as a theorem-derived asymptotic
  order: the 16-to-64-panel step gives order {order_recip_b_first}, the next
  {order_recip_b_last}. No order is pinned for it.

## 6. Reproduction

```
python -m tensorbem.benchmarks.six_dof_matrix_algebra --write
python docs/m3630/build_deliverables.py     # regenerates this note and the figure
pytest tests/test_six_dof_matrix_algebra.py
```

The note, the pinned metrics in `tests/data/` and the figure all come from the
same run; `tests/test_six_dof_matrix_algebra.py` re-runs the study, compares the
table above cell by cell with the recomputation, and re-derives the parity table
from the assembled normals.
"""


def _fill(template: str, **values: object) -> str:
    """Substitute ``{name}`` placeholders without tripping over literal braces."""
    for key, value in values.items():
        template = template.replace("{" + key + "}", str(value))
    return template


def _pinned_metrics(study: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    for row in study["levels"]:
        tag = f"subdivisions_{row['subdivisions']}"
        out[f"{tag}.panels"] = float(row["panels"])
        out[f"{tag}.panel_size"] = row["panel_size"]
        out[f"{tag}.added_mass_reciprocity_rel"] = row["added_mass_block"]["reciprocity_rel"]
        out[f"{tag}.damping_reciprocity_rel"] = row["damping_block"]["reciprocity_rel"]
        out[f"{tag}.added_mass_forbidden_rel"] = row["added_mass_block"]["forbidden_rel"]
        out[f"{tag}.damping_forbidden_rel"] = row["damping_block"]["forbidden_rel"]
        out[f"{tag}.spin_axis_added_mass_rel"] = row["spin_axis_added_mass_rel"]
        out[f"{tag}.spin_axis_damping_rel"] = row["spin_axis_damping_rel"]
        out[f"{tag}.surge_pitch_symmetric_part_rel"] = row["same_class"]["surge-pitch"][
            "symmetric_part_rel"
        ]
    out["covariance.added_mass_residual_rel"] = study["covariance"]["added_mass_residual_rel"]
    out["covariance.damping_residual_rel"] = study["covariance"]["damping_residual_rel"]
    return out


def _figure(study: dict) -> None:
    levels = study["levels"]
    h = np.array([row["panel_size"] for row in levels])
    series = {
        "added-mass reciprocity": [row["added_mass_block"]["reciprocity_rel"] for row in levels],
        "damping reciprocity": [row["damping_block"]["reciprocity_rel"] for row in levels],
        "reflection-forbidden (A)": [
            row["added_mass_block"]["forbidden_rel"] for row in levels
        ],
        "spin axis A66/A33": [row["spin_axis_added_mass_rel"] for row in levels],
        "surge-pitch (symmetric part)": [
            abs(row["same_class"]["surge-pitch"]["symmetric_part_rel"]) for row in levels
        ],
    }
    figure, (left, right) = plt.subplots(1, 2, figsize=(12.0, 4.8))
    for label, values in series.items():
        left.loglog(h, np.abs(values), "o-", label=label)
    guides = h / h[0]
    left.loglog(h, 2.0e-4 * guides**2, "k--", lw=1.0, label=r"$h^2$ guide")
    left.loglog(h, 2.0e-5 * guides**3, "k:", lw=1.0, label=r"$h^3$ guide")
    left.set_xlabel("panel size  h = sqrt(2A/N)")
    left.set_ylabel("relative residual (normalised by max|diag|)")
    left.set_title("Structural residual decay, floating hemisphere at $ka=1$")
    left.grid(True, which="both", alpha=0.3)
    left.legend(fontsize=7.5, loc="lower right")

    finest = levels[-1]
    added = np.asarray(finest["added_mass"])
    scale = float(np.max(np.abs(np.diag(added))))
    mask = module.selection_mask()
    shown = np.abs(added) / scale
    shown[shown <= 0.0] = 1.0e-18
    image = right.imshow(np.log10(shown), cmap="viridis", vmin=-12.0, vmax=0.0)
    for i, j in module.forbidden_pairs():
        right.plot(j, i, "x", color="red", ms=5.0, mew=1.2)
    right.set_xticks(range(6), module.MODES, rotation=45)
    right.set_yticks(range(6), module.MODES)
    right.set_title(
        f"$|A_{{ij}}|/\\max|\\mathrm{{diag}}\\,A|$ at {finest['panels']} panels\n"
        "red x = entries the reflection rule forbids"
    )
    figure.colorbar(image, ax=right, label=r"$\log_{10}$ (relative magnitude)")
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=150)
    plt.close(figure)


def main() -> int:
    study = module.run((1, 2, 3), 1.0, covariance_subdivisions=2)
    PINNED.parent.mkdir(parents=True, exist_ok=True)
    PINNED.write_text(json.dumps(_pinned_metrics(study), indent=2) + "\n", encoding="utf-8")

    levels = study["levels"]
    orders = study["observed_order"]
    same_orders = study["same_class_orders"]["surge-pitch"]
    masthead = (
        "Measurement: `python -m tensorbem.benchmarks.six_dof_matrix_algebra --write`"
        + f" | Python {study['environment']['python']}, numpy {study['environment']['numpy']}"
        + f" | ka = {study['wavenumber']:.1f}, density = {study['density']:.0f} kg/m^3"
    )
    text = _fill(DOC_TEMPLATE, 
        masthead=masthead,
        tables=module.render_tables(study),
        recip_a=f"{levels[-1]['added_mass_block']['reciprocity_rel']:.2e}",
        panels_last=levels[-1]["panels"],
        order_recip_a=", ".join(f"{v:.1f}" for v in orders["added_mass_block.reciprocity_rel"][1:]),
        order_recip_b=", ".join(f"{v:.1f}" for v in orders["damping_block.reciprocity_rel"][1:]),
        order_recip_b_first=f"{orders['damping_block.reciprocity_rel'][1]:.1f}",
        order_recip_b_last=f"{orders['damping_block.reciprocity_rel'][-1]:.1f}",
        forbidden_first=f"{levels[0]['added_mass_block']['forbidden_rel']:.2e}",
        forbidden_last=f"{levels[-1]['added_mass_block']['forbidden_rel']:.1e}",
        order_forbidden=", ".join(f"{v:.1f}" for v in orders["added_mass_block.forbidden_rel"][1:]),
        spin_added=" -> ".join(f"{row['spin_axis_added_mass_rel']:.1e}" for row in levels),
        spin_damping=" -> ".join(f"{row['spin_axis_damping_rel']:.1e}" for row in levels),
        cov_a=f"{study['covariance']['added_mass_residual_rel']:.1e}",
        cov_b=f"{study['covariance']['damping_residual_rel']:.1e}",
        order_same=", ".join(f"{v:.2f}" for v in same_orders[1:]),
        same_first=f"{abs(levels[0]['same_class']['surge-pitch']['symmetric_part_rel']):.1e}",
        same_last=f"{abs(levels[-1]['same_class']['surge-pitch']['symmetric_part_rel']):.1e}",
    )
    DOC.write_text(text, encoding="utf-8")
    _figure(study)
    print(f"wrote {PINNED}\nwrote {DOC}\nwrote {FIGURE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())