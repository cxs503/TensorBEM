"""Unit-invariance and dispersion arbitration of the three Michell paths.

Why this module exists
----------------------
``docs/m3125/wigley_normalization_audit.md`` established, as a one-off study,
that :func:`tensorbem.wigley.michell_wave_resistance` does not return a force:
under a change of base units its output scales with exponents
``(mass, length, time) = (1, +1, -2)`` where a force requires ``(1, -1, +2)``.
That study concluded the only possible repair is a missing factor ``g**2 / C``
with an undetermined dimensionless ``C``.

Two things were left undone, and this module supplies them as *executable*
gates rather than prose:

1. **The gate was never applied to the other two shipped paths.** There are
   three independent Michell implementations in the package -- the Wigley
   closed form (:mod:`tensorbem.wigley`), the general-hullform panel path
   (:mod:`tensorbem.michell`), and the standard-form amplitude path
   (:mod:`tensorbem.thin_ship_michell`). Only the first was audited. The
   unit-rescaling harness here (:func:`measure_force_exponents`) measures the
   exponent vector of each one, so the ledger is complete and machine-checked.

2. **``g**2 / C`` is not the whole defect.** A missing *constant* cannot repair
   a *spectrum*: the two legacy paths sample the Kochin function at a single
   wavenumber ``k = alpha = k0 * lambda``, while the steady free-surface
   condition ``U**2 phi_xx + g phi_z = 0`` forces the depth-decay wavenumber
   and the streamwise phase wavenumber apart, ``k = alpha**2 / k0``. At
   ``alpha = k0 * lambda`` that identity demands ``k = k0 * lambda**2``: the
   canonical pair used by the lambda form of Michell's integral. The legacy
   sampling therefore violates the free-surface condition by
   ``|k - alpha**2/k0| / k = lambda - 1`` -- a *relative* error that varies
   over the integration range, so no prefactor can remove it (measured in the
   report: 0.5, 1.0, 3.0 at ``lambda = 1.5, 2, 4``). The measured
   consequence is :func:`residual_spectrum_ratio`: after applying the ``g**2``
   factor the legacy result still disagrees with the standard form by a
   **Froude-dependent** factor, which a constant ``C`` cannot express.

What is measured, and against what
----------------------------------
The reference is :func:`tensorbem.thin_ship_michell.wave_resistance_from_
amplitude`, the repo's standard-form path (prefactor ``4 rho g**2/(pi U**2)``,
weight ``lambda**2/sqrt(lambda**2 - 1)``, ``k = k0 lambda**2``,
``alpha = k0 lambda``), whose lambda and theta forms are pinned to each other
by ``tests/test_thin_ship_michell.py``. **No claim is made here about the
absolute value of Michell's theory against experiment** -- that stays
``WITHHELD``; what is arbitrated is which implementations are *dimensionally
sound* and *obey the free-surface condition*, and how far the legacy ones are
from the standard form.

The repair recipe is implemented and validated here as
:func:`wigley_corrected_resistance`: the Wigley closed-form Kochin integrals
(:func:`tensorbem.wigley._michell_kochin_integrand_x` for the phase,
:func:`tensorbem.wigley._michell_kochin_z_integral` for the depth decay) are
evaluated at the canonical pair and integrated with the standard weight. It
reproduces the independent theta form to ~1e-9, i.e. the closed-form leg can be
repaired without touching the theory. The recipe is **not** wired into the
shipped function by this module (that changes pinned numbers in
``benchmarks/verified``, ``docs/wigley_wave_resistance.md`` and the
declared-number ledger -- see ``docs/michell_unit_arbitration.md``).

Run ``python -m tensorbem.benchmarks.michell_unit_arbitration`` to regenerate
``docs/michell_unit_arbitration.md`` and its ``numbers.json``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .. import michell, wigley
from ..thin_ship_michell import (
    WigleyHull,
    wave_resistance_from_amplitude,
    wave_resistance_theta_form,
    wigley_amplitude_function,
)

#: Unit exponents ``(mass, length, time)`` of a force, in the convention of
#: :func:`case_in_units`: a unit system whose length unit is ``s_l`` metres
#: makes a force of ``R`` newtons read ``R * s_t**2 / (s_m * s_l)``.
FORCE_EXPONENTS = (-1.0, -1.0, 2.0)

#: The Wigley III geometry of the verified anchors: ``L/B = 10``, ``B/T = 1.6``.
BASE_CASE: dict[str, float] = {
    "L": 3.0,
    "B": 0.30,
    "T": 0.1875,
    "rho": 1025.0,
    "g": 9.80665,
    "Fr": 0.30,
}

#: Unit scalings used to measure the exponents, one axis at a time.
UNIT_SCALES = {
    "mass": (2.0, 1.0, 1.0),
    "length": (1.0, 2.0, 1.0),
    "time": (1.0, 1.0, 2.0),
}


def case_in_units(case: dict[str, float], s_mass: float = 1.0,
                  s_length: float = 1.0, s_time: float = 1.0) -> dict[str, float]:
    """The same physical case written in a rescaled unit system.

    ``s_mass``, ``s_length``, ``s_time`` are the sizes of the new base units in
    kilograms, metres and seconds, so every numeric value carries the inverse
    power: a length reads ``L / s_length``, a density ``rho * s_length**3 /
    s_mass``, a speed ``U * s_time / s_length``. The Froude number is
    dimensionless and therefore invariant -- which is exactly why geometric
    scaling studies cannot detect the defect (``docs/m3125``, pitfall 2).
    """
    speed = math.sqrt(case["g"] * case["L"]) * case["Fr"]
    return {
        "L": case["L"] / s_length,
        "B": case["B"] / s_length,
        "T": case["T"] / s_length,
        "rho": case["rho"] * s_length**3 / s_mass,
        "g": case["g"] * s_time**2 / s_length,
        "U": speed * s_time / s_length,
        "Fr": case["Fr"],
    }


def _speed(case: dict[str, float]) -> float:
    return math.sqrt(case["g"] * case["L"]) * case["Fr"]


# --------------------------------------------------------------------------- #
# The three shipped paths
# --------------------------------------------------------------------------- #
def legacy_wigley_resistance(case: dict[str, float]) -> float:
    """``tensorbem.wigley.michell_wave_resistance`` (closed-form Kochin)."""
    return wigley.michell_wave_resistance(
        case["L"], case["B"], case["T"], _speed(case), case["rho"], case["g"]
    )


def legacy_panel_resistance(case: dict[str, float]) -> float:
    """``tensorbem.michell.michell_resistance`` through the panel path."""
    panels = michell.wigley_panels(case["L"], case["B"], case["T"], n_x=24, n_z=12)
    return michell.wave_resistance_from_panels(
        panels, _speed(case), case["T"], case["rho"], case["g"]
    )


def standard_form_resistance(case: dict[str, float]) -> float:
    """``tensorbem.thin_ship_michell`` standard-form amplitude integral."""
    hull = WigleyHull(L=case["L"], B=case["B"], T=case["T"])
    return wave_resistance_from_amplitude(
        wigley_amplitude_function(hull),
        _speed(case),
        lam_max=40.0,
        n_lam=240,
        rho=case["rho"],
        g=case["g"],
    )


PATHS = {
    "wigley_closed_form (legacy)": legacy_wigley_resistance,
    "michell_panels (legacy)": legacy_panel_resistance,
    "thin_ship_standard_form": standard_form_resistance,
}


def measure_force_exponents(path, case: dict[str, float] | None = None,
                            scales: dict[str, tuple[float, float, float]] | None = None):
    """Measure the unit exponents of ``path`` as ``(mass, length, time)``.

    Each exponent is the log-ratio slope of the output under a rescaling of one
    base unit, the measurement rule of ``docs/m3125`` §3: a quantity that is a
    force must read ``(1, -1, 2)``-compatible, i.e. ``FORCE_EXPONENTS``.
    """
    case = BASE_CASE if case is None else case
    scales = UNIT_SCALES if scales is None else scales
    base = path(case)
    out = []
    for axis in ("mass", "length", "time"):
        s_m, s_l, s_t = scales[axis]
        u = {"mass": s_m, "length": s_l, "time": s_t}[axis]
        scaled = path(case_in_units(case, s_m, s_l, s_t))
        out.append(math.log(scaled / base) / math.log(u))
    return tuple(float(v) for v in out)


def dispersion_residual(k, alpha, k0):
    """Relative violation ``|k - alpha**2/k0| / k`` of the steady free surface.

    The steady Neumann-Kelvin condition ``U**2 phi_xx + g phi_z = 0`` admits
    ``phi ~ exp(k z + i alpha x)`` only if ``k = alpha**2 / k0``; both are real
    for a wave and ``k`` is the decay rate used in the Kochin integral.
    """
    k = np.asarray(k, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    return np.abs(k - alpha * alpha / k0) / k


def wigley_corrected_resistance(case: dict[str, float], lam_max: float = 40.0,
                                n_quad: int = 360) -> float:
    """Repair recipe: the Wigley closed-form Kochin at the canonical pair.

    ``R_w = (4 rho g**2/(pi U**2)) int_1^{lam_max} |A|**2 lam**2 dlam /
    sqrt(lam**2 - 1)`` with ``A`` from :mod:`tensorbem.wigley`'s own closed-form
    ``x``- and ``z``-integrals at ``k = k0 lam**2`` (depth decay) and
    ``alpha = k0 lam`` (phase). The ``lam = 1 + s**2`` substitution removes the
    cusp, so the weight in ``s`` is ``2 lam**2 ds / sqrt(2 + s**2)``.

    One side of the hull only: the legacy closed form carries a factor two for
    the two sides, which is the convention question recorded in the report (a
    factor four in ``R_w``) and NOT resolved here.
    """
    L, B, T, rho, g = case["L"], case["B"], case["T"], case["rho"], case["g"]
    U = _speed(case)
    k0 = g / U**2
    nodes, weights = np.polynomial.legendre.leggauss(n_quad)
    s_max = math.sqrt(lam_max - 1.0)
    s = 0.5 * s_max * (nodes + 1.0)
    ws = 0.5 * s_max * weights
    lam = 1.0 + s * s
    integrand = np.empty(n_quad)
    for i in range(n_quad):
        alpha = k0 * lam[i]
        k_depth = k0 * lam[i] ** 2
        amplitude = (
            (-4.0 * B / L**2)
            * wigley._michell_kochin_integrand_x(alpha, L)
            * wigley._michell_kochin_z_integral(k_depth, T)
        )
        integrand[i] = 2.0 * lam[i] ** 2 * abs(amplitude) ** 2 / math.sqrt(2.0 + s[i] * s[i])
    prefactor = 4.0 * rho * g * g / (math.pi * U**2)
    return float(prefactor * np.sum(ws * integrand))


def residual_spectrum_ratio(fr: float) -> dict[str, float]:
    """How far the legacy value is from the standard form, before and after ``g**2``.

    ``after_prefactor`` multiplies the legacy value by ``g**2 / pi`` (the only
    factor dimensional analysis admits) and divides by the standard-form value.
    A value that still drifts with Froude number proves the defect is not a
    missing constant: it is the single-wavenumber spectrum.
    """
    case = dict(BASE_CASE)
    case["Fr"] = fr
    legacy = legacy_wigley_resistance(case)
    standard = standard_form_resistance(case)
    k0_factor = case["g"] ** 2 / math.pi
    return {
        "Fr": fr,
        "legacy": legacy,
        "standard": standard,
        "raw_ratio": standard / legacy,
        "after_prefactor": standard / (legacy * k0_factor),
        "corrected_recipe": wigley_corrected_resistance(case),
    }


def build_record() -> dict:
    """Every number of the arbitration, computed, never transcribed."""
    exponents = {name: measure_force_exponents(fn) for name, fn in PATHS.items()}
    lam = np.array([1.0, 1.5, 2.0, 4.0])
    k0 = BASE_CASE["g"] / _speed(BASE_CASE) ** 2
    record = {
        "case": BASE_CASE,
        "unit_scales": {k: list(v) for k, v in UNIT_SCALES.items()},
        "force_exponents": list(FORCE_EXPONENTS),
        "exponents": {name: list(v) for name, v in exponents.items()},
        "dimensionally_sound": {
            name: bool(np.allclose(v, FORCE_EXPONENTS, atol=1e-6, rtol=0.0))
            for name, v in exponents.items()
        },
        "dispersion": {
            "lambda": lam.tolist(),
            "canonical_residual": dispersion_residual(k0 * lam**2, k0 * lam, k0).tolist(),
            "legacy_residual": dispersion_residual(k0 * lam, k0 * lam, k0).tolist(),
            "analytic_legacy_residual": (lam - 1.0).tolist(),
        },
        "spectrum": [residual_spectrum_ratio(fr) for fr in (0.20, 0.25, 0.30, 0.35)],
        "theta_form_cross_check": {},
    }
    case = dict(BASE_CASE)
    hull = WigleyHull(L=case["L"], B=case["B"], T=case["T"])
    amp = wigley_amplitude_function(hull)
    k0 = case["g"] / _speed(case) ** 2

    def amplitude_theta(theta):
        lam = 1.0 / np.cos(theta)
        return amp(k0 * lam**2, k0 * lam)

    theta = wave_resistance_theta_form(
        amplitude_theta, _speed(case), n_theta=400, order=8,
        rho=case["rho"], g=case["g"], theta_max=math.acos(1.0 / 40.0),
    )
    corrected = wigley_corrected_resistance(case)
    record["theta_form_cross_check"] = {
        "corrected_closed_form": corrected,
        "theta_form": theta,
        "relative_difference": abs(corrected - theta) / abs(theta),
    }
    record["sides_factor"] = {
        "legacy_kochin_carries_both_sides": True,
        "effect_on_resistance": 4.0,
        "note": "legacy H = 2 * int int y_x; the standard form integrates one "
                "side.  Resolving which pairs with the textbook 4 rho g^2/pi U^2 "
                "constant needs an external far-field energy-flux reference and "
                "is recorded as WITHHELD, not guessed.",
    }
    return record


def render_markdown(record: dict) -> str:
    exp = record["exponents"]
    sound = record["dimensionally_sound"]
    lines = [
        "# Michell unit-invariance and dispersion arbitration",
        "",
        "Generated by `python -m tensorbem.benchmarks.michell_unit_arbitration`;",
        "every number below is read back from `numbers.json`, none is transcribed.",
        "",
        "## 1. Case",
        "",
        "Wigley III geometry of the verified anchors "
        f"(`L` = {record['case']['L']} m, `B` = {record['case']['B']} m, "
        f"`T` = {record['case']['T']} m, `rho` = {record['case']['rho']} kg/m^3, "
        f"`Fr` = {record['case']['Fr']}). The case is re-expressed in unit systems "
        "whose base units are scaled by the factors in §2, which is the only "
        "probe that can see a missing dimensional factor: the Froude number is "
        "invariant under it.",
        "",
        "## 2. Unit exponents (target = force = `(mass, length, time)` "
        f"{tuple(record['force_exponents'])})",
        "",
        "| path | measured exponents | dimensionally sound |",
        "|---|---|:--:|",
    ]
    for name in exp:
        vals = ", ".join(f"{v:+.6f}" for v in exp[name])
        lines.append(f"| `{name}` | ({vals}) | {'yes' if sound[name] else '**no**'} |")
    lines += [
        "",
        "## 3. Free-surface dispersion identity",
        "",
        "`phi ~ exp(k z + i alpha x)` satisfies `U^2 phi_xx + g phi_z = 0` only if",
        "`k = alpha^2/k0`. The canonical lambda form uses `(k, alpha) = "
        "(k0 lambda^2, k0 lambda)`; the legacy paths hand the *same* wavenumber "
        "to both roles, `(k0 lambda, k0 lambda)`.",
        "",
        "| lambda | canonical residual | legacy residual | `lambda - 1` |",
        "|---:|---:|---:|---:|",
    ]
    disp = record["dispersion"]
    for i, lam in enumerate(disp["lambda"]):
        lines.append(
            f"| {lam:g} | {disp['canonical_residual'][i]:.3e} | "
            f"{disp['legacy_residual'][i]:.3e} | {disp['analytic_legacy_residual'][i]:.3e} |"
        )
    lines += [
        "",
        "The legacy residual is exactly `lambda - 1`: a *relative* error that",
        "varies across the integration range, so no prefactor can absorb it.",
        "",
        "## 4. Consequence: the defect is a spectrum, not a constant",
        "",
        "| Fr | legacy `R_w` [N] | standard form [N] | standard/legacy | "
        "after `g^2/pi` | repaired recipe [N] |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in record["spectrum"]:
        lines.append(
            f"| {row['Fr']:.2f} | {row['legacy']:.6e} | {row['standard']:.6e} | "
            f"{row['raw_ratio']:.3f} | {row['after_prefactor']:.4f} | "
            f"{row['corrected_recipe']:.6e} |"
        )
    cross = record["theta_form_cross_check"]
    lines += [
        "",
        "The `after g^2/pi` column is the residue the earlier audit attributed to",
        "an undetermined constant `C`. It drifts with Froude number, so the",
        "`g^2/C` framing is incomplete: the wavenumber pairing must be fixed",
        "first, and only then can a constant be discussed.",
        "",
        "## 5. Repair recipe, cross-checked against an independent route",
        "",
        f"* Wigley closed form, canonical pair, one side: `{cross['corrected_closed_form']:.10e}` N",
        f"* independent theta form: `{cross['theta_form']:.10e}` N",
        f"* relative difference: `{cross['relative_difference']:.3e}`",
        "",
        "Two different quadratures (lambda with the `1 + s^2` cusp removal vs a",
        "graded theta rule) and two different amplitude evaluations agree to the",
        "quadrature floor quoted above, so the closed-form leg can be repaired",
        "without touching the theory.",
        "",
        "## 6. What is still WITHHELD",
        "",
        "* **Absolute scale.** Adopting the standard form fixes the value only",
        "  relative to `tensorbem.thin_ship_michell`; no experimental or",
        "  first-principles reference for a Wigley `C_w` ships with this repo, so",
        "  the absolute value stays `WITHHELD` (`benchmarks/verified/README.md`).",
        f"* **Sides convention.** The legacy Kochin carries both sides "
        f"(factor {record['sides_factor']['effect_on_resistance']:.0f} in `R_w`). "
        "  Settling whether the textbook",
        "  constant pairs with one or two sides needs a far-field energy-flux",
        "  derivation that this module does not contain -- recorded, not guessed.",
        "* **The shipped functions are not patched here.** Wiring the recipe in",
        "  changes pinned numbers in `benchmarks/verified/`, ",
        "  `docs/wigley_wave_resistance.md` and the declared-number ledger, which",
        "  is a re-pinning mission of its own.",
        "",
    ]
    return "\n".join(lines)


def main() -> dict:
    record = build_record()
    out_dir = Path("docs/michell_unit_arbitration")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "numbers.json").write_text(json.dumps(record, indent=2) + "\n")
    Path("docs/michell_unit_arbitration.md").write_text(render_markdown(record) + "\n")
    return record


if __name__ == "__main__":
    main()