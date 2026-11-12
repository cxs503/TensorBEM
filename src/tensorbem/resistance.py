"""Empirical calm-water resistance baselines: ITTC-57 friction and Holtrop (1984).

TensorBEM's own resistance physics is the Michell thin-ship wave resistance
(:mod:`tensorbem.thin_ship_michell`) obtained from a Rankine centreplane source
sheet, i.e. a *wave* resistance.  Total calm-water resistance additionally needs
a friction line and a roughness treatment, and any verification of a predicted
total resistance needs an independent prediction to compare against.  Both roles
are filled here by the route standard in ship practice:

* the ITTC-1957 model-ship correlation line
  ``C_F = 0.075/(log10 Re - 2)^2`` (ITTC 1957).  Holtrop-Mennen uses the same
  line, so a Holtrop comparison and a Michell prediction share their friction
  term *exactly* and can differ only in the wave term;
* Holtrop & Mennen (1982) with the Holtrop (1984) statistical re-analysis for
  the half angle of entrance, the form factor, the roughness allowance, the
  wave resistance and hence the total resistance.

Evidence level: **empirical regression, not a solved flow field.**  These
routines reproduce published algebraic formulae; they do not model a flow.
They are useful for a verification case exactly because they are independent of
the BEM kernel, but Holtrop's coefficients are regressions over a data set of
real (mostly fuller, 100-400 m) hullforms.  A mathematical hullform such as the
Wigley form -- ``iE = 6.3 deg``, ``vol/L^3 = 2.8e-3``, no bulb, no transom --
sits outside that data set, so :func:`holtrop_applicability` exists to be
reported *next to* any comparison, and a disagreement there is a property of the
baseline rather than evidence about the kernel.

Two published conventions matter at the 10-20 % level in ``R_W`` and both are
therefore selectable instead of silently hard-coded:

* ``include_c17`` -- the ``c17`` correction introduced in Holtrop (1984).
  ``c17 ~ 1.17`` for a 230 m container ship, so leaving it out changes ``R_W``
  by that factor;
* ``regime`` -- the Froude-dependent term of the exponential.  The widely
  reproduced Holtrop-Mennen (1982) form is
  ``m4 = c15 * 0.4 * exp(-0.034 Fn^-3.29)``; a legacy variant uses
  ``m2 = c15 * Cp^2 * exp(-0.1 Fn^-2)``.

The defaults are ``include_c17=True`` and ``regime="m4"`` (Holtrop 1984 as
cited).  :func:`holtrop_resistance` records the conventions it used, and callers
are expected to publish the variant grid when a comparison verdict depends on
it -- see ``docs/wigley_calm_water_resistance.md``.

References
----------
- ITTC (1957). *Report of the Resistance Committee*, Proc. 9th ITTC.
- Holtrop, J. & Mennen, G.G.J. (1982). "An approximate power prediction
  method." *International Shipbuilding Progress* 29(335), 166-170.
- Holtrop, J. (1984). "A statistical re-analysis of resistance and propulsion
  data." *International Shipbuilding Progress* 31(363), 272-276.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

#: Kinematic viscosity of sea water at 15 degC [m^2/s] (ITTC recommended
#: procedure 7.5-02-01-03, water density 1025 kg/m^3).
DEFAULT_KINEMATIC_VISCOSITY = 1.19e-6
DEFAULT_DENSITY = 1025.0
DEFAULT_GRAVITY = 9.80665
#: Standard model-ship roughness for the ITTC-78 correlation allowance [m].
DEFAULT_ROUGHNESS = 150.0e-6

#: Correlation-allowance conventions accepted by :func:`holtrop_resistance`.
CA_CONVENTIONS = ("none", "holtrop84", "ittc78")
#: Wave-resistance exponential-term conventions.
REGIME_TERMS = ("m4", "cp2")

__all__ = [
    "CA_CONVENTIONS",
    "DEFAULT_GRAVITY",
    "DEFAULT_KINEMATIC_VISCOSITY",
    "DEFAULT_DENSITY",
    "DEFAULT_ROUGHNESS",
    "REGIME_TERMS",
    "HoltropResistance",
    "holtrop_applicability",
    "holtrop_correlation_allowance",
    "holtrop_form_factor",
    "holtrop_half_angle_of_entrance",
    "holtrop_length_of_run",
    "holtrop_resistance",
    "holtrop_wave_resistance",
    "ittc57_friction_coefficient",
    "reynolds_number",
]


def reynolds_number(
    speed: float, length: float, viscosity: float = DEFAULT_KINEMATIC_VISCOSITY
) -> float:
    """``Re = U L / nu`` with all quantities in SI units."""
    if length <= 0.0:
        raise ValueError("length must be positive")
    if viscosity <= 0.0:
        raise ValueError("viscosity must be positive")
    return float(speed) * float(length) / float(viscosity)


def ittc57_friction_coefficient(reynolds: float) -> float:
    """ITTC-1957 friction coefficient ``C_F = 0.075 / (log10 Re - 2)^2``.

    The same line is used inside Holtrop-Mennen, which is what makes the
    friction term of the two routes identical.
    """
    re = float(reynolds)
    if not math.isfinite(re) or re <= 1.0e2:
        raise ValueError("Reynolds number must be finite and greater than 100")
    return 0.075 / (math.log10(re) - 2.0) ** 2


def holtrop_length_of_run(
    length: float, prismatic_coefficient: float, lcb_percent: float = 0.0
) -> float:
    """Length of run ``L_R = L (1 - Cp + 0.06 Cp lcb / (4 Cp - 1))``.

    ``lcb`` is in percent of ``L``, positive forward of ``0.5 L``.
    """
    length = float(length)
    cp = float(prismatic_coefficient)
    if length <= 0.0:
        raise ValueError("length must be positive")
    if not 0.0 < cp < 1.0:
        raise ValueError("prismatic coefficient must lie in (0, 1)")
    denominator = 4.0 * cp - 1.0
    if abs(denominator) < 1.0e-12:
        raise ValueError("Cp = 0.25 is outside the regression")
    return length * (1.0 - cp + 0.06 * cp * float(lcb_percent) / denominator)


def holtrop_half_angle_of_entrance(
    length: float,
    beam: float,
    draft: float,
    volume: float,
    prismatic_coefficient: float,
    waterplane_coefficient: float,
    lcb_percent: float = 0.0,
) -> float:
    """Half angle of entrance ``iE`` [deg] from the Holtrop (1984) regression.

    ``iE = 1 + 89 exp[-(L/B)^0.80856 (1-Cwp)^0.30484 (1-Cp-0.0225 lcb)^0.6367
    (L_R/B)^0.34574 (100 vol/L^3)^0.16302]``
    """
    length, beam, draft, volume = float(length), float(beam), float(draft), float(volume)
    cp, cwp, lcb = float(prismatic_coefficient), float(waterplane_coefficient), float(lcb_percent)
    if min(length, beam, draft, volume) <= 0.0:
        raise ValueError("length, beam, draft and volume must be positive")
    length_of_run = holtrop_length_of_run(length, cp, lcb)
    inner = (
        (length / beam) ** 0.80856
        * (1.0 - cwp) ** 0.30484
        * (1.0 - cp - 0.0225 * lcb) ** 0.6367
        * (length_of_run / beam) ** 0.34574
        * (100.0 * volume / length**3) ** 0.16302
    )
    return 1.0 + 89.0 * math.exp(-inner)


def holtrop_form_factor(
    length: float,
    beam: float,
    draft: float,
    volume: float,
    prismatic_coefficient: float,
    lcb_percent: float = 0.0,
    *,
    c_stern: float = 0.0,
    form: str = "1984",
) -> float:
    """Viscous form factor ``1 + k``.

    ``form="1984"`` (default, Holtrop 1984)::

        1 + k = 0.93 + 0.487118 c14 (B/L)^1.06806 (T/L)^0.46106
                      (L/L_R)^0.121563 (L^3/vol)^0.36486 (1-Cp)^-0.604247
        c14   = 1 + 0.011 C_stern

    ``form="1982"`` (Holtrop & Mennen 1982)::

        1 + k = c13 [0.93 + c12 (B/L)^0.92497 (0.95-Cp)^-0.521448
                          (1-Cp+0.0225 lcb)^0.6906]
        c12   = (T/L)^0.2228446                    (T/L > 0.05)
              = 48.20 (T/L-0.02)^2.078 + 0.479948  (T/L <= 0.05)
        c13   = 1 + 0.003 C_stern

    The two are not interchangeable: for the KCS hullform the 1982 expression
    returns ``1.014`` where the 1984 one returns ``1.166``, and hull practice
    quotes ``k ~ 0.14``.  Both are kept because published tables exist for each.
    """
    length, beam, draft, volume = float(length), float(beam), float(draft), float(volume)
    cp, lcb = float(prismatic_coefficient), float(lcb_percent)
    if min(length, beam, draft, volume) <= 0.0:
        raise ValueError("length, beam, draft and volume must be positive")
    if not 0.0 < cp < 1.0:
        raise ValueError("prismatic coefficient must lie in (0, 1)")
    if form not in ("1982", "1984"):
        raise ValueError("form must be '1982' or '1984'")
    t_over_l = draft / length
    if form == "1984":
        length_of_run = holtrop_length_of_run(length, cp, lcb)
        c14 = 1.0 + 0.011 * float(c_stern)
        return 0.93 + (
            0.487118
            * c14
            * (beam / length) ** 1.06806
            * t_over_l**0.46106
            * (length / length_of_run) ** 0.121563
            * (length**3 / volume) ** 0.36486
            * (1.0 - cp) ** -0.604247
        )
    if t_over_l > 0.05:
        c12 = t_over_l**0.2228446
    else:
        c12 = 48.20 * (t_over_l - 0.02) ** 2.078 + 0.479948
    c13 = 1.0 + 0.003 * float(c_stern)
    return c13 * (
        0.93
        + c12
        * (beam / length) ** 0.92497
        * (0.95 - cp) ** -0.521448
        * (1.0 - cp + 0.0225 * lcb) ** 0.6906
    )


def holtrop_correlation_allowance(
    length: float,
    block_coefficient: float,
    draft: float,
    *,
    bulb_coefficient: float = 1.0,
    convention: str = "none",
    roughness: float = DEFAULT_ROUGHNESS,
) -> float:
    """Correlation/roughness allowance ``C_A`` (dimensionless).

    Three declared options, because the two "Holtrop" formulations in
    circulation differ by more than a factor of two at model scale:

    ``convention="none"``
        ``C_A = 0``.  The default for this repository's model-scale
        mathematical hullforms: ``C_A`` is a full-scale roughness plus
        model-ship correlation term with no basis for a hull that has neither a
        prototype nor tank data.
    ``convention="holtrop84"``
        ``C_A = 0.006(L+100)^-0.16 - 0.00205
        + 0.003 sqrt(L/7.5) Cb^4 c2 sqrt(0.04-c4)``, ``c4 = min(T_F/L, 0.04)``.
    ``convention="ittc78"``
        ``C_A = [105 (k_s/L)^(1/3) - 0.64] 1e-3`` with ``k_s = 150e-6`` m, the
        ITTC-1978 roughness allowance Holtrop (1984) quotes for a new hull.
    """
    length = float(length)
    if length <= 0.0:
        raise ValueError("length must be positive")
    if convention not in CA_CONVENTIONS:
        raise ValueError(f"convention must be one of {CA_CONVENTIONS}")
    if convention == "none":
        return 0.0
    if convention == "holtrop84":
        c4 = min(float(draft) / length, 0.04)
        radicand = max(0.04 - c4, 0.0)
        return (
            0.006 * (length + 100.0) ** -0.16
            - 0.00205
            + 0.003
            * math.sqrt(length / 7.5)
            * float(block_coefficient) ** 4
            * float(bulb_coefficient)
            * math.sqrt(radicand)
        )
    if roughness <= 0.0:
        raise ValueError("roughness must be positive")
    return (105.0 * (float(roughness) / length) ** (1.0 / 3.0) - 0.64) * 1.0e-3


def holtrop_wave_resistance(
    length: float,
    beam: float,
    draft: float,
    volume: float,
    prismatic_coefficient: float,
    waterplane_coefficient: float,
    midship_coefficient: float,
    speed: float,
    *,
    lcb_percent: float = 0.0,
    bulb_area: float = 0.0,
    bulb_centre_height: float = 0.0,
    transom_area: float = 0.0,
    forward_draft: float | None = None,
    density: float = DEFAULT_DENSITY,
    gravity: float = DEFAULT_GRAVITY,
    include_c17: bool = True,
    regime: str = "m4",
) -> dict[str, float]:
    """Holtrop-Mennen wave resistance ``R_W`` [N] with every intermediate.

    ``R_W = c17 c1 c2 c5 rho g vol exp[m1 Fn^d + m4 cos(lam Fn^-2)]``

    with ``d = -0.9``, ``m4 = c15 * 0.4 * exp(-0.034 Fn^-3.29)`` for
    ``regime="m4"`` (Holtrop & Mennen 1982) or ``m4 = c15 Cp^2 exp(-0.1 Fn^-2)``
    for the legacy ``regime="cp2"``; ``c17 = 1`` when ``include_c17`` is false.
    ``c2`` (bulb) and ``c5`` (transom) are one for a bare mathematical hullform.
    """
    length, beam, draft, volume = float(length), float(beam), float(draft), float(volume)
    cp, cwp, cm = (
        float(prismatic_coefficient),
        float(waterplane_coefficient),
        float(midship_coefficient),
    )
    speed = float(speed)
    if min(length, beam, draft, volume) <= 0.0:
        raise ValueError("length, beam, draft and volume must be positive")
    if speed <= 0.0:
        raise ValueError("speed must be positive")
    if regime not in REGIME_TERMS:
        raise ValueError(f"regime must be one of {REGIME_TERMS}")
    lcb = float(lcb_percent)
    rho, g = float(density), float(gravity)
    t_forward = draft if forward_draft is None else float(forward_draft)

    fn = speed / math.sqrt(g * length)
    i_e = holtrop_half_angle_of_entrance(length, beam, draft, volume, cp, cwp, lcb)

    b_over_l = beam / length
    if b_over_l < 0.11:
        c7 = 0.229577 * b_over_l ** (1.0 / 3.0)
    elif b_over_l <= 0.25:
        c7 = b_over_l
    else:
        c7 = 0.5 - 0.0625 / b_over_l
    c1 = 2223105.0 * c7**3.78613 * (draft / beam) ** 1.07961 * (90.0 - i_e) ** -1.37565

    if float(bulb_area) > 0.0:
        denominator = beam * draft * (
            0.31 * math.sqrt(float(bulb_area)) + t_forward - float(bulb_centre_height)
        )
        if denominator <= 0.0:
            raise ValueError("bulbous-bow coefficients are inconsistent")
        c3 = 0.56 * float(bulb_area) ** 1.5 / denominator
    else:
        c3 = 0.0
    c2 = math.exp(-1.89 * math.sqrt(c3))
    c5 = 1.0 - 0.8 * float(transom_area) / (beam * draft * cm)

    if cp < 0.8:
        c16 = 8.07981 * cp - 13.8673 * cp**2 + 6.984388 * cp**3
    else:
        c16 = 1.73014 - 0.7067 * cp
    m1 = (
        0.0140407 * length / draft
        - 1.75254 * volume ** (1.0 / 3.0) / length
        - 4.79323 * b_over_l
        - c16
    )

    l_cubed_over_volume = length**3 / volume
    if l_cubed_over_volume < 512.0:
        c15 = -1.69385
    elif l_cubed_over_volume <= 1727.0:
        c15 = -1.69385 + (length / volume ** (1.0 / 3.0) - 8.0) / 2.36
    else:
        c15 = 0.0
    d = -0.9
    if regime == "m4":
        m4 = c15 * 0.4 * math.exp(-0.034 * fn**-3.29)
    else:
        m4 = c15 * cp**2 * math.exp(-0.1 * fn**-2.0)
    lam = 1.446 * cp - 0.03 * length / beam if length / beam < 12.0 else 1.446 * cp - 0.36
    if include_c17:
        c17 = (
            6919.3
            * cm**-1.3346
            * (volume / length**3) ** 2.00977
            * (length / beam - 2.0) ** 1.40692
        )
    else:
        c17 = 1.0
    exponent = m1 * fn**d + m4 * math.cos(lam * fn**-2.0)
    wave = c17 * c1 * c2 * c5 * rho * g * volume * math.exp(exponent)
    return {
        "Fn": fn,
        "iE_deg": i_e,
        "length_of_run": holtrop_length_of_run(length, cp, lcb),
        "c1": c1,
        "c2": c2,
        "c3": c3,
        "c5": c5,
        "c7": c7,
        "c15": c15,
        "c16": c16,
        "c17": c17,
        "m1": m1,
        "m4": m4,
        "d": d,
        "lam": lam,
        "exponent": exponent,
        "Rw": float(wave),
    }


def holtrop_applicability(
    length: float,
    beam: float,
    draft: float,
    volume: float,
    prismatic_coefficient: float,
    waterplane_coefficient: float,
    speed: float,
    *,
    lcb_percent: float = 0.0,
    gravity: float = DEFAULT_GRAVITY,
) -> dict[str, Any]:
    """Diagnostics: how far outside the regression's data set a hull lies.

    The thresholds are *reported, not applied*, so a caller can publish the
    flags next to a comparison and let the reader judge.  ``iE >= 5 deg``,
    ``vol/L^3 >= 1e-3`` and ``Fn <= 0.45`` are the ranges quoted by Holtrop &
    Mennen (1982); the absence of a bulb and of an immersed transom is recorded
    because ``c2``/``c5`` were fitted with them present in the data set.
    """
    length, beam, draft, volume = float(length), float(beam), float(draft), float(volume)
    cp, cwp = float(prismatic_coefficient), float(waterplane_coefficient)
    i_e = holtrop_half_angle_of_entrance(length, beam, draft, volume, cp, cwp, lcb_percent)
    fn = float(speed) / math.sqrt(float(gravity) * length)
    fine_volume = volume / length**3
    flags: dict[str, Any] = {
        "iE_deg": i_e,
        "iE_inside_fit_range": bool(i_e >= 5.0),
        "vol_over_L3": fine_volume,
        "vol_over_L3_inside_fit_range": bool(fine_volume >= 1.0e-3),
        "Fn": fn,
        "Fn_inside_fit_range": bool(fn <= 0.45),
        "lcb_percent": float(lcb_percent),
        "has_bulbous_bow": False,
        "has_immersed_transom": False,
    }
    flags["all_inside_fit_range"] = bool(
        flags["iE_inside_fit_range"]
        and flags["vol_over_L3_inside_fit_range"]
        and flags["Fn_inside_fit_range"]
    )
    return flags


@dataclass(frozen=True)
class HoltropResistance:
    """Holtrop (1984) calm-water resistance breakdown of one operating point."""

    speed: float
    Fn: float
    Re: float
    C_F: float
    form_factor: float
    C_A: float
    R_F: float
    R_A: float
    R_W: float
    R_T: float
    S: float
    C_T: float
    conventions: dict[str, Any] = field(default_factory=dict)
    wave_coefficients: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly view (wave coefficients nested under their own key)."""
        return {
            "speed": self.speed,
            "Fn": self.Fn,
            "Re": self.Re,
            "C_F": self.C_F,
            "form_factor": self.form_factor,
            "C_A": self.C_A,
            "R_F": self.R_F,
            "R_A": self.R_A,
            "R_W": self.R_W,
            "R_T": self.R_T,
            "wetted_area": self.S,
            "C_T": self.C_T,
            "conventions": dict(self.conventions),
            "holtrop_wave_coefficients": dict(self.wave_coefficients),
        }


def holtrop_resistance(
    length: float,
    beam: float,
    draft: float,
    volume: float,
    prismatic_coefficient: float,
    waterplane_coefficient: float,
    midship_coefficient: float,
    wetted_area: float,
    speed: float,
    *,
    block_coefficient: float | None = None,
    lcb_percent: float = 0.0,
    c_stern: float = 0.0,
    bulb_area: float = 0.0,
    bulb_centre_height: float = 0.0,
    transom_area: float = 0.0,
    forward_draft: float | None = None,
    density: float = DEFAULT_DENSITY,
    gravity: float = DEFAULT_GRAVITY,
    viscosity: float = DEFAULT_KINEMATIC_VISCOSITY,
    roughness: float = DEFAULT_ROUGHNESS,
    ca_convention: str = "none",
    include_c17: bool = True,
    regime: str = "m4",
    form: str = "1984",
) -> HoltropResistance:
    """Total calm-water resistance from the Holtrop (1984) regression.

    ``R_T = R_F (1 + k) + R_W + R_A``, with ``R_A = C_A 0.5 rho U^2 S`` and the
    appendage (``R_APP``) term zero for a bare hull.  ``C_A`` is zero unless a
    convention is asked for (:func:`holtrop_correlation_allowance`), because it
    is a full-scale correlation term.
    """
    speed = float(speed)
    if speed <= 0.0:
        raise ValueError("speed must be positive")
    area = float(wetted_area)
    if area <= 0.0:
        raise ValueError("wetted area must be positive")
    rho, g = float(density), float(gravity)
    cb = (
        volume / (float(length) * float(beam) * float(draft))
        if block_coefficient is None
        else float(block_coefficient)
    )
    fn = speed / math.sqrt(g * float(length))
    re = reynolds_number(speed, length, viscosity)
    c_f = ittc57_friction_coefficient(re)
    wave = holtrop_wave_resistance(
        length,
        beam,
        draft,
        volume,
        prismatic_coefficient,
        waterplane_coefficient,
        midship_coefficient,
        speed,
        lcb_percent=lcb_percent,
        bulb_area=bulb_area,
        bulb_centre_height=bulb_centre_height,
        transom_area=transom_area,
        forward_draft=forward_draft,
        density=rho,
        gravity=g,
        include_c17=include_c17,
        regime=regime,
    )
    form_factor = holtrop_form_factor(
        length,
        beam,
        draft,
        volume,
        prismatic_coefficient,
        lcb_percent,
        c_stern=c_stern,
        form=form,
    )
    c_a = holtrop_correlation_allowance(
        length,
        cb,
        draft if forward_draft is None else forward_draft,
        bulb_coefficient=wave["c2"],
        convention=ca_convention,
        roughness=roughness,
    )
    q = 0.5 * rho * speed**2 * area
    r_f = c_f * q
    r_a = c_a * q
    r_t = r_f * form_factor + r_a + wave["Rw"]
    return HoltropResistance(
        speed=speed,
        Fn=fn,
        Re=re,
        C_F=c_f,
        form_factor=form_factor,
        C_A=c_a,
        R_F=r_f,
        R_A=r_a,
        R_W=wave["Rw"],
        R_T=r_t,
        S=area,
        C_T=r_t / q,
        conventions={
            "form_factor": form,
            "include_c17": bool(include_c17),
            "regime": regime,
            "correlation_allowance": ca_convention,
            "viscosity": float(viscosity),
        },
        wave_coefficients=wave,
    )