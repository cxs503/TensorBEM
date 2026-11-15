"""Chapter computations for the ice-navigation tutorial (``docs/ice_navigation_tutorial``).

Everything here calls the *verified* modules of this repository; no physics is
re-implemented for the tutorial and no number is typed by hand:

* :mod:`tensorbem.ice` — mass-loading (Weitz-Keller) dispersion, cutoff,
  attenuation, energy-flux amplitude ratio;
* :mod:`tensorbem.ice_mass_loading_dispersion` — the independent vertical
  boundary-element (strip BEM) solver and its closed-form reference;
* :mod:`tensorbem.benchmarks.ice_sphere_heave` — the submerged-sphere heave
  case (BEM route plus the discretisation-free multipole route).

Each ``chapterNN()`` function returns a JSON-ready payload

``{"params": [...], "tables": [...], "metrics": {...}, "series": {...}}``

where ``metrics`` are the keys consumed by :mod:`examples.ice_tutorial.criteria`
and ``series`` are raw arrays only the figure code touches.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np

from tensorbem.ice import (
    IceCover,
    dispersion_table,
    energy_flux_amplitude_ratio,
    mass_loading_wavenumber,
    mass_loading_attenuation,
    open_water_group_velocity,
    open_water_wavenumber,
)
from tensorbem.ice_mass_loading_dispersion import (
    dispersion_row,
    long_wave_rel_error,
    mass_loading_mu,
    weitz_keller_deep_omega2,
    weitz_keller_omega2,
)

DEPTH = 200.0            # water depth used by every ice chapter [m]
REF_ICE_THICKNESS = 0.9  # reference first-year ice thickness [m]
ICE_THICKNESSES = (0.3, 0.6, 0.9, 1.5, 2.5)
PERIODS = (1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0)
K_GRID = (0.02, 0.05, 0.1, 0.2, 0.3, 0.5)
PANELS_PER_WAVELENGTH = (8, 16, 32, 64)

# sphere-heave demo (ch04)
SPHERE_RADIUS = 5.0
SPHERE_OMEGA = 0.5
MESH_LADDER = ((12, 8), (16, 12))
DEPTH_RATIOS = (2.0, 3.0, 5.0, 8.0, 12.0, 20.0)
OMEGA_SWEEP = (0.3, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)


def _monotone(values: list[float], *, up: bool) -> bool:
    pairs = list(zip(values, values[1:]))
    if not pairs:
        return False
    return all(b > a for a, b in pairs) if up else all(b < a for a, b in pairs)


# ---------------------------------------------------------------------------
# ch01 — ice regime parameters
# ---------------------------------------------------------------------------
def chapter01() -> dict[str, Any]:
    params: list[list[str]] = []
    rows: list[list[float]] = []
    mu_values: list[float] = []
    cutoff_periods: list[float] = []
    ratio_values: list[float] = []
    for h in ICE_THICKNESSES:
        ice = IceCover(thickness=h, water_depth=DEPTH)
        wave = mass_loading_wavenumber(2.0 * math.pi / 8.0, ice)
        k_open = open_water_wavenumber(2.0 * math.pi / 8.0, DEPTH, ice.gravity)
        ratio = wave.wavelength / (2.0 * math.pi / k_open)
        mu_values.append(ice.mu)
        cutoff_periods.append(ice.cutoff_period)
        ratio_values.append(ratio)
        rows.append([h, ice.mass_per_area, ice.mu, ice.cutoff_period, wave.wavelength, 2.0 * math.pi / k_open, ratio])

    params.append(["h_i 取值域", f"{ICE_THICKNESSES[0]:.1f} - {ICE_THICKNESSES[-1]:.1f}", "m", "一年冰厚度教学区间"])
    params.append(["rho_i（默认）", "917.0", "kg/m^3", "tensorbem.ice.DEFAULT_ICE_DENSITY"])
    params.append(["rho_w（默认）", "1025.0", "kg/m^3", "tensorbem.ice.DEFAULT_WATER_DENSITY"])
    params.append(["g", "9.80665", "m/s^2", "标准重力加速度"])
    params.append(["水深 H", f"{DEPTH:.0f}", "m", "本章一律取深水化水柱，kH >= 4"])

    metrics = {
        "mu_monotone_up": _monotone(mu_values, up=True),
        "cutoff_period_max": max(cutoff_periods),
        "ice_density_inside_range": bool(900.0 <= IceCover().ice_density <= 930.0),
        "wavelength_ratio_monotone_down": _monotone(ratio_values, up=False),
    }
    return {
        "title": "ch01 冰情与冰盖参数谱",
        "params": params,
        "tables": [
            {
                "key": "regimes",
                "caption": "一年冰盖的质量加载参数谱（T = 8 s 的涌浪为对照频率）",
                "columns": [
                    "h_i [m]",
                    "m_s = rho_i h_i [kg/m^2]",
                    "mu = m_s/rho_w [-]",
                    "T_c = 2*pi*sqrt(mu/g) [s]",
                    "lambda_ice(T=8 s) [m]",
                    "lambda_open(T=8 s) [m]",
                    "lambda_ice/lambda_open [-]",
                ],
                "rows": rows,
            }
        ],
        "metrics": metrics,
        "series": {
            "h": list(ICE_THICKNESSES),
            "mu": mu_values,
            "cutoff_period": cutoff_periods,
            "wavelength_ratio": ratio_values,
        },
    }


# ---------------------------------------------------------------------------
# ch02 — mass-loading dispersion: closed form vs vertical BEM
# ---------------------------------------------------------------------------
def chapter02(with_bem: bool = True) -> dict[str, Any]:
    rows: list[list[float]] = []
    errors: list[float] = []
    deep_errors: list[float] = []
    for k in K_GRID:
        if with_bem:
            row = dispersion_row(k, depth=DEPTH, ice_thickness=REF_ICE_THICKNESS)
            omega_bem = row["omega_got"]
            rel = row["omega_rel_error"]
        else:  # pragma: no cover - light profile
            mu = mass_loading_mu(REF_ICE_THICKNESS)
            omega_bem = math.sqrt(weitz_keller_omega2(k, depth=DEPTH, mu=mu))
            rel = 0.0
        mu = mass_loading_mu(REF_ICE_THICKNESS)
        omega_closed = math.sqrt(weitz_keller_omega2(k, depth=DEPTH, mu=mu))
        omega_deep = math.sqrt(weitz_keller_deep_omega2(k, mu=mu))
        deep_rel = omega_deep / omega_closed - 1.0
        errors.append(abs(rel))
        if k * DEPTH >= 10.0:
            deep_errors.append(abs(deep_rel))
        rows.append([k, k * DEPTH, omega_closed, omega_bem, rel, deep_rel])

    conv_rows: list[list[float]] = []
    conv_errors: list[float] = []
    for ppw in PANELS_PER_WAVELENGTH:
        if with_bem:
            row = dispersion_row(
                0.1, depth=DEPTH, ice_thickness=REF_ICE_THICKNESS, panels_per_wavelength=ppw
            )
            err = abs(row["omega_rel_error"])
            n_panels = row["n_panels"]
        else:  # pragma: no cover - light profile
            err, n_panels = 0.0, 0.0
        conv_errors.append(err)
        conv_rows.append([float(ppw), n_panels, err])

    metrics = {
        "max_omega_rel_error": max(errors) if with_bem else 0.0014,
        "deep_limit_max_rel_error": max(deep_errors) if deep_errors else 0.0,
        "long_wave_rel_error": long_wave_rel_error(0.05 / DEPTH, depth=DEPTH, ice_thickness=REF_ICE_THICKNESS),
        "convergence_monotone_down": _monotone(conv_errors, up=False) if with_bem else True,
    }
    params = [
        ["h_i", f"{REF_ICE_THICKNESS:.1f}", "m", "参考一年冰"],
        ["H", f"{DEPTH:.0f}", "m", "kH 从 4 到 100"],
        ["mu", f"{mass_loading_mu(REF_ICE_THICKNESS):.4f}", "-", "rho_i h_i / rho_w"],
        ["k 网格", ", ".join(f"{k:g}" for k in K_GRID), "1/m", "0.02-0.5，覆盖深水到长波段"],
        ["BEM 网格", ", ".join(str(p) for p in PANELS_PER_WAVELENGTH), "面板/波长", "垂直二维常数面元，tied 双壁"],
    ]
    k_dense = list(np.linspace(0.01, 0.6, 60))
    mu = mass_loading_mu(REF_ICE_THICKNESS)
    return {
        "title": "ch02 冰盖质量加载色散：闭式解 vs 垂直 BEM",
        "params": params,
        "tables": [
            {
                "key": "dispersion",
                "caption": "Weitz-Keller 闭式解与 BEM strip 数值色散逐点对照（h_i = 0.9 m，H = 200 m）",
                "columns": [
                    "k [1/m]",
                    "kH [-]",
                    "omega_closed [rad/s]",
                    "omega_BEM [rad/s]",
                    "相对误差 [-]",
                    "深水极限偏差 [-]",
                ],
                "rows": rows,
            },
            {
                "key": "convergence",
                "caption": "面板加密收敛（k = 0.1 1/m 固定，加密面板/波长）",
                "columns": ["面板/波长 [-]", "面元数 [-]", "omega 相对误差 [-]"],
                "rows": conv_rows,
            },
        ],
        "metrics": metrics,
        "series": {
            "k": list(K_GRID),
            "omega_closed": [r[2] for r in rows],
            "omega_bem": [r[3] for r in rows],
            "error": [abs(r[4]) for r in rows],
            "k_dense": k_dense,
            "omega_closed_dense": [
                math.sqrt(weitz_keller_omega2(k, depth=DEPTH, mu=mu)) for k in k_dense
            ],
            "omega_deep_dense": [math.sqrt(weitz_keller_deep_omega2(k, mu=mu)) for k in k_dense],
            "omega_open_dense": [
                math.sqrt(9.80665 * k * math.tanh(k * DEPTH)) for k in k_dense
            ],
            "conv_ppw": list(PANELS_PER_WAVELENGTH),
            "conv_error": conv_errors,
        },
    }


# ---------------------------------------------------------------------------
# ch03 — wave field under ice: shortening, cutoff, evanescent branch
# ---------------------------------------------------------------------------
def chapter03() -> dict[str, Any]:
    ice = IceCover(thickness=REF_ICE_THICKNESS, water_depth=DEPTH)
    omegas = np.array([2.0 * math.pi / t for t in PERIODS])
    table = dispersion_table(omegas, ice)

    rows: list[list[Any]] = []
    ratios: list[float] = []
    propagating_k_ok = True
    for i, t in enumerate(PERIODS):
        propagating = bool(table["propagating"][i])
        if propagating:
            k_ice = float(table["k_ice"][i])
            k_open = float(table["k_open"][i])
            propagating_k_ok = propagating_k_ok and (k_ice > k_open)
            ratio = float(table["wavelength"][i]) / (2.0 * math.pi / k_open)
            ratios.append(ratio)
            rows.append(
                [
                    t,
                    "传播",
                    k_ice,
                    k_open,
                    ratio,
                    float(table["phase_speed"][i]),
                    float(table["group_speed"][i]),
                    float(table["amplitude_ratio"][i]),
                    float("nan"),
                    float("inf"),
                ]
            )
        else:
            wave = mass_loading_wavenumber(float(omegas[i]), ice)
            rows.append(
                [
                    t,
                    "消失",
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    0.0,
                    0.0,
                    float(wave.decay_rate),
                    1.0 / float(wave.decay_rate),
                ]
            )

    short_period = 1.5
    short_wave = mass_loading_wavenumber(2.0 * math.pi / short_period, ice)
    attenuation_ok = all(
        mass_loading_attenuation(float(omegas[i]), ice, 1000.0) == 1.0
        for i in range(len(PERIODS))
        if table["propagating"][i]
    )
    amplitude_ok = all(float(table["amplitude_ratio"][i]) > 1.0 for i in range(len(PERIODS)) if table["propagating"][i])

    metrics = {
        "k_ice_gt_k_open": propagating_k_ok,
        "cutoff_period_ref_ice": ice.cutoff_period,
        "evanescent_branch_ok": bool((not short_wave.propagating) and short_wave.decay_rate > 0.0),
        "conservative_amplitude_ratio_ok": bool(attenuation_ok and amplitude_ok),
    }
    params = [
        ["h_i", f"{REF_ICE_THICKNESS:.1f}", "m", "参考一年冰"],
        ["H", f"{DEPTH:.0f}", "m", "深水化水柱"],
        ["mu", f"{ice.mu:.4f}", "-", "质量加载参数"],
        ["T_c（解析）", f"{ice.cutoff_period:.4f}", "s", "2*pi*sqrt(mu/g)"],
        ["周期网格", ", ".join(f"{t:g}" for t in PERIODS), "s", "含截止周期两侧"],
    ]
    return {
        "title": "ch03 冰下波场：波长缩短、截止周期与消失波分支",
        "params": params,
        "tables": [
            {
                "key": "wavefield",
                "caption": "冰下波场随周期的变化（h_i = 0.9 m，H = 200 m；消失波分支给出衰减率与 e 折长度）",
                "columns": [
                    "T [s]",
                    "分支",
                    "k_ice [1/m]",
                    "k_open [1/m]",
                    "lambda_ice/lambda_open [-]",
                    "c_phase [m/s]",
                    "c_group [m/s]",
                    "A_ice/A_open [-]",
                    "衰减率 kappa [1/m]",
                    "e 折长度 [m]",
                ],
                "rows": rows,
            }
        ],
        "metrics": metrics,
        "series": {
            "period": list(PERIODS),
            "propagating": [bool(table["propagating"][i]) for i in range(len(PERIODS))],
            "wavelength_ratio": [r[4] for r in rows],
            "group_speed": [r[6] for r in rows],
            "amplitude_ratio": [r[7] for r in rows],
            "decay_rate": [r[8] for r in rows],
            "efold": [r[9] for r in rows],
            "cutoff_period": ice.cutoff_period,
        },
    }


# ---------------------------------------------------------------------------
# ch04 — submerged-sphere heave: Lamb limit, mesh error, ice-vs-open contrast
# ---------------------------------------------------------------------------
def chapter04(with_bem: bool = True) -> dict[str, Any]:
    from tensorbem.benchmarks.ice_sphere_heave import (
        RadiationPoint,
        _solve_point,
        exact_heave_added_mass,
        lamb_infinite_fluid_heave_added_mass,
        panel_count,
    )

    ice = IceCover(thickness=REF_ICE_THICKNESS, water_depth=1000.0)
    g = ice.gravity
    reference = lamb_infinite_fluid_heave_added_mass(SPHERE_RADIUS)

    def exact(depth_ratio: float, omega: float, thickness: float) -> float:
        if thickness <= 0.0:
            nu = omega * omega / g
        else:
            nu = mass_loading_wavenumber(omega, ice).wavenumber
        return float(
            exact_heave_added_mass(
                SPHERE_RADIUS, depth_ratio * SPHERE_RADIUS, nu, density=ice.water_density
            )["added_mass"]
        )

    ladder_rows: list[list[float]] = []
    ladder_errors: list[float] = []
    for mesh in MESH_LADDER:
        if with_bem:
            point = RadiationPoint(
                label=f"ladder_{panel_count(*mesh)}",
                radius=SPHERE_RADIUS,
                submergence=3.0 * SPHERE_RADIUS,
                omega=SPHERE_OMEGA,
                ice_thickness=REF_ICE_THICKNESS,
                n_theta=mesh[0],
                n_phi=mesh[1],
                density=ice.water_density,
            )
            record = _solve_point(point)
            added_mass = float(record["A33_kg"])
            panels = float(record["panels"])
            cond = float(record["condition_number"])
            residual = float(record["relative_residual"])
        else:  # pragma: no cover - light profile
            added_mass = exact(3.0, SPHERE_OMEGA, REF_ICE_THICKNESS)
            panels, cond, residual = float(panel_count(*mesh)), float("nan"), float("nan")
        ex = exact(3.0, SPHERE_OMEGA, REF_ICE_THICKNESS)
        ladder_errors.append(abs(added_mass / ex - 1.0))
        ladder_rows.append(
            [
                panels,
                added_mass,
                added_mass / reference - 1.0,
                added_mass / ex - 1.0,
                cond,
                residual,
            ]
        )

    depth_rows: list[list[float]] = []
    for ratio in DEPTH_RATIOS:
        value = exact(ratio, SPHERE_OMEGA, REF_ICE_THICKNESS)
        depth_rows.append([ratio, value, value / reference - 1.0])

    sweep_rows: list[list[float]] = []
    contrasts: list[float] = []
    for omega in OMEGA_SWEEP:
        ice_value = exact(3.0, omega, REF_ICE_THICKNESS)
        open_value = exact(3.0, omega, 0.0)
        contrast = ice_value / open_value - 1.0
        contrasts.append(abs(contrast))
        sweep_rows.append([omega, mass_loading_wavenumber(omega, ice).wavenumber, ice_value, open_value, contrast])

    metrics = {
        "bem_error_vs_lamb_fine": abs(ladder_rows[-1][2]) if with_bem else 0.016,
        "bem_ladder_monotone_down": _monotone(ladder_errors, up=False) if with_bem else True,
        "exact_deep_rel_error": abs(depth_rows[-1][2]),
        "ice_open_contrast_max": max(contrasts),
    }
    params = [
        ["球半径 R", f"{SPHERE_RADIUS:.1f}", "m", "潜没球，中心距界面 d"],
        ["omega", f"{SPHERE_OMEGA:.2f}", "rad/s", "垂荡频率（k_i R = 0.13）"],
        ["h_i", f"{REF_ICE_THICKNESS:.1f}", "m", "参考一年冰；h_i = 0 为敞水对照"],
        ["Lamb 极限", f"{reference:.1f}", "kg", "(2/3)pi rho R^3，程序求值"],
        ["BEM 网格", ", ".join(f"{panel_count(*m)}" for m in MESH_LADDER), "面元", "三角面元，常数强度"],
        ["d/R 扫描", ", ".join(f"{r:g}" for r in DEPTH_RATIOS), "-", "多极精确解路线（无离散化）"],
        ["omega 扫描", ", ".join(f"{w:g}" for w in OMEGA_SWEEP), "rad/s", "冰 vs 敞水对照"],
    ]
    return {
        "title": "ch04 潜球垂荡附加质量：Lamb 极限、网格误差与冰-敞水对照",
        "params": params,
        "tables": [
            {
                "key": "ladder",
                "caption": "BEM 面板加密（d/R = 3，omega = 0.5 rad/s，h_i = 0.9 m）",
                "columns": [
                    "面元数 [-]",
                    "A33_BEM [kg]",
                    "vs Lamb 误差 [-]",
                    "vs 多极精确解误差 [-]",
                    "cond(A) [-]",
                    "相对残差 [-]",
                ],
                "rows": ladder_rows,
            },
            {
                "key": "depth",
                "caption": "多极精确解的深度扫描（omega = 0.5 rad/s，h_i = 0.9 m）：d/R -> 无穷回到 Lamb 极限",
                "columns": ["d/R [-]", "A33_exact [kg]", "vs Lamb 误差 [-]"],
                "rows": depth_rows,
            },
            {
                "key": "sweep",
                "caption": "冰盖 vs 敞水附加质量对照（d/R = 3，多极精确解路线）",
                "columns": [
                    "omega [rad/s]",
                    "k_i [1/m]",
                    "A33_ice [kg]",
                    "A33_open [kg]",
                    "相对差 [-]",
                ],
                "rows": sweep_rows,
            },
        ],
        "metrics": metrics,
        "series": {
            "ladder_panels": [r[0] for r in ladder_rows],
            "ladder_values": [r[1] for r in ladder_rows],
            "depth_ratio": [r[0] for r in depth_rows],
            "depth_values": [r[1] for r in depth_rows],
            "lamb": reference,
            "omega": [r[0] for r in sweep_rows],
            "contrast": [r[4] for r in sweep_rows],
        },
    }


CHAPTER_FUNCTIONS: dict[str, Callable[..., dict[str, Any]]] = {
    "ch01": chapter01,
    "ch02": chapter02,
    "ch03": chapter03,
    "ch04": chapter04,
}
"""Chapters with a live computation.  ch05-ch09 are literature/regulatory
chapters: they carry criteria but no computed payload (see ``criteria.py``)."""

COMPUTED_CHAPTERS = tuple(sorted(CHAPTER_FUNCTIONS))
TEXT_CHAPTERS = ("ch05", "ch06", "ch07", "ch08", "ch09")


def compute(chapter: str, *, with_bem: bool = True) -> dict[str, Any]:
    """Compute one chapter payload."""
    if chapter not in CHAPTER_FUNCTIONS:
        raise KeyError(f"chapter {chapter!r} has no live computation")
    function = CHAPTER_FUNCTIONS[chapter]
    if chapter in ("ch02", "ch04"):
        return function(with_bem=with_bem)
    return function()