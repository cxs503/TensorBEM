"""算例 CASE-BEM-CUMMINS-WIGLEY: Wigley 船体 6 自由度 Cummins 记忆核闭合.

物理背景
--------
浮体在频域内的辐射问题给出每个遭遇频率下的附加质量 ``A(ω)`` 与辐射阻尼
``B(ω)``. Cummins (1962) 把这对频域系数写成时域卷积的记忆核

    K(t) = (2/pi) integral_0^inf B(omega) cos(omega t) domega,

而因果性要求这两个分量不是彼此独立的: ``K(t)`` 是实的、因果的、衰减的,
于是 ``A(omega) - A_inf`` 必须是 ``B(omega)`` 的 Kramers-Kronig (Hilbert) 变换

    A(omega) - A_inf = (2/pi) PV integral_0^inf B(s) / (s^2 - omega^2) ds   (KK)

(Kotik & Mangulis 1962; Newman 1977 *Marine Hydrodynamics* ch.6; Ogilvie 1964
时域/频域等价性). 本算例把这三件事串成一条可机械裁判的链:

``C1`` 几何      Wigley 抛物线船型 ``Cb = 4/9``、``L/B = 10``、``B/T = 1.6``
                与网格排水体积一致性 (先验证几何再看水动力).
``C2`` 离散      面板分辨率闸门: 带顶频率的 ``k*sqrt(A_panel,max) <= 1``,
                否则该频段的面板解不具解释力.
``C3`` 因果      **算例判据**: 逐自由度
                ``|A_BEM - A_KK| / max|A_BEM| <= 10%``
                (tensor_case_pool id=145 "Wigley 船 Cummins 记忆核(6DOF)" 容差 <10%).
``C4`` 往返      ``B(omega) -> K(t) -> B(omega)`` 逐点相对误差 <= 2%.
``C5`` 自由衰减  时域 Cummins 卷积的自由衰减阻尼比与时域记录本身的
                ``B/(2(M+A)omega_n)`` 对比 <= 5% (同一份数据的另一条路线).
``C6`` 负控      (a) ``B == 0`` 必须给出零记忆核、零衰减; (b) 人为破坏 ``B``
                的高频段形状 (乘 1.3) 后 ``C3`` 必须判 FAIL -- 证明闸门咬得住.

方法要点 (相对朴素 KK 积分的三处真实改进)
-------------------------------------------
1. **不规则频率抑制**: 表面穿透船体的外场 BIE 在离散不规则频率处奇异,
   ``solve_radiation_3d(chief_points="auto")`` 用 CHIEF 内点行最小二乘抑制
   (John 1950; Hulme 1982). 不做这一步, 扫频里出现假的 A/B 尖峰, KK 残差被
   这些尖峰而不是物理支配.
2. **带限闭合**: ``kk_added_mass_closure`` (:mod:`tensorbem.causality`) 把
   ``s > s_max`` 的阻尼用幂律尾 ``B ~ C s^-p`` 拟合并以闭式主值积分补上
   (``p`` 由拟合给出; ``p <= 3`` 或拟合残差过大判为无效并退回零尾). 朴素做法
   把尾带当 0, 对峰值落在带内的表面穿透船体会给整条曲线带来偏置.
3. **``A_inf`` 只能由闭合定**: 截断扫频无法可靠外推 ``A_inf`` (它是 KK 关系里
   的一个自由常数), 因此判据用最小二乘闭合值 (一个自由参数, 不再多), 残差是
   **形状**残差; 另外把 ``A_inf + c/omega^2`` 尾拟合值作为独立诊断一并报告.

诚实边界
--------
* 参考值是**另一条路线** (BEM 直接解) 而不是实验/解析表: Wigley 船体辐射系数
  没有闭式解, 离线工作树里也没有可引用的实验表, 因此 L5 外部锚点记为
  WITHHELD; 本算例给出的是路线一致性证据 (direct BEM vs KK 重构 vs 时域).
* 由 ``A`` 反演 ``B`` 的那一支 (``kkr_damping``) 对截断极其敏感 (核
  ``s^2/(s^2-omega^2)`` 在大 s 不衰减), 只作诊断量报告, 不作判据.

入口
----
``python -m tensorbem.benchmarks.case_bem_cummins_wigley --out <dir> [--jobs N]``
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import json
import math
import platform
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from tensorbem import time_domain as td
from tensorbem.causality import (
    fit_added_mass_asymptote,
    fit_damping_power_tail,
    kk_added_mass_closure,
)
from tensorbem.free_surface_solver3d import solve_radiation_3d
from tensorbem.hydrostatic import (
    compute_displaced_volume_and_buoyancy,
    compute_waterplane_properties,
)
from tensorbem.wigley_hydrostatics import wigley_wetted_hull_mesh

__all__ = [
    "CASE_SLUG",
    "DOF_NAMES",
    "MESHES",
    "TOLERANCES",
    "Sweep",
    "solve_sweep",
    "geometry_gate",
    "panel_adequacy_gate",
    "kk_gate",
    "round_trip_gate",
    "free_decay_gate",
    "negative_control_gate",
    "sweep_grid",
    "grid_convergence_study",
    "run_case",
    "save_sweep",
    "load_sweep",
    "case_verdict",
    "gates_from_artifact",
    "render_markdown",
    "main",
]

CASE_SLUG = "CASE-BEM-CUMMINS-WIGLEY"

# --- mathematical Wigley hull (the standard Cummins/Wigley test body) -------
LENGTH = 3.0        # [m]
BEAM = 0.3          # [m]
DRAFT = 0.1875      # [m]
DENSITY = 1025.0    # [kg/m^3]
GRAVITY = 9.80665   # [m/s^2]
CB_ANALYTIC = 4.0 / 9.0
VOLUME_ANALYTIC = CB_ANALYTIC * LENGTH * BEAM * DRAFT   # (4/9)LBT = 0.075 m^3

DOF_NAMES = ("surge", "sway", "heave", "roll", "pitch", "yaw")

#: Production band and node count, pre-declared.
#:
#: Two independent constraints fix the band (m18249):
#:
#: * lower end -- below ``omega = 0.2 rad/s`` (``omega*sqrt(L/g) = 0.36``) the
#:   radiation damping of the horizontal modes is at the level of the BEM
#:   discretisation noise, so nothing is gained by going lower;
#: * upper end -- gate C5 compares the time-domain decay rate with the
#:   frequency-domain value ``B(omega_n)/(2 (m + A(omega_n)) omega_n)``.  For
#:   this 3 m model the *highest* rigid-body natural frequency (heave) sits near
#:   8.1 rad/s, so a band that stops at 8 rad/s cannot decide C5 at all: the
#:   previous artifact recorded ``zeta_frequency_domain = nan`` and scored a
#:   FAIL for a reason that was pure band coverage.  ``BAND[1] = 12 rad/s``
#:   covers every natural frequency of the model.
#:
#: The upper end is still bounded by discretisation: gate C2 checks
#: ``k*sqrt(A_panel) <= 1`` at ``BAND[1]``, i.e. at 12 rad/s with
#: ``k = omega**2/g = 14.7 1/m`` the medium mesh must keep its largest panel
#: below 6.8 cm.
#:
#: ``N_OMEGA`` is not a free knob either: gate C3 closes the Kramers-Kronig
#: integral with the *piecewise-linear interpolant* of the tabulated damping, so
#: the tabulated ``B(omega)`` has to resolve the resonance peaks of the hull.  The
#: previous artifact ran 13 nodes over 0.15..8 rad/s (1.7 decades, ~7.5 nodes per
#: decade) and its closure residual (3.5..16 %) was dominated by that
#: interpolation error, not by a causality violation -- see
#: :func:`grid_convergence_study`, which reports the residual as a function of
#: ``N_OMEGA`` so the claim is checkable instead of asserted.
MESHES: dict[str, tuple[int, int]] = {"coarse": (12, 4), "medium": (20, 6)}
BAND = (0.2, 12.0)
N_OMEGA = 41

TOLERANCES = {
    "kk": 0.10,           # case pool id=145, tol_policy <10%
    "round_trip": 0.02,   # Cummins transform self-consistency
    "free_decay": 0.05,   # time domain vs its own frequency-domain estimate
    "geometry": 0.01,     # Cb / L/B / B/T contract
    "volume": 0.02,       # meshed displaced volume vs (4/9)LBT
    "panel_kh": 1.0,      # k*sqrt(max panel area) at the top of the band
    "negative_control": 0.10,
}

DT = 0.01
DURATION = 40.0
TAU_MAX = 25.0


# ---------------------------------------------------------------------------
# radiation sweep
# ---------------------------------------------------------------------------


@dataclass
class Sweep:
    """Frequency-domain radiation sweep for the six rigid-body modes."""

    label: str
    n_sections: int
    n_levels: int
    panel_count: int
    max_panel_scale: float
    omega: np.ndarray
    added_mass: np.ndarray      # (n_omega, 6, 6)
    damping: np.ndarray         # (n_omega, 6, 6)
    solver_relative_residual: np.ndarray
    elapsed: float
    failed_frequencies: list[float] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "n_sections": self.n_sections,
            "n_levels": self.n_levels,
            "panel_count": self.panel_count,
            "max_panel_scale": self.max_panel_scale,
            "n_omega": int(self.omega.size),
            "omega": self.omega.tolist(),
            "elapsed_s": self.elapsed,
            "solver_relative_residual_max": float(
                np.max(self.solver_relative_residual)
            ),
            "failed_frequencies": self.failed_frequencies,
        }


#: Mesh identity fields that a sweep artifact must carry, so gate evaluation can
#: be replayed from the committed evidence instead of re-solving the BEM system.
SWEEP_METADATA = ("label", "n_sections", "n_levels", "panel_count", "max_panel_scale")


def save_sweep(sweep: Sweep, path: str | Path) -> Path:
    """Write a solved sweep to ``.npz``: coefficients *and* the mesh identity.

    The identity is part of the artifact on purpose.  Gate C2 reads
    ``max_panel_scale``, and C3's residual is only interpretable next to
    ``panel_count``; an artifact that stores only the coefficients cannot be
    re-judged later (see :func:`gates_from_artifact`).
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        target,
        omega=np.asarray(sweep.omega, dtype=float),
        added_mass=np.asarray(sweep.added_mass, dtype=float),
        damping=np.asarray(sweep.damping, dtype=float),
        solver_relative_residual=np.asarray(sweep.solver_relative_residual, dtype=float),
        label=np.array(str(sweep.label)),
        n_sections=np.array(int(sweep.n_sections)),
        n_levels=np.array(int(sweep.n_levels)),
        panel_count=np.array(int(sweep.panel_count)),
        max_panel_scale=np.array(float(sweep.max_panel_scale)),
        elapsed=np.array(float(sweep.elapsed)),
    )
    return target


def load_sweep(
    path: str | Path,
    *,
    label: str | None = None,
    n_sections: int | None = None,
    n_levels: int | None = None,
    panel_count: int | None = None,
    max_panel_scale: float | None = None,
) -> Sweep:
    """Rebuild a :class:`Sweep` from an artifact written by :func:`save_sweep`.

    Older artifacts stored the coefficients only.  Those can still be replayed,
    but every missing identity field has to be supplied by the caller: silently
    defaulting ``max_panel_scale`` would make gate C2 pass or fail on an
    invented mesh.
    """
    payload = np.load(Path(path), allow_pickle=False)

    def _metadata(name: str, fallback):
        if name in payload.files:
            return payload[name].item()
        if fallback is None:
            raise KeyError(
                f"sweep artifact {path} has no {name!r} field; pass it explicitly "
                "(it is needed to interpret the gates)"
            )
        return fallback

    sweep = Sweep(
        label=str(_metadata("label", label)),
        n_sections=int(_metadata("n_sections", n_sections)),
        n_levels=int(_metadata("n_levels", n_levels)),
        panel_count=int(_metadata("panel_count", panel_count)),
        max_panel_scale=float(_metadata("max_panel_scale", max_panel_scale)),
        omega=np.asarray(payload["omega"], dtype=float),
        added_mass=np.asarray(payload["added_mass"], dtype=float),
        damping=np.asarray(payload["damping"], dtype=float),
        solver_relative_residual=np.asarray(
            payload["solver_relative_residual"], dtype=float
        ),
        elapsed=float(payload["elapsed"].item()) if "elapsed" in payload.files else 0.0,
    )
    if sweep.added_mass.shape != (sweep.omega.size, 6, 6) or sweep.damping.shape != (
        sweep.omega.size,
        6,
        6,
    ):
        raise ValueError(
            f"sweep artifact {path} has coefficient shapes "
            f"{sweep.added_mass.shape}/{sweep.damping.shape}, expected "
            f"({sweep.omega.size}, 6, 6)"
        )
    return sweep


def sweep_grid(band: tuple[float, float] = BAND, n_omega: int = N_OMEGA) -> np.ndarray:
    """Log-spaced frequency nodes with both band ends exactly included."""
    low, high = (float(band[0]), float(band[1]))
    if not (0.0 < low < high):
        raise ValueError(f"band must satisfy 0 < low < high, got {band!r}")
    count = int(n_omega)
    if count < 3:
        raise ValueError("n_omega must be at least 3")
    grid = np.logspace(math.log10(low), math.log10(high), count)
    grid[0], grid[-1] = low, high
    return grid


def _refined_grid(
    grid: np.ndarray,
    added_mass: np.ndarray,
    damping: np.ndarray,
    *,
    n_refine: int = 0,
) -> np.ndarray:
    """Geometric midpoints of the intervals where the damping varies fastest.

    The KK closure integrates ``B`` with a piecewise-linear interpolant; the
    residual therefore concentrates on the intervals that straddle a resonance
    peak.  Ranking the intervals by the summed relative jump of ``B`` and
    inserting one midpoint each is a cheap, deterministic refinement and it keeps
    the adaptation tied to tabulated data only (nothing is extrapolated).
    """
    if n_refine <= 0 or grid.size < 3:
        return grid
    diag = np.zeros(grid.size - 1, dtype=float)
    for dof in range(min(6, damping.shape[1])):
        column = damping[:, dof, dof]
        scale = max(float(np.max(np.abs(column))), 1e-300)
        diag += np.abs(np.diff(column)) / scale
    order = np.argsort(diag)[::-1][: int(n_refine)]
    extra = np.sqrt(grid[order] * grid[order + 1])
    return np.unique(np.concatenate([grid, extra]))


def _solve_one(task: tuple[int, int, float]) -> tuple[float, Any, Any, float, int, float]:
    """Solve one frequency (module level so it is picklable)."""
    n_sections, n_levels, omega = task
    mesh = wigley_wetted_hull_mesh(
        LENGTH, BEAM, DRAFT, n_sections=n_sections, n_levels=n_levels
    )
    rad = solve_radiation_3d(
        mesh,
        omega**2 / GRAVITY,
        modes=tuple(range(6)),
        density=DENSITY,
        gravity=GRAVITY,
        chief_points="auto",
    )
    areas = np.asarray(mesh.areas, dtype=float)
    return (
        float(omega),
        np.asarray(rad.added_mass, dtype=float),
        np.asarray(rad.damping, dtype=float),
        float(getattr(rad, "relative_residual", 0.0)),
        int(mesh.panel_count),
        float(math.sqrt(float(np.max(areas)))),
    )


def solve_sweep(
    label: str,
    *,
    band: tuple[float, float] = BAND,
    n_omega: int = N_OMEGA,
    jobs: int = 1,
    omega_grid: Sequence[float] | None = None,
    refine_peaks: int = 0,
) -> Sweep:
    """Radiation sweep on the named mesh, CHIEF-regularised, optionally parallel.

    ``omega_grid`` overrides the log-spaced grid (used by the grid-convergence
    study).  ``refine_peaks`` adds one geometric midpoint per interval after the
    first pass, taken from the intervals where the damping jumps most (see
    :func:`_refined_grid`).
    """
    if label not in MESHES:
        raise ValueError(f"unknown mesh label {label!r}; expected one of {sorted(MESHES)}")
    n_sections, n_levels = MESHES[label]
    omega = (
        np.asarray(sorted(float(w) for w in omega_grid), dtype=float)
        if omega_grid is not None
        else sweep_grid(band, n_omega)
    )
    tasks = [(n_sections, n_levels, float(w)) for w in omega]
    started = time.time()

    def _run(items: list[tuple[int, int, float]]) -> list[tuple]:
        if jobs > 1:
            with ThreadPoolExecutor(max_workers=int(jobs)) as pool:
                return list(pool.map(_solve_one, items))
        return [_solve_one(task) for task in items]

    rows = _run(tasks)
    if int(refine_peaks) > 0:
        rows.sort(key=lambda row: row[0])
        grid = np.array([row[0] for row in rows], dtype=float)
        mass = np.array([row[1] for row in rows], dtype=float)
        damper = np.array([row[2] for row in rows], dtype=float)
        extra = [float(w) for w in _refined_grid(grid, mass, damper, n_refine=int(refine_peaks))]
        known = set(round(float(w), 12) for w in grid)
        tasks = [
            (n_sections, n_levels, w) for w in extra if round(w, 12) not in known
        ]
        if tasks:
            rows.extend(_run(tasks))
    rows.sort(key=lambda row: row[0])
    return Sweep(
        label=label,
        n_sections=n_sections,
        n_levels=n_levels,
        panel_count=rows[0][4],
        max_panel_scale=float(max(row[5] for row in rows)),
        omega=np.array([row[0] for row in rows], dtype=float),
        added_mass=np.array([row[1] for row in rows], dtype=float),
        damping=np.array([row[2] for row in rows], dtype=float),
        solver_relative_residual=np.array([row[3] for row in rows], dtype=float),
        elapsed=time.time() - started,
    )


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------


def geometry_gate() -> dict[str, Any]:
    """C1: the hull contract is checked before any hydrodynamic number."""
    n_sections, n_levels = MESHES["medium"]
    mesh = wigley_wetted_hull_mesh(
        LENGTH, BEAM, DRAFT, n_sections=n_sections, n_levels=n_levels
    )
    volume_mesh, _ = compute_displaced_volume_and_buoyancy(mesh)
    waterplane_area = float(compute_waterplane_properties(mesh)[0])
    cb_mesh = float(volume_mesh) / (LENGTH * BEAM * DRAFT)
    waterplane_analytic = (2.0 / 3.0) * BEAM * LENGTH
    checks = {
        "cb_relative": abs(cb_mesh - CB_ANALYTIC) / CB_ANALYTIC,
        "volume_relative": abs(float(volume_mesh) - VOLUME_ANALYTIC) / VOLUME_ANALYTIC,
        "l_over_b": LENGTH / BEAM,
        "b_over_t": BEAM / DRAFT,
        "waterplane_relative": abs(
            waterplane_area - waterplane_analytic
        ) / waterplane_analytic,
    }
    passed = (
        checks["cb_relative"] <= TOLERANCES["geometry"]
        and checks["volume_relative"] <= TOLERANCES["volume"]
        and abs(checks["l_over_b"] - 10.0) <= 1e-9
        and abs(checks["b_over_t"] - 1.6) <= 1e-9
    )
    return {
        "id": "C1",
        "title": "geometry precondition (Cb, L/B, B/T, displaced volume)",
        "threshold": (
            f"Cb within {TOLERANCES['geometry']:.0%} of 4/9; volume within "
            f"{TOLERANCES['volume']:.0%}; L/B=10; B/T=1.6"
        ),
        "passed": bool(passed),
        "values": {
            "displaced_volume_m3": float(volume_mesh),
            "displaced_volume_analytic_m3": VOLUME_ANALYTIC,
            "block_coefficient": cb_mesh,
            "block_coefficient_analytic": CB_ANALYTIC,
            "waterplane_area_m2": waterplane_area,
            "waterplane_area_analytic_m2": waterplane_analytic,
            **checks,
        },
    }


def panel_adequacy_gate(sweep: Sweep) -> dict[str, Any]:
    """C2: the top of the band must still be resolved by the panel mesh."""
    kh = float(sweep.omega[-1]) ** 2 / GRAVITY * sweep.max_panel_scale
    passed = kh <= TOLERANCES["panel_kh"]
    return {
        "id": "C2",
        "title": f"panel resolution at the top of the band ({sweep.label} mesh)",
        "threshold": f"k*sqrt(A_panel,max) <= {TOLERANCES['panel_kh']:g}",
        "passed": bool(passed),
        "values": {
            "omega_max": float(sweep.omega[-1]),
            "wavenumber_top": float(sweep.omega[-1]) ** 2 / GRAVITY,
            "panel_scale_m": sweep.max_panel_scale,
            "k_times_panel_scale": kh,
            "panels_per_wavelength": float(2.0 * math.pi / kh) if kh > 0 else float("inf"),
            "panel_count": sweep.panel_count,
        },
    }


def kk_gate(sweep: Sweep, *, corrupt: bool = False) -> dict[str, Any]:
    """C3: per-DOF Kramers-Kronig closure of the added mass (the case criterion)."""
    rows = []
    worst = 0.0
    worst_dof = ""
    for index, name in enumerate(DOF_NAMES):
        mass = sweep.added_mass[:, index, index]
        damper = sweep.damping[:, index, index]
        tail = fit_damping_power_tail(sweep.omega, damper)
        damper_used = damper
        if corrupt:
            # Pre-declared negative control: break the *shape* of the damping
            # band.  A constant rescale would be invisible to a closure that
            # normalises by the same scale, so the corruption has to be in shape.
            damper_used = damper * np.where(sweep.omega > 1.0, 1.3, 1.0)
        closure = kk_added_mass_closure(
            sweep.omega, mass, damper_used, damping_tail=tail
        )
        asymptote = fit_added_mass_asymptote(sweep.omega, mass)
        errors = np.abs(closure.relative_residual)
        index_worst = int(np.argmax(errors)) if errors.size else 0
        max_error = float(closure.max_relative_residual)
        rows.append(
            {
                "dof": int(index),
                "dof_name": name,
                "max_relative_residual": max_error,
                "residual_scale": float(closure.added_mass_scale),
                "worst_sample_omega": float(closure.sample_omegas[index_worst]),
                "worst_sample_reference": float(closure.measured_added_mass[index_worst]),
                "worst_sample_reconstructed": float(
                    closure.predicted_added_mass[index_worst]
                ),
                "worst_sample_absolute_error": float(
                    np.abs(closure.residual)[index_worst]
                ),
                "added_mass_inf_closure": float(closure.added_mass_inf),
                "added_mass_inf_closure_source": closure.added_mass_inf_source,
                "added_mass_inf_tail_fit": float(asymptote.added_mass_inf),
                "added_mass_inf_tail_fit_residual": float(asymptote.relative_residual),
                "added_mass_inf_cross_check": float(
                    abs(closure.added_mass_inf - asymptote.added_mass_inf)
                    / max(float(closure.added_mass_scale), 1e-300)
                ),
                "damping_tail_valid": bool(tail.valid),
                "damping_tail_exponent": float(tail.exponent),
                "damping_tail_coefficient": float(tail.coefficient),
                "damping_tail_reason": tail.reason,
                "sample_omega": closure.sample_omegas.tolist(),
                "reference_added_mass": closure.measured_added_mass.tolist(),
                "reconstructed_added_mass": closure.predicted_added_mass.tolist(),
                "relative_residual": closure.relative_residual.tolist(),
            }
        )
        if max_error > worst:
            worst = max_error
            worst_dof = name
    tolerance = TOLERANCES["kk"] if not corrupt else TOLERANCES["negative_control"]
    raw_passed = worst <= tolerance
    return {
        "id": "C3" if not corrupt else "C6b",
        "title": (
            "Kramers-Kronig closure of the added mass (6 DOF)"
            if not corrupt
            else "negative control: corrupted damping shape must fail the KK gate"
        ),
        "threshold": (
            f"max over samples and DOF |A_BEM-A_KK|/max|A_BEM| <= {tolerance:.0%}"
        ),
        "passed": bool(raw_passed) if not corrupt else bool(not raw_passed),
        "raw_gate_passed": bool(raw_passed),
        "values": {
            "mesh": sweep.label,
            "band": [float(sweep.omega[0]), float(sweep.omega[-1])],
            "worst_dof": worst_dof,
            "worst_relative_residual": worst,
            "per_dof": rows,
            "tail_fit_model": "B(s) = C s^-p on the upper quartile of the sweep",
            "added_mass_inf_estimator": "least-squares closure (one free parameter)",
            "added_mass_inf_cross_check_worst": float(
                max(row["added_mass_inf_cross_check"] for row in rows)
            ),
            "added_mass_inf_cross_check_note": (
                "C3 lets one free parameter (A_inf) absorb a constant offset, so a "
                "closure that only shifts A would still pass. The independent "
                "A_inf + c/omega^2 tail fit is therefore reported next to it: a "
                "closure A_inf far from the tail fit means the residual is being "
                "absorbed, not explained."
            ),
        },
    }


def round_trip_gate(sweep: Sweep, *, dof: int = 2) -> dict[str, Any]:
    """C4: B(omega) -> K(t) -> B(omega) round trip (Cummins transform closure)."""
    mass = sweep.added_mass[:, dof, dof]
    damper = sweep.damping[:, dof, dof]
    tau = np.arange(0.0, TAU_MAX + 0.5 * DT, DT)
    kernel = td.radiation_memory_kernel(sweep.omega, damper, tau)
    trip = td.fourier_round_trip(
        sweep.omega, mass, damper, float(np.interp(sweep.omega[-1], sweep.omega, mass)),
        dt=DT, tau_max=TAU_MAX,
    )
    max_error = float(trip.damping_relative_error_max)
    added_mass_error = float(trip.added_mass_relative_error_max)
    diagnostics = dict(getattr(trip, "diagnostics", {}) or {})
    causal = bool(diagnostics.get("causal", True))
    parity = float(diagnostics.get("even_parity_relative_error", 0.0) or 0.0)
    passed = max_error <= TOLERANCES["round_trip"] and causal
    return {
        "id": "C4",
        "title": f"Cummins round trip B->K->B ({DOF_NAMES[dof]})",
        "threshold": f"max pointwise relative error <= {TOLERANCES['round_trip']:.0%}",
        "passed": bool(passed),
        "values": {
            "dof": int(dof),
            "dof_name": DOF_NAMES[dof],
            "damping_relative_error_max": max_error,
            "added_mass_relative_error_max": added_mass_error,
            "round_trip_band": list(trip.band),
            "kernel_causal": causal,
            "kernel_even_parity_relative_error": parity,
            "kernel_tau_max": float(trip.tau_max),
            "kernel_dt": float(trip.dt),
        },
    }


#: Peaks below this fraction of the largest excursion are not part of the
#: physical decay: the memory kernel is truncated at ``omega_max``, which leaves
#: a small ripple at the truncation frequency (period ``2 pi / omega_max``) that
#: survives long after the hull mode has died away.  Fitting the logarithmic
#: decrement across that residue produced the recorded C5 failure (zeta = 0.027
#: from a mode whose measured decrement is 0.13): the peak spacing locked onto
#: the ripple, not onto the hull mode.
DECAY_AMPLITUDE_FLOOR = 1.0e-2


def _decay_zeta(
    time_array: np.ndarray,
    displacement: np.ndarray,
    *,
    amplitude_floor: float = DECAY_AMPLITUDE_FLOOR,
) -> tuple[float, float, int]:
    """Damping ratio from the logarithmic decrement of the positive peaks.

    Only peaks above ``amplitude_floor * max|displacement|`` are used, so a
    decaying record is fitted over the part that carries the hull mode instead
    of over whatever numerical ripple outlives it.
    """
    excursion = float(np.max(np.abs(displacement))) if displacement.size else 0.0
    threshold = float(amplitude_floor) * excursion
    peaks: list[tuple[float, float]] = []
    for i in range(1, displacement.size - 1):
        if (
            displacement[i] > displacement[i - 1]
            and displacement[i] >= displacement[i + 1]
            and displacement[i] > threshold
        ):
            peaks.append((float(time_array[i]), float(displacement[i])))
    if len(peaks) < 3:
        return float("nan"), float("nan"), len(peaks)
    first, last = peaks[0], peaks[-1]
    periods = (last[0] - first[0]) / (len(peaks) - 1)
    decrement = math.log(first[1] / last[1]) / (len(peaks) - 1)
    zeta = decrement / math.sqrt(4.0 * math.pi**2 + decrement**2)
    return float(zeta), float(periods), len(peaks)


def _restoring_and_mass(sweep: Sweep) -> tuple[float, float, float]:
    """Restoring stiffness C33 from the waterplane, displaced mass, A_inf fit."""
    n_sections, n_levels = MESHES[sweep.label]
    mesh = wigley_wetted_hull_mesh(
        LENGTH, BEAM, DRAFT, n_sections=n_sections, n_levels=n_levels
    )
    waterplane_area = float(compute_waterplane_properties(mesh)[0])
    restoring = DENSITY * GRAVITY * waterplane_area
    return restoring, DENSITY * VOLUME_ANALYTIC, waterplane_area


def free_decay_gate(sweep: Sweep, *, dof: int = 2) -> dict[str, Any]:
    """C5: time-domain Cummins free decay vs the frequency-domain damping ratio."""
    mass = sweep.added_mass[:, dof, dof]
    damper = sweep.damping[:, dof, dof]
    closure = kk_added_mass_closure(sweep.omega, mass, damper)
    added_mass_inf = float(closure.added_mass_inf)
    tau = np.arange(0.0, TAU_MAX + 0.5 * DT, DT)
    kernel = td.radiation_memory_kernel(sweep.omega, damper, tau)
    restoring, displaced_mass, _ = _restoring_and_mass(sweep)

    decay = td.cummins_free_decay(
        kernel,
        mass=displaced_mass,
        restoring=restoring,
        added_mass_inf=added_mass_inf,
        initial_displacement=1.0,
        dt=DT,
        duration=DURATION,
    )
    zeta_decay, damped_period, n_peaks = _decay_zeta(
        np.asarray(decay.time), np.asarray(decay.displacement)
    )
    if math.isfinite(damped_period) and damped_period > 0.0 and math.isfinite(zeta_decay):
        omega_d = 2.0 * math.pi / damped_period
        omega_n = omega_d / math.sqrt(max(1.0 - zeta_decay**2, 1e-12))
    else:
        omega_n = float("nan")
    in_band = bool(
        math.isfinite(omega_n) and sweep.omega[0] <= omega_n <= sweep.omega[-1]
    )
    if in_band:
        a_at = float(np.interp(omega_n, sweep.omega, mass))
        b_at = float(np.interp(omega_n, sweep.omega, damper))
        zeta_fd = b_at / (2.0 * (displaced_mass + a_at) * omega_n)
    else:
        a_at = b_at = zeta_fd = float("nan")
    if math.isfinite(zeta_decay) and math.isfinite(zeta_fd) and zeta_fd > 0.0:
        relative = abs(zeta_decay - zeta_fd) / zeta_fd
    else:
        relative = float("inf")
    # A gate that cannot be evaluated must say so: "outside the computed band"
    # and "the two routes disagree" are different failures and it was the first
    # one that produced the recorded FAIL (omega_n = 8.15 rad/s > band top 8).
    if not math.isfinite(omega_n):
        status = "undecidable: decay record too short for a damping ratio"
    elif not in_band:
        status = (
            "undecidable: natural frequency outside the computed band "
            f"({sweep.omega[0]:.3g}..{sweep.omega[-1]:.3g} rad/s)"
        )
    elif not (math.isfinite(zeta_fd) and zeta_fd > 0.0):
        status = "undecidable: non-positive frequency-domain damping at omega_n"
    else:
        status = "decided"
    passed = status == "decided" and math.isfinite(relative) and relative <= TOLERANCES["free_decay"]
    return {
        "id": "C5",
        "title": f"Cummins free decay vs frequency-domain damping ratio ({DOF_NAMES[dof]})",
        "threshold": f"|zeta_decay - zeta_fd|/zeta_fd <= {TOLERANCES['free_decay']:.0%}",
        "passed": bool(passed),
        "values": {
            "dof": int(dof),
            "dof_name": DOF_NAMES[dof],
            "zeta_decay": zeta_decay,
            "zeta_frequency_domain": zeta_fd,
            "relative_error": relative,
            "natural_frequency_in_band": in_band,
            "band_rad_s": [float(sweep.omega[0]), float(sweep.omega[-1])],
            "status": status,
            "damped_period_s": damped_period,
            "natural_frequency_rad_s": omega_n,
            "added_mass_at_natural": a_at,
            "damping_at_natural": b_at,
            "restoring_n_per_m": restoring,
            "displaced_mass_kg": displaced_mass,
            "added_mass_inf_used": added_mass_inf,
            "added_mass_inf_source": "KK closure (least-squares offset)",
            "n_peaks": n_peaks,
            "peak_amplitude_floor": DECAY_AMPLITUDE_FLOOR,
            "dt": DT,
            "duration": DURATION,
        },
    }


def negative_control_gate(sweep: Sweep, *, dof: int = 2) -> dict[str, Any]:
    """C6a: zero damping must give a zero kernel and a non-decaying record."""
    zeros = np.zeros_like(sweep.damping[:, dof, dof])
    mass = sweep.added_mass[:, dof, dof]
    tau = np.arange(0.0, TAU_MAX + 0.5 * DT, DT)
    kernel = td.radiation_memory_kernel(sweep.omega, zeros, tau)
    added_mass_inf = float(kk_added_mass_closure(sweep.omega, mass, zeros).added_mass_inf)
    restoring, displaced_mass, _ = _restoring_and_mass(sweep)
    decay = td.cummins_free_decay(
        kernel,
        mass=displaced_mass,
        restoring=restoring,
        added_mass_inf=added_mass_inf,
        initial_displacement=1.0,
        dt=DT,
        duration=DURATION,
    )
    zeta_zero, _, n_peaks = _decay_zeta(
        np.asarray(decay.time), np.asarray(decay.displacement)
    )
    kernel_max = float(np.max(np.abs(kernel.values)))
    tail_slice = np.abs(np.asarray(decay.displacement)[-int(0.2 * decay.displacement.size):])
    amplitude_ratio = float(np.max(tail_slice) / np.max(np.abs(decay.displacement)))
    passed = (
        kernel_max <= 1e-12
        and (not math.isfinite(zeta_zero) or zeta_zero <= 1e-3)
        and amplitude_ratio >= 0.98
    )
    return {
        "id": "C6a",
        "title": f"negative control: zero damping gives zero decay ({DOF_NAMES[dof]})",
        "threshold": "max|K| <= 1e-12, zeta_decay <= 1e-3, late/early amplitude ratio >= 0.98",
        "passed": bool(passed),
        "values": {
            "kernel_max_abs": kernel_max,
            "zeta_zero_damping": zeta_zero,
            "amplitude_ratio_late_over_early": amplitude_ratio,
            "n_peaks": n_peaks,
        },
    }


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def _mesh_convergence(coarse: Sweep, medium: Sweep) -> dict[str, Any]:
    """Ref vs got on the mesh axis: production mesh against the coarse one."""
    rows = []
    for index, name in enumerate(DOF_NAMES):
        mass_c = coarse.added_mass[:, index, index]
        mass_m = medium.added_mass[:, index, index]
        damp_c = coarse.damping[:, index, index]
        damp_m = medium.damping[:, index, index]
        rows.append(
            {
                "dof_name": name,
                "added_mass_relative_change_max": float(
                    np.max(np.abs(mass_m - mass_c)) / max(np.max(np.abs(mass_m)), 1e-300)
                ),
                "damping_relative_change_max": float(
                    np.max(np.abs(damp_m - damp_c)) / max(np.max(np.abs(damp_m)), 1e-300)
                ),
            }
        )
    return {
        "reference": f"{medium.label} mesh ({medium.panel_count} panels)",
        "got": f"{coarse.label} mesh ({coarse.panel_count} panels)",
        "per_dof": rows,
        "worst_added_mass_change": max(row["added_mass_relative_change_max"] for row in rows),
        "worst_damping_change": max(row["damping_relative_change_max"] for row in rows),
    }


def grid_convergence_study(
    *,
    label: str = "coarse",
    band: tuple[float, float] = BAND,
    node_counts: Sequence[int] = (13, 21, 33, 41, 61),
    jobs: int = 1,
) -> dict[str, Any]:
    """KK closure residual as a function of the number of frequency nodes.

    Gate C3 asks whether the tabulated damping is *causal*, and it closes the
    Kramers-Kronig integral with the piecewise-linear interpolant of the
    damping.  A residual that shrinks as the grid is refined is a statement
    about the interpolant; a residual that stalls is a statement about the
    BEM data.  Refining the node count and using nested log grids makes the two
    separable, which is what turns the gate from an assertion into a measurement.
    """
    rows: list[dict[str, Any]] = []
    for count in sorted(set(int(c) for c in node_counts)):
        sweep = solve_sweep(label, band=band, n_omega=int(count), jobs=jobs)
        gate = kk_gate(sweep)
        per_dof = {
            row["dof_name"]: float(row["max_relative_residual"])
            for row in gate["values"]["per_dof"]
        }
        rows.append(
            {
                "n_omega": int(sweep.omega.size),
                "mesh": label,
                "panel_count": int(sweep.panel_count),
                "omega_min": float(sweep.omega[0]),
                "omega_max": float(sweep.omega[-1]),
                "kk_residual_per_dof": per_dof,
                "kk_residual_max": float(max(per_dof.values())),
                "kk_gate_passed": bool(gate["passed"]),
                "elapsed_s": float(sweep.elapsed),
            }
        )
    return {
        "case": CASE_SLUG,
        "study": "kk closure vs frequency-node count",
        "band_rad_s": [float(band[0]), float(band[1])],
        "tolerance": TOLERANCES["kk"],
        "rows": rows,
    }


def case_verdict(gates: dict[str, Any]) -> dict[str, Any]:
    """Tally a gate dictionary into the case verdict (shared by run and replay)."""
    present = {key: value for key, value in gates.items() if value is not None}
    passed = [key for key, value in present.items() if value["passed"]]
    failed = [key for key, value in present.items() if not value["passed"]]
    return {
        "passed": not failed,
        "gates_passed": len(passed),
        "gates_total": len(present),
        "passed_list": passed,
        "failed_list": failed,
        "case_tolerance": TOLERANCES["kk"],
        "criterion_source": "tensor_case_pool id=145 (tol_policy <10%)",
    }


def gates_from_artifact(
    path: str | Path,
    *,
    dof_free_decay: int = 2,
    include_geometry: bool = True,
    corrupted_control: bool = True,
) -> dict[str, Any]:
    """Re-judge a committed sweep artifact without solving the BEM system again.

    Every gate in this case is a *function of the tabulated coefficients* plus
    the hull constants, so the verdict is reproducible from
    ``sweep_<label>.npz`` alone.  That is what makes the accepted numbers
    testable in CI (see ``tests/test_case_bem_cummins_wigley.py``) and what lets
    a later reader re-derive the verdict from the evidence that was actually
    committed instead of trusting the report.
    """
    sweep = load_sweep(path)
    gates: dict[str, Any] = {}
    if include_geometry:
        gates["C1"] = geometry_gate()
    gates["C2"] = panel_adequacy_gate(sweep)
    gates["C3"] = kk_gate(sweep)
    gates["C4"] = round_trip_gate(sweep, dof=int(dof_free_decay))
    gates["C5"] = free_decay_gate(sweep, dof=int(dof_free_decay))
    gates["C6a"] = negative_control_gate(sweep, dof=int(dof_free_decay))
    if corrupted_control:
        gates["C6b"] = kk_gate(sweep, corrupt=True)
    return {
        "case": CASE_SLUG,
        "artifact": str(path),
        "dof_free_decay": DOF_NAMES[int(dof_free_decay)],
        "sweep": sweep.summary(),
        "gates": gates,
        "verdict": case_verdict(gates),
    }


def run_case(
    *,
    out_dir: str | Path,
    jobs: int = 1,
    n_omega: int = N_OMEGA,
    band: tuple[float, float] = BAND,
    meshes: Sequence[str] = ("coarse", "medium"),
    figure: bool = True,
    save_sweeps: bool = True,
    dof_free_decay: int = 2,
    refine_peaks: int = 0,
    mission: str = "m17375",
) -> dict[str, Any]:
    """Run the whole case and write the artifacts; returns the results dict."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {
        "case": CASE_SLUG,
        "mission": str(mission),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "hull": {
            "form": "mathematical Wigley hull y = (B/2)(1-(2x/L)^2)(1-(z/T)^2)",
            "length_m": LENGTH,
            "beam_m": BEAM,
            "draft_m": DRAFT,
            "density_kg_m3": DENSITY,
            "gravity_m_s2": GRAVITY,
            "block_coefficient_analytic": CB_ANALYTIC,
            "displaced_volume_analytic_m3": VOLUME_ANALYTIC,
        },
        "settings": {
            "band_rad_s": [float(band[0]), float(band[1])],
            "n_omega": int(n_omega),
            "refine_peaks": int(refine_peaks),
            "meshes": {label: list(MESHES[label]) for label in meshes},
            "chief_points": "auto",
            "tolerances": TOLERANCES,
            "free_decay_dof": DOF_NAMES[int(dof_free_decay)],
            "dt": DT,
            "duration": DURATION,
            "tau_max": TAU_MAX,
        },
    }

    results["gates"] = {"C1": geometry_gate()}
    sweeps: dict[str, Sweep] = {}
    for label in meshes:
        sweep = solve_sweep(
            label, band=band, n_omega=n_omega, jobs=jobs, refine_peaks=refine_peaks
        )
        sweeps[label] = sweep
        results.setdefault("sweeps", {})[label] = sweep.summary()
        if save_sweeps:
            save_sweep(sweep, out_path / f"sweep_{label}.npz")

    production = sweeps[meshes[-1]]
    results["gates"]["C2"] = panel_adequacy_gate(production)
    results["gates"]["C3"] = kk_gate(production)
    if len(meshes) > 1:
        results["gates"]["C3_coarse"] = kk_gate(sweeps[meshes[0]])
    results["gates"]["C4"] = round_trip_gate(production, dof=int(dof_free_decay))
    results["gates"]["C5"] = free_decay_gate(production, dof=int(dof_free_decay))
    results["gates"]["C6a"] = negative_control_gate(production, dof=int(dof_free_decay))
    results["gates"]["C6b"] = kk_gate(production, corrupt=True)
    if len(meshes) > 1:
        results["mesh_convergence"] = _mesh_convergence(sweeps[meshes[0]], production)

    results["verdict"] = case_verdict(results["gates"])

    figure_name = _make_figure(sweeps, results, out_path) if figure else None
    results["figure"] = figure_name

    (out_path / "case_result.json").write_text(
        json.dumps(results, indent=2, sort_keys=False), encoding="utf-8"
    )
    (out_path / "CASE.md").write_text(render_markdown(results), encoding="utf-8")
    return results


def _make_figure(sweeps: dict[str, Sweep], results: dict[str, Any], out_path: Path) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - optional dependency
        return None
    production_label = list(sweeps)[-1]
    sweep = sweeps[production_label]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4))
    colours = plt.cm.viridis(np.linspace(0.05, 0.9, len(DOF_NAMES)))
    per_dof = results["gates"]["C3"]["values"]["per_dof"]
    for index, (name, colour) in enumerate(zip(DOF_NAMES, colours)):
        row = per_dof[index]
        scale = row["residual_scale"]
        axes[0].plot(
            row["sample_omega"],
            np.asarray(row["reference_added_mass"]) / scale,
            "o", ms=3.2, color=colour, label=f"{name} BEM",
        )
        axes[0].plot(
            row["sample_omega"],
            np.asarray(row["reconstructed_added_mass"]) / scale,
            "-", lw=1.3, color=colour, label=f"{name} KK",
        )
        axes[1].semilogy(
            row["sample_omega"],
            100.0 * np.asarray(row["relative_residual"]),
            "-o", ms=3.0, lw=1.3, color=colour, label=name,
        )
    axes[0].set_xscale("log")
    axes[0].set_xlabel(r"$\omega$ [rad/s]")
    axes[0].set_ylabel(r"$A_{ii}(\omega)/\max|A_{ii}|$")
    axes[0].set_title(
        f"{production_label} mesh ({sweep.panel_count} panels): BEM vs KK reconstruction"
    )
    axes[0].legend(fontsize=6.0, ncol=2)
    axes[0].grid(alpha=0.3)
    axes[1].axhline(
        100.0 * TOLERANCES["kk"], color="crimson", ls="--", lw=1.0,
        label=f"criterion {TOLERANCES['kk']:.0%}",
    )
    axes[1].set_xscale("log")
    axes[1].set_xlabel(r"$\omega$ [rad/s]")
    axes[1].set_ylabel("KK residual [% of max|A|]")
    axes[1].set_title("Kramers-Kronig closure residual per DOF")
    axes[1].legend(fontsize=6.5)
    axes[1].grid(alpha=0.3, which="both")
    fig.suptitle(f"{CASE_SLUG}: Cummins memory kernel + KK causality (Wigley hull)")
    fig.tight_layout()
    name = "fig_kk_closure.png"
    fig.savefig(out_path / name, dpi=150)
    plt.close(fig)
    return name


def _fmt(value: Any, digits: int = 4) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "n/a"
        return f"{value:.{digits}g}"
    return str(value)


def render_markdown(results: dict[str, Any]) -> str:
    """Render the case document from the numbers in ``results`` (no hand copy)."""
    gates = results["gates"]
    verdict = results["verdict"]
    hull = results["hull"]
    settings = results["settings"]
    lines: list[str] = []
    lines.append(f"# {CASE_SLUG}")
    lines.append("")
    lines.append(
        "Auto-rendered by `tensorbem.benchmarks.case_bem_cummins_wigley`; every "
        "number below is read from `case_result.json` produced by the same run."
    )
    lines.append("")
    lines.append("## 1. Case setup")
    lines.append("")
    lines.append("| item | value |")
    lines.append("| --- | --- |")
    lines.append(f"| hull form | {hull['form']} |")
    lines.append(f"| L / B / T [m] | {hull['length_m']} / {hull['beam_m']} / {hull['draft_m']} |")
    lines.append(f"| Cb (analytic) | {_fmt(hull['block_coefficient_analytic'])} |")
    lines.append(f"| displaced volume (analytic) [m^3] | {_fmt(hull['displaced_volume_analytic_m3'])} |")
    lines.append(f"| frequency band [rad/s] | {settings['band_rad_s'][0]} .. {settings['band_rad_s'][1]} |")
    lines.append(f"| frequency nodes | {settings['n_omega']} (log-spaced) |")
    lines.append(
        "| meshes | "
        + ", ".join(f"{label}: {n}x{m}" for label, (n, m) in settings["meshes"].items())
        + " |"
    )
    lines.append(f"| CHIEF irregular-frequency suppression | {settings['chief_points']} |")
    lines.append("")
    lines.append("Criterion source: " + verdict["criterion_source"] + ".")
    lines.append("")
    figure = results.get("figure")
    if figure:
        lines.append(f"![{figure}]({figure})")
        lines.append("")
    lines.append("## 2. Gates")
    lines.append("")
    lines.append("| gate | criterion | threshold | passed |")
    lines.append("| --- | --- | --- | --- |")
    for key, gate in gates.items():
        if gate is None:
            continue
        lines.append(
            f"| {key} | {gate['title']} | {gate['threshold']} | "
            f"{'PASS' if gate['passed'] else 'FAIL'} |"
        )
    lines.append("")
    lines.append("## 3. Reference vs got (KK closure, per DOF)")
    lines.append("")
    lines.append(
        "Reference = direct BEM added mass at the same sample frequency; got = KK "
        "reconstruction from the damping sweep with a power-law tail. Errors are "
        "normalised by `max|A|` of the DOF."
    )
    lines.append("")
    lines.append(
        "| DOF | max rel. residual | worst sample omega | A_BEM (ref) | A_KK (got) | "
        "abs. error | A_inf (closure) | tail exponent p | tail valid |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in gates["C3"]["values"]["per_dof"]:
        lines.append(
            f"| {row['dof_name']} | {100.0*row['max_relative_residual']:.2f}% | "
            f"{row['worst_sample_omega']:.4g} | {row['worst_sample_reference']:.6g} | "
            f"{row['worst_sample_reconstructed']:.6g} | {row['worst_sample_absolute_error']:.3g} | "
            f"{row['added_mass_inf_closure']:.6g} | {row['damping_tail_exponent']:.3g} | "
            f"{_fmt(row['damping_tail_valid'])} |"
        )
    lines.append("")
    if "C3_coarse" in gates:
        lines.append("### 3b. Same closure on the coarse mesh (band sensitivity / convergence)")
        lines.append("")
        lines.append("| DOF | coarse max rel. residual | medium max rel. residual |")
        lines.append("| --- | --- | --- |")
        for coarse_row, medium_row in zip(
            gates["C3_coarse"]["values"]["per_dof"], gates["C3"]["values"]["per_dof"]
        ):
            lines.append(
                f"| {medium_row['dof_name']} | {100.0*coarse_row['max_relative_residual']:.2f}% | "
                f"{100.0*medium_row['max_relative_residual']:.2f}% |"
            )
        lines.append("")
    convergence = results.get("mesh_convergence")
    if convergence:
        lines.append("## 4. Mesh convergence (ref vs got)")
        lines.append("")
        lines.append(
            f"Reference = {convergence['reference']}, got = {convergence['got']}; "
            "relative change of the diagonal coefficients over the same band."
        )
        lines.append("")
        lines.append("| DOF | added mass change | damping change |")
        lines.append("| --- | --- | --- |")
        for row in convergence["per_dof"]:
            lines.append(
                f"| {row['dof_name']} | {100.0*row['added_mass_relative_change_max']:.2f}% | "
                f"{100.0*row['damping_relative_change_max']:.2f}% |"
            )
        lines.append("")
    for key in ("C4", "C5", "C6a", "C6b"):
        gate = gates.get(key)
        if gate is None:
            continue
        lines.append(f"### {key}: {gate['title']}")
        lines.append("")
        lines.append(f"Threshold: {gate['threshold']} -> **{'PASS' if gate['passed'] else 'FAIL'}**")
        lines.append("")
        lines.append("| quantity | value |")
        lines.append("| --- | --- |")
        for name, value in gate["values"].items():
            if isinstance(value, (list, dict)):
                continue
            lines.append(f"| {name} | {_fmt(value)} |")
        lines.append("")
    lines.append("## 5. Verdict")
    lines.append("")
    lines.append(
        f"{verdict['gates_passed']} / {verdict['gates_total']} gates passed "
        f"({'all pass' if verdict['passed'] else 'FAILED: ' + ', '.join(verdict['failed_list'])})."
    )
    lines.append("")
    lines.append(
        "Case tolerance (KK closure) = "
        f"{100.0*verdict['case_tolerance']:.0f}%; coarse-mesh closure and the corrupted "
        "damping control are reported alongside so the verdict cannot be read as a "
        "self-fulfilling gate."
    )
    lines.append("")
    lines.append("## 6. Reproduction")
    lines.append("")
    lines.append("```")
    lines.append(
        "PYTHONPATH=src python -m tensorbem.benchmarks.case_bem_cummins_wigley "
        f"--out {results.get('output_dir', 'benchmarks/verified/casepool_bem_cummins_wigley')} "
        f"--jobs {results.get('jobs', 1)}"
    )
    lines.append("```")
    lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        default="benchmarks/verified/casepool_bem_cummins_wigley",
        help="artifact directory",
    )
    parser.add_argument("--jobs", type=int, default=1, help="parallel frequency solves")
    parser.add_argument("--n-omega", type=int, default=N_OMEGA)
    parser.add_argument("--omega-min", type=float, default=BAND[0])
    parser.add_argument("--omega-max", type=float, default=BAND[1])
    parser.add_argument(
        "--meshes",
        nargs="+",
        default=["coarse", "medium"],
        choices=sorted(MESHES),
    )
    parser.add_argument("--no-figure", action="store_true")
    parser.add_argument(
        "--mission",
        default="m17375",
        help="mission id written into the artifact metadata (provenance only)",
    )
    parser.add_argument(
        "--replay",
        default=None,
        metavar="NPZ",
        help="re-judge a committed sweep artifact instead of solving (no BEM solve)",
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--grid-study",
        nargs="+",
        type=int,
        default=None,
        metavar="N",
        help="run the KK-residual vs node-count study on these counts and exit",
    )
    parser.add_argument(
        "--refine-peaks",
        type=int,
        default=0,
        metavar="K",
        help="add K geometric midpoints at the intervals with the largest damping jump",
    )
    args = parser.parse_args(argv)
    if args.grid_study:
        study = grid_convergence_study(
            label=args.meshes[-1],
            band=(args.omega_min, args.omega_max),
            node_counts=args.grid_study,
            jobs=args.jobs,
        )
        out_path = Path(args.out)
        out_path.mkdir(parents=True, exist_ok=True)
        (out_path / "grid_convergence.json").write_text(
            json.dumps(study, indent=2), encoding="utf-8"
        )
        if not args.quiet:
            print(json.dumps(study, indent=2))
        return 0
    if args.replay:
        replay = gates_from_artifact(args.replay)
        if not args.quiet:
            print(json.dumps(replay["verdict"], indent=2))
        return 0 if replay["verdict"]["passed"] else 1
    results = run_case(
        out_dir=args.out,
        jobs=args.jobs,
        n_omega=args.n_omega,
        band=(args.omega_min, args.omega_max),
        meshes=tuple(args.meshes),
        figure=not args.no_figure,
        refine_peaks=args.refine_peaks,
        mission=args.mission,
    )
    verdict = results["verdict"]
    if not args.quiet:
        print(json.dumps(verdict, indent=2))
        c3 = results["gates"]["C3"]
        for row in c3["values"]["per_dof"]:
            print(
                f"{row['dof_name']:>6s}: KK residual {100.0*row['max_relative_residual']:6.2f}%"
                f"  tail p={row['damping_tail_exponent']:.2f}"
                f"  A_inf={row['added_mass_inf_closure']:.4g}"
            )
        print(f"artifacts: {args.out}")
    return 0 if verdict["passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
