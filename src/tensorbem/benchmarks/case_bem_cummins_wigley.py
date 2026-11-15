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
    high_band_truncation_budget,
    kk_added_mass_closure,
    local_tail_exponent,
)
from tensorbem.added_mass_limit import (
    added_mass_limit,
    half_immersed_sphere_added_mass,
    half_immersed_sphere_mesh,
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
    "solver_validity_gate",
    "panel_adequacy_gate",
    "kk_gate",
    "round_trip_gate",
    "free_decay_gate",
    "negative_control_gate",
    "extension_consistency_gate",
    "truncation_decidability_gate",
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

#: Rotation centre of every rigid-body mode in this case.  The radiation sweep
#: and the added-mass limit solve must use the *same* centre, otherwise the two
#: ``A_inf`` columns differ by a rigid-body transfer and the comparison is void.
CENTER = (0.0, 0.0, 0.0)

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

#: Panels needed by gate ``C2`` to reach ``omega = 20 rad/s``: there
#: ``k = omega**2/g = 40.8 1/m`` forces ``sqrt(A_panel) <= 2.45 cm``, which the
#: structured Wigley mesh reaches at ``60 x 16`` (3840 panels,
#: ``sqrt(A_max) = 2.32 cm``, admissible to ``20.54 rad/s``).
EXTENDED_MESH = (60, 16)
#: Extended band and node count.  The extension is a *pre-declared* experiment:
#: the 12 rad/s band top leaves the truncation budget of surge/sway/yaw
#: unbounded (their damping has not entered algebraic decay there), so the only
#: honest way to certify those DOF is a band whose top the mesh can still resolve.
EXTENDED_BAND = (0.2, 20.0)
EXTENDED_N_OMEGA = 55

TOLERANCES = {
    "kk": 0.10,           # case pool id=145, tol_policy <10%
    "round_trip": 0.02,   # Cummins transform self-consistency
    "free_decay": 0.05,   # time domain vs its own frequency-domain estimate
    "geometry": 0.01,     # Cb / L/B / B/T contract
    "volume": 0.02,       # meshed displaced volume vs (4/9)LBT
    "panel_kh": 1.0,      # k*sqrt(max panel area) at the top of the band
    "kk_budget": 0.05,
    # Independent A_inf anchor (image method).  The offset the band closure had
    # to absorb must be smaller than the case tolerance when measured against
    # the added-mass amplitude of the same DOF -- i.e. the free parameter the
    # fitted gate spent is not allowed to exceed the criterion it is judged by.
    "anchor": 0.10,
    # The anchor itself is refereed by a closed-form case (half-immersed sphere,
    # A_inf(heave) = m_full/4).  Measured 0.91 % at 828 panels, 2.0 % at 360.
    "anchor_analytic": 0.03,    # half the case tolerance: above this the un-solved band
                          # can produce the residual on its own, so the sample is
                          # not evidence either way
    "solver_residual": 1.0e-2,  # gate C0: the BIE solve must actually have converged
    "negative_control": 0.10,
    # The extended-band sweep must be the *same physics* as the committed one on
    # the band they share; otherwise comparing the two verdicts is void.  This is
    # a discretisation check, not a criterion of the case: it is normalised by
    # the amplitude of the DOF, so a 5 % bar means the 8x panel increase may not
    # move the tabulated coefficients by more than a twentieth of their peak.
    "extension_overlap": 0.05,
}

DT = 0.002
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
        center=CENTER,
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


def solver_validity_gate(sweep: Sweep) -> dict[str, Any]:
    """C0: a sweep whose BIE solves did not converge cannot be judged.

    The gates below read ``A``/``B`` as data.  If the linear system behind them
    was solved badly (rank-deficient LAPACK path, corrupted operator assembly),
    ``B`` can come out *negative* -- energy created instead of radiated -- and
    every downstream gate would then score garbage: C3 would report a closure
    residual that is a property of the failed solve, and C5 would either be
    undecidable or, worse, agree with itself.  This gate checks the residual the
    solver already reports and refuses the sweep if it is not small.

    It was added after a run of this same case on a cluster node returned
    ``max solver residual = 1.25`` and ``min B = -129.9`` (negative damping) on
    the *same* mesh, band and node count that give ``1.1e-3`` and ``B > 0``
    locally; without C0 the case verdict silently followed the machine.
    """
    residual = np.asarray(sweep.solver_relative_residual, dtype=float)
    worst = float(np.max(residual)) if residual.size else float("inf")
    damper = np.asarray(sweep.damping, dtype=float)
    off_diagonal = damper - np.eye(6) * damper
    min_damping = float(np.min(np.diagonal(damper, axis1=-2, axis2=-1)))
    tolerance = TOLERANCES["solver_residual"]
    passed = bool(worst <= tolerance and min_damping >= 0.0)
    return {
        "id": "C0",
        "title": "solver validity: converged BIE solves and non-negative damping",
        "threshold": f"max solver relative residual <= {tolerance:g} and min B >= 0",
        "passed": passed,
        "values": {
            "mesh": sweep.label,
            "solver_relative_residual_max": worst,
            "solver_relative_residual_median": float(np.median(residual)) if residual.size else float("nan"),
            "min_radiation_damping": min_damping,
            "n_frequencies": int(sweep.omega.size),
            "note": (
                "C2..C6 read A/B as data; this gate is the precondition that the "
                "data is a solution at all. A FAIL here voids the other gates "
                "instead of reinterpreting them."
            ),
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


def kk_gate(
    sweep: Sweep,
    *,
    corrupt: bool = False,
    window: Sequence[bool] | None = None,
    added_mass_inf: Sequence[float] | None = None,
    identifier: str | None = None,
) -> dict[str, Any]:
    """C3: per-DOF Kramers-Kronig closure of the added mass (the case criterion).

    The residual of a *band-limited* closure cannot be read as a causality
    statement at every sample frequency: the un-solved band ``s > s_max``
    contributes

        ``delta A(omega) = (2/pi) integral_{s_max}^inf B(s)/(s^2-omega^2) ds``,

    which is positive for every ``omega < s_max`` and diverges as
    ``omega -> s_max``.  :func:`tensorbem.causality.high_band_truncation_budget`
    bounds that missing term from the measured local decay exponent of the
    tabulated damping.  A sample therefore counts towards the gate only when the
    bound is below ``TOLERANCES["kk_budget"]`` (half the case tolerance); the
    remaining samples are reported as undecidable instead of as a violation.

    ``added_mass_inf`` supplies the asymptote from outside the band (the
    independent image-method limit).  The closure then spends **no** free
    parameter: with it, the gate asks whether the KK *relation* holds rather
    than whether one constant can absorb the disagreement.  ``identifier``
    relabels the gate (``"C3a"``) so the anchored and fitted verdicts cannot be
    confused in an artifact.

    ``window`` freezes the decidable mask, so the corrupted negative control
    (``corrupt=True``) is judged on exactly the samples where the clean sweep was
    judged -- otherwise a corruption could shrink its own window and escape.
    """
    budget_tolerance = float(TOLERANCES["kk_budget"])
    rows = []
    worst = 0.0
    worst_dof = ""
    decidable_all: list[bool] = []
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
        supplied = (
            None
            if added_mass_inf is None
            else float(np.asarray(added_mass_inf, dtype=float).reshape(-1)[index])
        )
        closure = kk_added_mass_closure(
            sweep.omega,
            mass,
            damper_used,
            damping_tail=tail,
            added_mass_inf="closure" if supplied is None else supplied,
        )
        asymptote = fit_added_mass_asymptote(sweep.omega, mass)
        errors = np.abs(closure.relative_residual)
        index_worst = int(np.argmax(errors)) if errors.size else 0
        max_error = float(closure.max_relative_residual)

        # Truncation budget of the *clean* data: the corruption is a statement
        # about the damping shape, not about how much band is missing, and
        # letting it move the window would make the control unfalsifiable.
        budget = high_band_truncation_budget(
            sweep.omega, damper, closure.sample_omegas
        )
        budget_relative = budget.relative(closure.added_mass_scale)
        decidable = np.isfinite(budget_relative) & (budget_relative <= budget_tolerance)
        if window is not None:
            frozen = np.asarray(list(window), dtype=bool).reshape(-1)
            if frozen.size != decidable.size:
                raise ValueError(
                    f"frozen window has {frozen.size} samples, this closure has "
                    f"{decidable.size}"
                )
            decidable = frozen
        decidable_all.append(bool(np.all(decidable)))
        decided_errors = errors[decidable]
        max_decided = (
            float(np.max(decided_errors)) if decided_errors.size else float("inf")
        )
        exponent, _ = local_tail_exponent(sweep.omega, damper)
        rows.append(
            {
                "dof": int(index),
                "dof_name": name,
                "max_relative_residual": max_error,
                "max_relative_residual_decidable": max_decided,
                "n_decidable": int(np.count_nonzero(decidable)),
                "n_samples": int(decidable.size),
                "measured_tail_exponent": float(exponent),
                "truncation_budget_admissible": bool(budget.admissible),
                "truncation_budget_reason": budget.reason,
                "truncation_budget_max_relative": (
                    float(np.nanmax(budget_relative[np.isfinite(budget_relative)]))
                    if np.any(np.isfinite(budget_relative))
                    else float("inf")
                ),
                "truncation_budget_sample_relative": [
                    float("inf") if not np.isfinite(v) else float(v)
                    for v in budget_relative
                ],
                "decidable_mask": [bool(v) for v in decidable],
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
                "free_parameters": 1 if supplied is None else 0,
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
        if max_decided > worst:
            worst = max_decided
            worst_dof = name
    tolerance = TOLERANCES["kk"] if not corrupt else TOLERANCES["negative_control"]
    # ``raw`` keeps the pre-existing all-sample statistic (comparable with the
    # artifacts published before this rule existed); the gate itself is decided
    # on the samples where the un-solved band cannot explain the residual.
    raw_worst = float(max(row["max_relative_residual"] for row in rows))
    raw_passed = raw_worst <= tolerance
    any_decidable = any(row["n_decidable"] > 0 for row in rows)
    decided_passed = bool(worst <= tolerance and any_decidable)
    if not any_decidable or not math.isfinite(worst):
        # Nothing (or nothing finite) can be certified from the band-limited data
        # alone, so the headline residual falls back to the measured one instead
        # of infinity.
        worst = raw_worst
        worst_dof = max(
            rows, key=lambda row: row["max_relative_residual"]
        )["dof_name"]
    fully_certified = bool(
        all(
            row["n_decidable"] == row["n_samples"] and row["truncation_budget_admissible"]
            for row in rows
        )
    )
    # The pass criterion stays the pre-declared one (max over *all* samples): the
    # truncation budget does not move the threshold, it labels how much of the
    # band the verdict actually certifies.  Reading "undecidable" as a failure
    # would punish the case for the part of the spectrum it never solved.
    if not any_decidable:
        status = (
            "pass, not certified: no sample has a truncation budget below "
            f"{budget_tolerance:.0%} of the added-mass scale -- the measured "
            "damping does not decay fast enough at the top of the band for the "
            "omitted integral above s_max to be bounded, so the agreement holds "
            "empirically but is not implied by band-limited data alone"
        )
    elif fully_certified:
        status = "decided and certified"
    else:
        status = "pass, certified only where the truncation budget is small"
    gate_id = identifier if identifier is not None else ("C3" if not corrupt else "C6b")
    if not corrupt:
        title = (
            "Kramers-Kronig closure of the added mass (6 DOF)"
            if identifier is None
            else "Kramers-Kronig closure of the added mass (6 DOF) with A_inf "
            "supplied by the independent image solve: zero free parameters"
        )
    else:
        title = "negative control: corrupted damping shape must fail the KK gate"
    return {
        "id": gate_id,
        "title": title,
        "threshold": (
            f"max over decidable samples and DOF |A_BEM-A_KK|/max|A_BEM| <= "
            f"{tolerance:.0%} (a sample is decidable when the closed-form bound on "
            f"the omitted s > s_max integral is <= {budget_tolerance:.0%})"
        ),
        "passed": bool(raw_passed) if not corrupt else bool(not raw_passed),
        "decidable_window_passed": (
            bool(decided_passed) if not corrupt else bool(not decided_passed)
        ),
        "fully_certified": fully_certified,
        "raw_gate_passed": bool(raw_passed),
        "status": status,
        "values": {
            "mesh": sweep.label,
            "band": [float(sweep.omega[0]), float(sweep.omega[-1])],
            "worst_dof": worst_dof,
            "worst_relative_residual": worst,
            "worst_relative_residual_all_samples": raw_worst,
            "all_dof_fully_decidable": bool(all(decidable_all)),
            "kk_budget_tolerance": budget_tolerance,
            "per_dof": rows,
            "tail_fit_model": "B(s) = C s^-p on the upper quartile of the sweep",
            "truncation_budget_model": (
                "B continued above s_max as C s^-p with p from the last three "
                "tabulated samples, anchored at the last sample; the omitted "
                "PV integral is then closed in form"
            ),
            "added_mass_inf_estimator": (
                "least-squares closure (one free parameter)"
                if identifier is None
                else "supplied by the independent image solve (zero free parameters)"
            ),
            "free_parameters": 0 if identifier is not None else 1,
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
    min_damping = float(np.min(damper)) if damper.size else float("nan")
    if min_damping < 0.0:
        # ``cummins_free_decay`` (rightly) refuses to build a kernel from a
        # negative damping.  Raising here used to abort the whole case run after
        # the sweep was already on disk, so the artifact directory kept the
        # *previous* JSON while the .npz files were overwritten -- a silent
        # provenance break.  An undecidable gate is the honest outcome; gate C0
        # records the reason.
        return {
            "id": "C5",
            "title": f"Cummins free decay vs frequency-domain damping ratio ({DOF_NAMES[dof]})",
            "threshold": f"|zeta_decay - zeta_fd|/zeta_fd <= {TOLERANCES['free_decay']:.0%}",
            "passed": False,
            "values": {
                "dof": int(dof),
                "dof_name": DOF_NAMES[dof],
                "status": (
                    "undecidable: non-physical negative radiation damping in the sweep "
                    f"(min B = {min_damping:.4g} N s/m); see gate C0"
                ),
                "min_radiation_damping": min_damping,
                "band_rad_s": [float(sweep.omega[0]), float(sweep.omega[-1])],
            },
        }
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


# ---------------------------------------------------------------------------
# independent A_inf anchor (image method)
# ---------------------------------------------------------------------------

_ANCHOR_CACHE: dict[tuple[int, int], dict[str, Any]] = {}


def anchor_self_check(nlat: int = 8, nlon: int = 24) -> dict[str, Any]:
    """Referee the anchor itself against a closed-form case.

    A half-immersed sphere is the one body whose infinite-frequency heave added
    mass is exact: the antisymmetric image closes it into a full sphere driven
    in rigid translation, so ``A_inf(heave) = m_full / 4``.  If the image
    machinery cannot reproduce that within ``TOLERANCES["anchor_analytic"]``,
    its numbers are not evidence about anything else either.
    """
    radius = 1.0
    density = DENSITY
    body = half_immersed_sphere_mesh(int(nlat), int(nlon), radius)
    result = added_mass_limit(body, "infinite_frequency", density=density)
    got = float(np.diag(np.asarray(result.added_mass_matrix, dtype=float))[2])
    exact = half_immersed_sphere_added_mass(radius, density)
    error = abs(got - exact) / exact
    return {
        "body": "half-immersed sphere, R = 1 m, free surface on the equator",
        "exact_rule": "A_inf(heave) = rho * (4/3) pi R^3 / 4",
        "panels": int(body.panel_count),
        "doubled_panels": int(result.doubled_panel_count),
        "added_mass_inf_heave": got,
        "added_mass_inf_exact": exact,
        "relative_error": float(error),
        "tolerance": float(TOLERANCES["anchor_analytic"]),
        "passed": bool(error <= TOLERANCES["anchor_analytic"]),
    }


def added_mass_inf_anchor(
    n_sections: int,
    n_levels: int,
    *,
    self_check: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """``A_inf`` of the six diagonal modes from the independent image solve.

    The guess this replaces is the least-squares offset of :func:`kk_gate`: a
    free parameter fitted to the same band that is being judged.  Here the same
    quantity is obtained from a *different* boundary-value problem (the free
    surface replaced by its high-frequency limit ``phi = 0``, closed exactly by
    the antisymmetric image), so the KK relation can be tested with no free
    parameter at all.
    """
    key = (int(n_sections), int(n_levels))
    cached = _ANCHOR_CACHE.get(key)
    if cached is not None:
        return cached
    mesh = wigley_wetted_hull_mesh(
        LENGTH, BEAM, DRAFT, n_sections=key[0], n_levels=key[1]
    )
    result = added_mass_limit(
        mesh, "infinite_frequency", center=CENTER, density=DENSITY
    )
    diagonal = np.diag(np.asarray(result.added_mass_matrix, dtype=float))
    payload = {
        "method": "antisymmetric image (phi = 0 on the calm waterplane) about z = 0",
        "center_m": list(CENTER),
        "n_sections": key[0],
        "n_levels": key[1],
        "panel_count": int(mesh.panel_count),
        "doubled_panel_count": int(result.doubled_panel_count),
        "doubled_volume_over_hull_volume": float(
            result.doubled_volume / max(abs(result.hull_volume), 1e-300)
        ),
        "condition_number": float(result.condition_number),
        "solver_relative_residual_l2": float(result.max_relative_residual_l2),
        "symmetry_error": float(result.symmetry_error),
        "added_mass_inf_diagonal": [float(value) for value in diagonal],
        "added_mass_inf": {
            name: float(diagonal[index]) for index, name in enumerate(DOF_NAMES)
        },
        "self_check": anchor_self_check() if self_check is None else self_check,
    }
    _ANCHOR_CACHE[key] = payload
    return payload


def anchor_for_sweep(sweep: Sweep, **kwargs: Any) -> dict[str, Any]:
    """Anchor for the mesh the sweep was actually solved on (artifact identity)."""
    return added_mass_inf_anchor(sweep.n_sections, sweep.n_levels, **kwargs)


def added_mass_inf_gate(
    sweep: Sweep,
    anchor: dict[str, Any],
    *,
    closure_rows: Sequence[dict[str, Any]] | None = None,
    identifier: str = "C7",
) -> dict[str, Any]:
    """C7: is the least-squares ``A_inf`` of the band closure physical?

    ``C3`` may spend one free parameter (the ``A_inf`` the KK relation needs but
    a finite band cannot supply).  This gate spends none: it compares that
    fitted parameter with the image-method limit.  Two normalisations are
    reported because they answer different questions -- against the added-mass
    amplitude of the same DOF (comparable with the C3 residual scale, used for
    the verdict) and against the anchor value itself (the estimator error).
    """
    self_check = anchor.get("self_check") or {}
    anchor_values = np.asarray(anchor["added_mass_inf_diagonal"], dtype=float)
    if closure_rows is None:
        closure_rows = kk_gate(sweep)["values"]["per_dof"]
    rows = []
    worst = 0.0
    worst_dof = ""
    for index, name in enumerate(DOF_NAMES):
        scale = float(closure_rows[index]["residual_scale"])
        fitted = float(closure_rows[index]["added_mass_inf_closure"])
        anchored = float(anchor_values[index])
        difference = abs(fitted - anchored)
        relative_scale = difference / max(scale, 1e-300)
        relative_anchor = difference / max(abs(anchored), 1e-300)
        rows.append(
            {
                "dof": int(index),
                "dof_name": name,
                "added_mass_inf_fitted": fitted,
                "added_mass_inf_anchor": anchored,
                "absolute_difference": float(difference),
                "relative_to_added_mass_scale": float(relative_scale),
                "relative_to_anchor": float(relative_anchor),
                "added_mass_scale": scale,
                "passed": bool(relative_scale <= TOLERANCES["anchor"]),
            }
        )
        if relative_scale > worst:
            worst = relative_scale
            worst_dof = name
    anchor_ok = bool(
        self_check.get("passed", False)
        and anchor["solver_relative_residual_l2"] <= TOLERANCES["solver_residual"]
        and math.isfinite(anchor["condition_number"])
    )
    passed = bool(anchor_ok and worst <= TOLERANCES["anchor"])
    if not anchor_ok:
        status = (
            "undecidable: the anchor is not validated (self-check "
            f"{100.0 * float(self_check.get('relative_error', float('nan'))):.2f}% vs "
            f"{100.0 * TOLERANCES['anchor_analytic']:.0f}% bar, solver residual "
            f"{anchor['solver_relative_residual_l2']:.2e})"
        )
    else:
        status = (
            "pass" if passed else "fail: the fitted A_inf is off the physical limit "
            f"by {100.0 * worst:.2f}% of the added-mass amplitude ({worst_dof})"
        )
    return {
        "id": identifier,
        "title": (
            "A_inf anchor: the band-closure least-squares offset vs the independent "
            "image-method limit"
        ),
        "threshold": (
            f"|A_inf(fit) - A_inf(anchor)| / max|A_ii| <= {TOLERANCES['anchor']:.0%} "
            "per DOF; anchor self-check (half-immersed sphere, exact m_full/4) "
            f"<= {TOLERANCES['anchor_analytic']:.0%}"
        ),
        "passed": passed,
        "status": status,
        "values": {
            "mesh": sweep.label,
            "n_sections": int(sweep.n_sections),
            "n_levels": int(sweep.n_levels),
            "worst_dof": worst_dof,
            "worst_relative_to_added_mass_scale": float(worst),
            "worst_relative_to_anchor": float(
                max(row["relative_to_anchor"] for row in rows)
            ),
            "anchor_validated": bool(self_check.get("passed", False)),
            "anchor_self_check": self_check,
            "anchor_condition_number": anchor["condition_number"],
            "anchor_solver_relative_residual_l2": anchor[
                "solver_relative_residual_l2"
            ],
            "anchor_symmetry_error": anchor["symmetry_error"],
            "anchor_method": anchor["method"],
            "anchor_center_m": anchor["center_m"],
            "per_dof": rows,
        },
    }


def _interp_diagonal(
    sweep: Sweep, omega: np.ndarray, dof: int
) -> tuple[np.ndarray, np.ndarray]:
    """Piecewise-linear interpolation of ``A_ii``/``B_ii`` of one DOF to ``omega``."""
    grid = np.asarray(sweep.omega, dtype=float)
    order = np.argsort(grid)
    mass = np.asarray(sweep.added_mass, dtype=float)[order][:, dof, dof]
    damper = np.asarray(sweep.damping, dtype=float)[order][:, dof, dof]
    return (
        np.interp(omega, grid[order], mass),
        np.interp(omega, grid[order], damper),
    )


def extension_consistency_gate(
    coarse_band: Sweep,
    extended: Sweep,
    *,
    identifier: str = "C8",
) -> dict[str, Any]:
    """C8: is the extended band the same physics as the committed sweep?

    Widening the band to ``omega = 20 rad/s`` is only worth anything if the wider
    sweep still reproduces the committed one where the two overlap.  The checking
    mesh is *finer* (3840 panels vs 480), so a disagreement on the shared band is a
    discretisation error of the **committed** sweep, not of the extension -- and if
    it is large the two verdicts cannot be compared at all: the residual that
    ``C3a`` measures would then be dominated by the mesh change instead of by the
    causality of the data.

    The criterion is normalised by the amplitude of each DOF (``max|A_ii|``,
    ``max|B_ii|`` over the shared band), so it is independent of where the DOF
    peaks and of the near-zero samples.
    """
    shared = np.asarray(
        [w for w in coarse_band.omega if extended.omega[0] <= w <= extended.omega[-1]],
        dtype=float,
    )
    if shared.size < 3:
        raise ValueError(
            "the two sweeps do not share at least three frequencies; the "
            "extension consistency check cannot be evaluated"
        )
    rows = []
    worst = 0.0
    worst_dof = ""
    for index, name in enumerate(DOF_NAMES):
        mass_ref, damper_ref = _interp_diagonal(extended, shared, index)
        mass_old, damper_old = _interp_diagonal(coarse_band, shared, index)
        mass_scale = max(float(np.max(np.abs(mass_ref))), 1e-300)
        damper_scale = max(float(np.max(np.abs(damper_ref))), 1e-300)
        mass_diff = np.abs(mass_ref - mass_old) / mass_scale
        damper_diff = np.abs(damper_ref - damper_old) / damper_scale
        rows.append(
            {
                "dof": int(index),
                "dof_name": name,
                "added_mass_relative_difference_max": float(np.max(mass_diff)),
                "damping_relative_difference_max": float(np.max(damper_diff)),
                "added_mass_difference_at_band_top": float(mass_diff[-1]),
                "damping_difference_at_band_top": float(damper_diff[-1]),
                "worst_omega_added_mass": float(shared[int(np.argmax(mass_diff))]),
                "worst_omega_damping": float(shared[int(np.argmax(damper_diff))]),
            }
        )
        local_worst = max(
            float(np.max(mass_diff)), float(np.max(damper_diff))
        )
        if local_worst > worst:
            worst = local_worst
            worst_dof = name
    tolerance = float(TOLERANCES["extension_overlap"])
    passed = bool(worst <= tolerance)
    return {
        "id": identifier,
        "title": (
            "extended band is the same physics: the resolved sweep reproduces the "
            "committed one on the shared band"
        ),
        "threshold": (
            "max over DOF and shared-band samples of |A,B(extended mesh) - "
            f"A,B(committed mesh)| / max|A,B| <= {tolerance:.0%}"
        ),
        "passed": passed,
        "status": (
            "pass: the panel refinement moves the shared band by "
            f"{100.0 * worst:.2f}% or less"
            if passed
            else "fail: the two bands disagree on the shared band by "
            f"{100.0 * worst:.2f}% ({worst_dof}); the verdicts are not comparable"
        ),
        "values": {
            "committed_mesh": {
                "label": coarse_band.label,
                "panels": int(coarse_band.panel_count),
                "band": [float(coarse_band.omega[0]), float(coarse_band.omega[-1])],
            },
            "extended_mesh": {
                "label": extended.label,
                "panels": int(extended.panel_count),
                "band": [float(extended.omega[0]), float(extended.omega[-1])],
            },
            "n_shared_frequencies": int(shared.size),
            "worst_dof": worst_dof,
            "worst_relative_difference": float(worst),
            "tolerance": tolerance,
            "per_dof": rows,
        },
    }


def truncation_decidability_gate(
    coarse_band: Sweep,
    extended: Sweep,
    *,
    closure_rows: Sequence[dict[str, Any]] | None = None,
    reference_rows: Sequence[dict[str, Any]] | None = None,
    identifier: str = "C9",
) -> dict[str, Any]:
    """C9: does the wider band move the damping into algebraic decay?

    ``C3a`` can only certify the samples whose truncation budget is small, and
    that budget is ``inf`` whenever the measured damping slope ``p`` at the band
    top is not ``> 2.05`` (the omitted integral ``int^inf B(s) ds/(s^2-omega^2)``
    does not converge for ``p <= 2``).  On the 0.2..12 rad/s sweep that happens
    for surge, sway and yaw, i.e. three of the six DOF are *not* certified.

    This gate is the audit of the extension itself: per DOF it reports the
    measured slope at the old and at the new band top together with the budget
    that follows, and it refuses to let the extension lose ground -- a DOF that
    was certifiable at 12 rad/s may not become uncertifiable at 20.

    Passing does **not** mean every DOF became certifiable: it means the
    extension was judged and nothing got worse.  Which DOF actually gained is
    reported in ``newly_decidable``; if that list is empty the extension is an
    honest negative result (the horizontal modes are still not in decay).
    """
    from tensorbem.causality import high_band_truncation_budget, local_tail_exponent

    if closure_rows is None:
        closure_rows = kk_gate(extended)["values"]["per_dof"]
    if reference_rows is None:
        reference_rows = kk_gate(coarse_band)["values"]["per_dof"]
    rows = []
    newly: list[str] = []
    lost: list[str] = []
    for index, name in enumerate(DOF_NAMES):
        old = reference_rows[index]
        new = closure_rows[index]
        budget = high_band_truncation_budget(
            extended.omega, extended.damping[:, index, index]
        )
        exponent, _ = local_tail_exponent(
            extended.omega, extended.damping[:, index, index]
        )
        scale = max(float(new["residual_scale"]), 1e-300)
        budget_relative = budget.relative(scale)
        finite = budget_relative[np.isfinite(budget_relative)]
        old_admissible = bool(old["truncation_budget_admissible"])
        new_admissible = bool(budget.admissible)
        if new_admissible and not old_admissible:
            newly.append(name)
        if old_admissible and not new_admissible:
            lost.append(name)
        rows.append(
            {
                "dof": int(index),
                "dof_name": name,
                "exponent_at_12": float(old["measured_tail_exponent"]),
                "exponent_at_20": float(exponent),
                "budget_admissible_at_12": old_admissible,
                "budget_admissible_at_20": new_admissible,
                "budget_max_relative_at_20": (
                    float(np.max(finite)) if finite.size else float("inf")
                ),
                "budget_reason": budget.reason,
                "n_decidable_at_12": int(old["n_decidable"]),
                "n_decidable_at_20": int(new["n_decidable"]),
                "n_samples": int(new["n_samples"]),
                "worst_anchored_residual": float(
                    new["max_relative_residual_decidable"]
                ),
            }
        )
    passed = not lost
    return {
        "id": identifier,
        "title": (
            "band extension 12 -> 20 rad/s: does the damping reach algebraic decay "
            "so the truncation budget becomes finite?"
        ),
        "threshold": (
            "no DOF may lose truncation-budget admissibility (a DOF is certifiable "
            f"only when the measured slope p > 2.05 and the budget <= "
            f"{TOLERANCES['kk_budget']:.0%} of the added-mass scale)"
        ),
        "passed": bool(passed),
        "status": (
            ("widened: " + ", ".join(newly) + " became certifiable")
            if newly and not lost
            else (
                "no widening: " + ", ".join(lost) + " lost certifiability"
                if lost
                else "no widening: no DOF reached algebraic decay by 20 rad/s"
            )
        ),
        "values": {
            "band_at_12": [float(coarse_band.omega[0]), float(coarse_band.omega[-1])],
            "band_at_20": [float(extended.omega[0]), float(extended.omega[-1])],
            "decidable_dof_at_12": int(
                sum(1 for row in reference_rows if row["n_decidable"] > 0)
            ),
            "decidable_dof_at_20": int(
                sum(1 for row in closure_rows if row["n_decidable"] > 0)
            ),
            "newly_decidable": newly,
            "lost_decidability": lost,
            "per_dof": rows,
        },
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
    gates["C0"] = solver_validity_gate(sweep)
    gates["C2"] = panel_adequacy_gate(sweep)
    gates["C3"] = kk_gate(sweep)
    anchor = anchor_for_sweep(sweep)
    gates["C3a"] = kk_gate(
        sweep, added_mass_inf=anchor["added_mass_inf_diagonal"], identifier="C3a"
    )
    gates["C7"] = added_mass_inf_gate(
        sweep, anchor, closure_rows=gates["C3"]["values"]["per_dof"]
    )
    gates["C4"] = round_trip_gate(sweep, dof=int(dof_free_decay))
    gates["C5"] = free_decay_gate(sweep, dof=int(dof_free_decay))
    gates["C6a"] = negative_control_gate(sweep, dof=int(dof_free_decay))
    if corrupted_control:
        gates["C6b"] = kk_gate(
            sweep,
            corrupt=True,
            window=[
                all(row["decidable_mask"][k] for row in gates["C3"]["values"]["per_dof"])
                for k in range(len(gates["C3"]["values"]["per_dof"][0]["decidable_mask"]))
            ],
        )
    return {
        "case": CASE_SLUG,
        "artifact": str(path),
        "added_mass_inf_anchor": anchor,
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
    results["gates"]["C0"] = solver_validity_gate(production)
    if len(meshes) > 1:
        results["gates"]["C0_coarse"] = solver_validity_gate(sweeps[meshes[0]])
    if not results["gates"]["C0"]["passed"]:
        # A sweep that failed C0 is not evidence about the physics, so C2..C6
        # are *voided* rather than evaluated: scoring them would label a failed
        # linear solve "non-causal" (C3) or "undecidable" (C5) and the artifact
        # would hide the real defect behind an ordinary-looking FAIL.
        for key in ("C2", "C3", "C4", "C5", "C6a", "C6b"):
            results["gates"][key] = {
                "id": key,
                "title": f"{key} voided: the sweep failed gate C0",
                "threshold": "not evaluated",
                "passed": False,
                "voided": True,
                "values": {
                    "voided_by": "C0",
                    "note": (
                        "the BIE solve that produced A/B did not converge or gave "
                        "negative damping; re-run the sweep on a validating node"
                    ),
                },
            }
    else:
        results["gates"]["C2"] = panel_adequacy_gate(production)
        results["gates"]["C3"] = kk_gate(production)
        anchor = anchor_for_sweep(production)
        results["added_mass_inf_anchor"] = anchor
        results["gates"]["C3a"] = kk_gate(
            production,
            added_mass_inf=anchor["added_mass_inf_diagonal"],
            identifier="C3a",
        )
        results["gates"]["C7"] = added_mass_inf_gate(
            production, anchor, closure_rows=results["gates"]["C3"]["values"]["per_dof"]
        )
        if len(meshes) > 1:
            results["gates"]["C3_coarse"] = kk_gate(sweeps[meshes[0]])
            coarse_anchor = anchor_for_sweep(sweeps[meshes[0]])
            results["gates"]["C3a_coarse"] = kk_gate(
                sweeps[meshes[0]],
                added_mass_inf=coarse_anchor["added_mass_inf_diagonal"],
                identifier="C3a_coarse",
            )
        results["gates"]["C4"] = round_trip_gate(production, dof=int(dof_free_decay))
        results["gates"]["C5"] = free_decay_gate(production, dof=int(dof_free_decay))
        results["gates"]["C6a"] = negative_control_gate(production, dof=int(dof_free_decay))
        results["gates"]["C6b"] = kk_gate(
            production,
            corrupt=True,
            window=[
                all(
                    row["decidable_mask"][k]
                    for row in results["gates"]["C3"]["values"]["per_dof"]
                )
                for k in range(
                    len(results["gates"]["C3"]["values"]["per_dof"][0]["decidable_mask"])
                )
            ],
        )
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
    if "C3a" in gates and "C7" in gates:
        lines.append(
            "### 3c. A_inf anchor (independent image solve) and the "
            "zero-free-parameter closure"
        )
        lines.append("")
        lines.append(
            "`C3` estimates `A_inf` by least squares on the same band it judges "
            "(one free parameter).  `C3a` supplies it from the antisymmetric-image "
            "solve instead, so the KK relation is tested with no free parameter; "
            "`C7` is the same comparison read the other way round -- how far off "
            "the fitted parameter is."
        )
        lines.append("")
        lines.append(
            "| DOF | A_inf (fit) | A_inf (anchor) | abs. diff | diff / max\\|A\\| | "
            "diff / \\|A_inf(anchor)\\| | KK residual, A_inf fitted (C3) | "
            "KK residual, A_inf anchored (C3a) |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for fitted_row, anchored_row, anchor_row in zip(
            gates["C3"]["values"]["per_dof"],
            gates["C3a"]["values"]["per_dof"],
            gates["C7"]["values"]["per_dof"],
        ):
            lines.append(
                f"| {fitted_row['dof_name']} | "
                f"{anchor_row['added_mass_inf_fitted']:.6g} | "
                f"{anchor_row['added_mass_inf_anchor']:.6g} | "
                f"{anchor_row['absolute_difference']:.3g} | "
                f"{100.0*anchor_row['relative_to_added_mass_scale']:.2f}% | "
                f"{100.0*anchor_row['relative_to_anchor']:.2f}% | "
                f"{100.0*fitted_row['max_relative_residual']:.2f}% | "
                f"{100.0*anchored_row['max_relative_residual']:.2f}% |"
            )
        lines.append("")
        self_check = gates["C7"]["values"]["anchor_self_check"]
        lines.append(
            "Anchor self-check (half-immersed sphere, exact `A_inf(heave) = "
            f"m_full/4`): {self_check['panels']} panels, error "
            f"{100.0*self_check['relative_error']:.2f}% vs the "
            f"{100.0*self_check['tolerance']:.0f}% bar -> "
            f"{'PASS' if self_check['passed'] else 'FAIL'}."
        )
        lines.append("")
        lines.append(
            f"Anchored closure worst DOF: **{gates['C3a']['values']['worst_dof']}** at "
            f"{100.0*gates['C3a']['values']['worst_relative_residual']:.2f}% "
            f"(criterion {100.0*TOLERANCES['kk']:.0f}%)."
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
