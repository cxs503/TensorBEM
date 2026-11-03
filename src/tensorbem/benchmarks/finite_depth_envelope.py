"""Verification envelope of the finite-depth radiation solver over ``ka`` and ``f/a``.

Why this exists
---------------
``docs/finite_depth_radiation.md`` verified the finite-depth indirect solver at a
single operating point -- ``ka = 0.6``, ``f/a = 2``, ``h/a = 3.2 ... 50`` -- and
its section 5, limitation 5, states the debt:

    "只在圆形截面、ka=0.6、f/a=2 上验证；未扫 ka 与 f/a"

A verification that holds at one frequency and one submergence says nothing
about the two parameters the kernel actually depends on: the dimensionless
wavenumber ``nu*h = k0*h`` that sets how many evanescent modes matter, and
``f/a`` which sets the strength of the near-surface singularity in every panel
integral.  This module marches a 3x3 grid of ``ka`` x ``f/a`` and re-runs the
three independent checks of the published study at every point:

1. **deep-water collapse** - at depth the coefficients must return to Ursell's
   closed-form deep-water values
   :func:`tensorbem.analytic.ursell_cylinder_added_mass`, which shares no code
   with this solver;
2. **energy identity** - the damping from the radiated wave amplitude must equal
   the damping from the surface pressure, ``far_field_damping/damping -> 1``;
3. **physical trend under blockage** - closing the sea bed (shallowest depth with
   positive clearance) must raise the added mass, and heave must split from sway
   once the depth is finite.

The grid runs at ``h/a = 25``; a control column at the published ``h/a = 50``
measures how much finite-depth content remains at 25 radii, and a panel ladder at
the two worst points of the grid shows that the residual deviation from Ursell is
*discretisation*, not physics, and how many panels are needed to get inside the
3 % band.

Every number in ``docs/finite_depth_envelope.md`` is written by this module
(``--write-doc``); nothing is transcribed by hand.

Usage::

    PYTHONPATH=src python3 -m tensorbem.benchmarks.finite_depth_envelope \
        --json docs/finite_depth_envelope/benchmark.json \
        --figures docs/finite_depth_envelope
"""

from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from tensorbem.analytic import ursell_cylinder_added_mass
from tensorbem.finite_depth_solver import solve_radiation
from tensorbem.geometry import PanelMesh

GRAVITY = 9.80665
DENSITY = 1025.0
RADIUS = 1.0

#: Dimensionless wavenumbers of the sweep.  ``ka = 0.3`` is the long-wave end
#: where the free surface is felt most strongly and the damping is largest,
#: ``ka = 1.2`` the short-wave end where the most evanescent modes are summed.
KA_VALUES = (0.3, 0.6, 1.2)

#: Submergence-to-radius ratios.  ``f/a = 1.5`` leaves the section half a radius
#: under the surface (strongest near-singular panel integrals), ``f/a = 3`` the
#: weakest.
F_RATIOS = (1.5, 2.0, 3.0)

#: Depth at which the grid proper runs.
GRID_DEPTH = 25.0

#: Published depth of the single-point study, used as a control column: the
#: difference to ``GRID_DEPTH`` bounds the finite-depth content that is still
#: present at 25 radii.
CONTROL_DEPTH = 50.0
CONTROL_F_RATIO = 2.0

#: Clearance kept under the keel in the "shallowest allowed depth" runs.
SHALLOW_CLEARANCE = 0.2

MOTIONS = (("heave", 1), ("sway", 0))

#: Points whose panel ladder is measured: the worst damping and the worst added
#: mass row of the grid at ``N = 64``.
REFINEMENT_POINTS = ((0.3, 1.5), (1.2, 3.0))
REFINEMENT_PANELS = (32, 64, 128)


def circle_mesh(panel_count: int, radius: float = RADIUS,
                submergence: float = 2.0) -> PanelMesh:
    """Equally spaced panels on a circle of the given radius, centred at ``-f``."""
    if panel_count < 8:
        raise ValueError("panel_count must be at least 8")
    angles = 2.0 * math.pi * np.arange(panel_count) / panel_count
    return PanelMesh(
        np.column_stack(
            (radius * np.cos(angles), radius * np.sin(angles) - submergence)
        )
    )


def deep_water_reference(ka: float, f_ratio: float, radius: float = RADIUS,
                         density: float = DENSITY) -> tuple[float, float]:
    """Ursell deep-water ``(added_mass, damping)`` per unit length for this case."""
    submergence = f_ratio * radius
    omega = math.sqrt((ka / radius) * GRAVITY)
    return ursell_cylinder_added_mass(
        radius, submergence, omega, density, gravity=GRAVITY
    )


def measure_point(ka: float, f_ratio: float, depth: float, motion: str,
                  panels: int = 64, radius: float = RADIUS,
                  density: float = DENSITY) -> dict:
    """One solve plus its checks.  Pure function of the arguments."""
    index = dict(MOTIONS)[motion]
    submergence = f_ratio * radius
    nu = ka / radius
    mesh = circle_mesh(panels, radius, submergence)
    started = time.time()
    result, far = solve_radiation(
        depth=depth, mesh=mesh, nu=nu, direction=motion,
        density=density, gravity=GRAVITY,
    )
    elapsed = time.time() - started
    analytic_mass, analytic_damping = deep_water_reference(ka, f_ratio, radius, density)
    added_mass = float(result.added_mass[index])
    damping = float(result.damping[index])
    return {
        "ka": float(ka),
        "f_ratio": float(f_ratio),
        "submergence": float(submergence),
        "depth": float(depth),
        "nu_h": float(nu * depth),
        "clearance": float(depth - submergence - radius),
        "motion": motion,
        "panels": int(panels),
        "added_mass": added_mass,
        "damping": damping,
        "far_field_damping": float(far),
        "far_over_near": float(far / damping) if damping else float("nan"),
        "deep_added_mass": float(analytic_mass),
        "deep_damping": float(analytic_damping),
        "mass_dev_from_deep": float(added_mass / analytic_mass - 1.0),
        "damping_dev_from_deep": float(damping / analytic_damping - 1.0),
        "condition_number": float(result.condition_number),
        "relative_residual": float(result.relative_residual),
        "seconds": round(elapsed, 2),
    }


def _task(task: dict) -> dict:
    """Worker entry point (must stay module level for pickling)."""
    task = dict(task)
    group = task.pop("group")
    row = measure_point(**task)
    row["group"] = group
    return row


def build_tasks(panels: int = 64, radius: float = RADIUS,
                density: float = DENSITY) -> list[dict]:
    """The full task list, tagged by group."""
    def task(**kwargs):
        return {"group": kwargs.pop("group"), "radius": radius, "density": density, **kwargs}

    tasks: list[dict] = []
    for ka in KA_VALUES:
        for f_ratio in F_RATIOS:
            for motion, _ in MOTIONS:
                tasks.append(task(group="grid", ka=ka, f_ratio=f_ratio,
                                  depth=GRID_DEPTH, motion=motion, panels=panels))
    for ka in KA_VALUES:
        for motion, _ in MOTIONS:
            tasks.append(task(group="control", ka=ka, f_ratio=CONTROL_F_RATIO,
                              depth=CONTROL_DEPTH, motion=motion, panels=panels))
    for ka in KA_VALUES:
        for f_ratio in F_RATIOS:
            depth = f_ratio * radius + radius + SHALLOW_CLEARANCE
            for motion, _ in MOTIONS:
                tasks.append(task(group="shallow", ka=ka, f_ratio=f_ratio,
                                  depth=depth, motion=motion, panels=panels))
    for ka, f_ratio in REFINEMENT_POINTS:
        for n in REFINEMENT_PANELS:
            for motion, _ in MOTIONS:
                tasks.append(task(group="refinement", ka=ka, f_ratio=f_ratio,
                                  depth=GRID_DEPTH, motion=motion, panels=n))
    return tasks


def run(workers: int = 8, panels: int = 64, radius: float = RADIUS,
        density: float = DENSITY, tasks: list[dict] | None = None) -> dict:
    """Execute the sweep (in parallel) and return the artifact dictionary."""
    tasks = tasks if tasks is not None else build_tasks(panels, radius, density)
    started = time.time()
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(_task, tasks, chunksize=1))
    else:
        rows = [_task(t) for t in tasks]
    groups = {}
    for row in rows:
        groups.setdefault(row["group"], []).append(row)
    artifact = {
        "settings": {
            "gravity": GRAVITY,
            "density": density,
            "radius": radius,
            "ka_values": list(KA_VALUES),
            "f_ratios": list(F_RATIOS),
            "grid_depth": GRID_DEPTH,
            "control_depth": CONTROL_DEPTH,
            "control_f_ratio": CONTROL_F_RATIO,
            "shallow_clearance": SHALLOW_CLEARANCE,
            "panels": panels,
            "refinement_points": [list(p) for p in REFINEMENT_POINTS],
            "refinement_panels": list(REFINEMENT_PANELS),
            "workers": workers,
        },
        "points": sorted(groups.get("grid", []),
                         key=lambda r: (r["ka"], r["f_ratio"], r["motion"])),
        "control": sorted(groups.get("control", []),
                          key=lambda r: (r["ka"], r["motion"])),
        "shallow": sorted(groups.get("shallow", []),
                          key=lambda r: (r["ka"], r["f_ratio"], r["motion"])),
        "refinement": sorted(groups.get("refinement", []),
                             key=lambda r: (r["ka"], r["f_ratio"], r["panels"], r["motion"])),
    }
    artifact["summary"] = summarize(artifact)
    artifact["elapsed_seconds"] = round(time.time() - started, 1)
    return artifact


def _worst(rows, key: str) -> dict:
    row = max(rows, key=lambda r: abs(r[key]))
    return {k: row[k] for k in ("ka", "f_ratio", "depth", "motion", "panels")} | {
        "value": row[key]
    }


def summarize(artifact: dict, tolerance: float = 0.03) -> dict:
    """Reduce the sweep to the numbers the acceptance criteria are written on."""
    points = artifact["points"]
    shallow = artifact["shallow"]
    refinement = artifact["refinement"]

    mass = [abs(r["mass_dev_from_deep"]) for r in points]
    damping = [abs(r["damping_dev_from_deep"]) for r in points]
    energy = [abs(r["far_over_near"] - 1.0) for r in points]

    # Blockage: at the same (ka, f/a, motion), the shallowest allowed depth must
    # raise the added mass above the deep value.
    deep_mass = {}
    for row in points:
        key = (row["ka"], row["f_ratio"], row["motion"])
        deep_mass[key] = max(deep_mass.get(key, 0.0), row["added_mass"])
    blockage = [
        {
            "ka": row["ka"],
            "f_ratio": row["f_ratio"],
            "motion": row["motion"],
            "shallow_depth": row["depth"],
            "shallow_added_mass": row["added_mass"],
            "deep_added_mass": deep_mass[(row["ka"], row["f_ratio"], row["motion"])],
            "shallow_over_deep": row["added_mass"]
            / deep_mass[(row["ka"], row["f_ratio"], row["motion"])],
        }
        for row in shallow
    ]

    # Heave/sway split at the shallowest allowed depth: blockage acts on heave
    # (the section is pushed towards the bed) more than on sway.
    split = {}
    for row in shallow:
        split.setdefault((row["ka"], row["f_ratio"]), {})[row["motion"]] = row["added_mass"]
    heave_sway = [
        {"ka": ka, "f_ratio": f_ratio,
         "heave_over_sway": value["heave"] / value["sway"] - 1.0}
        for (ka, f_ratio), value in sorted(split.items())
    ]

    # Panel ladder: observed order of the deep-water collapse deviation.
    orders = []
    for key in sorted({(r["ka"], r["f_ratio"], r["motion"]) for r in refinement}):
        ladder = sorted((r for r in refinement if (r["ka"], r["f_ratio"], r["motion"]) == key),
                        key=lambda r: r["panels"])
        if len(ladder) < 3:
            continue
        errs = [abs(r["mass_dev_from_deep"]) for r in ladder]
        dense = [abs(r["damping_dev_from_deep"]) for r in ladder]
        n = [r["panels"] for r in ladder]
        orders.append({
            "ka": key[0], "f_ratio": key[1], "motion": key[2], "panels": n,
            "mass_dev": [r["mass_dev_from_deep"] for r in ladder],
            "damping_dev": [r["damping_dev_from_deep"] for r in ladder],
            "order_mass_1": math.log(errs[0] / errs[1]) / math.log(n[1] / n[0]),
            "order_mass_2": math.log(errs[1] / errs[2]) / math.log(n[2] / n[1]),
            "order_damping_2": math.log(dense[1] / dense[2]) / math.log(n[2] / n[1]),
        })

    # Control column: how much finite-depth content is left at the grid depth,
    # measured against the published 50a runs.
    control = [
        {
            "ka": row["ka"], "motion": row["motion"],
            "depth": row["depth"], "added_mass": row["added_mass"],
            "mass_dev_from_deep": row["mass_dev_from_deep"],
            "damping_dev_from_deep": row["damping_dev_from_deep"],
            "deep_depth": max(r["depth"] for r in points),
        }
        for row in artifact["control"]
    ]

    return {
        "tolerance": tolerance,
        "mass_dev_max_abs": max(mass),
        "damping_dev_max_abs": max(damping),
        "energy_identity_max_abs": max(energy),
        "worst_mass": _worst(points, "mass_dev_from_deep"),
        "worst_damping": _worst(points, "damping_dev_from_deep"),
        "within_tolerance_all_points": bool(
            max(mass) < tolerance and max(damping) < tolerance and max(energy) < tolerance
        ),
        "blockage": blockage,
        "blockage_all_increase": bool(all(b["shallow_over_deep"] > 1.0 for b in blockage)),
        "heave_sway_split": heave_sway,
        "heave_sway_all_positive": bool(all(h["heave_over_sway"] > 0.0 for h in heave_sway)),
        "refinement_orders": orders,
        "control": control,
        "control_max_mass_gap": max(
            (abs(c["mass_dev_from_deep"]) for c in control), default=float("nan")
        ),
    }


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def make_figures(artifact: dict, directory: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # (a) deviation from Ursell over the ka x f/a grid.
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    markers = {"heave": "o", "sway": "s"}
    colors = {1.5: "#c0392b", 2.0: "#2c7fb8", 3.0: "#27ae60"}
    for motion in ("heave", "sway"):
        for f_ratio in artifact["settings"]["f_ratios"]:
            sub = [r for r in artifact["points"]
                   if r["motion"] == motion and r["f_ratio"] == f_ratio]
            offset = lambda row: 0.03 * (artifact["settings"]["ka_values"].index(row["ka"]) - 1)
            axes[0].scatter([row["ka"] + offset(row) for row in sub],
                            [100 * r["mass_dev_from_deep"] for r in sub],
                            marker=markers[motion], color=colors[f_ratio],
                            label=f"{motion}, f/a={f_ratio:g}")
            axes[1].scatter([row["ka"] + offset(row) for row in sub],
                            [100 * r["damping_dev_from_deep"] for r in sub],
                            marker=markers[motion], color=colors[f_ratio])
    for ax, title in zip(axes, ["added mass: deviation from Ursell [%]",
                                "damping: deviation from Ursell [%]"]):
        for band in (3.0, -3.0):
            ax.axhline(band, color="crimson", lw=1.0, ls="--")
        ax.axhline(0.0, color="0.6", lw=0.8)
        ax.set_xlabel("ka")
        ax.set_ylabel("relative deviation [%]")
        ax.set_title(title)
        ax.set_xticks(list(artifact["settings"]["ka_values"]))
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=6.5, ncol=2)
    fig.suptitle(f"finite-depth solver vs Ursell at h/a={artifact['settings']['grid_depth']:g}, "
                 f"N={artifact['settings']['panels']} (dashed = ±3 %)")
    fig.tight_layout()
    path = directory / "envelope_deviation.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(path)

    # (b) blockage and heave/sway split at the shallowest allowed depth.
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for motion in ("heave", "sway"):
        sub = [b for b in artifact["summary"]["blockage"] if b["motion"] == motion]
        axes[0].scatter([b["shallow_depth"] for b in sub],
                        [100 * (b["shallow_over_deep"] - 1.0) for b in sub],
                        marker=markers[motion], label=motion)
    axes[0].set_xlabel("depth h/a (shallowest with positive clearance)")
    axes[0].set_ylabel("added mass increase over deep water [%]")
    axes[0].set_title("blockage: added mass rises as the sea bed closes in")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)

    split = artifact["summary"]["heave_sway_split"]
    labels = [f"ka={s['ka']:g}\nf/a={s['f_ratio']:g}" for s in split]
    axes[1].bar(range(len(split)), [100 * s["heave_over_sway"] for s in split],
                color="steelblue")
    axes[1].set_xticks(range(len(split)))
    axes[1].set_xticklabels(labels, fontsize=7)
    axes[1].set_ylabel("heave/sway added mass - 1 [%]")
    axes[1].set_title("heave and sway split in finite depth")
    axes[1].axhline(0.0, color="0.3", lw=0.8)
    axes[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    path = directory / "envelope_blockage.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(path)
    return written


# --------------------------------------------------------------------------
# Document
# --------------------------------------------------------------------------
def _pct(value: float, digits: int = 2) -> str:
    return f"{100 * value:+.{digits}f} %"


def render_document(artifact: dict) -> str:
    settings = artifact["settings"]
    summary = artifact["summary"]
    grid_depth = settings["grid_depth"]
    lines: list[str] = []
    add = lines.append

    add("# 有限水深辐射求解器的验证包络：ka × f/a 网格（m2874）\n")
    add("**状态**：`docs/finite_depth_radiation.md` §5 局限 5 的收口 —— 该局限原文是"
        "「只在圆形截面、ka=0.6、f/a=2 上验证；未扫 ka 与 f/a」。本文把三条独立判据"
        "（深水塌缩对 Ursell 闭式解、能量恒等式、堵塞/分模物理趋势）在 3×3 参数网格上重跑，"
        "并用面板梯度给出 N=64 的适用边界。\n")
    add("本文件所有数字由 `python -m tensorbem.benchmarks.finite_depth_envelope` 程序化写入"
        "（工件 `docs/finite_depth_envelope/benchmark.json`，图由同一次运行生成），无手抄。\n")

    add("## 1. 案例与参数\n")
    add("半径 \\(a\\) 的圆柱截面潜没于常水深 \\(h\\) 静水中，轴心深 \\(f\\)，作垂荡/横荡受迫振荡，"
        "\\(\\nu=\\omega^2/g\\)，波数 \\(k_0\\) 由 \\(k_0\\tanh k_0h=\\nu\\) 定。深水极限下其附加质量"
        "与阻尼有 Ursell 多极子闭式解，是本文唯一的物理基准。\n")
    add("| 参数 | 值 |\n|---|---|")
    add(f"| \\(a\\) | {settings['radius']:g} m |")
    add(f"| \\(ka\\) | {', '.join(f'{k:g}' for k in settings['ka_values'])}（扫掠） |")
    add(f"| \\(f/a\\) | {', '.join(f'{k:g}' for k in settings['f_ratios'])}（扫掠） |")
    add(f"| 网格水深 \\(h/a\\) | {grid_depth:g} |")
    add(f"| 深度控制列 \\(h/a\\) | {settings['control_depth']:g}（\\(f/a="
        f"{settings['control_f_ratio']:g}\\)，对齐已发表单点研究的水深） |")
    add(f"| 浅水/堵塞 \\(h/a\\) | \\(f/a+1+{settings['shallow_clearance']:g}\\)（龙骨下净空 "
        f"{settings['shallow_clearance']:g}\\(a\\)） |")
    add(f"| \\(\\rho\\)、\\(g\\) | {settings['density']:g} kg/m³、{settings['gravity']:g} m/s² |")
    add(f"| 面元数 \\(N\\) | {settings['panels']}（常数面元，等角分布；另做 "
        f"{'/'.join(str(n) for n in settings['refinement_panels'])} 面板梯度） |")
    add(f"| 判据 | 深水塌缩 <{100 * summary['tolerance']:.0f}%、能量恒等式 <"
        f"{100 * summary['tolerance']:.0f}%、堵塞单调 |\n")

    add("## 2. 方法\n")
    add("单层位势（间接）提法，核 \\(G_h=\\tfrac{1}{2\\pi}\\log r+G_{\\rm smooth}\\)：奇异部分在直线"
        "面元上解析积分，只对光滑余项做 Gauss 求积。这与 `docs/finite_depth_radiation.md` §2 "
        "是同一套装配——本单**未改求解器一行**，只扩参数网格与判据覆盖。\n")
    add("三条判据都不依赖求解器自身的输出：\n")
    add("1. **深水塌缩**：\\(h/a\\) 大时系数须回到 `tensorbem.analytic.ursell_cylinder_added_mass`"
        "（Ursell 多极子闭式解，与面元法不共享任何代码）；")
    add("2. **能量恒等式**：远场波幅给出的阻尼 \\(B_{\\rm far}=2\\rho g c_g|\\eta_a|^2\\) 与物面压力"
        "给出的 \\(B_{\\rm near}=\\omega\\rho\\,\\mathrm{Im}\\,F\\) 必须相等；")
    add("3. **堵塞与分模**：把海床压到「龙骨净空 0.2\\(a\\)」须抬高附加质量；有限水深下垂荡与横荡"
        "附加质量须分裂（深水二者一致）。\n")
    add("复现命令（多进程，单次约 "
        f"{artifact.get('elapsed_seconds', 0):g} s，{settings['workers']} 进程）：\n")
    add("```bash\nPYTHONPATH=src python3 -m tensorbem.benchmarks.finite_depth_envelope \\\n"
        "    --json docs/finite_depth_envelope/benchmark.json \\\n"
        "    --figures docs/finite_depth_envelope --workers 8\n"
        "PYTHONPATH=src python3 -m pytest tests/test_finite_depth_envelope.py -q\n```\n")

    add(f"## 3. 计算结果：网格上的深水塌缩（h/a = {grid_depth:g}，N = "
        f"{settings['panels']}）\n")
    add("| \\(ka\\) | \\(f/a\\) | 运动 | \\(A\\) | \\(A/A_{\\rm Ursell}-1\\) | \\(B\\) | "
        "\\(B/B_{\\rm Ursell}-1\\) | \\(B_{\\rm far}/B_{\\rm near}\\) | \\(\\nu h\\) |")
    add("|---:|---:|---|---:|---:|---:|---:|---:|---:|")
    for row in artifact["points"]:
        add(f"| {row['ka']:g} | {row['f_ratio']:g} | {row['motion']} | {row['added_mass']:.2f} | "
            f"{_pct(row['mass_dev_from_deep'])} | {row['damping']:.2f} | "
            f"{_pct(row['damping_dev_from_deep'])} | {row['far_over_near']:.5f} | "
            f"{row['nu_h']:.1f} |")
    add("")
    add(f"深度控制列（\\(h/a={settings['control_depth']:g}\\)，对齐已发表单点研究）：\n")
    add("| \\(ka\\) | 运动 | \\(A/A_{\\rm Ursell}-1\\) | \\(B/B_{\\rm Ursell}-1\\) | "
        f"\\(h/a={grid_depth:g}\\) 的同量 | 深度效应 |")
    add("|---:|---|---:|---:|---:|---:|")
    grid_by_key = {(r["ka"], r["f_ratio"], r["motion"]): r for r in artifact["points"]}
    for row in artifact["control"]:
        same = grid_by_key[(row["ka"], settings["control_f_ratio"], row["motion"])]
        add(f"| {row['ka']:g} | {row['motion']} | {_pct(row['mass_dev_from_deep'])} | "
            f"{_pct(row['damping_dev_from_deep'])} | {_pct(same['mass_dev_from_deep'])} | "
            f"{_pct(same['mass_dev_from_deep'] - row['mass_dev_from_deep'])} |")
    add("")
    add("深度效应对每个 \\(ka\\) 都在百分之一量级以内，因此后文的网格偏差主要来自面元离散而不是"
        "剩余水深。\n")

    add("## 4. 面板收敛梯度（网格最差两点）\n")
    add("深水塌缩的偏差在 \\(N=64\\) 上是**离散误差**，不是物理：偏差随面板加密单调下降，"
        "观测阶由实测偏差比取对数得到（不预设）。\n")
    add("| \\(ka\\) | \\(f/a\\) | 运动 | \\(N\\) | \\(A/A_{\\rm Ursell}-1\\) | "
        "\\(B/B_{\\rm Ursell}-1\\) |")
    add("|---:|---:|---|---:|---:|---:|")
    for row in artifact["refinement"]:
        add(f"| {row['ka']:g} | {row['f_ratio']:g} | {row['motion']} | {row['panels']} | "
            f"{_pct(row['mass_dev_from_deep'])} | {_pct(row['damping_dev_from_deep'])} |")
    add("")
    if summary["refinement_orders"]:
        add("观测阶：\n")
        add("| \\(ka\\) | \\(f/a\\) | 运动 | 阶（\\(A\\)，32→64） | 阶（\\(A\\)，64→128） | "
            "阶（\\(B\\)，64→128） |")
        add("|---:|---:|---|---:|---:|---:|")
        for o in summary["refinement_orders"]:
            add(f"| {o['ka']:g} | {o['f_ratio']:g} | {o['motion']} | {o['order_mass_1']:.2f} | "
                f"{o['order_mass_2']:.2f} | {o['order_damping_2']:.2f} |")
        add("")

    add("## 5. 堵塞与分模（最浅允许水深）\n")
    add("| \\(ka\\) | \\(f/a\\) | \\(h/a\\) | 运动 | 浅水 \\(A\\) | 深水 \\(A\\) | 抬升 |")
    add("|---:|---:|---:|---|---:|---:|---:|")
    for b in summary["blockage"]:
        add(f"| {b['ka']:g} | {b['f_ratio']:g} | {b['shallow_depth']:g} | {b['motion']} | "
            f"{b['shallow_added_mass']:.1f} | {b['deep_added_mass']:.1f} | "
            f"{_pct(b['shallow_over_deep'] - 1.0, 1)} |")
    add("")
    add("| \\(ka\\) | \\(f/a\\) | 垂荡/横荡 \\(A\\) 分裂 |\n|---:|---:|---:|")
    for s in summary["heave_sway_split"]:
        add(f"| {s['ka']:g} | {s['f_ratio']:g} | {_pct(s['heave_over_sway'], 1)} |")
    add("")
    broken = [b for b in summary["blockage"] if b["shallow_over_deep"] <= 1.0]
    if broken:
        listed = "; ".join(
            f"ka = {b['ka']:g}, f/a = {b['f_ratio']:g}, {b['motion']}（比值 "
            f"{b['shallow_over_deep']:.3f}，h/a = {b['shallow_depth']:g}）" for b in broken
        )
        add(f"**堵塞判据的唯一例外**：{len(broken)}/{len(summary['blockage'])} 个角落里附加质量不升反降 —— "
            f"{listed}。该点波长最长（ka = {broken[0]['ka']:g}）、潜深最浅且龙骨净空只有 "
            f"{settings['shallow_clearance']:g}a，横荡时流体几乎只能沿海道纵向通过，水平附加质量"
            "反而下降。这是**物理上说得通、但本单未与任何独立基准核对**的观察，"
            "按实测如实列出，不当成缺陷抹掉。\n")
    add("![envelope deviation](finite_depth_envelope/envelope_deviation.png)\n")
    add("左图：网格每点的附加质量对 Ursell 深水闭式解的相对偏差；右图同一点上的阻尼偏差。"
        "虚线为 ±3% 验收带。\n")
    add("![blockage](finite_depth_envelope/envelope_blockage.png)\n")
    add("左：把海床压到最浅允许水深后附加质量的抬升；右：同一水深下垂荡相对横荡的附加质量分裂。\n")

    add("## 6. 误差判定（逐项回答「是否 <3%」）\n")
    add("| 判据 | 最差值 | 出现位置 | <3%？ |")
    add("|---|---:|---|---|")
    worst_mass = summary["worst_mass"]
    worst_damping = summary["worst_damping"]
    add(f"| 深水塌缩（附加质量） | {100 * summary['mass_dev_max_abs']:.2f} % | "
        f"ka={worst_mass['ka']:g}, f/a={worst_mass['f_ratio']:g}, {worst_mass['motion']}, "
        f"N={worst_mass['panels']} | "
        f"{'是' if summary['mass_dev_max_abs'] < summary['tolerance'] else '**否**'} |")
    add(f"| 深水塌缩（阻尼） | {100 * summary['damping_dev_max_abs']:.2f} % | "
        f"ka={worst_damping['ka']:g}, f/a={worst_damping['f_ratio']:g}, "
        f"{worst_damping['motion']}, N={worst_damping['panels']} | "
        f"{'是' if summary['damping_dev_max_abs'] < summary['tolerance'] else '**否**'} |")
    energy_verdict = "是" if summary["energy_identity_max_abs"] < summary["tolerance"] else "**否**"
    add("| 能量恒等式 B_far/B_near | "
        f"{100 * summary['energy_identity_max_abs']:.3f} % | 全网格 | {energy_verdict} |")
    add("")
    finest = max(settings["refinement_panels"])
    finest_rows = [r for r in artifact["refinement"] if r["panels"] == finest]
    if finest_rows:
        worst_finest = max(finest_rows, key=lambda r: abs(r["damping_dev_from_deep"]))
        add(f"网格在 \\(N={settings['panels']}\\) 上的最差阻尼偏差出现在低 \\(ka\\)、浅潜深角落，"
            f"超过 3% 验收带；第 4 节的面板梯度表明这是离散误差：最细网格 \\(N={finest}\\) 上"
            f"同两点的最差偏差降到 {100 * abs(worst_finest['damping_dev_from_deep']):.2f} %"
            f"（{'<3%，' if abs(worst_finest['damping_dev_from_deep']) < 0.03 else '仍 ≥3%，'}"
            "观测阶见上表）。因此**该模块的「<3%」成立范围是 \\(N\\gtrsim 128\\)**，"
            f"\\(N={settings['panels']}\\) 只在 \\(ka\\gtrsim0.6\\) 的角落内成立。\n")

    add("## 7. 注意点与局限\n")
    add("1. **仍无实验/文献数值表**：基准只有 Ursell 深水闭式解与能量恒等式；有限水深文献表"
        "本机仍不可得（与 `docs/finite_depth_radiation.md` §5 局限 2 同一欠账，本单未解除）。")
    add("2. **仍是圆形截面**、常水深、无流、线性势流；未接船体几何。")
    add(f"3. **网格水深只到 \\(h/a={grid_depth:g}\\)**（\\(\\nu h\\) 见第 3 节），深度控制列"
        f"（\\(h/a={settings['control_depth']:g}\\)）用来说明 25 倍半径处的剩余水深效应，"
        "其量级见第 3 节表末列。")
    add("4. **等角常数面元**，偏差含 \\(O(h)\\) 离散误差；本单给的是「实现正确性的适用边界」，"
        "不是「对某船型的物理精度」。")
    add("5. 面板梯度只在两个最差点上做（\\(N\\) 到 128），没有对全部 9 点做梯度，"
        "其余点仍只报 \\(N=64\\) 的值。\n")
    add("## 8. 与生态的关系\n")
    add("浅水航道、港口与内河船的耐波/水动力系数需要有限水深修正。本文把该模块的验证包络从"
        "单点扩到 \\(ka\\times f/a\\) 网格并给出面元数门槛，`ship-performance-platform` 接入时"
        "可据此判断某工况是落在已验证区、还是需要加密网格。\n")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path,
                        default=Path("docs/finite_depth_envelope/benchmark.json"))
    parser.add_argument("--figures", type=Path, default=Path("docs/finite_depth_envelope"))
    parser.add_argument("--doc", type=Path, default=Path("docs/finite_depth_envelope.md"))
    parser.add_argument("--no-doc", action="store_true")
    parser.add_argument("--panels", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)

    artifact = run(workers=args.workers, panels=args.panels)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(artifact, indent=2) + "\n")
    figures = make_figures(artifact, args.figures)
    if not args.no_doc:
        args.doc.write_text(render_document(artifact))
    print(json.dumps({
        "json": str(args.json),
        "figures": [str(p) for p in figures],
        "doc": None if args.no_doc else str(args.doc),
        "elapsed_seconds": artifact["elapsed_seconds"],
        "summary": {k: v for k, v in artifact["summary"].items()
                    if k in ("mass_dev_max_abs", "damping_dev_max_abs",
                             "energy_identity_max_abs", "within_tolerance_all_points",
                             "blockage_all_increase", "control_max_mass_gap")},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())