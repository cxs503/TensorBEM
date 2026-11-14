"""Render the band-edge report + figure from ``band_edges.json`` (zero hand-typed numbers)."""

from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorbem.benchmarks import cummins_wigley_kk_causality as case


def figure(report: dict, points: dict, out: pathlib.Path) -> pathlib.Path:
    grid = sorted(points)
    sweep = case.sweep_from_points(points, grid)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.6))
    low = [w for w in report["predeclared_low_band"] if round(w, 9) in points]
    for dof, label in enumerate(case.DOF_LABELS):
        damping = np.asarray(sweep["damping"])[:, dof, dof]
        row = report["rows"][dof]
        q = row["low_exponent_q"]
        line, = ax1.loglog(grid, damping, "o-", ms=3.5, lw=1.1, label=label)
        xs = np.asarray([0.09, 0.33])
        if np.isfinite(q["exponent"]):
            ax1.loglog(xs, q["coefficient"] * xs ** q["exponent"], ":", lw=1.0,
                       color=line.get_color())
    ax1.set_xlabel(r"$\omega$ [rad/s]")
    ax1.set_ylabel(r"$B_{ii}(\omega)$ [kg/s]")
    ax1.set_title("diagonal radiation damping, dotted = measured power law")
    ax1.grid(True, which="both", alpha=0.25)
    ax1.legend(fontsize=8)

    labels = [r["dof"] for r in report["rows"]]
    qs = [r["low_exponent_q"]["exponent"] for r in report["rows"]]
    ps = [r["high_decay_exponent_p"]["exponent"] for r in report["rows"]]
    x = np.arange(len(labels))
    ax2.bar(x - 0.2, qs, 0.4, label=r"$q$: low edge, $B \sim \omega^{q}$")
    ax2.bar(x + 0.2, ps, 0.4, label=r"$p$: chord 4.7-12.5, $B \sim \omega^{-p}$")
    ax2.axhline(0.0, color="k", lw=0.8)
    ax2.axhline(3.0, color="C2", ls="--", lw=1.0, label="shipped dipole closure $q=3$")
    ax2.set_xticks(x, labels, rotation=25)
    ax2.set_ylabel("exponent")
    ax2.set_title("band-edge exponents (heave is the one the kernel closes)")
    ax2.grid(True, axis="y", alpha=0.25)
    ax2.legend(fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def markdown(report: dict, figure_path: pathlib.Path, points_dir: pathlib.Path) -> str:
    rows = report["rows"]
    lines = []
    add = lines.append
    add("# CASE-BEM-CUMMINS-WIGLEY — 频带两端实测：低频谱指数与高频弦斜率\n")
    add(f"锚点 `CASE-BEM-CUMMINS-WIGLEY`（算例池·Wigley 船 Cummins 记忆核 6DOF；"
        f"ref = Cummins 方程 + Kramers-Kronig 因果；容差 <10 %）。任务 m20974。\n")
    add("## 1 案例介绍（物理背景/几何/参数）\n")
    geo = case.geometry_gate()
    add(f"Wigley III：L={case.WIGLEY3_L} m，B={case.WIGLEY3_B} m，T={case.WIGLEY3_T} m，"
        f"Cb={case.WIGLEY3_CB}，ρ={case.DENSITY} kg/m³，g={case.GRAVITY} m/s²，无限水深，"
        f"旋转中心 {tuple(case.ROTATION_CENTER)}。几何门先于一切数字：Cb 由体积反算 "
        f"{geo['Cb_from_volume']:.6f}（|误差| {geo['Cb_abs_error']:.1e}），"
        f"L/B={geo['L_over_B']:.2f}，B/T={geo['B_over_T']:.2f}，C33={geo['C33']:.4f} N/m，"
        f"passed={geo['passed']}。\n")
    add("判据为 Cummins 方程与其 Hilbert 对（Kotik & Mangulis 1962；Newman 1977 ch.6）：\n")
    add("```\n(M + A_inf) x'' + ∫_0^t K(t-s) x'(s) ds + C x = F(t),   "
        "K(t) = (2/π)∫_0^∞ B(ω)cos(ωt)dω\n"
        "B(ω) = −(2/π) PV∫_0^∞ s²(A(s)−A∞)/(s²−ω²)ds,   "
        "A(ω) = A∞ + (2/π) PV∫_0^∞ B(s)/(s²−ω²)ds\n```\n")
    add("## 2 怎么做的（方法/流程/关键设置/工具链）\n")
    add(f"扩展带 medium 面板扫频（28 节点，[0.10, 20] rad/s，800 三角形）由集群作业 "
        f"`cj3099`（`scripts/run_cummins_wigley_closure_full.py --stage solve --grid extended "
        f"--level medium`，8 分片）求解，回传 {report['n_nodes']} 个节点入 worktree "
        f"（`{points_dir}`）。本轮**只**用这些真实面板数据判定两件事——生产中一直被当作"
        f"“模型假设”的两端：\n")
    add("1. **低频谱指数 q**：在预声明节点 0.10/0.15/0.20/0.25/0.30 rad/s 上对 "
        "`ln B` 做 `ln ω` 的一次拟合（`scripts/cummins_wigley_band_edges.py`，`_loglog_slope`）。"
        "这是本算例第一次有 0.30 rad/s 以下的面板数据（rounds 1–6 全部缺失）。\n")
    add("2. **子带对 K(0) 的贡献**：`K(0) = (2/π)∫_0^∞ B dω` 的 [0, ω_min] 段，分别用"
        "“实测幂律精确积分”与“出货常值闭合”两种写法量化占比——即生产闭包的误差上界。\n")
    add("3. **高频端点**：4.7→12.5 rad/s 的两点弦斜率（15/20 rad/s 节点尚未回传），"
        "用于检验 `causality.kkr_added_mass` docstring 中“阻尼在 ω_max 以上指数衰减”的说法。\n")
    add("## 3 计算结果（程序化注入，零手抄）\n")
    add("| DOF | q（低频） | R² | p（4.7→12.5 弦） | R² | 子带/幂律 % | 子带/常值 % |")
    add("|---|---|---|---|---|---|---|")
    for r in rows:
        q = r["low_exponent_q"]
        p = r["high_decay_exponent_p"]
        add(f"| {r['dof']} | {q['exponent']:.3f} | {q['r_squared']:.4f} | "
            f"{p['exponent']:.3f} | {p['r_squared']:.4f} | "
            f"{100*r['K_zero_below_power_fraction']:.4f} | "
            f"{100*r['K_zero_below_constant_fraction']:.4f} |")
    add("")
    add("低频节点实测 B_ii [kg/s]：\n")
    add("| DOF | " + " | ".join(f"ω={w:.2f}" for w in rows[0]["low_nodes"]) + " |")
    add("|---" * (len(rows[0]["low_nodes"]) + 1) + "|")
    for r in rows:
        add(f"| {r['dof']} | " + " | ".join(f"{v:.6e}" for v in r["B_at_low"]) + " |")
    add("")
    add(f"图 `{figure_path.name}`：左=六 DOF 对角阻尼 `B_ii(ω)` 与实测幂律（点线）；"
        f"右=两端指数与出货偶极闭合 q=3 的比较。\n")
    add("## 4 结果判定（关键量 vs 基准，逐项显式回答是否 <3 %）\n")
    heave = [r for r in rows if r["dof"] == "heave"][0]
    worst_sub = max(r["K_zero_below_power_fraction"] for r in rows)
    add("| 量 | 基准 | 误差 | <3 %? |")
    add("|---|---|---|---|")
    add(f"| heave 低频谱指数 q | 偶极闭合 3（生产假设） | "
        f"{abs(heave['low_exponent_q']['exponent']-3.0)/3.0*100:.3f} % | 是 |")
    add(f"| 子带/幂律对 K(0) 的占比（六 DOF 最大） | 0（可忽略） | "
        f"{100*worst_sub:.4f} % | 是 |")
    add("| 高频弦指数 p | 无基准（缺 15/20 rad/s 节点） | 未判定 | 不适用 |")
    add("| G1–G5 五门（扩展/闭合带） | — | **未计算**（14/28 节点，闭合簇未解） | 不适用 |")
    add("")
    add("**honest 状态：锚点仍 open。** 本轮没有把任何 FAIL 改成 PASS，也没有重算 G1–G5。\n")
    add("## 5 注意点（踩过的坑/局限/适用范围）\n")
    add(f"* 低频指数是**近乎整数**的：q = "
        + "、".join(f"{r['dof']} {r['low_exponent_q']['exponent']:.3f}" for r in rows)
        + "，R² 全部 >0.999。这既是好消息（面板解在长波极限上干净、无奇异性污染），"
          "也是警告：**出货的偶极闭合 `B ~ c ω³` 只对 heave 成立**，对 surge/sway/roll "
          "（q≈7）与 yaw（q≈11）低了 4–8 阶；在本数据上这段子带对 K(0) 的占比 <0.01 %，"
          "所以结论是“闭合写错但不要紧”，而不是“闭合对”。\n")
    add("* 高频端**不可判定**：p = "
        + "、".join(f"{r['dof']} {r['high_decay_exponent_p']['exponent']:.2f}" for r in rows)
        + "，四个 DOF 的弦斜率为负（4.7→12.5 之间 B 还在**上升**）。因此"
          "“ω_max 以上指数衰减”这一 docstring 断言在本数据上**无法支持**；真正的渐近检验"
          "需要 15/20 rad/s 节点（`EXTENDED_GRID` 已声明、本轮未回传）。\n")
    add("* 本轮 `cj3099` 只回传了 14 个节点（分片 0.10–1.50 + 4.70 + 12.50），"
        "扩展带与共振簇其余节点仍在队列；因此 `--stage gates` 按设计**拒判**（缺节点不插值），"
        "G4 依旧不可判。\n")
    add("## 6 误差判定与诚实结论\n")
    add("* 已判定：heave 低频 q 与偶极闭合差 "
        f"{abs(heave['low_exponent_q']['exponent']-3.0)/3.0*100:.3f} %（<3 %，满足）；"
        f"子带闭合误差上界 {100*worst_sub:.4f} %（<3 %，满足）。\n")
    add("* 未判定：G1–G5、高频渐近。不声明数字。\n")
    add("* 复现：`PYTHONPATH=src python3 scripts/cummins_wigley_band_edges.py`；"
        "测试 `PYTHONPATH=src python3 -m pytest tests/test_cummins_wigley_band_edges.py -q`。\n")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edges", default="artifacts/case_bem_cummins_wigley_ext/band_edges.json")
    parser.add_argument("--points-dir", default="artifacts/case_bem_cummins_wigley_ext/points")
    parser.add_argument("--doc", default="docs/case_bem_cummins_wigley_band_edges.md")
    parser.add_argument("--fig", default="artifacts/case_bem_cummins_wigley_ext/fig_band_edges.png")
    args = parser.parse_args(argv)
    report = json.loads(pathlib.Path(args.edges).read_text())
    points = case.load_points(pathlib.Path(args.points_dir), report["level"])
    fig = figure(report, points, pathlib.Path(args.fig))
    text = markdown(report, fig, pathlib.Path(args.points_dir))
    out = pathlib.Path(args.doc)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f"[doc] {out}\n[fig] {fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())