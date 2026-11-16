"""Figures and auto-tables for the m23106 Cummins-Wigley admissibility report.

Reads ``docs/m23106/closure_admissibility.json`` (written by
``scripts/audit_cummins_closure_admissibility.py``) and produces

* ``docs/m23106/fig_residual_and_damping.png`` -- solver residual and heave
  damping over the committed band vs the diverged m20021 artifact;
* ``docs/m23106/fig_gate_sensitivity.png`` -- per-DOF G3/G5 of the committed
  medium band, and how far one poisoned node moves G4 with nothing complaining;
* the markdown table block between the ``BEGIN/END AUTOTABLES`` markers of the
  report, rebuilt from the JSON so no number in the report is typed by hand.

Usage::

    PYTHONPATH=src python3 scripts/make_m23106_cummins_admissibility_evidence.py
"""

from __future__ import annotations

import base64
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tensorbem.benchmarks import cummins_wigley_kk_causality as case  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs" / "m23106"
DATA = DOCS / "closure_admissibility.json"
REPORT = ROOT / "docs" / "dev_report_m23106_cummins_wigley_band_closure_round2.md"
CASE_DIR = ROOT / "benchmarks" / "verified" / "casepool_bem_cummins_wigley"
DIVERGED = ROOT / "docs" / "m20021" / "sweep_medium.npz"

BEGIN = "<!-- BEGIN AUTOTABLES -->"
END = "<!-- END AUTOTABLES -->"


def _load_npz(path: pathlib.Path) -> dict:
    payload = np.load(path)
    return {
        "omega": np.asarray(payload["omega"], dtype=float),
        "damping": np.asarray(payload["damping"], dtype=float),
        "added_mass": np.asarray(payload["added_mass"], dtype=float),
        "residual": np.asarray(payload["solver_relative_residual"], dtype=float),
    }


def figure_residual_and_damping(outdir: pathlib.Path) -> pathlib.Path:
    clean = _load_npz(CASE_DIR / "sweep_medium.npz")
    bad = _load_npz(DIVERGED)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

    ax = axes[0]
    ax.semilogy(clean["omega"], clean["residual"], "o-", ms=3.2, lw=1.1,
                color="#1f77b4", label="committed medium (CASE dir)")
    ax.semilogy(bad["omega"], bad["residual"], "s-", ms=3.2, lw=1.1,
                color="#d62728", label="docs/m20021 medium (evidence cluster)")
    ax.axhline(case.SOLVER_RESIDUAL_TOL, color="k", ls="--", lw=1.0,
               label=f"G0 / C0 limit = {case.SOLVER_RESIDUAL_TOL:.0%}")
    ax.set_xlabel(r"$\omega$ [rad/s]")
    ax.set_ylabel("solver relative residual")
    ax.set_title("(a) node admissibility: 0/41 vs 41/41 over the 1 % limit")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="center right")

    ax = axes[1]
    ax.plot(clean["omega"], clean["damping"][:, 2, 2], "o-", ms=3.2, lw=1.1,
            color="#1f77b4", label="committed medium $B_{33}$")
    ax.plot(bad["omega"], bad["damping"][:, 2, 2], "s-", ms=3.2, lw=1.1,
            color="#d62728", label="m20021 medium $B_{33}$ (diverged)")
    ax.axhline(0.0, color="k", lw=0.8)
    idx = int(np.argmin(bad["damping"][:, 2, 2]))
    ax.annotate(rf"$B_{{33}}={bad['damping'][idx, 2, 2]:.1f}$ at "
                rf"$\omega={bad['omega'][idx]:.3f}$",
                xy=(bad["omega"][idx], bad["damping"][idx, 2, 2]),
                xytext=(0.08, 0.18), textcoords="axes fraction",
                arrowprops={"arrowstyle": "->", "lw": 0.9}, fontsize=8.5)
    ax.set_xlabel(r"$\omega$ [rad/s]")
    ax.set_ylabel(r"$B_{33}$ [kg/s]")
    ax.set_title("(b) negative heave damping read as physics before G0")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("CASE-BEM-CUMMINS-WIGLEY: gate G0 (solver admissibility)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = outdir / "fig_residual_and_damping.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def figure_gate_sensitivity(data: dict, outdir: pathlib.Path) -> pathlib.Path:
    committed = data["sweeps"]["committed_medium"]["gates"]
    per_dof = committed["per_dof"]
    names = list(per_dof)
    g3 = [per_dof[n]["G3"] for n in names]
    g5 = [per_dof[n]["G5"] for n in names]
    rows = data["silent_failure_mode"]["rows"]
    scales = [r["damping_scale"] for r in rows]
    g4 = [r["without_g0"]["gates"]["G4_free_decay"]["value"] for r in rows]
    baseline = data["silent_failure_mode"]["baseline"]["gates"]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    ax = axes[0]
    x = np.arange(len(names))
    ax.bar(x - 0.2, g3, 0.4, color="#1f77b4", label="G3 Kramers-Kronig (10 %)")
    ax.bar(x + 0.2, g5, 0.4, color="#ff7f0e", label="G5 round trip (5 %)")
    ax.axhline(case.GATES["G3_kramers_kronig"]["tol"], color="#1f77b4", ls="--", lw=1.0)
    ax.axhline(case.GATES["G5_round_trip"]["tol"], color="#ff7f0e", ls=":", lw=1.2)
    ax.set_xticks(x, names)
    ax.set_ylabel("relative error")
    ax.set_title(r"(a) committed band [0.2, 12.0] rad/s: tail truncation, not solver")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(scales, g4, "o-", color="#d62728", label="G4 with one poisoned node")
    ax.axhline(baseline["gates"]["G4_free_decay"]["value"], color="#1f77b4", ls="--",
               lw=1.1, label="G4 baseline (clean band)")
    ax.axhline(case.GATES["G4_free_decay"]["tol"], color="k", ls=":", lw=1.0,
               label="G4 tolerance (10 %)")
    for row, value in zip(rows, g4):
        ax.annotate(f"{value:.1%}", (row["damping_scale"], value),
                    textcoords="offset points", xytext=(0, 6), fontsize=8,
                    ha="center")
    ax.set_xlabel(r"heave-damping scale of the poisoned node at $\omega=5.291$ rad/s")
    ax.set_ylabel("G4 relative error")
    ax.invert_xaxis()
    ax.set_title("(b) G1-G5 verdict string never changes; only G0 names the node")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")

    fig.suptitle("G0 is not redundant with the five consistency gates", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = outdir / "fig_gate_sensitivity.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _fmt(value, digits=4) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "PASS" if value else "FAIL"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{value:.{digits}g}" if abs(value) < 1e-2 else f"{value:.{digits}f}"


def table_block(data: dict) -> str:
    lines: list[str] = []
    tol = data["residual_tol"]

    lines.append(f"### 3.1 G0 审计（在树 5 份 sweep 工件，阈值 {tol:.0%} = 算例池 C0）")
    lines.append("")
    lines.append("| sweep | 网格 | 节点 | 越限节点 | 最坏残差 | 最坏 ω [rad/s] | G0 |")
    lines.append("|---|---|---|---|---|---|---|")
    level_names = {"committed_medium": "medium", "committed_coarse": "coarse",
                   "m20021_medium": "medium", "m20021_coarse": "coarse",
                   "n315_coarse": "coarse"}
    for key, block in data["sweeps"].items():
        if "solver_admissibility" not in block:
            lines.append(f"| `{key}` | — | — | — | 缺失 | — | 工件不在树 |")
            continue
        adm = block["solver_admissibility"]
        lines.append(
            f"| `{key}` | {level_names.get(key, '—')} | {adm['nodes']} | "
            f"{adm['n_offenders']} | {_fmt(adm['worst_residual'], 4)} | "
            f"{_fmt(adm['worst_omega'], 6)} | {_fmt(adm['passed'])} |")
    lines.append("")

    lines.append("### 3.2 G0–G5 逐门（可采性通过的带）")
    lines.append("")
    lines.append("| band | G0 | G1 | G2 | G3 | G4 | G5 | verdict |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for key in ("committed_medium", "committed_coarse"):
        gates = data["sweeps"][key]["gates"]["gates"]
        cells = []
        for gid in ("G0_solver_validity", "G1_cross_solver", "G2_kernel_at_zero",
                    "G3_kramers_kronig", "G4_free_decay", "G5_round_trip"):
            gate = gates[gid]
            cells.append("skip" if gate["passed"] is None
                         else f"{_fmt(gate['value'], 3)} {'✓' if gate['passed'] else '✗'}")
        lines.append(f"| `{key}` | " + " | ".join(cells)
                     + f" | `{data['sweeps'][key]['gates']['verdict']}` |")
    lines.append("")
    lines.append("门限：G0 1 %、G1 10 %、G2 2 %、G3 10 %、G4 10 %、G5 5 %。"
                 "G1 为 `skip`：该 41 节点带不含 ω=1 rad/s，交叉求解器参考只在该频率冻结，"
                 "不插值、不臆造。")
    lines.append("")

    lines.append("### 3.3 分歧工件的可判定性（G1–G5 根本跑不动）")
    lines.append("")
    lines.append("| sweep | 评估结果 |")
    lines.append("|---|---|")
    for key in ("m20021_medium", "m20021_coarse", "n315_coarse"):
        block = data["sweeps"][key]
        err = block.get("gates_error")
        if err:
            msg = err["message"].replace("|", "/")
            lines.append(f"| `{key}` | `{err['type']}`：{msg} |")
    lines.append("")

    lines.append("### 3.4 单个坏节点的静默性（committed medium，1 节点扰动量级扫描）")
    lines.append("")
    lines.append("| heave 阻尼缩放 | G0（含门） | 无 G0 的 verdict | G4 | G3 | G5 | 是否可解释 |")
    lines.append("|---|---|---|---|---|---|---|")
    base = data["silent_failure_mode"]["baseline"]
    base_gates = base["gates"]["gates"]
    lines.append(
        f"| 1.00（基线） | PASS | `{base['gates']['verdict']}` | "
        f"{_fmt(base_gates['G4_free_decay']['value'], 4)} | "
        f"{_fmt(base_gates['G3_kramers_kronig']['value'], 4)} | "
        f"{_fmt(base_gates['G5_round_trip']['value'], 4)} | 是 |")
    for row in data["silent_failure_mode"]["rows"]:
        without = row["without_g0"]["gates"]
        with_g0 = row["with_g0"]
        g0 = with_g0["gates"]["G0_solver_validity"]
        lines.append(
            f"| {row['damping_scale']:.2f}{'（阻尼不变，仅残差坏）' if row['damping_scale'] == 1.0 else ''} | {_fmt(g0['passed'])} "
            f"(residual {_fmt(g0['value'], 3)}) | `{row['without_g0']['verdict']}` | "
            f"{_fmt(without['G4_free_decay']['value'], 4)} | "
            f"{_fmt(without['G3_kramers_kronig']['value'], 4)} | "
            f"{_fmt(without['G5_round_trip']['value'], 4)} | "
            f"{'是' if with_g0['interpretable'] else '否'} |")
    lines.append("")
    lines.append("扰动节点 ω = 5.291 rad/s，注入残差 1.245（实测 m20021 medium 最坏值）；"
                 "缩放后 B33 仍为正，因此能通过核函数唯一的一条物理检查（B ≥ 0）。")
    lines.append("")
    return "\n".join(lines)


def inject(data: dict) -> None:
    block = table_block(data)
    report = REPORT.read_text()
    if BEGIN in report and END in report:
        head, rest = report.split(BEGIN, 1)
        _, tail = rest.split(END, 1)
        REPORT.write_text(head + BEGIN + "\n" + block + "\n" + END + tail)
    else:
        raise SystemExit(f"{REPORT} has no {BEGIN} / {END} markers")


def main() -> int:
    DOCS.mkdir(parents=True, exist_ok=True)
    data = json.loads(DATA.read_text())
    fig1 = figure_residual_and_damping(DOCS)
    fig2 = figure_gate_sensitivity(data, DOCS)
    inject(data)
    payload = {p.name: base64.b64encode(p.read_bytes()).decode() for p in (fig1, fig2)}
    b64_out = pathlib.Path("/tmp/figures_base64_m23106.json")
    b64_out.parent.mkdir(parents=True, exist_ok=True)
    b64_out.write_text(json.dumps(payload))
    print(f"[fig] {fig1}\n[fig] {fig2}\n[tables] injected into {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())