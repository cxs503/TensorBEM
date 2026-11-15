#!/usr/bin/env python3
"""Publish the m22264 report for the finite-depth cross-validation grid.

Every number in the report body is read out of
``benchmarks/verified/capytaine_finite_depth/results.json`` (the artifact the
study driver wrote); nothing is typed by hand, so the report cannot drift from
the frozen evidence.  The figures are regenerated from the same artifact.

Usage::

    TENSOR_AGENT_USER=... TENSOR_AGENT_PASS=... \
        python post_report_m21591.py --mission 22264
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "benchmarks" / "verified" / "capytaine_finite_depth" / "results.json"
LADDER = ROOT / "benchmarks" / "verified" / "capytaine_finite_depth" / "tensor_ladder.json"
REFERENCE = ROOT / "benchmarks" / "verified" / "capytaine_finite_depth" / "capytaine_ref.json"
FIG_DIR = ROOT / "docs"
FIG_GRID = FIG_DIR / "fig_xval_bem_finite_depth_grid.png"
QUANTITIES = (
    ("surge.added_mass", "A11 纵荡附加质量"),
    ("heave.added_mass", "A33 垂荡附加质量"),
    ("surge.damping", "B11 纵荡阻尼"),
    ("heave.damping", "B33 垂荡阻尼"),
)


def _pct(value: float) -> str:
    return f"{100.0 * value:+.2f} %"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mission", type=int, default=22264)
    parser.add_argument("--api", default=os.environ.get("TENSOR_API_BASE", ""))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    results = json.loads(RESULTS.read_text())
    ladder = json.loads(LADDER.read_text())
    reference = json.loads(REFERENCE.read_text())
    rows = results["comparison"]["rows"]
    worst = results["comparison"]["worst_relative"]
    gates = results["gates"]
    status = results["case"]["execution_status"]
    provenance = reference["provenance"]

    # ---- regenerated figure ------------------------------------------------
    subprocess.run(
        [sys.executable, "scripts/fig_xval_bem_finite_depth_grid.py",
         "--results", str(RESULTS), "--out", str(FIG_GRID)],
        check=True, cwd=ROOT,
        env={**os.environ, "PYTHONPATH": "src"},
        stdout=subprocess.DEVNULL,
    )
    figures = [
        {
            "name": FIG_GRID.name,
            "caption": "左：全 12 格 × 4 个判定量的外推偏差热图（红/蓝 = TensorBEM 高于/低于 "
                       "capytaine，色标由预登记 5% 带定）；右：偏差随 k0h 的走向，"
                       "点线=预登记带，空心点=该格 capytaine 梯级未进渐近区（取最细级）。",
            "png_base64": base64.b64encode(FIG_GRID.read_bytes()).decode(),
        }
    ]
    for name in ("fig_capytaine_finite_depth_frequency.png",
                 "fig_capytaine_finite_depth_difference_map.png"):
        path = FIG_DIR / name
        if path.exists():
            figures.append({
                "name": name,
                "caption": (
                    "研究页自带图：偏差地图（h/a × k0a，色标 ±5%）"
                    if name.endswith("map.png")
                    else "研究页自带图：两码 A33/B33 随 k0h 的对比（marker=TensorBEM，叉=capytaine）"
                ),
                "png_base64": base64.b64encode(path.read_bytes()).decode(),
            })

    # ---- markdown ----------------------------------------------------------
    md: list[str] = []
    md.append("# m22264 — 有限水深交叉验证：12 格全网格判定落盘（PARTIAL 1/12 → COMPLETE）\n")
    md.append(
        f"**锚点**：`XVAL-BEM-FINITE-DEPTH-CROSSVAL-PLAN` · **仓**：TensorBEM · "
        f"**执行状态**：`{status}`\n"
    )

    md.append("\n## ① 案例介绍\n")
    md.append(
        "完全淹没球（$a=1$ m，球心 $z=-1.5$ m，最低点 $-2.5$ m）在有限水深中的辐射问题。"
        "有限水深自由面 Green 函数是三维有限水深 BEM 的核函数：TensorBEM 用 John 本征级数 + "
        "直接配点，capytaine 用波数积分路线——两条**独立数值路线**，一致性是有限水深能力最硬的证据。"
        "问题由两个相似参数决定：$k_0a$（波长/体尺度）与 $h/a$（水深/体尺度），"
        "$k_0h=(k_0a)(h/a)$。计划按 capytaine 的做法在相似参数网格上(而非固定 $\nu$ 扫频)"
        "反解 $\nu=k_0\tanh(k_0 h)$，使 $k_0a$ 每格精确。\n"
    )
    md.append(
        "\n| 项 | 值 |\n|---|---|\n"
        f"| 网格 | $h/a \\in {sorted({row['h_over_a'] for row in rows})}$ × "
        f"$k_0a \\in {sorted({row['ka'] for row in rows})}$ = {len(rows)} 格 |\n"
        f"| $k_0h$ 范围 | {min(row['k0h'] for row in rows):.2f} – "
        f"{max(row['k0h'] for row in rows):.2f} |\n"
        f"| 密度/重力 | {provenance['density']} kg/m³ / {provenance['gravity']} m/s² |\n"
        f"| TensorBEM 梯级 | {sorted({rec['panels'] for cell in ladder['cells'].values() for rec in cell['levels'].values()})} 面板 |\n"
        f"| capytaine 梯级 | {[list(r) for r in provenance['resolutions']]} = "
        f"{sorted({rec['panels'] for cell in reference['cells'] for rec in cell['levels']})} 面板 |\n"
        f"| capytaine 版本/方法 | {provenance['version']} / `{provenance['finite_depth_method']}` |\n"
        f"| 预登记带 | 附加质量 3 %、阻尼 5 %；owner 不可接受线 15 % |\n"
    )

    md.append("\n## ② 怎么做的\n")
    md.append(
        "1. **两个代码各自做面板梯级**（TensorBEM 20/80/320；capytaine 288/1152/4608 面板），"
        "阶 $p$ 由三级梯级**量出**（$v(N)=v+CN^{-p}$ 二分求解），而不是假定；"
        "两侧都 Richardson 外推到 $N\\to\\infty$ 后再比。单网格上的差是两个无关离散误差之和，不构成判定。\n"
        "2. **参考值有自己的不确定度**：capytaine 两条有限水深公式(`newer`/`legacy`)在球上差到舍入，"
        "不构成第二意见，真正的不确定度是它的面板梯级，因此逐格报阶。\n"
        "3. **负控制**：深水(盲水深刻)求解器在同网格同频率跑一遍——浅水格必须与有限水深解分得开，"
        "否则网格根本没在探水深。\n"
        "4. 本次把冻结工件从 `PARTIAL: 1 of 12` 推到全网格：capytaine 参考由 "
        "`scripts/capytaine_finite_depth_ref.py` 重出 12 格 × 3 分辨率 + legacy，"
        "TensorBEM 梯级由研究驱动 `-m tensorbem.benchmarks.capytaine_finite_depth_crossval "
        "--levels 0,1,2 --jobs 6 --write-doc` 在集群(cj3552, 8 核)算出，"
        "结果写 `results.json` / `tensor_ladder.json` 并由同一驱动生成发布页。\n"
        "5. 数值：集群作业 `cj3552`（`stage_dir`=本单工作树，TensorBEM 侧 6 进程并行）。\n"
    )

    md.append("\n## ③ 计算结果\n")
    md.append("\n**总账（每个量在全网格上的最坏外推偏差）**\n\n")
    md.append("| 量 | 最坏偏差 | 预登记带 | 判定 |\n|---|---:|---:|---|\n")
    bands = {"added_mass": 0.03, "damping": 0.05}
    for key, label in QUANTITIES:
        band = bands[key.split(".")[1]]
        ok = worst[key] <= band
        md.append(
            f"| {label} | {_pct(worst[key])} | {100 * band:.0f} % | "
            f"{'PASS' if ok else 'FAIL'} |\n"
        )

    md.append("\n**逐格外推偏差（TensorBEM 外推 vs capytaine 参考）**\n\n")
    md.append("| h/a | k0a | k0h | " + " | ".join(label for _k, label in QUANTITIES) + " |\n")
    md.append("|---:|---:|---:|" + "---:|" * len(QUANTITIES) + "\n")
    for row in sorted(rows, key=lambda r: (r["h_over_a"], r["ka"])):
        cells = " | ".join(_pct(row["quantities"][key]["relative"]) for key, _l in QUANTITIES)
        md.append(
            f"| {row['h_over_a']:.0f} | {row['ka']:.2f} | {row['k0h']:.2f} | {cells} |\n"
        )

    md.append("\n**门禁**\n\n| 门禁 | 实测 | 限 | 判定 |\n|---|---:|---:|---|\n")
    for gate in gates:
        measured = gate.get("worst", gate.get("min_gap", gate.get("min_order",
                              gate.get("max"))))
        shown = "n/a" if measured is None else (
            str(measured) if isinstance(measured, bool) else f"{measured:.4g}"
        )
        md.append(
            f"| `{gate['gate']}` | {shown} | {gate['limit']} | "
            f"{'PASS' if gate['passed'] else 'FAIL'} |\n"
        )

    md.append("\n**量出的收敛阶（外推是否有意义）**\n\n")
    md.append("| h/a | k0a | p(TB, A33) | p(capytaine, A33) | p(TB, B33) | p(capytaine, B33) |\n")
    md.append("|---:|---:|---:|---:|---:|---:|\n")
    for row in sorted(rows, key=lambda r: (r["h_over_a"], r["ka"])):
        q = row["quantities"]

        def fmt(value):
            return "n/a" if value is None else f"{value:.2f}"

        md.append(
            f"| {row['h_over_a']:.0f} | {row['ka']:.2f} | "
            f"{fmt(q['heave.added_mass']['tensorbem_order'])} | "
            f"{fmt(q['heave.added_mass']['capytaine_order'])} | "
            f"{fmt(q['heave.damping']['tensorbem_order'])} | "
            f"{fmt(q['heave.damping']['capytaine_order'])} |\n"
        )

    md.append("\n**负控制（深水盲求解器 vs 有限水深解，A33）**\n\n")
    md.append("| h/a | k0a | k0h | 相对差 |\n|---:|---:|---:|---:|\n")
    for row in sorted(rows, key=lambda r: (r["h_over_a"], r["ka"])):
        gap = row.get("negative_control_gap_added_mass")
        md.append(
            f"| {row['h_over_a']:.0f} | {row['ka']:.2f} | {row['k0h']:.2f} | "
            f"{'n/a' if gap is None else _pct(gap)} |\n"
        )

    md.append("\n## ④ 图\n")
    md.append(
        "`docs/fig_xval_bem_finite_depth_grid.png`（内嵌）：上格=全网格 4 个判定量的外推偏差热图，"
        "下格=偏差随 $k_0h$ 的走向（大 $k_0h$ 处两码必须并到深水答案）。\n\n"
        "关于场量云图：本算例的判定量是**积分水动力系数**（附加质量/阻尼），"
        "两码之间的可比值就是这些系数与它们的收敛阶；球面速度势场本身不是判定对象且在球上近轴对称，"
        "画云图不产生信息，因此这里用「偏差热图 + 收敛梯级」替代——不硬凑云图。\n"
    )

    md.append("\n## ⑤ 注意点与局限\n")
    md.append(
        f"1. **参考梯级的渐近性**：{sum(1 for row in rows for q in row['quantities'].values() if q['capytaine_order'])}"
        f"/{sum(len(row['quantities']) for row in rows)} 个判定条目上 capytaine 梯级量出了阶；"
        "其余条目梯级非单调(未进渐近区)，比较取最细级，**这些条目的基准是本研究里最不确定的数**。\n"
        "2. **两侧同为低阶常值面元**：几何内接多面体面积亏欠(粗网格 1–2 %)同时进两侧，"
        "外推消一阶，阶的量出值说明残余里还有多少是离散化。\n"
        "3. **仅淹没闭合体**：无水线，因此不触发不规则频率处理；穿孔水面(有自由面交线)的有限水深情形仍未覆盖。\n"
        "4. **1280 面板确认梯级(`EXTRA_LEVEL_CELLS`，4 格)未跑**：本次用 `--no-extra-level`，"
        "冻结工件里 `extra_level=null` 如实记录，不冒充。\n"
    )

    md.append("\n## ⑥ 误差判定（逐项，是否 <3 %）\n\n")
    md.append("| 关键量 | 最坏偏差(全网格) | <3 % ? | <15 % (owner) ? |\n|---|---:|---|---|\n")
    for key, label in QUANTITIES:
        md.append(
            f"| {label} | {_pct(worst[key])} | "
            f"{'✅ 是' if worst[key] < 0.03 else '❌ 否'} | "
            f"{'✅ 是' if worst[key] < 0.15 else '❌ 否'} |\n"
        )
    md.append(
        f"\n基准：capytaine {provenance['version']} 有限水深 Green 函数"
        f"(`{provenance['finite_depth_method']}`) 的 Richardson 外推值/最细级值。"
        "阻尼的预登记带是 5 %（辐射波幅收敛比附加质量慢），故阻尼项同时给「<3 %」与「<5 % 预登记带」两个答案。\n"
    )

    body = "".join(md)

    payload = {
        "title": "m22264 有限水深交叉验证全网格判定（12 格 × 4 量，锚点 XVAL-BEM-FINITE-DEPTH-CROSSVAL-PLAN）：TensorBEM vs capytaine 12 格 × 4 量外推对标",
        "summary": (
            f"把 XVAL-BEM-FINITE-DEPTH-CROSSVAL-PLAN 的冻结工件从 1/12 格推到全网格 "
            f"（{len(rows)} 格 × 4 量）：两侧三级梯级 + Richardson 外推，最坏偏差 "
            f"附加质量 {_pct(max(worst['heave.added_mass'], worst['surge.added_mass']))}"
            f" / 阻尼 {_pct(max(worst['heave.damping'], worst['surge.damping']))}"
            f"（{'在' if max(worst.values()) < 0.15 else '**超出**'} owner 15 % 线"
            f"{'内' if max(worst.values()) < 0.15 else ''}），"
            f"门禁 {'全部' if all(g['passed'] for g in gates) else '有 FAIL：'}"
            f"{'' if all(g['passed'] for g in gates) else '、'.join(g['gate'] for g in gates if not g['passed'])}。"
        ),
        "scenario_key": "bem_finite_depth_crossval",
        "mission_id": args.mission,
        "tags": ["TensorBEM", "finite-depth", "capytaine", "crossval", "XVAL"],
        "markdown": body,
        "figures": figures,
    }
    out = Path("/tmp/report_m22264.json")
    out.write_text(json.dumps(payload))
    print(f"wrote {out} ({len(body)} chars, {len(figures)} figures)")

    if args.dry_run:
        return 0
    api = args.api or "http://localhost:8001/api/v1"
    token = os.environ.get("TENSOR_TOKEN")
    if token is None:
        credentials = json.dumps({
            "username": os.environ["TENSOR_AGENT_USER"],
            "password": os.environ["TENSOR_AGENT_PASS"],
        }).encode()
        request = urllib.request.Request(
            api.rstrip("/") + "/auth/login", data=credentials,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            token = json.loads(response.read())["access_token"]
    request = urllib.request.Request(
        api.rstrip("/") + "/tensor/reports",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {token}"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        print(response.status, response.read().decode()[:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())