"""Publish the m20485 progress report (figures embedded as base64 PNG)."""
from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path

BASE = os.environ["TENSOR_API_BASE"].rstrip("/")
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]
ROOT = Path(__file__).resolve().parent
NUM = json.loads((ROOT / "docs/case_pool_records/numbers.json").read_text())
A = NUM["anchor_numbers"]
P = NUM["case_pool"]


def figure(name: str, caption: str) -> dict:
    png = base64.b64encode((ROOT / name).read_bytes()).decode()
    return {"name": Path(name).name, "caption": caption, "png_base64": png}


def login() -> str:
    body = json.dumps({"username": USER, "password": PASS}).encode()
    req = urllib.request.Request(f"{BASE}/auth/login", data=body,
                                headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.load(resp)
    return data.get("access_token") or data.get("token")


def table_anchor() -> str:
    rows = ["| 网格细分 | 面元数 | max 入口相对误差 | 体积误差 | 接受 |", "|---|---|---|---|---|"]
    for level in A["orientation_ladder"]:
        rows.append(f"| {level['subdivision']} | {level['panels']} | "
                    f"{level['max_entry_rel_error']:.3e} | {level['volume_rel_error']:.3e} | "
                    f"{'是' if level['within_tolerance'] else '否'} |")
    return "\n".join(rows)


def table_momentum() -> str:
    rows = ["| 构型 | route P 最大入口相对误差 |", "|---|---|"]
    for cfg in A["momentum_route"]:
        rows.append(f"| {cfg['configuration']} | {cfg['max_relative_error']:.2e} |")
    return "\n".join(rows)


def table_pool() -> str:
    rows = ["| case_code | 修复前悬空路径 | 修复后 | verdict |", "|---|---|---|---|"]
    for code in P["cases"]:
        rows.append(f"| {code} | {P['missing_before'].get(code, 0)} | "
                    f"{P['missing_after'].get(code, 0)} | `{P['verdicts'][code]}` |")
    return "\n".join(rows)


markdown = f"""# m20485 · 刚体质量矩阵锚点复核 + 算例池证据链修复（TensorBEM）

## ① 案例介绍

锚点 `CASE-BEM-RIGID-BODY-MASS`（算例池·刚体质量矩阵，容差 <3%）。物理对象是浮体六自由度运动方程
`(-ω²(M+A(ω)) + iωB(ω) + C)ξ = F_exc` 中的刚体惯性块 `M_RB`：

```
M_RB = [[ m I₃ , -m S(r_g) ],
        [ m S(r_g) , I_g + m S(r_g) S(r_g)ᵀ ]]
```

`S(·)` 为反对称矩阵，`r_g` 为相对参考点的重心偏移。参考取 Fossen (2011) *Handbook of Marine Craft
Hydrodynamics and Motion Control* §3.3.1 eq.(3.44)–(3.53)（同构见 Newman 1977 §4.5、
Journée & Massie 2001 §3.2）。算例体：三轴椭球半轴 (60, 12, 8) m，ρ=1025 kg/m³，
重心偏移 (5, 0, −3) m；另设旋转体（yaw 20°/pitch 15°/roll 25°，全惯量张量）与双组件箱体。

本单同时处理该锚所属的**算例池证据链契约**：注册表 `tensorbem.benchmarks.case_pool_registry`
承诺每个锚的入口/证据/文档路径，并在 `tests/test_case_pool_registry.py` 里被逐个检查。

## ② 怎么做的

1. 先核对总账（`feature_ledger` 中该 slug 为 `open`）与仓内实际：求解、对标、证据包、9 个测试均已在树内，
   故不做重复求解，改做「独立复算复核 + 机器可核闭环」。
2. **锚点复核**：干净目录重跑 `casepool_bem_rigid_body_mass`，与提交的 `results.json`/`verdict.json`
   逐项比对（仅忽略 `environment` 段），结果完全一致。
3. **缺陷定位**：`_verdict_of` 在嵌套 `verdict` 块里对第一个缺失的键直接 `return None`，
   导致 `CASE-BEM-ANNULUS-LAPLACE`、`CASE-BEM-HULME-HEMISPHERE` 的判定无法被机器回答；
   `registry.missing_paths()` 另报出 4 个 case 共 10 条不存在的路径（9 张派生图 + 1 个 run.log）。
4. **修复**：verdict 抽取改为「明文标量 → 布尔旗标 → criteria 折叠 → tiers → 如实兜底」五级，
   永不返回 `None`，且无布尔的 criteria 块不凭空造判定；路径修复不改证据主张，而是用各模块**自己的绘图函数**
   以已提交的 `results.json` 为唯一输入重新生成，并写 sha256 审计；`run.log` 由证据里保存的同次运行记录物化。
5. 工具链：Python 3.11 / numpy / matplotlib(Agg) / pytest；复核与修复均为单机轻量（<10 s / <30 s）。

## ③ 计算结果

锚点（旋转体，网格收敛阶梯，容差 3%）：

{table_anchor()}

动量定义独立路线（`mass_matrix_from_momentum`，round-off 门 1e-9）：

{table_momentum()}

判据汇总：`worst_accepted_entry_rel_error = {A['worst_accepted_entry_rel_error']:.4f}`
（< 容差 {A['tolerance']:.2f}）；负对照「翻转耦合块符号」最大入口相对误差 2.0（100% 偏离，必须被拒），
「反对称污染」能量恒等式 <1e-9 通过但动量残差 >0.1 失败——证明判据具备分辨力。

算例池注册表（修复前 → 后）：

{table_pool()}

## ④ 结果图

图 1：锚点复核。左＝`M_RB` 36 个入口相对误差与网格体积误差随细分收敛，红点线为 3% 池容差，
红叉标出未接受层级；右＝动量定义独立路线在 4 个构型上与出厂组装的偏差（均在机器精度，远低于 1e-9 门）。

图 2：注册表健康度。左＝各锚「注册了但盘上不存在」的证据路径数（修复前 4 个锚共 10 条 → 0）；
右＝注册表能否回答「最近一次判定」（修复前 2 个锚返回 `None` → 现 7/7 可答）。

（本任务为证据链/注册表修复，非场量求解，故无流场云图；数值曲线与审计图即为本任务的场量等价物。）

## ⑤ 注意点

* 接受层级是细分 3：细分 1/2 误差 19.9% / 5.5% 明显超差，说明该容差判据只在收敛后成立，引用时不可脱离网格层级。
* `verdict` 折叠只看显式布尔，不做阈值/量级判断；它回答 pass 与否，不替代证据审阅。
* 重新生成的 PNG 不保证与丢失原图逐位相同（matplotlib 版本/后端差异），但同源同数据，审计记录了 sha256。
* `CASE-BEM-NEUMANN-KELVIN` 的证据包本身无 verdict 字段，注册表只能如实答 `see artifact`；
  `CASE-BEM-SUBMERGED-SPHERE-HAVELOCK` 的 economy 档为 FAIL，其 `verdict.honest_note` 保留未掩盖。
* 未做：驱动与证据版本一致性门（驱动改动而证据未重生成时无法发现）。

## ⑥ 误差判定

| 关键量 | 参考 | 计算 | 相对误差 | 是否 <3% |
|---|---|---|---|---|
| 基础算例（盒体）36 入口 | 闭式分块形式 | 出厂组装 | 0.0 | 是 |
| 同一算例，网格求积路线 | 闭式分块形式 | 散度定理表面积分 | 4.30e-16 | 是 |
| 旋转体（接受层级 sub=3）36 入口 | 闭式分块形式 | 出厂组装 | 1.42e-2 | 是 |
| 旋转体体积（几何前置校验） | 解析椭球体积 | 网格求积 | 8.61e-3 | 是 |
| 动量定义路线（4 构型） | 出厂组装 | route P | ≤1.15e-16 | 是（远超 1e-9 门） |
| 负对照：翻转耦合符号 | 闭式形式 | 错误约定 | 2.0 | 否（预期必须失败） |

结论：锚点验收（全部 36 入口 <3%）**通过**，最坏接受值 1.42%。

## 产物与复现

* 代码：`src/tensorbem/benchmarks/case_pool_registry.py`（verdict 抽取修复）、
  `scripts/repair_case_pool_evidence.py`（派生证据重建 + sha256 审计）、
  `tests/test_case_pool_verdict_extraction.py`（四类 schema + 全锚可答 + 不粉饰 FAIL）。
* 证据：`docs/case_pool_records/{{evidence_repair.json,numbers.json,make_figures.py}}`；
  重建的 10 个文件见 `evidence_repair.json`。
* 报告：`docs/dev_report_m20485_case_pool_evidence.md`。commit `1b8ace4f`（分支 `agent/m20485`）。
* 测试：`pytest tests/test_case_pool_registry.py tests/test_case_pool_verdict_extraction.py
  tests/test_rigid_body_mass*.py tests/test_case_rigid_body_mass.py tests/test_doc_numbers_rigid_body.py`
  → 94 passed（基线该注册表文件 4 failed / 8 passed）。
"""

report = {
    "title": "m20485 · 刚体质量矩阵锚点复核与算例池证据链修复（TensorBEM）",
    "summary": ("CASE-BEM-RIGID-BODY-MASS 干净重跑逐位复现提交证据（最坏入口误差 1.42% < 3%，负对照仍被拒）；"
                "修复算例池注册表 verdict 抽取返回 None 的缺陷，并用各模块自带绘图函数从已提交数据重建 "
                "10 个悬空派生证据（含 sha256 审计），注册表由 4 failed 转为全绿。"),
    "scenario_key": "BEM-RIGID-BODY-MASS",
    "mission_id": 20485,
    "tags": ["TensorBEM", "case-pool", "rigid-body-mass", "evidence-integrity", "verification-debt"],
    "markdown": markdown,
    "figures": [
        figure("docs/case_pool_records/fig_anchor_mass_matrix.png",
               "锚点复核：M_RB 入口相对误差 vs 3% 池容差（左）；动量定义独立路线偏差（右）"),
        figure("docs/case_pool_records/fig_case_pool_health.png",
               "算例池注册表健康度：悬空证据路径数（左）与 verdict 可答性（右），修复前→后"),
    ],
}

token = login()
body = json.dumps(report).encode()
req = urllib.request.Request(f"{BASE}/tensor/reports", data=body,
                             headers={"Content-Type": "application/json",
                                      "Authorization": f"Bearer {token}"})
with urllib.request.urlopen(req, timeout=120) as resp:
    print(resp.status, resp.read().decode()[:600])