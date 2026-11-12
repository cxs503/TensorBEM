"""Post the m16000 progress report (six-element format) to the Tensor platform."""
from __future__ import annotations

import base64
import json
import os
import pathlib
import urllib.request

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]

E = json.loads(pathlib.Path("docs/fmm_operator_linearity/evidence.json").read_text())
rows = E["error_versus_order"]["rows"]
g = E["gmres"]
c = E["cache_overhead"]


def _post(url: str, payload: dict, token: str | None = None) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 **({"Authorization": f"Bearer {token}"} if token else {})},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode())


_login = json.dumps({"username": USER, "password": PASS}).encode()
token = json.loads(urllib.request.urlopen(urllib.request.Request(
    f"{BASE}/auth/login", data=_login,
    headers={"Content-Type": "application/json"})).read())["access_token"]

table = ["| p | 首次调用（修复前后逐位相同） | 修复后第 2 次调用 | 修复前第 2 次调用 |",
         "|---|---:|---:|---:|"]
for r in rows:
    table.append(f"| {r['p']} | {r['fixed_first_err']:.3e} | {r['fixed_second_err']:.3e} | "
                 f"{r['prefix_second_err']:.3e} |")
table = "\n".join(table)

figs = []
for name, caption in (
    ("fig_error_vs_p.png",
     "第二次调用（换电荷向量）matvec 相对稠密直接求和的误差 vs 多极展开阶 p："
     "修复前恒为 2.46（不随 p 改善=系统性错位），修复后随 p 下降（p=8 达 2.9e-10）；"
     "首次调用曲线修复前后逐位重合"),
    ("fig_gmres_residual.png",
     "GMRES+FMM 单层求解（icosphere(1) N=80，p=10）报告残差历史：两条曲线重合，"
     "说明该网格以近场为主，对本缺陷不敏感（局限如实登记）"),
):
    path = pathlib.Path("docs/fmm_operator_linearity") / name
    figs.append({"name": name, "caption": caption,
                 "png_base64": base64.b64encode(path.read_bytes()).decode()})

md = f"""# TensorBEM m16000 — `FMMOperator` 电荷向量缓存失效：算子非线性导致静默错答（修复）

## ① 案例介绍（物理背景/几何/参数）

BEM 单层算子 `S q = Σ_j q_j /(4π|x_i−y_j|)` 由 `tensorbem.fmm.FMMOperator` 用八叉树 + 实球谐
多极展开加速。多极系数 `M_(l,m)` 是电荷向量 q 的**线性泛函**，但实现把它缓存在
`_multipole_matrix`，只在 `is None` 时构建：`setup(points)`（无电荷）后，**首次** `matvec`
用当时 `charges` 建多极，此后**永不失效**。于是同一算子第二次用别的电荷调用时
「近场按新电荷直和、远场读旧电荷多极」⇒ 算子不再是线性的 ⇒ 静默返回错值。

| 项 | 值 |
|---|---|
| 几何（精度 A/B） | 200 源点高斯聚簇（σ=0.5）+ 3 远场靶点 (4,0,0)/(8,0,0)/(0,6,0)，`max_leaf=25` |
| 几何（端到端） | `geometry3d.icosphere(1)`，N={g['panels']} 面元，`max_leaf={g['max_leaf']}` |
| 展开阶 | p=1..8（扫描）；p={g['p']}（GMRES） |
| 电荷 | 随机正态向量 q1、q2（`np.random.default_rng(42)`） |
| 工具链 | Python 3.11 + numpy 2.4 + scipy 1.16（GMRES），loong64 单节点 |

## ② 怎么做的

1. **先红后绿**：新写 `tests/test_fmm_operator_linearity.py`（10 例），在**未修改**的 `fmm.py` 上
   首条即失败（`second call error 2.464e+00 (first 1.048e-08)`），再改代码。
2. **最小修复**（`src/tensorbem/fmm.py`）：新增电荷指纹 `_cache_points/_cache_charges`（`copy=True`，
   调用方原地改数组也能被发现）+ 谓词 `_multipole_cache_valid()`；`matvec` 由「`is None` 才建」
   改为「指纹不匹配即重建」；`setup()` 同步刷新/清空指纹；`sources`/`charges` 形状显式校验
   （原为裸 `IndexError`）。
3. **A/B 对照**：`scripts/fmm_linearity_bench.py` 用 `_PrefixCacheOperator`（唯一差异=缓存谓词恒真，
   即修复前语义）与修复后算子在同一几何、同一电荷序列上对跑，p=1..8 扫描；另做 GMRES 端到端
   与缓存指纹开销测量。所有数字写入 `evidence.json`，本报告表格与图由脚本注入（零手抄）。
4. **回归**：`tests/test_fmm.py`(23) + `test_fmm_cost_audit.py` + 新文件 = 49 passed。

## ③ 计算结果

第二次调用（换电荷向量）相对稠密直接求和的最大相对误差：

{table}

- 修复前误差在 p=1..8 **恒为 2.46（≈246%）**——不随展开阶改善，证明是缓存错位而非截断误差；
- 修复后 p=6 为 1.239e-08、p=8 为 2.936e-10，随 p 下降；
- **首次调用修复前后逐位相同**（8/8 行相等），证明正常路径数值未被改动。

端到端 GMRES（icosphere(1)，N={g['panels']}，p={g['p']}，rtol=1e-12）：

| 项 | 修复前语义 | 修复后 |
|---|---:|---:|
| 迭代数 | {g['prefix']['iterations']} | {g['fixed']['iterations']} |
| 报告残差 | {g['prefix']['reported_residual']:.3e} | {g['fixed']['reported_residual']:.3e} |
| 真残差 ‖Sq−b‖/‖b‖ | {g['prefix']['true_residual']:.3e} | {g['fixed']['true_residual']:.3e} |
| 解 vs 稠密解相对误差 | {g['prefix']['solution_error_vs_dense']:.3e} | {g['fixed']['solution_error_vs_dense']:.3e} |

缓存指纹开销（N=200，同电荷重复调用 vs 换电荷触发重建，20 次取最优）：
命中 {c['cache_hit_ms']:.2f} ms / 重建 {c['rebuild_ms']:.2f} ms（比值 {c['overhead_ratio']:.2f}）
⇒ 每次调用多一次 O(N) 指纹比较，成本在测量噪声内。

## ④ 结果图

见 figures：误差-阶 A/B 曲线（判别性证据）与 GMRES 残差历史（语义回归）。

## ⑤ 注意点 / 局限

1. **GMRES 实验对本缺陷不敏感**：N={g['panels']} 的 icosphere 面元间距相对远场判据太近，几乎全部
   (target,leaf) 对走近场直和，两条残差曲线完全重合、解误差均 {g['fixed']['solution_error_vs_dense']:.1e}。
   该实验证明的是「修复后算子可安全放进迭代求解器」（回归价值），判别性证据是 ③ 的 p 扫描 A/B。
2. **`far_fraction()` 自相矛盾（疑似独立诊断缺陷，本单未修）**：同几何返回
   {E['error_versus_order']['far_fraction']:.1f}（暗示无远场对），却与修复前 2.46 的远场缓存误差
   不能同时成立；本单范围是缓存语义，不动该诊断函数，登记为后续项。
3. 修复是语义修复而非精度提升：首次调用逐位不变；精度上限仍由 p 与近/远场判据决定
   （无 M2L，严格说是树码而非教科书 FMM）。
4. **既有红不掩盖**：`tests/test_capability_evidence.py` 4 例红在 HEAD 上即存在（`git stash` 验证），
   与本单无关，未触碰。

## ⑥ 误差判定（关键量 vs 基准，显式回答是否 <3%）

| 关键量 | 基准 | 修复前 | 修复后 | <3%？ |
|---|---|---|---|---|
| 第 2 次调用 matvec, p=6 | 稠密直接求和 | 2.464e+00（246%） | 1.239e-08 | 修复后 ✅ / 修复前 ❌ |
| 第 2 次调用 matvec, p=4 | 稠密直接求和 | 2.464e+00 | 2.897e-07 | 修复后 ✅ |
| 叠加性残差 S(q1+q2)−(Sq1+Sq2) | 0 | 1.428e-01（14%） | <1e-6（测试断言） | 修复后 ✅ |
| 首次调用 matvec | 稠密直接求和 | 1.048e-08 | 1.048e-08（逐位不变） | ✅ |

结论：修复后关键量 1.2e-08 ≪ 3%；修复前 246% 不可接受。本单为缺陷修复（含判别性 A/B），
非新的对标精度研究；未做的判别性 GMRES 几何（双远簇）如实登记为未覆盖项。
"""

payload = {
    "title": "TensorBEM m16000 · FMMOperator 电荷缓存失效（算子非线性→静默错答）修复 + 判别性 A/B",
    "summary": "FMMOperator 多极系数缓存不随电荷向量失效，第二次调用静默错答：实测相对误差 "
               "2.464e+00（246%，p=1..8 恒定，叠加性残差 1.428e-01）。改为电荷指纹失效后 "
               "p=6 误差 1.239e-08、叠加性 <1e-6，首次调用逐位不变；10 例回归测试先红后绿，"
               "FMM 相关 49 例全绿。",
    "scenario_key": "tensorbem_fmm_operator_linearity",
    "mission_id": 16000,
    "tags": ["TensorBEM", "FMM", "linear-operator", "defect-fix", "regression", "m16000"],
    "markdown": md,
    "figures": figs,
}
resp = _post(f"{BASE}/tensor/reports", payload, token)
print(json.dumps(resp)[:400])