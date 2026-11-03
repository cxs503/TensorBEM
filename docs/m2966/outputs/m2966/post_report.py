"""Build and POST the m2966 platform report (numbers injected from the evidence JSON)."""

import base64
import io
import json
import os
import urllib.request
from pathlib import Path

from PIL import Image

ROOT = Path("/root/tp-agent/work/m2966")
EVID = json.loads((ROOT / "outputs/m2966/rankine_primitive_check.json").read_text())
URS = json.loads((ROOT / "outputs/m2966/ursell_table_64_128_256.json").read_text())

# ---- figure: shrink below the ~60 KB payload guidance
img = Image.open(ROOT / "outputs/m2966/fig_rankine_referee.png").convert("RGB")
img.thumbnail((1500, 700))
buf = io.BytesIO()
img.save(buf, format="PNG", optimize=True)
if buf.tell() > 90_000:
    buf = io.BytesIO()
    img.quantize(colors=128).save(buf, format="PNG", optimize=True)
png_b64 = base64.b64encode(buf.getvalue()).decode()
print("figure png bytes:", len(buf.getvalue()), "base64:", len(png_b64))

exact = EVID["closed_form_vs_mpmath"]
conv = EVID["uniform_gauss_referee"]
cop = [r for r in conv["rows"] if r["xi"] == 0.5 and r["h"] == 0.0]
REF = {"added_mass": URS["reference"]["added_mass"], "damping": URS["reference"]["damping"]}

exact_tbl = "\n".join(
    f"| {r['label']} | ({r['xi']:g}, {r['h']:g}) | {r['closed_form']:.15f} | "
    f"{r['mpmath_40dps']:.15f} | {r['abs_error']:.2e} |"
    for r in exact
)
conv_tbl = "\n".join(
    f"| {r['pieces']} | {r['err_sing_on_edge']:.3e} | {r['err_sing_inside_piece']:.3e} |"
    for r in cop
)
panel_tbl = "\n".join(
    f"| {r['panels']} | {r['added_mass']:.2f} | {r['added_mass_relative_error']:.3e} | "
    f"{r['damping']:.2f} | {r['damping_relative_error']:.3e} | {r['near_over_far_field_damping']:.4f} |"
    for r in URS["rows"]
)
mfs_tbl = "\n".join(
    f"| {r['panels']} | {r['added_mass']:.2f} | {r['added_mass_relative_error']:.3e} | "
    f"{r['damping']:.2f} | {r['damping_relative_error']:.3e} |"
    for r in URS["mfs"]
)

md = f"""# TensorBEM 三条长期红测试结清：闭式板块积分的判据错了，不是积分错了

**任务** m2966（自找活 · models 席） · **仓** TensorBEM · **分支** agent/m2966 · **commit** c14834e

## ① 案例介绍

TensorBEM 的二维自由液面间接（源分布）求解器把单层位势 `∫ log|P-Q| ds` 在直板块上**解析积分**
（`tensorbem.free_surface_solver._log_panel_primitive`），以消掉自板块 `log r` 奇异性带来的位势偏差
（`docs/ursell_quadrature_provenance.md`：旧均匀 24 点 Gauss 方案把 Ursell 128 面元附加质量误差自
7.72e-3 挪到 8.51e-3）。这条闭式积分由一个守卫测试把关，但该测试在本仓长期报红（m2918 报告已留证
「3 条与本单无关的既存红」并明确建议作为下一单入口）。本单即结清这三条红：

1. `test_analytic_rankine_panel_integral_is_exact` —— 闭式积分 vs 均匀分片 Gauss 参照，容差 1e-8；
2. `test_pinned_cases_reference_existing_docs` —— 文档必须指向数字钉扎文件 `docs/benchmark_pins.json`；
3. `test_ursell_doc_reproducibility::test_document_contains_the_generated_rows` —— 文档 §4.6 的 MFS 行必须是生成器输出。

几何/参数（Ursell 潜没圆柱，深水）：半径 a=1 m、潜深 f=2 m（a/f=0.5）、ka=0.6、ρ=1025 kg/m³、g=9.80665 m/s²，
常数面元 N=64/128/256；参照解为 Ursell 多极子闭式解。板块积分侧的自变量为场点沿板坐标 ξ、
垂距 h、板长 L=1。

## ② 怎么做的

1. **先判定哪一侧错**：用 `mpmath.quad`（tanh-sinh，40 位有效位，奇异点 s=ξ 作为分段点）独立积分 6 个构型
   （含自板块居中/端点、场点在板端点、共面偏心、近共面 h=1e-3、光滑板块 h=0.3），与闭式解比绝对差。
2. **量化参照侧的收敛机制**：均匀分片 Gauss（每片 20 点），分片数 N=25…1600，两种网格
   （奇异点恰在片边界 / 网格错动使奇异点落入片内），用实测首末误差反解观测阶（不预设理论阶）。
3. **改判据而不是降标准**：新测试改用任意精度参照（`pytest.importorskip("mpmath")`，容差相对 1e-13），
   并**新增**一条测试把均匀 Gauss 参照自身的 O(1/N) 分辨率上限钉住（阶、N=400 残差、单片 24 点偏差三个断言）。
4. **文档侧**：按 `python -m tensorbem.benchmarks.ursell_convergence_table --panels 64 128 256` 的现值
   逐字恢复 MFS 三行（原表被手工改成一行「64 面元」却填 128 面元数值，且 64 面元行丢失），
   加勘误说明；补「数字钉在 `docs/benchmark_pins.json`」指针与复核命令；
   并在 `tests/test_ursell_doc_reproducibility.py` 新增**列数守卫**——每出现一个面元数列就必须有对应 MFS 行。
5. **负控**：把历史缺陷（删 64 面元行 + 改标签）重新注入文档，确认两条守卫同时报红，再还原。

## ③ 计算结果

### 3.1 闭式 `_log_panel_primitive` vs 40 位任意精度求积（零手抄，`outputs/m2966/rankine_primitive_check.json`）

| 构型 | (ξ, h) | 闭式解 | mpmath（40 dps） | 绝对差 |
|---|---|---|---|---|
{exact_tbl}

最大绝对偏差 **{EVID['max_abs_error']:.2e}**，最大相对偏差 {EVID['max_rel_error']:.2e}。

### 3.2 均匀分片 Gauss 参照在共面板（ξ=0.5, h=0）上的分辨率

| 分片数 N | 奇异点在片边界 | 奇异点在片内部 |
|---|---|---|
{conv_tbl}

渐近段（N≥100）实测观测阶：片边界 **{conv['coplanar_panel_observed_order_sing_on_edge']:.3f}**、
片内部 **{conv['coplanar_panel_observed_order_sing_inside_piece']:.3f}**（全段拟合 {conv['coplanar_panel_observed_order_full_range']:.2f}，
因 N=25/50 属预渐近段而偏大，故剔除）。单片 24 点 Gauss 跨奇异点偏差 {conv['single_panel_24pt_bias']:.2e}（纯自板块极端量级）。

### 3.3 文档表与代码的一致性（Ursell，现值重算）

| N | A | A 相对误差 | B | B 相对误差 | 近/远场阻尼比 |
|---|---|---|---|---|---|
{panel_tbl}
| 参考解（Ursell） | {REF['added_mass']:.2f} | — | {REF['damping']:.2f} | — | — |

| MFS N | A | A 相对误差 | B | B 相对误差 |
|---|---|---|---|---|
{mfs_tbl}

面元行三人值与文档逐字一致；MFS 行恢复后与生成器逐字一致（原「64 面元」行错填 128 面元数值）。

### 3.4 测试判决

- 修改前：`tests/test_documented_benchmarks.py + tests/test_ursell_doc_reproducibility.py` → **3 failed, 8 passed**。
- 修改后：同两文件 **13 passed**（44 s）。负控（注入历史缺陷）→ 2 failed，确认守卫有效。
- 另：`tests/test_doc_claims.py` 3 条仍红，**与本单无关且非本单引入**——它要 `artifacts/ursell_doc_claims.json`，
  该文件在 m2186 落盘时被门禁滤除（`FileNotFoundError`），属独立欠账，未修（超出本单白名单 `artifacts/`）。

## ④ 结果图

左图：均匀分片 Gauss 参照误差 vs 分片数（log-log）。共面 `log r` 两类网格都呈斜率 −1；过载红线为旧测试的
1e-8 阈值——参照本身在 N=1600 时仍差 1.9e-6，阈值不可达。光滑板块（h=0.3）同一规则直达机器精度（灰线）。
右图：6 个构型上闭式解与 40 位求积的绝对偏差，全部 ≤2.2e-16，判据线 1e-13 之上留出三个数量级余量。

## ⑤ 注意点（坑 / 局限）

1. **判据比被测对象更差是最贵的坑**：这条红测试挂着「闭式积分」的名字，实际考的是参照的收敛能力；
   长期红态会让真实回归沉没（m2918 只能把它登记为「与本单无关的既存红」）。修判据必须同时给出
   *为什么旧判据不可达* 的可执行证据，故新增 O(1/N) 定阶测试，而不是简单放宽容差。
2. **预渐近段会骗人**：N=25/50 两点使全段拟合给出 1.69，远高于真实阶；定阶必须自己选渐近窗口并写进结论。
3. **单片偏差与净效应差三个量级**：单片 24 点跨奇异点为 4.4e-2（纯自板块），而它在整个物面位势里的净效应是
   1e-4 级——引用文献数字时必须说清是哪一个。
4. **文档手改是最大的腐化源**：m2609 把 MFS 行手工改成「64 面元 + 128 面元数值」，逐字守卫只约束
   *存在* 的行，抓不到「行被删并被改名」。新增列数守卫才真正堵住这一路。
5. 适用范围：仅二维直线板块的 `log r` 单层位势；h>0 光滑情形无此限制；近共面开关为 h≤1e-14，
   h=1e-3 已走一般分支且同为机器精度。
6. **未做**：`tests/test_doc_claims.py` 的 3 条红（缺 `artifacts/ursell_doc_claims.json`，需 112 s live 重生成，
   且 `artifacts/` 不在本单白名单）未修，仅留证。

## ⑥ 误差判定（是否 <3%）

| 关键量 | 基准 | 本单结果 | 相对误差 | 是否 <3% |
|---|---|---|---|---|
| 闭式板块积分（6 构型） | 40 位 mpmath tanh-sinh | 最大绝对偏差 {EVID['max_abs_error']:.2e} | {EVID['max_rel_error']:.1e} | **是**（机器精度） |
| 闭式板块积分（最差构型 ξ=0.25,h=0） | mpmath | — | 1.3e-16 | **是** |
| 文档 MFS 行（64/128/256 面元，A、B 共 6 值） | 生成器现值 | 逐字相同 | 0 | **是** |
| 文档面元行（64/128/256，A、B 共 6 值） | 现值重算 | 逐字相同 | 0 | **是** |
| Ursell 面元解 A（128 面元） | Ursell 多极子闭式解 | 2566.83 vs 2545.17 | 8.51e-3 | **是** |
| Ursell 面元解 B（128 面元） | Ursell 多极子闭式解 | 2754.47 vs 2718.15 | 1.34e-2 | **是** |
| 均匀 Gauss 参照（N=400）对 N=1600 的残差 | — | 7.51e-6 vs 1.88e-6 | 75%（未收敛） | 否 —— 正是它不能当判据的证据 |

结论：被测的闭式积分在机器精度内正确（远优于 3%）；三条长期红测试全部转绿，且各有负控；
唯一未收敛的量是**参照侧**的均匀 Gauss，已作为「不可用作判据」的正面证据固化进测试。
"""

payload = {
    "title": "TensorBEM 三条长期红测试结清：闭式板块积分判据纠错（40 位参照 2.2e-16）+ 文档 MFS 行恢复",
    "summary": (
        "TensorBEM 三条既存红测试全部转绿：①闭式 log 板块积分被 40 位 tanh-sinh 求积验证到 2.2e-16，"
        "原 1e-8 判据不可达的根因是均匀分片 Gauss 参照在共面板上只有 O(1/N)（N=400 仍差 7.5e-6）；"
        "②文档 §4.6 被手改成一行『64 面元』却填 128 面元数值，现按生成器恢复 64/128/256 三行并加勘误；"
        "③文档补 docs/benchmark_pins.json 钉扎指针。新增两条守卫（参照分辨率上限定阶、MFS 列数守卫），"
        "并做负控确认能抓回历史缺陷。另留证一条无关红：tests/test_doc_claims.py 缺 artifacts/ursell_doc_claims.json。"
    ),
    "scenario_key": "tensorbem/validation/red-test-clearance-log-panel-integral",
    "mission_id": 2966,
    "tags": [
        "tensorbem",
        "validation-debt",
        "quadrature",
        "log-singularity",
        "documentation-drift",
        "red-tests",
    ],
    "markdown": md,
    "figures": [
        {
            "name": "fig_rankine_referee.png",
            "caption": (
                "左：均匀分片 Gauss 参照在共面 log r 板块上的误差 vs 分片数（log-log），两类网格观测阶 1.000，"
                "旧 1e-8 阈值（红点划线）不可达；光滑板块（h=0.3）同一规则直达机器精度。"
                "右：闭式 _log_panel_primitive 与 40 位任意精度求积的绝对偏差，6 构型全部 ≤2.2e-16。"
            ),
            "png_base64": png_b64,
        }
    ],
}

body = json.dumps(payload).encode()
print("payload bytes:", len(body))
tok = open("/tmp/tok.txt").read().strip()
base = os.environ["TENSOR_API_BASE"]
req = urllib.request.Request(
    base + "/tensor/reports",
    data=body,
    headers={"Content-Type": "application/json", "Authorization": "Bearer " + tok},
)
try:
    resp = json.load(urllib.request.urlopen(req, timeout=120))
    print("POST ok:", json.dumps(resp)[:300])
    rid = resp.get("id") or (resp.get("report") or {}).get("id")
except urllib.error.HTTPError as exc:
    print("HTTP", exc.code, exc.read()[:500].decode(errors="replace"))
    raise
Path("/tmp/report_m2966.json").write_bytes(body)
print("report id:", rid)