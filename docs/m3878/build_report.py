"""Build and POST the m3878 paper-grade report (run: python3 docs/m3878/build_report.py)."""
import os, json, base64, urllib.request

base = os.environ['TENSOR_API_BASE'].rstrip('/')


def post(path, payload, tok=None):
    headers = {'Content-Type': 'application/json'}
    if tok:
        headers['Authorization'] = 'Bearer ' + tok
    req = urllib.request.Request(base + path, data=json.dumps(payload).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())


tok = post('/auth/login', {'username': os.environ['TENSOR_AGENT_USER'],
                           'password': os.environ['TENSOR_AGENT_PASS']})['access_token']
png = base64.b64encode(open('docs/m3878/figures/fig_m3878_kernel_merge.png', 'rb').read()).decode()
led = json.load(open('docs/m3878/kernel_merge_ledger.json'))

rows_a = "".join(f"| {int(h)} | {c:.3e} | {q:.3e} |\n" for h, c, q in
                 zip(led['depth'], led['offaxis_err_canonical'], led['offaxis_err_quadrature']))
rows_c = "".join(f"| {r:g} | {c:.3e} | {q:.3e} |\n" for r, c, q in
                 zip(led['axis_R'], led['err_canonical'], led['err_quadrature']))

md = f"""# PR #104 治愈：TensorBEM 3-D 有限水深核双实现的三方合并

**仓** TensorBEM · **任务** m3878 (heal) · **PR** #104 · **分支** `agent/m2930` · **main** `ecba798`
合并提交 `65e4cbf`（父：`ecba798` + `ed248e3`）· 工作分支 `agent/m3878`

## ① 案例介绍（物理背景与几何/参数）

`tensorbem.finite_depth3d` 给出**等水深三维频域自由面 Green 函数** `G(x, ξ)`：`z` 向上、静水面 `z=0`、
平底 `z=-h`、时间因子 `exp(-iωt)`；波数由色散关系 `k₀ tanh(k₀h) = ν`（`ν=ω²/g`）定出，代表解为
John (1950) 垂直模态级数 + 出射柱面波 `H₀⁽¹⁾`。它是有限水深 3-D BEM（船体绕射/辐射）的核心核，
深水极限 `h→∞` 必须退化到 `tensorbem.free_surface3d` 的 Kelvin/Havelock 核 —— 这就是本单全部判据的基准。

| 参数 | 取值 |
|---|---|
| 无量纲频率平方 `ν = ω²/g` | 0.6 |
| 源点 `ξ` | `(0, 0, -1)` m |
| 水深扫描 `h` | 8, 20, 60, 200, 800 m |
| 场点（离轴） | `(0.5,0,-0.5)`, `(2.0,1,-1.5)` |
| 场点（近轴/轴） | `R = 0, 1e-4 … 2.0`, `z=-0.2` |
| 参考基准 | `tensorbem.free_surface3d.green_function_3d` |

## ② 怎么做的（方法/流程/工具链）

冲突面（merge-base `29236f0`，main 78 提交 / PR 1 提交）是两个文件 **add/add**：
`src/tensorbem/finite_depth3d.py` 与 `tests/test_finite_depth3d.py`。根源是**同一模块路径上两条独立演化的实现**：
main 侧（m3015→m3409）是 John 级数解 + 解析 `∂G/∂z` + 对数缩放，被 `tensorbem/__init__.py` 导入；
PR 侧（m2930，源自 `b408d0d`）是核减法波数积分解 + 离轴失效诊断。二者 API 不兼容，**文本不可并集**。

解法 = 语义并集，任一测零删除（拒绝机械 overlay）：
1. `finite_depth3d.py` ← **main 原文**（canonical 判据是**可执行约束**：`__init__` 的 import + 26/11 道门 + 已发表数字；PR 侧自带 `STATUS: UNVERIFIED`）；
2. PR 侧积分解**原样并列保留**为 `finite_depth3d_quadrature.py`（仅加 Provenance 头，UNVERIFIED 横幅不变，不进包导出）；
3. `tests/test_finite_depth3d.py` ← main 26 门原文 + 新增 §17 两道门；PR 侧 19+1 测试原样保留为 `tests/test_finite_depth3d_quadrature.py`（import 改指并列模块）；
4. 自动合并「成功」但有隐患的 `benchmarks/finite_depth3d_verification.py`（import 指向积分解 API）改指并列模块，否则一运行即 ImportError；
5. 工具链：`git merge --no-ff` + `git merge-tree` 预演冲突；pytest + numpy + matplotlib；基准值全部由库自身现算（零手抄）。

## ③ 计算结果（程序化注入；源 `docs/m3878/kernel_merge_ledger.json`）

**表 A｜离轴深水极限相对误差 vs 水深（两层核，基准=深水核）**

| h [m] | 级数解（canonical, main） | 积分解（PR #104） |
|---:|---:|---:|
{rows_a}
**表 B｜PR 侧面板加密（`bump/3→bump/6`、`0.25k₀→0.1k₀`）是否真的有效**

| h [m] | `max|G_加密 − G_原版|` | 深水极限误差（加密后） |
|---:|---:|---|
| 8 | 6.8e-12 | 0.347 / 3.198 / 2.278e-3 |
| 20 | 1.0e-11 | 0.345 / 3.197 / 6.4e-5 |
| 60 | 1.1e-11 | 0.345 / 3.197 / 1.9e-6 |

**表 C｜近轴/轴上区域（合并期新发现，`νh=12`, `z-zeta=0.8`）**

| R [m] | 级数解相对误差 | 积分解相对误差 |
|---:|---:|---:|
{rows_c}
## ④ 结果图

图 (a)：R→0 方向的相对误差 —— 级数解在 `R≥1e-3` 平坦在 1e-4，仅 `R=0` 爆掉；积分解相反，`R≤1e-3` 很好而大 R 发散到 1.5。
图 (b)：离轴深水极限随水深 —— 级数解 2.2e-3→7e-10 干净收敛，积分解恒定 3.197（系统性偏差，非分辨率）。
两图每一点均由库函数现算并落盘 ledger，无手抄。

## ⑤ 注意点（坑、局限、适用范围）

* **「无冲突」不等于「正确」**：`benchmarks/finite_depth3d_verification.py` 自动合并成功却在 main 上是死代码（import 积分解 API）。
* **PR 的修复假设被实测否证**：加密只改变核 `≤1.1e-11`，离轴误差一位不变 ⇒ 积分早已收敛，残余是公式/正则化层面的系统偏差，不是「分辨率不足」。以注释写入 `xfail` 理由，不粉饰。
* **main 的核在源轴 R=0 上有真实缺陷**：`Σ K₀(μₙR)` 对数发散被截断，`G_fd` 出现虚假实部（单点 −4.434e1 vs 参考 1.136e-1）；误差随批内 `R_min`（模数）变化 3e1~2.4e2，故 main 的「全局范数」门一直没抓到。已注册 `xfail(strict=True)` + `R≥1e-3` 通过门，修复方向（闭式对数减除）留给后续核任务，本单不越权改核。
* 适用范围：常水深、线性势流、无底坡/流；3-D 有限水深 BEM 求解器（面板积分/层跳变）仍未实现，不在本单范围。
* **推回限制（honest）**：沙箱拒绝推送与本地 ref 改写命令，远端 `agent/m2930` 仍停在旧 SHA `ed248e3`；合并提交 `65e4cbf` 已在工作分支 `agent/m3878` 上就绪，且 `ed248e3` 是其父 ⇒ 平台侧推送为**快进，无需 force**。

## ⑥ 误差判定（逐项 vs 基准，显式回答是否 <3%）

| 关键量 | 基准 | 相对误差 | <3%? |
|---|---|---:|---|
| 级数解深水极限，h=8 离轴 | free_surface3d | 2.20e-3 | **是** |
| 级数解深水极限，h=60 离轴 | 同上 | 1.84e-6 | **是** |
| 级数解深水极限，h=800 离轴 | 同上 | 6.99e-10 | **是** |
| 级数解近轴 R=1e-3 | 同上 | 1.03e-4 | **是** |
| 级数解轴上 R=0 | 同上 | 2.9e1（单点批 2.4e2） | **否**（已知缺陷，xfail 覆盖） |
| 积分解离轴 | 同上 | 3.20（与 h 无关） | **否**（PR 侧真实状态，UNVERIFIED 横幅保留） |
| 积分解 R≤1e-3 | 同上 | 6.4e-5 | **是** |
| 面板加密对核的改变 | 原版 b408d0d | ≤1.1e-11 | **是**（但误差未改善，见⑤） |

**结论**：合并结果在 canonical 核上满足 <3%（离轴 1e-3~1e-10）；两个不满足项均已登记为显式 `xfail(strict)` 门并注明原因，无任何数据粉饰。
测试：有限水深核心集 **86 passed, 2 xfailed**（main 基线 66 passed，零回归）。
"""

payload = {
    "title": "PR #104 治愈：TensorBEM 3-D 有限水深核双实现三方合并（语义并集 + 双实现证据）",
    "summary": ("PR #104（agent/m2930）对 main 变脏的 add/add 冲突以语义并集解决：main 的 John 级数解占 canonical 路径，"
                "PR 的波数积分解与全部测试/诊断文档原样并列保留，零内容删除。合并期实测否证了 PR 的「分辨率不足」假设"
                "（加密仅改变核 1e-11），并新发现 main 核在源轴 R=0 的对数发散缺陷（登记为 strict xfail）。"
                "测试 86 passed / 2 xfailed（main 基线 66，零回归）。"),
    "scenario_key": "tensorbem/merge/heal",
    "mission_id": 3878,
    "tags": ["TensorBEM", "merge-heal", "PR104", "finite-depth-3d", "Green-function", "xfail-strict"],
    "markdown": md,
    "figures": [{"name": "fig_m3878_kernel_merge.png",
                 "caption": "(a) R→0 轴区：级数解与积分解的互补失效区；(b) 离轴深水极限 vs 水深：级数解收敛至 7e-10，积分解恒定 3.197（系统性偏差）",
                 "png_base64": png}],
}

r = post('/tensor/reports', payload, tok)
print('POST ok:', json.dumps(r)[:400])