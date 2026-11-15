import base64, json, os, pathlib, urllib.request

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
if not BASE.endswith("/api/v1"):
    BASE = BASE.rstrip("/") + "/api/v1"

def post(path, data=None, token=None):
    req = urllib.request.Request(BASE + path,
        data=json.dumps(data).encode() if data else None, method="POST",
        headers={"Content-Type": "application/json",
                 **({"Authorization": f"Bearer {token}"} if token else {})})
    return json.loads(urllib.request.urlopen(req, timeout=300).read())

tok = post("/auth/login", {"username": os.environ["TENSOR_AGENT_USER"],
                           "password": os.environ["TENSOR_AGENT_PASS"]})
tok = tok.get("access_token") or tok.get("token")
print("login ok", len(tok or ""))

# ---- numbers injected programmatically from the committed artifact
d = json.load(open("benchmarks/verified/casepool_bem_cummins_wigley/case_result.json"))
v = d["gates"]["C5c"]["values"]; mw = v["multi_window"]; bar = mw["bar"]
rows = sorted(mw["per_floor"], key=lambda r: -r["floor"])
tbl = "\n".join(f"| {r['floor']*100:g} % | {r['zeta']:.5f} | {r['n_peaks']} |" for r in rows)
gates = d["gates"]
npass = sum(1 for g in gates.values() if g.get("passed"))
ntot = len(gates)
heave = json.load(open("benchmarks/verified/casepool_bem_cummins_wigley/decay_dof_audit.json"))["modes"]["heave"]["values"]

def fig(name, caption):
    return {"name": name, "caption": caption,
            "png_base64": base64.b64encode(pathlib.Path(f"outputs/m22091/{name}").read_bytes()).decode()}

md = f"""# CASE-BEM-CUMMINS-WIGLEY — C5c 自由衰减闸门的独立审计与失效注入

**锚点** `CASE-BEM-CUMMINS-WIGLEY` ｜ **仓** TensorBEM ｜ **commit** `5eb8f18a` ｜ **mission** m22091

## 1 案例介绍

Wigley 数学船型（$y=\\frac{{B}}{{2}}(1-(2x/L)^2)(1-(z/T)^2)$），$L=3.0$ m、$B=0.3$ m、$T=0.1875$ m，
$C_b=4/9$，排水体积 $\\nabla=0.075$ m³，$\\rho=1025$ kg/m³，$g=9.80665$ m/s²。求解 6 自由度刚性辐射的
频域系数 $A(\\omega)$、$B(\\omega)$（频带 0.2–12 rad/s，41 频点，medium 网格 20×6）。

算例判据来自 Cummins (1962) 时域卷积方程与 Kramers–Kronig 因果对，容差 <10%。
Cummins 意义上 $K(t)=\\frac{{2}}{{\\pi}}\\int_0^\\infty B(\\omega)\\cos\\omega t\\,d\\omega$，
时域自由衰减由 $K(t)$ 卷积驱动。

**本单对象**：闸门 C5c（自由衰减阻尼比的多窗口共识估计器），它是整个算例
{npass}/{ntot} 闭合结论所依赖的最新一件证据。C5c 是为「救回」roll 模态而新增的，
一个只被观察到 pass 的闸门与图章无法区分，故本单做**验证欠账**。

roll 模态的物理难点：$I=0.81163$ kg·m²、恢复刚度 $k=48.686$ N·m/rad（$GM=0.065$ m），
$\\omega_n={v['natural_frequency_rad_s']:.3f}$ rad/s，阻尼比很小，衰减不是单指数。

## 2 怎么做的

**（a）独立重建执行路径**（不信任报告叙事，只读代码）：
`oscillator_properties(3)` → `kk_added_mass_closure`（$A_\\infty={v['added_mass_inf_used']:.5f}$）
→ `radiation_memory_kernel`（由 $B(\\omega)$ 余弦变换）→ `cummins_free_decay`（$dt={v['dt']}$，
$T={v['duration']}$ s）→ `_decay_zeta_multi_window`（三个振幅地板各取对数衰减率、取中位数）
→ 频域路线 $\\zeta_{{fd}}=B(\\omega_n)/[2(I+A(\\omega_n))\\omega_n]$（对扫频网格插值）→ 判定。

**（b）契约 vs 实现对照**：逐条比对 docstring 声明的判据与判定代码实际执行的条件。

**（c）失效注入**：把判定规则抽出为纯函数 `c5c_decision()`（无副作用、不依赖求解器），
对 7 条分支逐一注入反例。

**（d）mutation check**：把聚合逻辑回退到修复前语义，确认新测试**会**失败——否则测试只是执行代码。

**（e）工件复算**：用提交的 `sweep_medium.npz` 重算 C5c，逐字段与提交值 `==` 比对。

## 3 计算结果

### 3.1 发现并修复的缺陷

| 编号 | 严重度 | 内容 | 处理 |
|---|---|---|---|
| F1 | 高 | docstring 声明「跨窗口 spread ≤ 5%」为判据，判定代码**从未执行**该条 —— 按契约审计代码的人会相信存在收敛性检查 | 契约改写为如实描述实执行判据 + 新增披露字段 |
| F2 | 中 | 偶数窗口时「中位数」取 `sorted[len//2]` = 两个中的**较大者**，对单调序列系统性上偏 | 改真中位数（`statistics.median`） |
| F3 | 中 | 仅 1 个合格窗口时返回 `converged=True, spread=0.0`，**假称**窗口一致 | 改 `converged=False, spread=nan` |
| F4 | 限制 | 两条路线都源自**同一组** $B(\\omega)$，C5c 是**自洽性**检查，非精度证明 | 已记录，不在本单修 |

### 3.2 roll 三窗口实测（程序注入，非手抄）

| 振幅地板 | $\\zeta$ | 峰数 |
|---|---|---|
{tbl}

spread = **{mw['spread']:.2%}**（判据 bar = {bar:.0%}），**单调递减**——这是非单指数衰减的
物理签名（看得越长，表观衰减率越低），不是散点噪声。

| 量 | 值 |
|---|---|
| consensus $\\zeta$（三窗中位数） | {v['zeta_decay_consensus']:.6f} |
| $\\zeta_{{fd}}$（阻抗路线） | {v['zeta_frequency_domain']:.6f} |
| 相对误差 | {v['relative_error']:.4%} |
| `multi_window.converged` | {mw['converged']} |
| `convergence_claimed`（新增） | {v['convergence_claimed']} |
| `non_exponential_decay`（新增） | {v['non_exponential_decay']} |
| 判定 | {v['status']} |

### 3.3 失效注入（7 条分支全部被证伪过一次）

| 注入 | 期望 | 实测 |
|---|---|---|
| in_band, rel=0, 3 窗 | decided / pass | decided / pass |
| rel == bar（闭区间） | pass | pass |
| rel = bar×(1+1e−9) | beyond_bar / fail | beyond_bar / fail |
| rel = 0.30 | beyond_bar / fail | beyond_bar / fail |
| in_band=False | undecidable / fail | undecidable / fail |
| n_qualifying=1 | undecidable / fail | undecidable / fail |
| rel = inf / nan | fail | fail |

### 3.4 工件复算一致性

重算 C5c 后与提交值逐字段 `==` 比对：`zeta_decay_consensus`、`zeta_frequency_domain`、
`relative_error`、`status`、`converged`、`spread` **全部不变**；仅新增 3 个披露字段。
提交工件 diff = 5 行插入。

## 4 结果图

![fig1](fig1) ![fig2](fig2) ![fig3](fig3)

**图 1**：三个振幅窗口的 $\\zeta$ 随窗口放宽单调下降（{rows[0]['zeta']:.4f} → {rows[-1]['zeta']:.4f}），
与 $\\zeta_{{fd}}$（虚线）的 ±5% 带重叠的是中位数点，但 spread 达 {mw['spread']:.1%}——
「共识」是趋势的中点，不是收敛。
**图 2**：三窗口峰数 8/15/28 均 ≥5，故闸门有 ≥2 窗口可用；这也解释了为何放宽窗口是必要的。
**图 3**：7 条判定分支的 pass/fail 图示。

## 5 注意点

1. **F1 是本次最重要的发现**：放宽本身在物理上可辩护（非单指数衰减下强制小 spread 会误杀
   正确的阻抗对），但**未披露的放宽**就是缺陷。现在从代码 docstring、闸门 values、测试
   三处都能读到这一事实，不再依赖某一份 dev_report 的叙事。
2. **C5c 通过 ≠ 精度证明**：0.46% 是两条**归约路线**（时域卷积 / 频域阻抗）的自洽性数字，
   两路输入是同一组 $B(\\omega)$。对 $B(\\omega)$ 本身精度的证据仍只能来自 C10（Capytaine 外部锚点）
   与 C3（KK 因果闭合）。
3. **中位数不是好估计量**：对单调漂移序列取中位数等于「取趋势中点」，其与 $\\zeta_{{fd}}$ 吻合
   在统计上是可能的巧合。更物理的做法是在最长窗口做 Prony/矩阵铅笔拟合并取最靠近虚轴的极点，
   本单未做。
4. **本机负载纪律**：load15 全程 70–100，重计算（全量 BEM 测试）已递交集群 cj3896；本机只跑
   <8 核 <10 min 的快测试。

## 6 误差判定

| 关键量 | 基准 | 误差 | 是否 <3% |
|---|---|---|---|
| C5c roll consensus $\\zeta$ vs $\\zeta_{{fd}}$ | 频域阻抗路线（**自洽基准**） | **{v['relative_error']:.3%}** | **是**（但仅为自洽性） |
| heave C5 $\\zeta_{{decay}}$ vs $\\zeta_{{fd}}$ | 同上 | {heave['relative_error']:.2%} | 否（>3%，<5% bar 内） |
| C5c roll consensus $\\zeta$ vs C5 默认单窗口 $\\zeta$ | 默认 1% 窗口 | {abs(v['zeta_decay_consensus']-rows[0]['zeta'])/rows[0]['zeta']:.2%} | 否 |
| C5c 跨窗口 spread | 判据 bar 5% | {mw['spread']:.1%} | 否（**未收敛**，已披露） |

**显式回答**：唯一 <3% 的量是最关键量 consensus $\\zeta$ vs $\\zeta_{{fd}}$（0.46%），
但**必须注明**该基准是同一组系数的另一条归约，不是实验/文献基准，故**不能**据此宣称
阻尼比精度 <3%。跨窗口 spread 56.3% 远超 5% bar，说明窗口未收敛——这一点本单不做粉饰，
并已通过 `convergence_claimed=False` 固化进工件。

**结论**：算例 {npass}/{ntot} 闸门通过的**数值结论不变**，但可审计性改变。
disposition：approve with notes。
"""

report = {
    "title": "CASE-BEM-CUMMINS-WIGLEY · C5c 自由衰减闸门独立审计：契约与实现不一致的修复与失效注入",
    "summary": "发现并修复 C5c 闸门 docstring 声明但代码未执行的收敛判据，修复偶数窗口伪中位数与单窗口假共识，补齐 7 条分支失效注入与 mutation check；已提交 roll 数字与通过判定不变，新增 convergence_claimed/non_exponential_decay 披露，明确 0.46% 为自洽性而非精度证明。",
    "scenario_key": "case_bem_cummins_wigley_c5c_audit",
    "mission_id": 22091,
    "tags": ["TensorBEM", "CASE-BEM-CUMMINS-WIGLEY", "audit", "verification", "free-decay", "gate-integrity"],
    "markdown": md,
    "figures": [
        fig("fig1_window_drift.png", "C5c roll：三窗口 zeta 单调漂移 56.3%，共识中位数落入 zeta_fd 的 ±5% 带"),
        fig("fig2_peaks.png", "三窗口捕获峰数 8/15/28，均满足 ≥5 峰合格条件"),
        fig("fig3_branches.png", "C5c 判定规则 7 条分支的失效注入结果"),
    ],
}
r = post("/tensor/reports", report, token=tok)
print(json.dumps(r, ensure_ascii=False)[:600])
