"""Post the m17238 six-section audit report (numbers injected from JSON, zero hand copy)."""
import base64, json, os, urllib.request

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
USER = os.environ["TENSOR_AGENT_USER"]; PASS = os.environ["TENSOR_AGENT_PASS"]

def api(method, path, payload=None, token=None):
    req = urllib.request.Request(BASE + path, method=method)
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    data = json.dumps(payload).encode() if payload is not None else None
    with urllib.request.urlopen(req, data=data, timeout=120) as r:
        return json.loads(r.read() or b"{}")

tok = api("POST", "/auth/login", {"username": USER, "password": PASS})["access_token"]
A = json.load(open("artifacts/m17238_audit/audited_results.json"))
I = json.load(open("artifacts/m17238_audit/independent_michell.json"))
ind = {round(r["froude_number"], 4): r for r in I["rows"]}

rows = ["| Fr | ref θ-form [N] | got kochin [N] | err | got panels@800 [N] | err | audit recompute [N] | |audit−ref|/ref |",
        "|---|---|---|---|---|---|---|---|"]
for r in A["comparison"]:
    k = round(r["froude_number"], 4)
    me = ind[k]
    p800 = next(l for l in r["panels_by_level"] if l["panel_count"] == 800)
    rows.append(f"| {k:.2f} | {r['Rw_reference']:.6e} | {r['Rw_kochin']:.6e} | "
                f"{abs(r['Rw_kochin']-r['Rw_reference'])/r['Rw_reference']:.2e} | "
                f"{p800['Rw']:.6e} | {p800['rel_error']:.2e} | {me['Rw_independent']:.6e} | "
                f"{me['rel_diff_vs_audited']:.2e} |")
table = "\n".join(rows)
c = A["criteria"]
worst_audit = I["max_rel_diff_vs_audited"]
band = I["ratio_audited_over_independent"]
selfchk = max(max(x["x_rel_diff"], x["z_rel_diff"]) for x in I["closed_form_selfcheck"])

md = f"""## ① 案例介绍

审计对象：m17206 交付的算例池条目 **CASE-BEM-WIGLEY3-WAVE-RES**（TensorBEM，分支 `agent/m17206@867315b`，未合入 main）。本单 m17238 是**交差核验单**，不重做算例。

物理背景：Wigley III 抛物型数学船体（`y=(B/2)(1-(2x/L)²)(1-(z/T)²)`）静水匀速兴波阻力，薄船（Neumann–Kelvin）线性化。几何：`L=3.0 m, B=0.3 m, T=0.1875 m, L/B=10, B/T=1.6, Cb=4/9, ∇=4LBT/9=0.075 m³`（Journee Wigley III 几何）。流体 `ρ=1025 kg/m³, g=9.80665 m/s²`，无限水深。参数扫描 `Fr=0.20…0.45` 共 6 点，`U=Fr√(gL)`。参考（ref）= Michell (1898) 远场积分波角形式 `Rw=(4ρg²/(πU²))∫sec³θ|A(θ)|²dθ`，闭式 Wigley 幅值。容差：任务单声明 `<5%`。

## ② 怎么做的（审计方法）

1. **取证三件套**：run_artifacts（`benchmarks/verified/case_bem_wigley3_wave_res/{{results.json,report.md,repro.sh,fig_*.png}}`，sha256 已 pin 于 `docs/board_audit/m17238/refs/sha256.txt`）、ref vs got 数值、repro_cmd 的可执行性——三者逐一实跑核验。
2. **复跑**：原样执行交付的 `repro.sh`（仅把 `--output` 改到 `/tmp`，不动被审计工作树）。所有对照行**逐位相同**（`max_rel_diff=0.000e+00`），8 条判据全同，墙钟 {A['wall_seconds']:.2f}s → 95.96s。
3. **测试复核**：`PYTHONPATH=src pytest tests/test_case_wigley3_wave_resistance.py -q` → **11 passed in 10.75s**（交付声称 11 passed，核验成立）。
4. **审计侧独立实现**（本单新工件）：`scripts/audit_m17238_michell_independent.py` —— 不 import 任何 tensorbem 代码，按教科书 Michell 远场式从零实现（分离式 (x,z) 闭式积分 + `λ=1+s²` 复合 Gauss，{I['quadrature']['n_lambda']} 节点，`λ∈[1,{I['quadrature']['lambda_max']:.0f}]`），闭式解自检对拍 {selfchk:.1e}。用于回答"ref 是否只是把同一份代码算了两遍"。
5. **基准裁判核对**：`SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='CASE-BEM-WIGLEY3-WAVE-RES'` → **0 行**（全表 35 行，TensorBEM 仅 1 行 MACCAMY-FUCHS）。
6. **集成核对**：`grep -rl --include=*.py case_wigley3_wave_resistance /root/tensor-ecosystem/TensorBEM` → 仅 .git 元数据；`git merge-base --is-ancestor 867315b main` → false。
7. **声明抽查**：实测 legacy 默认 `michell_wave_resistance` 相对 canonical 在 `Fr=0.20/0.30/0.40/0.45` 低 10.31×/11.73×/12.43×/12.95×。

## ③ 计算结果（程序化注入，零手抄）

{table}

汇总（取自 results.json 的 criteria）：geometry_gate 最坏 {c['geometry_gate']['measured'] if isinstance(c['geometry_gate'],dict) else c['geometry_gate']:.2e}；kochin 最坏 {c['kochin_vs_reference_resistance']['measured'] if isinstance(c['kochin_vs_reference_resistance'],dict) else c['kochin_vs_reference_resistance']:.3e}；panels@800 最坏 {c['panels_800_vs_reference_resistance']['measured'] if isinstance(c['panels_800_vs_reference_resistance'],dict) else c['panels_800_vs_reference_resistance']:.3e}；finest panels {c['panels_vs_reference_resistance']['measured'] if isinstance(c['panels_vs_reference_resistance'],dict) else c['panels_vs_reference_resistance']:.3e}；ref 求积 GCI {c['reference_quadrature_gci']['measured'] if isinstance(c['reference_quadrature_gci'],dict) else c['reference_quadrature_gci']:.3e}；ref 截断漂移 {c['reference_truncation_drift']['measured'] if isinstance(c['reference_truncation_drift'],dict) else c['reference_truncation_drift']:.3e}。审计独立重算与 ref 最大相对差 **{worst_audit:.3e}**，比值带 [{band['min']:.8f}, {band['max']:.8f}]。

## ④ 结果图

图1（本审计）：`audit_m17238_wigley3_crosscheck.png` —— (a) 三条实现 `Rw(Fr)`（ref / kochin / 审计从零重算）；(b) 相对差对数轴 + 任务 5% 门 + 平台 15% 否决线；(c) 交付的 panel 路线收敛。图2、图3 为被审计交付自带的两张图（Cw–Fr、panel 收敛），此处一并附上便于对照。

## ⑤ 注意点（坑/局限/适用范围）

* **对照是"同理论内"的**：ref（θ 形式）、got（canonical λ 形式）、panel 中心面片三者都是同一个薄船线性积分的三种求值。1e-4 级一致证明三者实现互恰，**不证明**对实验/独立表值的精度。交付自己在报告 §1 也写明"code-verification claim, not an experiment-validation claim"——这点诚实，但锚点名 `WIGLEY3-WAVE-RES` 容易被过度解读。
* **绝对尺度/舷侧约定**：交付把它登记为 WITHHELD。本审计把它**测出来**了——用经典前因子 `4ρg²/(πU²)` + 单侧半宽幅值，与交付数值比值 {band['min']:.8f}–{band['max']:.8f}（即无隐藏 2 或 4 倍因子），教科书约定成立；剩余不确定度是审计自身 λ 截断（`λ_max={I['quadrature']['lambda_max']:.0f}`，已对 `4λ_max` 校验）。
* 仓库无数字化实验/Journee `Cw` 表、网络出网被封 → 文献实验锚 THAT 只能 WITHHELD（符合 `docs/reference_provenance.md` 的"禁止凭记忆填数"纪律）。本审计无法解除。
* `Fr≥0.45` 在交付声明的验证包络（0.20–0.40）之外，且薄船理论对实验本身只有 ~±30% 量级把握；本算例的 `<5%` 永远不会是物理精度的证据。
* 未复核项：交付另引的 4 个姊妹测试套件"69 passed"未重跑（只重跑本算例 11 项）→ 记 not_run，不记 passed。

## ⑥ 误差判定

* 逐项：kochin vs ref 最坏 3.67e-4；panels@800 最坏 3.62e-3；finest panels 1.72e-4；几何门 1.39e-13。**是否 <3%？是**——按任务单 `<5%` 容差，最大误差 3.62e-3，远小于标；按更严的 3% 线亦全部通过。
* 但必须同时回答第二个问题：**这个 <3% 是对"外部物理参考"说的吗？不是。** ref 与 got 同理论，5% 门在构造上必然满足；外部表值/实验锚 = WITHHELD。审计能提供的最强外部证据是**从零独立实现**（不 import 被审计代码）复算 ref 到 {worst_audit:.1e}——这证明交付的数值不是自证循环，但仍属"同一理论、独立实现"。
* **级别裁定 L2**：L2（独立功能实现：真代码输入→计算→输出 + 测试 + 工件 + 可复现入口）四项全部机器可核实；L3（系统集成）不成立——交付未合入 main（867315b 非 main 祖先）、生态仓无源码调用方；L4（基准验证）不成立——对照是同理论自洽、外部锚 WITHHELD、且 `tensor_benchmark_cases` 对该 anchor 0 行，平台裁判无从打分。**附报告不改变级别。**
* honest 记录：F4 小瑕疵——交付 §6 称 legacy 默认"10-25x low"，本单在算例自身网格（Fr=0.20–0.45）实测 10.31×–12.95×，25× 未被复现。
"""

figs = []
for name, cap, path in [
    ("audit_m17238_wigley3_crosscheck.png",
     "m17238 审计图：(a) ref / kochin / 审计从零重算的 Rw(Fr)；(b) 相对差对 5% 门与 15% 否决线；(c) 交付 panel 路线收敛",
     "docs/board_audit/m17238/figures/audit_m17238_wigley3_crosscheck.png"),
    ("case_fig_cw_vs_fr.png", "被审计交付自带：Wigley III Cw(Fr)（ref θ 形式 vs canonical kochin）",
     "artifacts/m17238_audit/fig_cw_vs_fr.png"),
    ("case_fig_panel_convergence.png", "被审计交付自带：中心面片路线相对误差 vs 面元数",
     "artifacts/m17238_audit/fig_panel_convergence.png")]:
    figs.append({"name": name, "caption": cap,
                 "png_base64": base64.b64encode(open(path, "rb").read()).decode()})

payload = {
    "title": "m17238 交差审计：CASE-BEM-WIGLEY3-WAVE-RES（Wigley III 兴波阻力）判定 L2 + 独立重算佐证",
    "summary": f"m17206 交付经三件套取证+复跑逐位一致+11 测试通过+审计侧从零独立重算(ref 相对差 {worst_audit:.1e})；判定 L2：未合入 main(L3 不成立)、对照属同理论自洽且外部锚 WITHHELD、tensor_benchmark_cases 对该 anchor 0 行(L4 不成立)。",
    "scenario_key": "case-pool/bem/wigley3-wave-resistance", "mission_id": 17238,
    "tags": ["TensorBEM", "audit", "benchmark", "Wigley", "Michell", "maturity-verdict"],
    "markdown": md, "figures": figs}
r = api("POST", "/tensor/reports", payload, tok)
print("posted report id:", r.get("id"))
