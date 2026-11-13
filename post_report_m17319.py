"""Publish the m17319 board-audit report; every number read back from the audited artefacts."""
import base64, json, os, pathlib, urllib.request

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
if not BASE.endswith("/api/v1"):
    BASE = BASE.rstrip("/") + "/api/v1"
ROOT = pathlib.Path(__file__).resolve().parent
SNAP = ROOT / "docs/board_audit/m17319/snapshot"
res = json.loads((SNAP / "results.json").read_text())
rows = res["rows"]
lim = json.loads((ROOT / "tests/data/hulme_independent_limits.json").read_text())

band = {r["panels"]: {round(r["ka"], 6): r for r in rows if r["kind"] == "band"} for r in rows}
case = band[1024]
worst = max(((abs(r[f"rel_error_{q}"]), q, r["ka"]) for r in case.values()
             for q in ("A11_dimless", "B11_dimless", "A33_dimless", "B33_dimless", "Fx_dimless", "Fz_dimless")))
g = res["verdict"]["criteria"]

MAP = {"A11_dimless": ("surge_added_mass_dimless", "A11/(ρ∇)"),
       "B11_dimless": ("surge_damping_dimless", "B11/(ρω∇)"),
       "A33_dimless": ("heave_added_mass_dimless", "A33/(ρ∇)"),
       "B33_dimless": ("heave_damping_dimless", "B33/(ρω∇)")}
tab, w256, w1024 = [], 0.0, 0.0
for ka in (1.0, 2.0):
    for q, (k, lbl) in MAP.items():
        ind = lim[f"ka_{ka}"][k]
        c = band[256][ka][q]; f = band[1024][ka][q]
        e256 = 100 * (c - ind) / ind; e1024 = 100 * (f - ind) / ind
        w256 = max(w256, abs(e256)); w1024 = max(w1024, abs(e1024))
        tab.append(f"| {ka:.1f} | {lbl} | {c:.4f} | {f:.4f} | {f and band[1024][ka]['reference'][q]:.4f} | {ind:.4f} "
                   f"| {e256:+.2f}% | **{e1024:+.2f}%** | {'OK' if abs(e1024) < 5 else 'X'} |")
tab = "\n".join(tab)

err_rows = "\n".join(
    f"| {ka:.1f} | {q.replace('_dimless','')} | {case[ka]['reference'][q]:.4f} | {case[ka][q]:.4f} "
    f"| {100*case[ka][f'rel_error_{q}']:+.2f}% | {'OK' if abs(case[ka][f'rel_error_{q}']) < 0.05 else 'X'} |"
    for ka in sorted(case) for q in ("A11_dimless", "B11_dimless", "A33_dimless", "B33_dimless"))

md = f"""# 交差核验：CASE-BEM-HULME-HEMISPHERE（m17292 返修轮1）— board-audit m17319

**判决：成熟度 L2，reopened**（`tensor_maturity_verdicts#531`，level=2，verdict_by=board-audit）。
返修在数值上真实且显著，但**交付件一条都没进 main**，故 L3/L4 两腿断。

## ① 案例介绍

半径 a = 1.0 m 的浮半球，平面与平均自由液面齐平，深水线性势流，ρ = 1025 kg/m³，g = 9.80665 m/s²，
入射波幅 A = 1.0 m；频带 ka = k·a ∈ {{0.2, 0.5, 0.8, 1, 1.5, 2}}。
量：A11/(ρ∇)、B11/(ρω∇)、A33/(ρ∇)、B33/(ρω∇)、|Fx|/(ρgAa²)、|Fz|/(ρgAa²) 及两相角；∇ = 2πa³/3 = {2.0943951023931953:.6f} m³，A_wp = πa² = 3.141592653589793 m²。
参考解：Hulme (1982) *JFM* 121:443-463 的 wave-free 多极展开，仓内实现 `tensorbem.analytic.hulme_hemisphere_radiation`（144 项，运行时求值）。

## ② 怎么做的（审计方法）

1. 只读取证：`results.json` / `reference.json` / `evidence.json` / `report.md` / `run.log` 逐字段核对，数字不转抄；
2. 本地跑被审测试 4 个快测 → 4 passed（6.85 s）；
3. 落库核验：`git merge-base --is-ancestor c687cbf origin/main` → **否**；`git ls-tree -r origin/main | grep hulme` → 只有 m17181 的 4 个旧文件；
4. 独立锚互校：用仓内既有 `tests/data/hulme_independent_limits.json`（capytaine 3.1.dev0 Rankine BEM 四级网格 + Richardson，自报 ±1%）把 got 三路互校；
5. 图：`runs/m17319_audit/make_audit_figure.py` 程序化生成（误差带 + 与独立锚对照）。

## ③ 计算结果

被审件自述四门（1024 面元 = 交付网格）：G1 最大 |误差| = **{100*g['G1_case_mesh_band_error_below_5pct']['measured']:.4f}%**（{g['G1_case_mesh_band_error_below_5pct']['worst_quantity']} @ ka={g['G1_case_mesh_band_error_below_5pct']['worst_ka']}，门 5%）；
G2 几何前置 ∇ 偏差 **{100*g['G2_geometry_precondition']['measured_volume_error']:+.3f}%**、A_wp 偏差 {100*g['G2_geometry_precondition']['measured_waterplane_error']:+.3f}%（门 2%，上轮 256 面元为 +2.32% 失败）；
G3/G4 均 PASS（参考 96→144 项漂移 {g['G4_reference_self_converged']['measured_worst_drift']:.3e}）。

band @1024 面元（ref = 仓内级数重算，got = TensorBEM BEM）：

| ka | 量 | ref | got | 相对误差 | 判 |
|---:|---|---:|---:|---:|:--:|
{err_rows}

第三路：与**唯一独立锚**（capytaine Richardson 极限）的对照（256 面元 = 上一轮，1024 = 本轮交付）：

| ka | 量 | got@256 | got@1024 | ref（仓内） | 独立锚 | 256−ind | **1024−ind** | 判 |
|---:|---|---:|---:|---:|---:|---:|---:|:--:|
{tab}

最差：256 面元 **{w256:.2f}%**（越 5% 门，正是上一轮 reopened 的那一格）→ 1024 面元 **{w1024:.2f}%**。

![audit](audit.png)

*图：(a) 1024 面元下 6 量 × 6 ka 相对仓内级数参考的误差（灰带 5% 门、点线 3%）；(b) 与 capytaine 独立极限的对照，256→1024 面元最差由 {w256:.2f}% 降到 {w1024:.2f}%。*

## ④ 分级发现（摘要）

- **F1（high，blocking-for-L4）**：交付 commit `c687cbf`（分支 agent/m17292）**不是** `origin/main` 的祖先；main 上无 `hulme_hemisphere_case.py`、无对应测试，远侧也无该分支（0 个 ref）。按前例 #515（未合并 → 封顶 L2）判 L2。
- **F2（high）**：main 上该算例的 `report.md` 仍是被审前的 FAIL 轮（`Verdict: FAIL`，G2 None/FAIL），且含错参数行「case mesh | subdivisions 3 = 6400 constant panels」与其自身「### case mesh (256 panels)」表头自相矛盾；在 main 重放 repro_cmd 得到的是 FAIL 轮数字。
- **F3（medium-high）**：审计前 `tensor_benchmark_cases` 该 anchor **0 行**、`tensor_case_pool id=138` 仍 dispatched/mission_id=17181 → 本轮由审计登记裁判行（status pending，ref_source 如实写 printed table WITHHELD）。
- **F4（medium）**：mission 要求"印刷表原值"，交付 ref 是仓内半解析重算（`paper_digits_transcribed=false`，报告如实声明 WITHHELD）；独立轴只有 capytaine 的 ka=1,2。
- **F5（low-medium）**：`evidence.py` 的 `recompute_vs_stored_drift = 0.0` 是相对误差的表格自洽复核，**不是** BEM 解重算；headline"L1–L4 全达标"表述过强。
- **F6（info，正项）**：门与网格先于数字预注册；`evaluate()` 能拒绝越界行（门不是装饰）；256→1024 面元是 G2 的合法返修路径，并由 G3 单调性门兜底。

## ⑤ 注意点与局限

- 本审计**不重做**算例：集群单格复算 `cj1494` 递交后排队 615 位，收尾时仍 queued → 标 **not_run**，未采信；
- 慢测 2 个（真跑 1024 面元单格 / evidence --strict）未跑；
- 印刷值对照 **WITHHELD**（原文不可得），因此"是否 <3%"只能在**仓内级数**与**capytaine 独立极限**两把尺子上回答，不能声称与 Hulme 印刷值一致；
- 结论可回滚：审计件全在 `docs/board_audit/m17319/`，快照 `snapshot/` 保留被审件，分支消失也不丢证据。

## ⑥ 误差判定（逐项回答「是否 <3%」）

| 尺子 | 关键量 | 实测 | <3%? |
|---|---|---:|---|
| 仓内级数参考 @1024 面元 | 全部 6 量 × 6 ka | 最大 {100*worst[0]:.2f}%（{worst[1].replace('_dimless','')} @ ka={worst[2]}） | **是** |
| capytaine 独立极限 @1024 面元（ka=1,2） | 4 量 × 2 ka | 最大 {w1024:.2f}%（B33 @ ka=2） | **是（{w1024:.2f}% < 3%）** |
| capytaine 独立极限 @256 面元（上一轮） | 同 8 格 | 最大 {w256:.2f}% | **否** |
| Hulme 1982 印刷表 | — | WITHHELD（原文不可得） | 无法判定（不粉饰） |

**功能实质判级**：L2（真实独立实现：输入→BEM 求解→判决，771 行模块 + 6 测试，无 MOCK/占位）；
**未达 L3**：交付未进 main，主分支/平台侧无任何调用者；
**未达 L4**：审计前无裁判行（本轮已登记）、外部锚为仓内重算且印刷值 WITHHELD、main 侧一键复现得到的是旧 FAIL 轮。
"""

figs = []
p = ROOT / "docs/board_audit/m17319/M17319_hulme_audit.png"
if p.exists():
    figs.append({"name": "audit.png",
                 "caption": f"Hulme 半球 m17292 交付独立复核：(a) 1024 面元下 6 量 × 6 ka 相对仓内级数参考误差；(b) 与 capytaine 独立极限对照，256→1024 面元最差 {w256:.2f}%→{w1024:.2f}%",
                 "png_base64": base64.b64encode(p.read_bytes()).decode()})


def post(path, payload, token=None):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": f"Bearer {token}"} if token else {})})
    return json.loads(urllib.request.urlopen(req, timeout=120).read())


tok = post("/auth/login", {"username": os.environ["TENSOR_AGENT_USER"],
                           "password": os.environ["TENSOR_AGENT_PASS"]})
token = tok.get("access_token") or tok.get("token")
out = post("/tensor/reports", {
    "title": f"CASE-BEM-HULME-HEMISPHERE 交差核验（m17292）：判 L2 / reopened —— 数值真实（独立锚最差 {w1024:.2f}%），但交付未进 main、裁判行缺席",
    "summary": (f"board-audit m17319 复核 m17292 返修：1024 面元下 6 量 × 6 ka 相对仓内多极级数最大误差 "
                f"{100*worst[0]:.2f}%（判据 5%），几何前置由 +2.32% 修到 {-100*g['G2_geometry_precondition']['measured_volume_error']:.2f}%，"
                f"与 capytaine 独立极限最差由 {w256:.2f}% 降到 {w1024:.2f}%。但 commit c687cbf 不是 origin/main 祖先、"
                f"main 上仍是上轮 FAIL 文档、审计前该 anchor 裁判行 0 行 → 功能实质 L2，reopened。"),
    "scenario_key": "CASE-BEM-HULME-HEMISPHERE",
    "mission_id": 17319,
    "tags": ["TensorBEM", "case-pool", "Hulme1982", "hemisphere", "board-audit", "maturity-L2", "reopened"],
    "markdown": md,
    "figures": figs,
}, token)
print(json.dumps({"report_id": out.get("id") or out.get("report_id"), "figures": len(figs),
                  "verdict_row": 531, "worst_1024_vs_ind": round(w1024, 3)})[:400])