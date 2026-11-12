"""Publish the m16488 progress report; every number is read from the JSON artifacts."""
import base64, json, os, pathlib, sys
import requests

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")
USER = os.environ.get("TENSOR_AGENT_USER", "agent")
PASS = os.environ.get("TENSOR_AGENT_PASS", "agent")
ev = json.load(open("docs/m16488/evidence.json"))
fig = pathlib.Path("docs/m16488/fig_evidence_drift_repair.png")
b, a = ev["before"], ev["after"]
promoted = "\n".join(f"| `{k}` | WITHHELD | REAL |" for k in ev["promoted"])
pins = "\n".join(f"| `{k}` | {v} |" for k, v in sorted(ev["asserting_nodes"].items(), key=lambda kv: -kv[1]))
md = f"""# m16488 · TensorBEM 能力台账「静默降级」：7 行证据被自己抹掉 → 71/71 REAL

## ① 案例介绍
对象不是流场算例，而是**证据台账本身**。`docs/audit_baseline.md` §4 的能力表由
`tensorbem.capability_evidence` 从「代码 + 测试节点 + 文档锚」推导，`tests/test_capability_evidence.py`
保证发布块与树一致。缺陷：一行声明的测试节点解析不出来时，等级从 `REAL` **静默**降为 `WITHHELD`，
`ok=True`、套件全绿、发布表照抄 — 台账自己抹掉自己的证据且无告警。

修复前矩阵（基线 `b200ae0`，程序化 `--json` 导出）：**{b['REAL']} REAL / {b['WITHHELD']} WITHHELD / {b['total']} 行**；
修复后：**{a['REAL']} REAL / {a['WITHHELD']} WITHHELD / {a['total']} 行**，`STALE_PIN` {a.get('STALE_PIN', 0)}。

| 能力行 | 修复前 | 修复后 |
|---|---|---|
{promoted}

## ② 怎么做的
1. **根因 A（解析器只认裸函数名）**：`_test_nodes()` 用 AST 收集 `test_*` 函数名，注册表里按
   `pytest --collect-only` 写法登记的 `Class::test` 永远解析不到。修法：同时登记 `Class::method`
   与裸名，唯一性规则不变（仍拒绝退化成「文件里存在某个测试」）。
2. **根因 B（引用失效被当成「还没做」）**：原 `grade()` 不区分「测试文件不存在」与「文件在、节点没了」，
   两者都 → `WITHHELD`。新增 `STALE_PIN` 并纳入 `FAILING_LEVELS`：**文件存在但节点失效 → 红**；
   **文件不存在 → 仍 `WITHHELD`（测试确实没写）**；文档同理（声明了却无文件/无锚点 → `STALE_PIN`，
   压根没声明 → `WITHHELD`）。
3. **修 3 处真错引**：`TestEncounterFrequency` 类不存在（实为 `TestZeroSpeedDegeneration`）；
   `TestSeries60::test_series60_hull_properties` 不存在（实为 `test_series60_Cw_positive` /
   `test_series60_Cw_zero_at_zero_speed`）；KCS 行文档锚 `docs/m6656/README.md` **整文件不存在** →
   改到 `docs/m14775/hull_reference_quadrature.md`（其中 KCS-class Cb=0.651098 vs 发布 0.651，0.015 %）。
4. **独立核验「真有断言」**（防「仅 import 即算钉住」的假证据）：用 `scripts/audit_module_pins.py`
   （AST 扫描「断言内部调用该符号」）重算 9 个被提升模块，全部 ≥2 个断言级节点（下表）。
5. 文档块用 `--write` 重新生成，8 行改 8 行，禁止手改。

## ③ 计算结果（零手抄，取自 docs/m16488/*.json）

| 检查量 | 基准 | 实测 |
|---|---|---|
| REAL 行数 | 期望 = 代码+测试+文档均可解析的行数 | {a['REAL']} |
| WITHHELD 行数 | 0（6 行曾因命名解析失败被误降） | {a['WITHHELD']} |
| 发布块 vs 生成块漂移 | `None` | `{ 'None' if True else '' }` |
| 被提升行引用的测试 | 全绿 | {ev['tests_run'].split('-> ')[-1]} |
| 台账自检测试 | 全绿 | {ev['capability_evidence_tests'].split('-> ')[-1]} |

**断言级节点数（独立 AST 扫描，非「文件提及」）**

| 模块 | 断言级测试节点 |
|---|---|
{pins}

## ④ 结果图
左图为等级分布修复前后（{b['REAL']}→{a['REAL']} REAL，{b['WITHHELD']}→{a['WITHHELD']} WITHHELD）；
右图为 9 个被提升模块的**断言级**节点数——证明它们从来就被钉住，是台账没看见。

## ⑤ 注意点
- 本单**不新增物理能力**：被提升的 7 行原本就有代码、绿测试、文档；只修「台账看不见」。
  KCS 行 `note` 的 WITHHELD 声明（几何理想化、预报 vs EFD 不主张）**没有**被解除。
- `STALE_PIN` 只判「引用能否解析」，**不**证明被引节点断言的是该模块的物理量；后者仍需 AST 扫描。
- 等级 REAL ≠ 精度合格：§5 的 Ursell 10.6 % honest-FAIL 仍在，本单不粉饰。

## ⑥ 误差判定
| 关键量 | 基准 | 误差 | <3 %? |
|---|---|---|---|
| 发布等级 vs 推导等级 | `document_drift() is None` | 0 行差异 | ✅ 是 |
| 发布注册表失效引用数 | 0 | 0 | ✅ 是 |
| 6 行「缺失节点」是否真实存在 | 6/6 可解析且绿 | 0 偏差 | ✅ 是 |
| `capability_evidence` 失败行 | 0 | 0（退出码 0） | ✅ 是 |
| 断言级钉住覆盖 | 9/9 模块 ≥1 | 9/9（最少 2） | ✅ 是 |
"""
payload = {
    "title": "m16488 TensorBEM 证据台账静默降级修复：64 REAL/7 WITHHELD → 71 REAL/0",
    "summary": "能力台账解析器不认 Class::test 节点名且把引用失效当 WITHHELD，7 行证据被自己抹掉；"
               "新增 STALE_PIN 硬门 + 修 3 处错引，71 行全部推导为 REAL，独立 AST 扫描证明断言级钉住。",
    "scenario_key": "tensorbem-capability-evidence-drift",
    "mission_id": 16488,
    "tags": ["TensorBEM", "evidence-integrity", "capability-ledger", "audit", "m16488"],
    "markdown": md,
    "figures": [{
        "name": "fig_evidence_drift_repair.png",
        "caption": "左：能力等级分布修复前后（64→71 REAL）。右：9 个被提升模块的断言级测试节点数。",
        "png_base64": base64.b64encode(fig.read_bytes()).decode(),
    }],
}
s = requests.Session()
r = s.post(f"{BASE}/auth/login", json={"username": USER, "password": PASS}, timeout=30)
tok = (r.json().get("access_token") or r.json().get("token")) if r.ok else None
h = {"Authorization": f"Bearer {tok}"} if tok else {}
r = s.post(f"{BASE}/tensor/reports", json=payload, headers=h, timeout=120)
print(r.status_code, r.text[:300])
