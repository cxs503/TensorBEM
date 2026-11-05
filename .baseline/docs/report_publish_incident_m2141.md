# 报告投递异常记录（m2141，2026-11-03）

本单成果（TensorBEM 静水恢复矩阵闭式解复核）已落盘于 commit `fb4c1b1`
（`docs/hydrostatic_verification.md`、`hydrostatic_verification_report.json`、
`docs/figures/hydrostatic_{convergence,exactness}.png`、
`tests/test_hydrostatic_convergence.py`、`src/tensorbem/benchmarks/hydrostatic_{convergence,figures}.py`）。

## 平台侧问题（只记录，不修：TensorPlatform 由 owner 指定专人重构）

对 `POST /api/v1/tensor/reports` 的三次投递均未写入本单内容：

| 尝试 | 返回 | 现象 |
|---|---|---|
| 1（body: title=静水恢复矩阵闭式解复核, mission_id=2141） | HTTP 500 | `UniqueViolation: uq_reports_mission_md5`，Key `(mission_id, md5(markdown)) = (2123, cc486ef6…)` —— 与提交内容无关的 mission 2123 判重 |
| 2（markdown 追加工件行） | HTTP 200, id=2096, mission_id=2123 | 返回记录标题为「TensorLBM 自由液面重力波…」，与本单内容无关 |
| 3（markdown 追加投递戳 + mission_id=2141） | HTTP 200, id=2097, mission_id=2141 | 返回记录标题仍为「TensorLBM 自由液面重力波…」 |

`GET /tensor/reports?limit=40` 检索本单标题 0 命中，确认本单报告未入库。
推断：报告服务在插入/回显路径复用了与请求体无关的缓存或请求上下文
（mission_id 被改写为 2123、title/markdown 回显他人记录），属平台缺陷。

**因此本单的报告纪律以 worktree 工件为准**：`docs/hydrostatic_verification.md`
已按六要素撰写（案例介绍 / 方法 / 数字表 / 图 / 注意点 / 误差判定，含"是否 <3%"
逐项回答），报告 JSON 证据与图在 commit `fb4c1b1` 内可直接复核
（`PYTHONPATH=src python -m tensorbem.benchmarks.hydrostatic_figures --out-root .` 可整体重生成）。
