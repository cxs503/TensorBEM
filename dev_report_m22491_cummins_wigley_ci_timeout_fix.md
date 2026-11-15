# CASE-BEM-CUMMINS-WIGLEY — CI 超时修复

**Mission**: m22491 | **Anchor**: `CASE-BEM-CUMMINS-WIGLEY` | **仓**: TensorBEM

## 1 问题发现

`test_production_artifact_replays_to_the_published_verdict` 在 CI 中持续超时（30s 门限），导致算例裁决无法在自动化流程中复现。

### 根因分析

`gates_from_artifact()` 默认计算全部 16+ 个闸门，其中三项是重计算：

| 闸门 | 计算内容 | 耗时 | 测试是否需要 |
|------|---------|------|------------|
| C3a | 锚定 KK 闭合（需 BEM 镜像求解 A_inf） | ~58s | ❌ |
| C5b | 3 自由度自由衰减审计 | ~8s | ❌ |
| C5c | roll 模式鲁棒衰减 | ~31s | ❌ |

而该测试仅验证 C2, C3, C4, C5, C6a, C6b —— 这些闸门**不需要**锚定求解或多自由度衰减。

总耗时 ~100s >> 30s 超时门限。

## 2 修复方案

### 2.1 `gates_from_artifact` 新增 `include_anchor` 参数

```python
def gates_from_artifact(
    path: str | Path,
    *,
    dof_free_decay: int = 2,
    include_geometry: bool = True,
    include_anchor: bool = True,          # ← 新增
    corrupted_control: bool = True,
    include_decay_modes: bool = True,
    include_external_anchor: bool = True,
    extended_path: str | Path | None = None,
) -> dict[str, Any]:
```

当 `include_anchor=False` 时，跳过 C3a 和 C7 闸门（两者都需要 `anchor_for_sweep()` → BEM 镜像求解）。

### 2.2 测试更新

```python
def test_production_artifact_replays_to_the_published_verdict() -> None:
    # Skip the anchor BEM solve (~60s) and multi-DOF decay (~30s) since this test
    # only verifies C2, C3, C4, C5, C6a, C6b -- none of which need them.
    replay = case.gates_from_artifact(
        ARTIFACT_DIR / "sweep_medium.npz",
        include_anchor=False,
        include_decay_modes=False,
    )
```

## 3 验证结果

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| 测试耗时 | >30s (TIMEOUT) | 7.79s |
| 闸门验证 | 无法完成 | C2/C3/C4/C5/C6a/C6b 全部 PASS |
| 裁决一致性 | 无法验证 | 与 published case_result.json 一致 |

```
tests/test_case_bem_cummins_wigley.py::test_production_artifact_replays_to_the_published_verdict PASSED [100%]
======================== 1 passed, 2 warnings in 7.79s =========================
```

## 4 改动文件

- `src/tensorbem/benchmarks/case_bem_cummins_wigley.py`: 新增 `include_anchor` 参数，条件化 C3a/C7 计算
- `tests/test_case_bem_cummins_wigley.py`: 测试跳过不必要的重计算

## 5 后续建议

1. **其他超时测试**: `test_anchor_is_mesh_converged_between_the_case_meshes` 和 `test_replay_artifact_reports_the_anchor_gates` 也因 BEM 求解超时，建议类似处理（增加 skipif 标记或拆分轻量/重量测试）
2. **记忆核积分优化**: `_integrate_cummins` 中的卷积是 O(N²) 复杂度，可用 FFT 加速到 O(N log N)，但目前不是 CI 超时的主要瓶颈
3. **锚定缓存持久化**: `_ANCHOR_CACHE` 是进程内缓存，CI 每次重启都需重新求解。可考虑将锚定结果序列化到 `anchor_audit.json`（已存在），`gates_from_artifact` 优先从文件加载

## 6 复现

```bash
cd /root/tp-agent/work/m22491
PYTHONPATH=src python3 -m pytest tests/test_case_bem_cummins_wigley.py::test_production_artifact_replays_to_the_published_verdict -v --timeout=60
```
