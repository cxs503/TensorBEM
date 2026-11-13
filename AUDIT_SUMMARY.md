# 交差审计总结：m17178 CASE-BEM-ANNULUS-LAPLACE

**审计任务**：m17234  
**审计对象**：m17178（CASE-BEM-ANNULUS-LAPLACE·算例池·环形域 Laplace（多连通）·TensorBEM）  
**审计时间**：2026-11-13  
**审计人**：board-audit

---

## 审计结论

**成熟度等级**：**L4（基准验证）** ✅

**maturity_gate 裁定**：**confirmed（消账）**

---

## 逐项核验结果

### ① 工件存在性 ✅

| 工件 | 路径 | 大小 | 状态 |
|---|---|---|---|
| results.json | benchmarks/verified/bem_annulus_laplace/results.json | 47KB | ✅ 存在 |
| case.json | benchmarks/verified/bem_annulus_laplace/case.json | 2.6KB | ✅ 存在 |
| report.md | benchmarks/verified/bem_annulus_laplace/report.md | 11KB | ✅ 存在 |
| posted_report.md | benchmarks/verified/bem_annulus_laplace/posted_report.md | 11KB | ✅ 存在 |
| convergence.png | benchmarks/verified/bem_annulus_laplace/figures/convergence.png | 107KB | ✅ 存在 |
| field_error.png | benchmarks/verified/bem_annulus_laplace/figures/field_error.png | 175KB | ✅ 存在 |
| sensitivity_wall_ratio.png | benchmarks/verified/bem_annulus_laplace/figures/sensitivity_wall_ratio.png | 56KB | ✅ 存在 |
| sensitivity_reference_length.png | benchmarks/verified/bem_annulus_laplace/figures/sensitivity_reference_length.png | 58KB | ✅ 存在 |

### ② 数值精度 ✅

**预算点（64/32 面板）误差**：

| 算例 | q (L2) | q (Linf) | u (max) | flux | 容差 | 判定 |
|---|---|---|---|---|---|---|
| radial_log | 0.18% | 0.22% | 0.04% | 0.11% | 3% | ✅ 全优 |
| angular_cos | 0.63% | 0.87% | 0.06% | ≈0% | 3% | ✅ 全优 |

**所有误差均远优于 3% 容差**（最差的 0.87% 仅为容差的 29%）。

### ③ 收敛阶 ✅

| 算例 | 拟合阶 | 阈值 | 判定 |
|---|---|---|---|
| radial_log | 1.959 | >1.5 | ✅ 二阶 |
| angular_cos | 2.022 | >1.5 | ✅ 二阶 |

符合常数面元 BEM 理论预期。

### ④ 调和性验证 ✅

| 算例 | 相对残差 | 阈值 | 判定 |
|---|---|---|---|
| radial_log | 6.96e-16 | <1e-6 | ✅ 通过 |
| angular_cos | 1.20e-15 | <1e-6 | ✅ 通过 |

闭式解确实调和。

### ⑤ 测试通过性 ✅

```
24 passed, 1 warning in 17.56s
```

测试文件：
- tests/test_case_annulus_laplace.py
- tests/test_multiloop_annulus.py

### ⑥ 报告完整性 ✅

报告六段齐全：
1. ✅ Problem and Reference
2. ✅ Method
3. ✅ Parameters (complete)
4. ✅ Reference vs computed
5. ✅ Sensitivity
6. ✅ Failure modes

### ⑦ repro_cmd 可执行性 ✅

```bash
PYTHONPATH=src python -m tensorbem.benchmarks.case_annulus_laplace \
    --outdir benchmarks/verified/bem_annulus_laplace
```

命令定义在 case.json，工件已生成。

---

## 实现文件

| 文件 | 大小 | 作用 |
|---|---|---|
| src/tensorbem/benchmarks/case_annulus_laplace.py | 47KB | 算例池外壳（本单新增） |
| src/tensorbem/benchmarks/annulus_laplace.py | 20KB | 闭式解定义 |
| src/tensorbem/multiloop.py | 14KB | 多环求解器（复用） |

---

## 关键发现

1. **域守卫补强**：`evaluate_multiloop_field` 原先没有域守卫，孔洞内/外环外的探针会返回解析延拓值。本单补了 `_point_in_polygon` + `domain_membership` 守卫。

2. **负控实验**：
   - 内环法向未翻转：线性系统残差不变，但 q 与通量错到量级级别
   - 孔洞内探针：被域守卫拒绝

3. **敏感性研究**：
   - 壁厚比 `b/a ∈ {0.2…0.95}`：误差与条件数随环变薄的走势
   - 核参考长度 `L ∈ {1.0…8.0}`：量化对数容量退化

4. **诚实缺口**：参考解页码未核（书不可达），用调和性测量补偿。

---

## 判级依据

**L4（基准验证）**：
- ✅ 完整算例（2 算例 × 5 加密层级 × 48 探针）
- ✅ 误差分析（4 类误差指标 × 2 算例 × 5 层级）
- ✅ 收敛研究（拟合阶 1.959 / 2.022）
- ✅ 敏感性研究（壁厚比、核参考长度）
- ✅ 负控实验（法向翻转、域守卫）
- ✅ 可复现（repro_cmd + 24 测试全过）
- ✅ 报告六段齐全

**未达 L5**：本单是算例池条目（验证既有能力），非新研究结果。

---

## verdict 落表

```sql
INSERT INTO tensor_maturity_verdicts 
(repo, scope, mission_id, level, evidence, verdict_by)
VALUES 
('TensorBEM', 'm17178 CASE-BEM-ANNULUS-LAPLACE', 17234, 4, '<JSON>', 'board-audit');
```

**执行结果**：id=2, created_at=2026-11-13 02:49:50

---

## 报告发布

- **report_id**: 16231
- **submission_id**: 6d65748955134a8991af324282788de8
- **API**: POST $TENSOR_API_BASE/tensor/reports
- **状态**: ✅ 已发布

---

## 结论

m17178 的 CASE-BEM-ANNULUS-LAPLACE 算例池条目**完全满足验收标准**：
- 数值精度远优于容差
- 收敛性符合理论
- 测试覆盖完整
- 报告六段齐全
- 工件可复现

**verdict 已落表，maturity_gate 裁定 confirmed（消账）**。
