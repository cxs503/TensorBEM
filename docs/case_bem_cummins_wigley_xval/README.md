# CASE-BEM-CUMMINS-WIGLEY：外部程序锚点（capytaine，同网格）

Gate `C10` 是本算例第一个**非自证**证据：把同一套面板（同一顶点、同一三角形）交给独立实现的自由面 BEM（capytaine）重解 6 自由度辐射问题，再与已提交的 TensorBEM medium 扫频逐格比对。同网格 ⇒ 残差是**方法差**，不是离散差。

| 项 | 值 |
| --- | --- |
| 外来程序 | capytaine 3.1.dev0（commit `None`） |
| 平台 | Linux-6.6.52-1.7.lns23.loongarch64-loongarch64-with-glibc2.38 |
| 比对层级 | `medium`（480 面板，与 TensorBEM medium 完全相同：`True`） |
| 变体 | `lid`（capytaine 内部自由面 lid，抑制不规则频率） |
| 频带 | [0.2, 12.0] rad/s，41 节点 |
| 旋转中心 | [0.0, 0.0, 0.0] |

## 1 逐格比对（归一化到该格带内幅值）

| 量 | 层级 | 变体 | 最差格 | 最差偏差 | 出现频率 (rad/s) | 全频带最大偏差 | 计分格 | ≤3% 格数 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 6x6 A/B | `coarse` | `no_lid` | yaw←yaw | 7.79% | 9.7785 | 7.79% | 20 | 7 |
| 6x6 A/B | `coarse` | `lid` | yaw←yaw | 7.79% | 9.7785 | 7.79% | 20 | 9 |
| 6x6 A/B | `medium` | `no_lid` | heave←heave | 6.09% | 10.8325 | 6.09% | 20 | 16 |
| 6x6 A/B | `medium` | `lid` | heave←heave | 6.09% | 10.8325 | 6.09% | 20 | 16 |
| 6x6 A/B | `fine` | `no_lid` | roll←roll | 9.59% | 7.1930 | 9.59% | 20 | 11 |
| 6x6 A/B | `fine` | `lid` | roll←roll | 9.59% | 7.1930 | 9.59% | 20 | 12 |

## 2 判据与裁决

* 判据：`max_{cells,omega} |X_capytaine - X_tensorbem| / max_omega |X_tensorbem| <= 10% (case criterion), null cells excluded (amplitude < 1% of the largest diagonal amplitude)`
* 门限：10%（算例池 id=145 容差）
* 实测最差偏差：**6.09%**（heave←heave，ω = 10.8325 rad/s）
* 裁决：**PASSED**
* ≤3% 的计分格：16 / 20（80.0%）

## 3 带顶分歧的归因（可伪证）

预测：若带顶分歧是**外来程序**的不规则频率伪迹（TensorBEM 用 CHIEF 点抑制，capytaine 用内部自由面 lid），则加 lid 后分歧必须不增。

| 量 | 值 |
| --- | --- |
| no_lid 全带最大偏差 | 6.09% |
| lid 全带最大偏差 | 6.09% |
| 超门限频点数 | 0 |
| 加 lid 后改善的频点数 | 0 |
| 假设判定 | **无法检验（该层级 capytaine 未生成 lid 面板）** |
| lid 效果 `coarse` | 12 面板，A 1628.6945%，B 2556.2412% |
| lid 效果 `medium` | 0 面板，A 0.0000%，B 0.0000% |
| lid 效果 `fine` | 46 面板，A 143.9000%，B 64.2536% |

## 4 面板阶梯（方法差 vs 离散差）

| 层级 | 面板数 | 最差偏差 | 出现频率 (rad/s) |
| --- | --- | --- | --- |
| `coarse` | 192 | 7.79% | 9.7785 |
| `medium` | 480 | 6.09% | 10.8325 |
| `fine` | 1080 | 9.59% | 7.1930 |

capytaine 自身 medium→fine 的带内最大相对变化：added mass 121.69%，damping 81.79%。

![figure](fig_case_bem_cummins_wigley_xval.png)

## 5 注意点

1. **同网格是本闸门的关键**：两个程序读的是同一组顶点与三角形（参考文件里存了几何 sha256，`geometry_matches` 每次重算比对），因此偏差不含网格差。
2. **结构零格不计分**：以最大对角幅值的 1% 为阈，避免用数值零去除自身幅值制造假偏差；这些格的绝对偏差单列。
3. **capytaine 的矩阵不正对称**（其为逐列求解的直接 BEM），TensorBEM 侧按需对称化；对称化误差已记录在参考文件里，量级见第 4 节。
4. **承认探针**：本单开工前在 ω≈2 rad/s 上做过 1 个 heave 探针（A33 0.6%），但判据、归一化与门限在本次全量比对之前就已写进代码；探针未用于选择频带或门限。
5. **L5 的边界**：这是**独立程序**锚点，不是实验锚点。它排除共享实现错误，但不能排除两程序共有的建模假设错误（线性势流、无限水深、无粘）。
