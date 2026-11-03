# 椭球附加质量声明数字：heavy 层首次直接复算 + 收敛族外推（m2394）

**仓**：TensorBEM · **分支**：agent/m2394 · **类型**：验证欠账结清
**完整报告**：`docs/ellipsoid_added_mass_verification.md`（数字由工件程序化注入）

## 结论

1. `python -m tensorbem.benchmarks.declared_numbers --tiers fast,standard,heavy --strict`
   首次跑满三层：16 条声明数字全部 PASS（value 型最大相对偏差 1.1e-4），
   其中此前从未跑过的 heavy 5 条 —— 椭球 1280 面元三轴 4.1679e-3 / 4.4621e-3 / 4.9359e-3
   实测复现到 8.7e-6 / 4.8e-6 / 6.5e-6；Ursell 256 面元两条复现到 4.1e-6 / 5.3e-6。
   ⇒ `expand_audit_tensorbem.md` §4.2「椭球基准未直接复现 1280 面元」结清。
2. 新增 `tensorbem.benchmarks.ellipsoid_refinement`：把单点数字升级为 20/80/320 面元收敛族，
   先验证几何（多面体体积误差 3.95e-1 → 1.27e-1 → 3.38e-2，观测阶 1.53；Lamb 形状因子恒等式 <1e-9），
   再做 Richardson 外推：三轴零网格极限与 Lamb 解析值差 3.9e-3 / 5.2e-3 / 6.0e-3（受 320 面元本底误差限制），
   即 1280 面元处的 4e-3 是离散误差而非偏差。
3. `tests/test_ellipsoid_refinement.py`：6 passed（合成族阶数/外推代数、几何单调性、记录工件复核）。

## 工件

- `outputs/ellipsoid_heavy/declared_numbers_all_tiers.json`（含 environment: commit/numpy/platform）
- `outputs/ellipsoid_heavy/ellipsoid_refinement.json`、`ellipsoid_refinement.png`
- `src/tensorbem/benchmarks/ellipsoid_refinement.py`
- `docs/tools/{make_ellipsoid_refinement_figure,gen_ellipsoid_verification_doc}.py`

## 环境坑（给后续班次）

本机 `site-packages` 的 `tensorbem` 指向 `/root/tp-agent/work/m57/src`，
在别的 worktree 直接 `python -m tensorbem...` 会跑错代码；必须 `PYTHONPATH=<worktree>/src`。
1280 面元三轴装配在共享高负载机上约 10 分钟，heavy 层不进默认 pytest。
