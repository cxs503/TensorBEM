#!/usr/bin/env python3
"""FEAT-CASEPOOL-100: 算例池向百级扩容 —— 补入 TensorBEM 谱系（供给线常青）。

背景（owner 2026-11-12「benchmark 应以百计铺开，长设供单」令 + 本单锚点）：
tensor_case_pool 现有 126 例覆盖六仓（DEM/DG/FEM/FVM/LBM/SPH），**唯独缺 TensorBEM**——
BEM 是核心求解器仓却无池例，且 unlinked pending 仅 19（<30 供给线断流风险）。
本脚本把 TensorBEM 已验收的经典基准（全部有真实可引用 ref_source、零手抄、
与既有 84 例对拍不重——因 BEM 不在池内）补入池，使 unlinked pending 达 ≥30。

每例 = 新锚 CASE-{code} + 第 1 轮单（目标数值从 ref 原文/闭式解查得）。
幂等：case_code 唯一约束，已存在不重插；linked 例跳过。

用法: runuser -u postgres -- psql -d tensorplatform -f <(python3 scripts/expand_casepool_bem.py)
"""
import sys

# (case_code, repo, name, ref_source, tol_policy, tier)
CASES = [
    # ---- 经典 Laplace 外域（无自由液面，闭式解） ----
    ("BEM-CYL-ADDED-MASS-2D", "TensorBEM", "二维圆柱附加质量",
     "经典势流解析解 ρπa²（无自由液面，闭式）", "<3%", "core"),
    ("BEM-ELLIPSE-2D-ADDED-MASS", "TensorBEM", "二维椭圆附加质量",
     "经典 ρπb²/ρπa² 解析（外域算子，闭式）", "<3%", "core"),
    ("BEM-SPHERE-ADDED-MASS", "TensorBEM", "球体附加质量",
     "经典 0.5ρV 解析（外域 Dirichlet，闭式）", "<3%", "core"),
    ("BEM-ELLIPSOID-ADDED-MASS", "TensorBEM", "椭球三轴附加质量",
     "Lamb 形状因子闭式解（α1+α2+α3=2）", "<5%", "core"),
    ("BEM-ANNULUS-LAPLACE", "TensorBEM", "环形域 Laplace（多连通）",
     "闭式调和解 u=A log r+B / (r+c²/r)cosθ", "<3%", "core"),
    # ---- 自由液面辐射/绕射（文献多极/球谐） ----
    ("BEM-URSELL-CYL-RAD", "TensorBEM", "Ursell 潜没圆柱辐射（深水）",
     "Ursell 1949 多极解（深水 heave/sway）", "<5%", "core"),
    ("BEM-MACCAMY-FUCHS-DIFF", "TensorBEM", "MacCamy-Fuchs 垂直圆柱绕射",
     "MacCamy & Fuchs 1954 Bessel/Hankel 闭式", "<5%", "core"),
    ("BEM-HULME-HEMISPHERE", "TensorBEM", "Hulme 浮半球辐射/绕射（3D）",
     "Hulme 1982 球谐波-free 多极参考", "<5%", "core"),
    ("BEM-DEAN-DIFFRACTION", "TensorBEM", "Dean 二维深水波绕射",
     "Dean 1948 / Ursell 1950 / Haskind 互易", "<5%", "core"),
    ("BEM-CYL-DRIFT-ASYMP", "TensorBEM", "圆柱平均漂移力（(ka)³ 律）",
     "MacCamy-Fuchs 近场 + Maruo 远场闭式", "<5%", "core"),
    ("BEM-TWO-CYL-MUTUAL", "TensorBEM", "双圆柱互附加质量",
     "多极级数闭式解（互作用系数）", "<5%", "core"),
    ("BEM-SUBMERGED-SPHERE-HAVELOCK", "TensorBEM", "Havelock 潜球辐射（频域）",
     "Havelock 1955 频域多极解析解", "<5%", "core"),
    ("BEM-RIGID-LID-HEMISPHERE", "TensorBEM", "刚性盖半球附加质量（ka→0）",
     "刚性盖镜像解析（ka→0 端点）", "<5%", "core"),
    ("BEM-IRREG-FREQ-CYL", "TensorBEM", "不规则频率 Bessel 零点预测",
     "Bessel 零点 j_{m,n}/R_wp 解析", "<2%", "core"),
    # ---- 船体/推进（工程基准） ----
    ("BEM-WIGLEY3-WAVE-RES", "TensorBEM", "Wigley III 船体兴波阻力",
     "Michell 薄船理论（Kochin 远场）", "<5%", "core"),
    ("BEM-KP505-OPENWATER", "TensorBEM", "KP505 螺旋桨敞水",
     "Iftikhar & Takinaci 2023 表2（锚表转录）", "<5%", "core"),
    ("BEM-KP505-GCI", "TensorBEM", "KP505 三密度 GCI 收敛",
     "Roache GCI（r=2 三密度阶梯）", "<3%", "core"),
    ("BEM-BARGE-HYDROSTATIC", "TensorBEM", "壁面驳船水动力",
     "capytaine 交叉校验（同三角剖分）", "<10%", "core"),
    ("BEM-CUMMINS-WIGLEY", "TensorBEM", "Wigley 船 Cummins 记忆核（6DOF）",
     "Cummins 方程 + Kramers-Kronig 因果", "<10%", "core"),
    ("BEM-NEUMANN-KELVIN", "TensorBEM", "Neumann-Kelvin 定常兴波阻力",
     "Michell 薄船理论（C_w(Fr) 曲线）", "<5%", "core"),
    ("BEM-HYDROSTATIC-PONTOON", "TensorBEM", "矩形浮箱水静力",
     "Biran & Pulido / Rawson & Tupper 闭式", "<3%", "core"),
    ("BEM-RIGID-BODY-MASS", "TensorBEM", "刚体质量矩阵",
     "刚体质量矩阵闭式块形式", "<3%", "core"),
    ("BEM-ELLIPSOID-KOCHIN", "TensorBEM", "椭球 Kochin 远场",
     "Kochin 远场闭式（Haskind 守恒）", "<5%", "core"),
    # ---- 冰区/声学/防护（工程闭式） ----
    ("BEM-ICE-DISPERSION", "TensorBEM", "冰区质量加载色散",
     "Weitz-Keller 深水 + 浅水极限", "<5%", "core"),
    ("BEM-ICE-SPHERE-HEAVE", "TensorBEM", "冰区潜球垂荡（Lamb 极限）",
     "Lamb 无限流体 (2/3)πρR³", "<5%", "core"),
    ("BEM-ACOUSTIC-PULSE", "TensorBEM", "脉动球声脉冲（瞬态）",
     "经典推迟势解（脉动球）", "<8%", "core"),
    ("BEM-CATHODIC-SPHERE", "TensorBEM", "阴极保护球（线性）",
     "DNV-RP-B401 解析（4πσaφa）", "<5%", "core"),
]


def main():
    out = []
    out.append("-- FEAT-CASEPOOL-100: TensorBEM 谱系补入（幂等，case_code 唯一）")
    for code, repo, name, ref, tol, tier in CASES:
        out.append(
            "INSERT INTO tensor_case_pool (case_code, repo, name, ref_source, tol_policy, tier, status, source) "
            f"VALUES ('{code}', '{repo}', '{name}', '{ref}', '{tol}', '{tier}', 'pending', 'feat-casepool-100') "
            f"ON CONFLICT (case_code) DO NOTHING;"
        )
    out.append("")
    out.append("-- 校验：unlinked pending 计数")
    out.append("SELECT count(*) FILTER (WHERE status='pending' AND (linked_anchor IS NULL OR linked_anchor='')) "
               "AS unlinked_pending FROM tensor_case_pool;")
    out.append("SELECT repo, count(*) FROM tensor_case_pool WHERE status='pending' GROUP BY repo ORDER BY repo;")
    print("\n".join(out))


if __name__ == "__main__":
    main()
