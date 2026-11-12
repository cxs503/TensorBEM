import base64, hashlib, json, os, pathlib, urllib.request
D = pathlib.Path(__file__).resolve().parent
S = json.loads((D / "study.json").read_text())
png = (D / "fig_alpha_axis.png").read_bytes()
h, f = S["cells"]["wigley3_hull"], S["cells"]["wigley3_freq"]
g, cb, pre = S["geometry"], S["cross_builder"], S["pre_fix_reference"]
md = f"""# TensorBEM：Wigley III 船型定义缺陷裁定与修复（α 轴：垂向 → 纵向，m16935）

## ① 案例介绍
TensorBEM 有**两个** Wigley III 生成器：`tensorbem.wigley3_hull`（多体/耐波路径）与
`tensorbem.wigley3_freq`（频域验证路径，仓库全部公开 Wigley III 数字的来源）。
两者把丰满度参数 α 分别加在 $\\zeta=z/T$ 与 $\\xi=2x/L$ 上，**块系数都等于 0.543** ——
所以大家会去核的那个数看不出差别；但它们是两条**不同的船**。

判定依据（先几何后数值）：仓库自己的船型定义（`docs/wigley3_freq_verification_report.md` §1
"fuller ends than the standard parabolic Wigley (Wigley I)"，
$y=(B/2)(1-\\xi^2)(1-\\zeta^2)(1+c\\xi^2)$）+ Journée (1992) 尺寸表
（L=3.0 m, B=0.30 m, T=0.1875 m, Cb=0.543）。若剖面保持标准抛物线（Cm=2/3），
则 $C_{{wp}}=C_b/C_m=0.8145$；把 α 加在垂向只在**改变剖面**的情况下保持 Cwp=2/3，
与"更丰满的**端部**"自相矛盾。故 `wigley3_hull` 是缺陷方。

## ② 怎么做的
1. 修 `wigley3_half_breadth`：α 作用于 $\\xi^2$（纵向），与文档定义一致。
2. 同步修 `compute_wigley3_properties` 的闭式解（该函数原来的 A_wp、I_xx、I_yy、KB
   是按旧几何推的）：$A_{{wp}}=BL(2/3+2\\alpha/15)$、
   $I_{{xx}}=(B^3L/24)(32/35+32\\alpha/105+32\\alpha^2/385+32\\alpha^3/3003)$、
   $I_{{yy}}=(BL^3/8)(4/15+4\\alpha/35)$、$KB=5T/8$（纵向丰满度下剖面形状不变，故 KB 与 α 无关）。
3. 新增几何门 `tests/test_wigley3_hull_alpha_axis.py`（6 项）：两生成器逐点恒等、
   Cwp=2/3+2α/15 且距退役配方 >15%（回退哨兵）、Cb/Cm/Cwp 三角一致、
   五个闭式解 vs 同一曲面的 Gauss–Legendre 积分、网格体积、以及两生成器的**辐射收敛交叉验证**。
4. 相关钉桩更新：`test_multibody_frequency_dependence_3d` 的 A_wp 由 2BL/3 改为 BL(2/3+2α/15)；
   `tests/test_wigley3_recalled_ref.py` 的几何门改为"冻结记录保留退役缺陷 + 现行生成器必须已修复"，
   并把冻结件里 `wigley3_hull` 的单元标注为**修复前历史**（capytaine 冻结网格签名同理）。

## ③ 计算结果（全部由 `docs/wigley3_hull_alpha_axis/study.json` 注入）
| 量 | 值 |
|---|---|
| 水面线系数 Cwp（修复后，积分） | {g['Cwp_now']:.7f} |
| Cwp（解析 $2/3+2\\alpha/15$，α={S['alpha']:.4f}） | {g['Cwp_analytic']:.7f} |
| Cwp（退役配方，α 加垂向） | {g['Cwp_retired_recipe']:.7f} |
| Cb / Cm | {g['Cb']:.4f} / {g['Cm']:.4f} |
| 水线面面积 $A_{{wp}}$ [m²] | {g['A_wp']:.5f} |
| KB/T | {g['KB_over_T']:.4f} |
| 两生成器半宽最大差 $\\max|\\Delta y|$ | {g['max_abs_y_difference_vs_wigley3_freq']:.3e} |

ω=1.0 rad/s 升沉（$\\rho\\nabla$={S['density']*S['analytic_volume']:.4f} kg）：

| 生成器 | 面元 | $A_{{33}}/(\\rho\\nabla)$ | $B_{{33}}/(\\rho\\nabla\\omega)$ | 网格体积误差 | 墙钟 |
|---|---|---|---|---|---|
| wigley3_hull | {h['panels']} | {h['A33_over_rhoV']:.6f} | {h['B33_over_rhoV_omega']:.6f} | {100*h['mesh_volume_rel_error']:.2f}% | {h['wall_s']} s |
| wigley3_freq | {f['panels']} | {f['A33_over_rhoV']:.6f} | {f['B33_over_rhoV_omega']:.6f} | {100*f['mesh_volume_rel_error']:.2f}% | {f['wall_s']} s |

**跨生成器分裂**：A33 {100*cb['A33_rel_split']:+.2f}%，B33 {100*cb['B33_rel_split']:+.2f}%
（修复前冻结件 level-20 单元：{pre['A33_over_rhoV_wigley3_hull']:.4f} vs {pre['A33_over_rhoV_wigley3_freq']:.4f}
→ {100*pre['A33_rel_split']:+.2f}%；m16892 叙述里的 1.51 vs 2.07 = 36% 是它另一组单元）。

## ④ 结果图
左：修复后的静水线 $y(x,0)/(B/2)$ vs 退役配方（Cwp {g['Cwp_now']:.4f} vs {g['Cwp_retired_recipe']:.4f}）。
右：同一 ω 下两生成器的 $A_{{33}}/(\\rho\\nabla)$，分裂已塌缩到 {100*cb['A33_rel_split']:+.2f}%。
图 sha256 = `{hashlib.sha256(png).hexdigest()}`。

## ⑤ 注意点（局限）
- **未重跑 capytaine 冻结件**：`benchmarks/verified/wigley3_recalled_ref/capytaine_ref.json`
  是在退役几何上取的，现按"修复前历史记录"保留并由测试显式豁免；重取需重跑
  `scripts/wigley3_recalled_ref_capytaine.py`（本单未做）。
- 760 面元 `wigley3_freq` 求解超出单次 118 s 墙钟（hull 760 已单独测得
  $A_{{33}}/(\\rho\\nabla)$=2.0654），故本单的交叉验证取 ~200 面元；结论方向不受影响，
  但**未宣称** 760 面元级别的跨生成器收敛。
- 只跑了 `wigley or multibody or seakeeping or capability or doc_claim` 子集（exit 0）+ 新增门，
  未跑全量套件。
- 线性势流、无粘、深水；α 的物理解释按仓库文档与 Journée 尺寸表，**未**取得 Journée (1992)
  原始图表做逐点核对（离线不可得）——这是本单裁定的最大外部依赖。

## ⑥ 误差判定（是否 <3%）
| 关键量 | 基准 | 误差 | <3% |
|---|---|---|---|
| 半宽场 $y(x,z)$，hull vs freq | 同一曲面（文档定义） | {g['max_abs_y_difference_vs_wigley3_freq']:.2e} | 是（机器精度） |
| Cwp | 解析 2/3+2α/15 | {abs(g['Cwp_now']/g['Cwp_analytic']-1):.2e} | 是 |
| Cb·(Cm=2/3) vs Cwp | Cb/Cm 恒等 | <1e-12 | 是 |
| A33 跨生成器 | 彼此 | {100*abs(cb['A33_rel_split']):.2f}% | 是 |
| B33 跨生成器 | 彼此 | {100*abs(cb['B33_rel_split']):.2f}% | 是 |
| A33 vs 独立码 capytaine | m16892 冻结（同一曲面 = freq 生成器） | −0.9%（继承自 m16892） | 是 |

复算：`PYTHONPATH=src python docs/wigley3_hull_alpha_axis/study.py`；
`PYTHONPATH=src pytest tests/test_wigley3_hull_alpha_axis.py -q` → 6 passed。
"""
(D / "REPORT.md").write_text(md)
base = os.environ["TENSOR_API_BASE"].rstrip("/")
tok = json.loads(urllib.request.urlopen(urllib.request.Request(
    base + "/auth/login",
    data=json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                     "password": os.environ["TENSOR_AGENT_PASS"]}).encode(),
    headers={"Content-Type": "application/json"})).read())["access_token"]
body = {"title": "TensorBEM Wigley III 船型定义缺陷裁定与修复：α 轴垂向→纵向（Cwp 0.667→0.815，跨生成器 A33 分裂 +1.44%）",
        "summary": "m16892 几何门发现的 Wigley III 双生成器缺陷已裁定并修复：α 作用轴从 ζ 改为 ξ，"
                   "闭式静水力同步订正（A_wp/I_xx/I_yy/KB=5T/8），新增 6 项几何门与回退哨兵；"
                   "修复后两生成器半宽逐点恒等（5.55e-17），ω=1.0 升沉 A33 跨生成器分裂 +1.44%、B33 +0.46%（均 <3%）。",
        "scenario_key": "tensorbem-wigley3-hull-alpha-axis",
        "mission_id": 16935,
        "tags": ["TensorBEM", "Wigley III", "geometry", "verification", "m16935"],
        "markdown": md,
        "figures": [{"name": "fig_alpha_axis.png",
                     "caption": "左：修复后静水线 vs 退役配方（Cwp 0.8145 vs 0.6667）；右：ω=1.0 两生成器 A33/(ρ∇) 分裂塌缩到 +1.44%。",
                     "png_base64": base64.b64encode(png).decode()}]}
req = urllib.request.Request(base + "/tensor/reports", data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json",
                                      "Authorization": "Bearer " + tok})
try:
    print("REPORT", urllib.request.urlopen(req).read()[:400])
except urllib.error.HTTPError as e:
    print("HTTP", e.code, e.read()[:600])
