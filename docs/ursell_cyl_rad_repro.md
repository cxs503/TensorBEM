# Ursell 潜没圆柱辐射基准（深水）— 复现指南

## 一键复现

```bash
cd /root/tp-agent/work/m17179
PYTHONPATH=src python -m tensorbem.benchmarks.ursell_cylinder --panels 128 --ka 0.6
```

## 收敛性研究

```bash
cd /root/tp-agent/work/m17179
PYTHONPATH=src python -c "
import json
from tensorbem.benchmarks.ursell_cylinder import run
for n in [32, 64, 128]:
    r = run(panel_count=n, ka=0.6)
    print(f'Panels={n}: A_err={r[\"panel_added_mass_relative_error\"]:.4%}, B_err={r[\"panel_damping_relative_error\"]:.4%}')
"
```

## 参数敏感性（ka 扫描）

```bash
cd /root/tp-agent/work/m17179
PYTHONPATH=src python -c "
import json
from tensorbem.benchmarks.ursell_cylinder import run
for ka in [0.2, 0.4, 0.6, 0.8, 1.0]:
    r = run(panel_count=128, ka=ka)
    print(f'ka={ka}: A_err={r[\"panel_added_mass_relative_error\"]:.4%}, B_err={r[\"panel_damping_relative_error\"]:.4%}')
"
```

## 生成报告图

```bash
cd /root/tp-agent/work/m17179
PYTHONPATH=src python make_ursell_figures.py
```

## 运行测试

```bash
cd /root/tp-agent/work/m17179
PYTHONPATH=src python -m pytest tests/test_ursell_convergence.py tests/test_ursell_convergence_doc_pins.py -v
```

## 工件路径

| 工件 | 路径 |
|------|------|
| 收敛性数据 | `runs/ursell_cyl_rad_ka06.json` |
| ka 扫描数据 | `runs/ursell_cyl_rad_ka_sweep.json` |
| 收敛性图 | `docs/figures/ursell_cyl_rad_convergence.png` |
| 误差 vs ka 图 | `docs/figures/ursell_cyl_rad_error_vs_ka.png` |
| 系数 vs ka 图 | `docs/figures/ursell_cyl_rad_coefficients_vs_ka.png` |
| 完整报告 | `docs/ursell_cyl_rad_report.md` |

## 参考值

- Ursell 1949 多极子解析解（`tensorbem.analytic.ursell_cylinder_added_mass`）
- ka=0.6, a/f=0.5: A_ref=2545.1672 kg/m, B_ref=2718.1540 kg/(m·s)

## 容差判据

- 附加质量误差 < 5%: ✅ (实测 0.85%)
- 阻尼误差 < 5%: ✅ (实测 1.34%)
