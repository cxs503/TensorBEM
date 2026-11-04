"""Additive patch: track the B11' observed order in the Hulme refinement study.

Table 10 of docs/marine_bem_benchmarks.md prints five observed orders
(A33'/B33'/A11'/B11'/Fz) but the study only computed four.  This patch adds
the missing fifth series without touching any existing field.

Run once from the repository root:  ``python outputs/verif_m3631/patch_fh.py``
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
p = REPO_ROOT / "src" / "tensorbem" / "benchmarks" / "floating_hemisphere.py"
text = p.read_text()

pairs = [
    (
        "    err_a11: list[float] = []\n    err_fz: list[float] = []\n",
        "    err_a11: list[float] = []\n    err_b11: list[float] = []\n    err_fz: list[float] = []\n",
    ),
    (
        "        err_a11.append(e_a11)\n        err_fz.append(e_fz)\n",
        "        err_a11.append(e_a11)\n        err_b11.append(e_b11)\n        err_fz.append(e_fz)\n",
    ),
    (
        "    orders_fz = _observed_orders(h_list, err_fz)\n",
        "    orders_b11 = _observed_orders(h_list, err_b11)\n    orders_fz = _observed_orders(h_list, err_fz)\n",
    ),
    (
        '        row["observed_order_fz"] = orders_fz[idx]\n',
        '        row["observed_order_b11"] = orders_b11[idx]\n        row["observed_order_fz"] = orders_fz[idx]\n',
    ),
]

for old, new in pairs:
    assert text.count(old) == 1, f"anchor not unique/found: {old[:60]!r}"
    text = text.replace(old, new)

p.write_text(text)
print("patched OK")
