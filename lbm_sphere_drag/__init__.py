"""LBM sphere drag benchmark framework.

This module provides the reference correlation and benchmark framework for
verifying LBM sphere drag force against the Schiller-Naumann correlation
(Clift-Grace-Weber standard).

The actual LBM simulation requires TensorLBM infrastructure; this module
provides the reference values, test structure, and verification framework.

GAP-LBM-SPHERE anchor:
  ref: Clift-Grace-Weber 球阻力标准关联（Schiller-Naumann 0.1<Re<1000）
  判据: C_D vs 关联值 <8%
  曲面边界力法+球体绕流验证
"""

from .schiller_naumann import (
    schiller_naumann_cd,
    schiller_naumann_valid_range,
    clift_grace_weber_cd,
    drag_force_from_cd,
    reynolds_number,
)

__all__ = [
    "schiller_naumann_cd",
    "schiller_naumann_valid_range",
    "clift_grace_weber_cd",
    "drag_force_from_cd",
    "reynolds_number",
]
