"""Reproducible TensorBEM benchmarks."""

from tensorbem.benchmarks import rigid_body_doc_ledger  # noqa: F401
from tensorbem.benchmarks import kp505_w20_gci  # noqa: F401
from tensorbem.benchmarks import case_bem_rigid_lid_hemisphere  # noqa: F401

__all__ = [
    "rigid_body_doc_ledger",
    "kp505_w20_gci",
    "case_bem_rigid_lid_hemisphere",
]
