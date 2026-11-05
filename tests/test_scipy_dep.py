"""Dependency regression: scipy must be installable at runtime (PR #163)."""
import scipy  # noqa: F401


def test_scipy_importable():
    assert scipy.__version__
