"""Pipeline E2E smoke test (mission m4624).

Two checks only, by design:
  1. the repository-root smoke marker module imports and carries the expected
     mission constant (proves "repo root file + tests/ file" code path works);
  2. a trivial arithmetic sanity assertion (proves the test runner itself runs).

Nothing else is touched by this test; it is a wiring smoke, not a physics check.
"""
import os
import sys

import pytest

# The marker lives at the repository root. Under ``python -m pytest`` the CWD is
# already on sys.path, but a bare ``pytest`` invocation only prepends the test
# directory, so add the root explicitly to keep both invocations green.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import smoke_marker_m4624  # noqa: E402  (import after sys.path setup)


def test_marker_constant_matches_mission_number():
    assert smoke_marker_m4624.PIPELINE_SMOKE_M4624 == "4624"


def test_marker_constant_is_digit_string():
    value = smoke_marker_m4624.PIPELINE_SMOKE_M4624
    assert isinstance(value, str)
    assert value.isdigit()
    assert int(value) == 4624


def test_arithmetic_sanity():
    assert 1 + 1 == 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))