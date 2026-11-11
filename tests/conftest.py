"""Make the checked-out ``src`` tree win over any pre-installed ``tensorbem``.

Without this the test session can silently exercise a *different* worktree's
copy of the package (editable install pointing elsewhere), which would make
the evidence meaningless.  CI installs the package from this checkout anyway,
so the prepend is a no-op there.
"""
from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))