"""Suite-wide fixtures.

``artifacts/`` is git-ignored on purpose (see ``.gitignore``): verification
reports are build products, not sources. CI, however, runs ``pytest`` before
the benchmark steps that populate that directory, so every test that reads a
generated artifact failed on a clean checkout - that is exactly how ``main``
went red in CI run 41619203219 (``tests/test_doc_claims.py``, three tests,
``FileNotFoundError: artifacts/ursell_doc_claims.json``).

The session fixture below closes the loop: if the published-number artifact is
absent it is measured from the code before the tests that read it run. The
suite is then self-contained on a fresh clone and in CI, and a *stale* cached
artifact is still caught, because ``test_live_benchmark_reproduces_the_artifact``
re-runs the benchmark and demands agreement to 1e-9.

Cost: one Ursell sweep (64/128/256 panels) on the first run in a checkout,
cached afterwards in ``artifacts/``.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session", autouse=True)
def published_number_artifact() -> None:
    """Ensure ``artifacts/ursell_doc_claims.json`` exists before it is read."""
    from tensorbem import doc_claims  # noqa: PLC0415

    if doc_claims.ARTIFACT.exists():
        return
    doc_claims.ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    doc_claims.write_artifact(doc_claims.ARTIFACT)
