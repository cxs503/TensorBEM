"""Wageningen B-series open-water anchor (Oosterveld & van Oossanen, 1975).

This module is the *judge* used by the W25 propeller open-water study
(``benchmarks/w25/b4_70_error_surface``): it evaluates the published 39-term
``KT`` / 47-term ``KQ`` regression of the B-screw series so that every BEM number
can be scored against an external quantitative anchor without a single number
being typed into a report.

Anchor data live in ``benchmarks/anchors/wageningen_b_series.json``; see
``tools/build_wageningen_anchor.py`` for the machine-extraction chain (published
NSMB tables -> platform reference implementation -> extraction by importlib, no
retyping) and ``tests/test_wageningen_anchor.py`` for the adjudication of the
regression itself against physics anchors.

Model
-----
    KT = sum_i C_i J^s_i (P/D)^t_i (Ae/Ao)^u_i Z^v_i        (39 terms)
    KQ = sum_i D_i J^s_i (P/D)^t_i (Ae/Ao)^u_i Z^v_i        (47 terms)
    eta0 = J KT / (2 pi KQ)

Published validity domain: Z = 2..7, Ae/Ao = 0.30..1.05, P/D = 0.5..1.4,
Re = 2e6 referred to the 0.75 R chord, hub ratio 0.18..0.20, smooth blades.

Beyond the zero-thrust advance ratio the regression turns over (``KT < 0`` and
eventually ``KQ < 0``); that is extrapolation, not physics.  :class:`B4Series`
therefore exposes :attr:`B4Series.j0` (first zero of ``KT``) and
:meth:`B4Series.scoring_range` so scoring code can separate "model error" from
"outside the anchor's meaningful range" instead of dividing by ~0.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path

__all__ = [
    "ANCHOR_PATH",
    "anchor_payload",
    "anchor_sha256",
    "table_fingerprints",
    "verify_tables",
    "B4Series",
]

#: Anchor artefact shipped with the package.
ANCHOR_PATH = (
    Path(__file__).resolve().parents[2]
    / "benchmarks"
    / "anchors"
    / "wageningen_b_series.json"
)


def anchor_sha256(path: str | Path | None = None) -> str:
    """sha256 of the anchor artefact on disk."""

    target = Path(path) if path is not None else ANCHOR_PATH
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=4)
def anchor_payload(path: str | None = None) -> dict:
    """Parsed anchor artefact (cached)."""

    target = Path(path) if path is not None else ANCHOR_PATH
    return json.loads(target.read_text())


def table_fingerprints(payload: dict) -> dict:
    """Recompute the coefficient-table fingerprints stored in the artefact."""

    out = {}
    for key, name in (("kt_terms", "kt_table_sha256"), ("kq_terms", "kq_table_sha256")):
        rows = [
            [float(row[0])] + [int(v) for v in row[1:]] for row in payload[key]
        ]
        serialised = json.dumps(
            [[float(r[0])] + [int(v) for v in r[1:]] for r in rows],
            separators=(",", ":"),
        )
        out[name] = hashlib.sha256(serialised.encode()).hexdigest()
    out["n_kt_terms"] = len(payload["kt_terms"])
    out["n_kq_terms"] = len(payload["kq_terms"])
    return out


def verify_tables(payload: dict | None = None) -> dict:
    """Check term counts and fingerprints against what the artefact declares."""

    payload = payload if payload is not None else anchor_payload()
    stored = payload["fingerprints"]
    actual = table_fingerprints(payload)
    mismatches = [
        key for key in ("kt_table_sha256", "kq_table_sha256") if stored[key] != actual[key]
    ]
    return {
        "n_kt_terms": actual["n_kt_terms"],
        "n_kq_terms": actual["n_kq_terms"],
        "fingerprint_match": not mismatches,
        "mismatched_keys": mismatches,
        "declared": {k: stored[k] for k in ("kt_table_sha256", "kq_table_sha256")},
        "recomputed": {
            k: actual[k] for k in ("kt_table_sha256", "kq_table_sha256")
        },
    }


def _poly(terms, advance: float, pitch_ratio: float, area_ratio: float, blades: int) -> float:
    total = 0.0
    for coefficient, s_j, t_pd, u_ae, v_z in terms:
        total += (
            coefficient
            * advance**s_j
            * pitch_ratio**t_pd
            * area_ratio**u_ae
            * blades**v_z
        )
    return total


@dataclass(frozen=True)
class B4Series:
    """One Wageningen B-series member: ``Z`` blades, ``Ae/Ao``, ``P/D``."""

    blades: int = 4
    expanded_area_ratio: float = 0.70
    pitch_ratio: float = 1.0
    payload: dict | None = None  # injected for tests; None -> shipped artefact

    # -- coefficients -------------------------------------------------------
    @property
    def tables(self) -> tuple:
        payload = self.payload if self.payload is not None else anchor_payload()
        return payload["kt_terms"], payload["kq_terms"]

    def kt(self, advance: float) -> float:
        """``K_T = T / (rho n^2 D^4)`` from the O&O (1975) regression."""

        return _poly(
            self.tables[0], float(advance), self.pitch_ratio,
            self.expanded_area_ratio, self.blades,
        )

    def kq(self, advance: float) -> float:
        """``K_Q = Q / (rho n^2 D^5)`` from the O&O (1975) regression."""

        return _poly(
            self.tables[1], float(advance), self.pitch_ratio,
            self.expanded_area_ratio, self.blades,
        )

    def eta0(self, advance: float) -> float:
        """Open-water efficiency; ``nan`` where the regression gives ``K_Q <= 0``."""

        kq = self.kq(advance)
        if kq <= 0.0:
            return float("nan")
        return advance * self.kt(advance) / (2.0 * math.pi * kq)

    # -- range handling -----------------------------------------------------
    @property
    def j0(self) -> float:
        """Zero-thrust advance ratio (bisection on ``K_T``)."""

        return self.zero_thrust_advance_ratio()

    def zero_thrust_advance_ratio(self, j_hi: float = 3.0) -> float:
        if self.kt(0.0) <= 0.0:
            return 0.0
        lo, hi = 0.0, j_hi
        if self.kt(hi) > 0.0:
            return float("nan")
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if self.kt(mid) > 0.0:
                lo = mid
            else:
                hi = mid
            if hi - lo < 1e-13:
                break
        return 0.5 * (lo + hi)

    def scoring_range(self) -> dict:
        """Advance ratios where both anchor coefficients are positive.

        ``KT > 0`` and ``KQ > 0`` are the conditions under which a *relative*
        deviation is a meaningful score; past them the regression is an
        extrapolation and relative error diverges.
        """

        j0 = self.j0
        j_kq = float("nan")
        lo, hi = 0.0, 3.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if self.kq(mid) > 0.0:
                lo = mid
            else:
                hi = mid
            if hi - lo < 1e-13:
                break
        j_kq = 0.5 * (lo + hi)
        return {"j_zero_thrust": j0, "j_zero_torque": j_kq, "usable_max_j": min(j0, j_kq)}

    # -- provenance --------------------------------------------------------
    def provenance(self) -> dict:
        payload = self.payload if self.payload is not None else anchor_payload()
        return {
            "anchor_id": payload["anchor_id"],
            "citation": payload["citation"],
            "provenance": payload["provenance"],
            "fingerprints": payload["fingerprints"],
            "anchor_sha256": anchor_sha256() if self.payload is None else "injected",
            "geometry": {
                "Z": self.blades,
                "Ae_Ao": self.expanded_area_ratio,
                "P_D": self.pitch_ratio,
            },
        }


def b4_70(pitch_ratio: float = 1.0) -> B4Series:
    """The W25 case: B4-70 (Z=4, Ae/Ao = 0.70) at the requested ``P/D``."""

    return B4Series(blades=4, expanded_area_ratio=0.70, pitch_ratio=pitch_ratio)