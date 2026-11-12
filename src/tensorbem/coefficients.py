"""Hydrodynamic coefficient interchange layer (``GAP-BEM-COEFF-IO``).

Why this module exists
----------------------
Every wave-load result TensorBEM produces (added mass ``A_ij``, radiation
damping ``B_ij``, wave exciting force ``F_i``) currently leaves the library in
whatever ad-hoc JSON a benchmark driver happens to write.  Nothing can read a
coefficient file *back*: the verification drivers therefore compare against
literature numbers that were **transcribed by hand** (e.g.
``benchmarks/wigley3_complete_verification.py::_WAMIT_A33_MASS_RATIO = 0.42``),
which is exactly what the repository's no-hand-transcription rule forbids, and
it makes a third-party reference value impossible to re-use across cases.

This module is the missing wire: one canonical in-memory container plus
readers/writers for the three coefficient dialects the field actually
exchanges,

* ``.json``  -- canonical native container, exact (float ``repr``) round trip;
* ``.1``     -- WAMIT added-mass/damping coefficient lines, ``PER I J A [B]``;
* ``.3``     -- WAMIT exciting-force lines, ``PER BETA I |F| phase Re Im``;
* ``.tec``   -- Nemoh ``RadiationCoefficients.tec`` / ``ExcitationForce.tec``.

Conventions (all stated so a reader cannot guess wrong)
------------------------------------------------------
* Frequencies are angular, ``omega`` [rad/s], increasing.  WAMIT dialect files
  carry the period ``T = 2*pi/omega``; the conversion is done here, once.
* ``added_mass`` is the physical ``A_ij(omega)`` [kg, kg*m, kg*m^2] and
  ``damping`` the physical ``B_ij(omega)`` [kg/s, ...]; the WAMIT dialect
  stores them *non-dimensional* as ``A_ij/(rho*L**k)`` and
  ``B_ij/(omega*rho*L**k)`` with ``k = 3`` for two translational DOFs, ``4`` for
  one translational and one rotational, ``5`` for two rotational DOFs.
* Exciting forces are normalised as ``F_i/(rho*g*A*L**m)`` with ``m = 2``
  translational, ``3`` rotational; ``.3`` phases are in degrees, headings in
  degrees, and the complex value stored is the modulus/phase pair (the two
  trailing columns are cross-checked, not trusted).
* The heading angle follows the WAMIT/capytaine sign convention: the incident
  wave travels along ``(cos(beta), sin(beta))`` in the ``x-y`` plane.  Nothing
  in this module re-phases a solver's output -- it stores what it is given.

What this module deliberately does **not** do
---------------------------------------------
* It does not implement the *block* variant of the WAMIT ``.1`` file (six rows
  of six columns per period per matrix) -- only the line variant, which is what
  an independent implementation in this environment (capytaine 3.1's
  ``io.wamit.export_wamit_1``) emits and what BEMRosetta/Nemoh post-processing
  chains consume.  Without an authoritative block-format sample in-repo we
  would be unable to verify such a parser, so it is not written.
* It does not re-derive coefficients from a potential field: use
  :func:`from_radiation_3d` / :func:`stack_frequencies` to build a container
  from solver results.

Integrity
---------
A container carries ``provenance``; :meth:`HydrodynamicCoefficients.fingerprint`
hashes the numeric payload only, and :meth:`write_json` stamps it.  Reading a
JSON container re-hashes and fails closed on mismatch, so a silently edited
"reference" number cannot be smuggled into a verification run.

CLI::

    python -m tensorbem.coefficients FILE...       # summarise + verify containers
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "COEFFICIENT_SCHEMA_VERSION",
    "DOF_ORDER",
    "CoefficientIntegrityError",
    "HydrodynamicCoefficients",
    "combine_slices",
    "from_radiation_3d",
    "stack_frequencies",
    "wamit_k_exponent",
]

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

COEFFICIENT_SCHEMA_VERSION = 1

DOF_ORDER: tuple[str, ...] = ("surge", "sway", "heave", "roll", "pitch", "yaw")
_ROTATIONAL = frozenset(("roll", "pitch", "yaw"))
_TRANSLATIONAL = frozenset(("surge", "sway", "heave"))

#: Relative precision an external dialect file can carry by construction: the
#: dialect writers use ``%.6e``, i.e. 7 significant digits, so a value reads
#: back to better than ``5e-7`` relative no matter who wrote it.
EXTERNAL_DIALECT_PRECISION = 5.0e-7


class CoefficientIntegrityError(ValueError):
    """Raised when a container's stored fingerprint does not match its payload."""


def _require_finite(name: str, array: NDArray[Any]) -> NDArray[Any]:
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _positive_finite(name: str, value: Any, *, allow_none: bool = False) -> float | None:
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"{name} must be a positive finite real number")
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive finite real number")
    out = float(value)
    if not math.isfinite(out) or out <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return out


def wamit_k_exponent(label_i: str, label_j: str) -> int:
    """Return the WAMIT nondimensionalisation exponent ``k`` for a coefficient pair.

    ``3`` for two translational DOFs, ``4`` for a translational/rotational pair,
    ``5`` for two rotational DOFs (WAMIT convention, mirrored by capytaine's
    ``io.wamit.K_LOOKUP``).
    """

    for label in (label_i, label_j):
        if label not in DOF_ORDER:
            raise ValueError(f"unknown degree of freedom {label!r}; expected one of {DOF_ORDER}")
    rot_i = label_i in _ROTATIONAL
    rot_j = label_j in _ROTATIONAL
    if rot_i and rot_j:
        return 5
    if rot_i or rot_j:
        return 4
    return 3


def wamit_m_exponent(label: str) -> int:
    """Excitation-force regularity exponent ``m`` (``2`` translational, ``3`` rotational)."""

    if label not in DOF_ORDER:
        raise ValueError(f"unknown degree of freedom {label!r}; expected one of {DOF_ORDER}")
    return 3 if label in _ROTATIONAL else 2


@dataclass(frozen=True)
class HydrodynamicCoefficients:
    """Frequency-domain hydrodynamic coefficients for one body (or one DOF subset).

    Parameters
    ----------
    omega:
        Angular frequencies ``(Nf,)`` in rad/s, strictly increasing and positive.
    added_mass:
        ``(Nf, ndof, ndof)`` physical added mass ``A_ij``.
    damping:
        ``(Nf, ndof, ndof)`` physical radiation damping ``B_ij``.
    dof_labels:
        ``ndof``-long tuple of names drawn from :data:`DOF_ORDER`, in the order
        the matrices are indexed.
    excitation:
        Optional ``(Nf, Nbeta, ndof)`` complex exciting force.
    headings_deg:
        ``Nbeta``-long heading angles in degrees (required with ``excitation``).
    rho, g, length_scale, water_depth:
        Environment used for every nondimensionalisation.  ``water_depth=None``
        means infinite depth.
    provenance:
        Free-form JSON-serialisable audit trail (solver, case, git ref, ...).
    """

    omega: FloatArray
    added_mass: FloatArray
    damping: FloatArray
    dof_labels: tuple[str, ...]
    excitation: ComplexArray | None = None
    headings_deg: FloatArray | None = None
    rho: float = 1025.0
    g: float = 9.80665
    length_scale: float = 1.0
    water_depth: float | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    # -- validation ---------------------------------------------------------
    def __post_init__(self) -> None:
        omega = np.asarray(self.omega, dtype=float).reshape(-1)
        if omega.size == 0:
            raise ValueError("omega must contain at least one frequency")
        _require_finite("omega", omega)
        if np.any(omega <= 0.0):
            raise ValueError("omega entries must be strictly positive")
        if omega.size > 1 and np.any(np.diff(omega) <= 0.0):
            raise ValueError("omega must be strictly increasing")

        labels = tuple(str(label) for label in self.dof_labels)
        if not labels:
            raise ValueError("dof_labels must be non-empty")
        duplicates = {label for label in labels if labels.count(label) > 1}
        if duplicates:
            raise ValueError(f"dof_labels contains duplicates: {sorted(duplicates)}")
        for label in labels:
            if label not in DOF_ORDER:
                raise ValueError(f"unknown degree of freedom {label!r}; expected one of {DOF_ORDER}")
        ndof = len(labels)
        nf = omega.size

        added_mass = np.asarray(self.added_mass, dtype=float)
        damping = np.asarray(self.damping, dtype=float)
        for name, array in (("added_mass", added_mass), ("damping", damping)):
            if array.shape != (nf, ndof, ndof):
                raise ValueError(
                    f"{name} must have shape {(nf, ndof, ndof)} (Nf, ndof, ndof); got {array.shape}"
                )
            _require_finite(name, array)

        excitation = None
        headings = None
        if self.excitation is not None:
            excitation = np.asarray(self.excitation, dtype=np.complex128)
            if excitation.ndim != 3 or excitation.shape[0] != nf or excitation.shape[2] != ndof:
                raise ValueError(
                    f"excitation must have shape {(nf, '<Nbeta>', ndof)}; got {excitation.shape}"
                )
            if not np.all(np.isfinite(excitation.real)) or not np.all(np.isfinite(excitation.imag)):
                raise ValueError("excitation must contain only finite values")
            if self.headings_deg is None:
                raise ValueError("headings_deg is required when excitation is given")
            headings = np.asarray(self.headings_deg, dtype=float).reshape(-1)
            if headings.size != excitation.shape[1]:
                raise ValueError(
                    "headings_deg length must match excitation.shape[1] "
                    f"({headings.size} != {excitation.shape[1]})"
                )
            _require_finite("headings_deg", headings)
        elif self.headings_deg is not None:
            raise ValueError("headings_deg was given without excitation")

        object.__setattr__(self, "omega", omega)
        object.__setattr__(self, "added_mass", added_mass)
        object.__setattr__(self, "damping", damping)
        object.__setattr__(self, "dof_labels", labels)
        object.__setattr__(self, "excitation", excitation)
        object.__setattr__(self, "headings_deg", headings)
        object.__setattr__(self, "rho", _positive_finite("rho", self.rho))
        object.__setattr__(self, "g", _positive_finite("g", self.g))
        object.__setattr__(self, "length_scale", _positive_finite("length_scale", self.length_scale))
        object.__setattr__(
            self, "water_depth", _positive_finite("water_depth", self.water_depth, allow_none=True)
        )
        object.__setattr__(self, "provenance", dict(self.provenance))

    # -- basic accessors ----------------------------------------------------
    @property
    def frequencies(self) -> FloatArray:
        return self.omega

    @property
    def periods(self) -> FloatArray:
        return 2.0 * math.pi / self.omega

    @property
    def ndof(self) -> int:
        return len(self.dof_labels)

    def dof_pairs(self) -> tuple[tuple[int, int, str, str], ...]:
        """Enumerate ``(i, j, label_i, label_j)`` for every stored coefficient pair."""

        return tuple(
            (i, j, label_i, label_j)
            for i, label_i in enumerate(self.dof_labels)
            for j, label_j in enumerate(self.dof_labels)
        )

    # -- integrity ----------------------------------------------------------
    def payload(self) -> dict[str, Any]:
        """Return the canonical, JSON-serialisable numeric payload.

        ``omega`` and both coefficient matrices are compared by their exact
        shortest round-trip ``repr`` (Python's ``json`` for floats), so the
        payload is deterministic across platforms for identical input.
        """

        payload: dict[str, Any] = {
            "schema_version": COEFFICIENT_SCHEMA_VERSION,
            "omega": self.omega.tolist(),
            "dof_labels": list(self.dof_labels),
            "added_mass": self.added_mass.tolist(),
            "damping": self.damping.tolist(),
        }
        if self.excitation is not None:
            payload["excitation"] = {
                "real": self.excitation.real.tolist(),
                "imag": self.excitation.imag.tolist(),
                "headings_deg": self.headings_deg.tolist(),
            }
        return payload

    def fingerprint(self) -> str:
        """SHA-256 of the canonical payload -- changes if any number changes."""

        blob = json.dumps(self.payload(), sort_keys=True, allow_nan=False, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    # -- canonical JSON -----------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Return the full native container (payload + environment + fingerprint)."""

        document = self.payload()
        document["environment"] = {
            "rho": self.rho,
            "g": self.g,
            "length_scale": self.length_scale,
            "water_depth": self.water_depth,
        }
        document["fingerprint"] = self.fingerprint()
        document["provenance"] = dict(self.provenance)
        return document

    @classmethod
    def from_dict(cls, document: Mapping[str, Any], *, verify: bool = True) -> "HydrodynamicCoefficients":
        """Rebuild a container from :meth:`to_dict` output (integrity-checked)."""

        try:
            version = int(document["schema_version"])
        except KeyError as exc:  # pragma: no cover - defensive
            raise ValueError("container is missing schema_version") from exc
        if version != COEFFICIENT_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported coefficient schema_version {version}; "
                f"this build speaks {COEFFICIENT_SCHEMA_VERSION}"
            )
        environment = document.get("environment", {})
        excitation_block = document.get("excitation")
        container = cls(
            omega=np.asarray(document["omega"], dtype=float),
            added_mass=np.asarray(document["added_mass"], dtype=float),
            damping=np.asarray(document["damping"], dtype=float),
            dof_labels=tuple(document["dof_labels"]),
            excitation=(
                None
                if excitation_block is None
                else np.asarray(excitation_block["real"], dtype=float)
                + 1j * np.asarray(excitation_block["imag"], dtype=float)
            ),
            headings_deg=(
                None if excitation_block is None else np.asarray(excitation_block["headings_deg"], dtype=float)
            ),
            rho=environment.get("rho", cls.rho),
            g=environment.get("g", cls.g),
            length_scale=environment.get("length_scale", cls.length_scale),
            water_depth=environment.get("water_depth"),
            provenance=document.get("provenance", {}),
        )
        stored = document.get("fingerprint")
        if verify and stored is not None and stored != container.fingerprint():
            raise CoefficientIntegrityError(
                "coefficient payload does not match its stored fingerprint "
                f"({stored[:16]}... != {container.fingerprint()[:16]}...)"
            )
        return container

    def write_json(self, path: str | Path) -> Path:
        """Write the native container; every float is exact (shortest ``repr``)."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return target

    @classmethod
    def read_json(cls, path: str | Path, *, verify: bool = True) -> "HydrodynamicCoefficients":
        """Read a native container and verify its fingerprint by default."""

        document = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(document, verify=verify)

    # -- WAMIT .1 (added mass + damping) ------------------------------------
    def _wamit_1_lines(self) -> list[str]:
        lines: list[str] = []
        kk = {
            (i, j): wamit_k_exponent(label_i, label_j)
            for i, j, label_i, label_j in self.dof_pairs()
        }
        for f_index, period in enumerate(self.periods):
            for i, j, label_i, label_j in self.dof_pairs():
                k = kk[(i, j)]
                norm = self.rho * self.length_scale**k
                a_norm = float(self.added_mass[f_index, i, j]) / norm
                b_norm = float(self.damping[f_index, i, j]) / (float(self.omega[f_index]) * norm)
                # The dialect numbers DOFs absolutely (1..6), not by position in
                # whatever subset this container happens to hold, and its two
                # index columns are (radiating, influenced) -- the order
                # capytaine's own exporter uses.
                w_i = DOF_ORDER.index(label_j) + 1  # column 2: radiating DOF
                w_j = DOF_ORDER.index(label_i) + 1  # column 3: influenced DOF
                lines.append(f"{period:20.12e} {w_i:5d} {w_j:5d} {a_norm:20.12e} {b_norm:20.12e}")
        return lines

    def write_wamit_1(self, path: str | Path) -> Path:
        """Write WAMIT-dialect ``.1`` lines: ``PER I J A_nondim B_nondim``."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(self._wamit_1_lines()) + "\n", encoding="utf-8")
        return target

    @classmethod
    def read_wamit_1(
        cls,
        path: str | Path,
        *,
        rho: float = 1025.0,
        length_scale: float = 1.0,
        g: float = 9.80665,
        water_depth: float | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> "HydrodynamicCoefficients":
        """Parse a WAMIT-DIALECT line-format ``.1`` file into physical coefficients.

        Only pairs that appear in the file are populated; missing pairs stay at
        zero, and the DOF set is the sorted union of the indices present.
        A ``period <= 0`` entry (the infinite-frequency/zero-frequency limits the
        dialect allows) is rejected rather than silently mapped to ``inf``.
        """

        rows: list[tuple[float, int, int, float, float]] = []
        for lineno, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
            line = raw.strip()
            if not line or line.startswith(("#", "!", "%")):
                continue
            fields = line.split()
            if len(fields) == 4:
                fields = fields + ["0.0"]
            if len(fields) != 5:
                raise ValueError(f"{path}:{lineno}: expected 4 or 5 fields, got {len(fields)}")
            try:
                period = float(fields[0])
                i = int(fields[1])
                j = int(fields[2])
                a_norm = float(fields[3])
                b_norm = float(fields[4])
            except ValueError as exc:
                raise ValueError(f"{path}:{lineno}: cannot parse WAMIT .1 line {raw!r}") from exc
            if not period > 0.0:
                raise ValueError(
                    f"{path}:{lineno}: period {period!r} is not positive; "
                    "the infinite/zero-frequency WAMIT limit blocks are not supported"
                )
            if not (1 <= i <= len(DOF_ORDER) and 1 <= j <= len(DOF_ORDER)):
                raise ValueError(f"{path}:{lineno}: DOF index out of range 1..6 (i={i}, j={j})")
            rows.append((period, i, j, a_norm, b_norm))
        if not rows:
            raise ValueError(f"{path}: no coefficient lines found")

        # Longest period == lowest frequency first: the container stores omega
        # ascending, and the dialect stores the period, so the sort order flips.
        periods = sorted({round(row[0], 12) for row in rows}, reverse=True)
        present = sorted({row[1] for row in rows} | {row[2] for row in rows})
        labels = tuple(DOF_ORDER[index - 1] for index in present)
        index_of = {label: position for position, label in enumerate(labels)}
        omega = np.array([2.0 * math.pi / period for period in periods], dtype=float)

        added_mass = np.zeros((len(periods), len(labels), len(labels)), dtype=float)
        damping = np.zeros((len(periods), len(labels), len(labels)), dtype=float)
        period_index = {period: position for position, period in enumerate(periods)}
        for period, i, j, a_norm, b_norm in rows:
            f = period_index[round(period, 12)]
            # Column 2 is the radiating DOF, column 3 the influenced one.
            label_j = DOF_ORDER[i - 1]  # radiating   -> second matrix axis
            label_i = DOF_ORDER[j - 1]  # influenced  -> first matrix axis
            k = wamit_k_exponent(label_i, label_j)
            norm = float(rho) * float(length_scale) ** k
            added_mass[f, index_of[label_i], index_of[label_j]] = a_norm * norm
            damping[f, index_of[label_i], index_of[label_j]] = b_norm * omega[f] * norm

        return cls(
            omega=omega,
            added_mass=added_mass,
            damping=damping,
            dof_labels=labels,
            rho=rho,
            g=g,
            length_scale=length_scale,
            water_depth=water_depth,
            provenance=dict(provenance or {}),
        )

    # -- WAMIT .3 (exciting force) ------------------------------------------
    def _wamit_3_lines(self) -> list[str]:
        assert self.excitation is not None and self.headings_deg is not None
        lines: list[str] = []
        for f_index, period in enumerate(self.periods):
            for b_index, beta in enumerate(self.headings_deg):
                for d_index, label in enumerate(self.dof_labels):
                    m = wamit_m_exponent(label)
                    w_index = DOF_ORDER.index(label) + 1
                    norm = self.rho * self.g * 1.0 * self.length_scale**m
                    value = complex(self.excitation[f_index, b_index, d_index]) / norm
                    mag = abs(value)
                    phase_deg = math.degrees(math.atan2(value.imag, value.real))
                    lines.append(
                        f"{period:20.12e} {float(beta):12.6f} {w_index:5d} "
                        f"{mag:20.12e} {phase_deg:20.12e} {value.real:20.12e} {value.imag:20.12e}"
                    )
        return lines

    def write_wamit_3(self, path: str | Path) -> Path:
        """Write WAMIT-dialect ``.3`` lines: ``PER BETA I |F| phase Re Im`` (nondim)."""

        if self.excitation is None:
            raise ValueError("container has no excitation; cannot write a .3 file")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(self._wamit_3_lines()) + "\n", encoding="utf-8")
        return target

    @classmethod
    def read_wamit_3(
        cls,
        path: str | Path,
        *,
        rho: float = 1025.0,
        g: float = 9.80665,
        length_scale: float = 1.0,
        wave_amplitude: float = 1.0,
        conjugate: bool = False,
        provenance: Mapping[str, Any] | None = None,
    ) -> "HydrodynamicCoefficients":
        """Parse a WAMIT-dialect ``.3`` file.

        ``conjugate=True`` absorbs a sign convention difference in the phase
        column: WAMIT's own convention is ``e^{+i*omega*t}``, so a file written by
        an ``e^{-i*omega*t}`` code (capytaine; also this library) carries the
        exciting force with the *opposite* phase sign when it is re-expressed in
        WAMIT's convention.  The modulus is unaffected either way, which is why
        the convention is a switch here and not a silently applied default.

        The modulus/phase columns are authoritative (WAMIT's own definition);
        the trailing ``Re``/``Im`` columns are only cross-checked against them and
        the two are never mixed.  Added mass and damping are not carried by a
        ``.3`` file and are therefore returned as zero.
        """

        rows: list[tuple[float, float, int, float, float]] = []
        for lineno, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
            line = raw.strip()
            if not line or line.startswith(("#", "!", "%")):
                continue
            fields = line.split()
            if len(fields) < 5:
                raise ValueError(f"{path}:{lineno}: expected at least 5 fields, got {len(fields)}")
            period = float(fields[0])
            beta = float(fields[1])
            dof = int(fields[2])
            mag = float(fields[3])
            phase_deg = float(fields[4])
            if not period > 0.0:
                raise ValueError(f"{path}:{lineno}: period {period!r} is not positive")
            if not 1 <= dof <= len(DOF_ORDER):
                raise ValueError(f"{path}:{lineno}: DOF index out of range 1..6 (i={dof})")
            if len(fields) >= 7:
                re_part = float(fields[5])
                im_part = float(fields[6])
                scale = max(mag, 1.0)
                if abs(re_part - mag * math.cos(math.radians(phase_deg))) > 1e-4 * scale or abs(
                    im_part - mag * math.sin(math.radians(phase_deg))
                ) > 1e-4 * scale:
                    raise ValueError(
                        f"{path}:{lineno}: Re/Im columns disagree with |F|/phase "
                        f"({re_part:g}{im_part:+g}j vs {mag:g}@{phase_deg:g}deg)"
                    )
            rows.append((period, beta, dof, mag, phase_deg))
        if not rows:
            raise ValueError(f"{path}: no excitation lines found")

        periods = sorted({round(row[0], 12) for row in rows}, reverse=True)
        headings = sorted({round(row[1], 10) for row in rows})
        present = sorted({row[2] for row in rows})
        labels = tuple(DOF_ORDER[index - 1] for index in present)
        index_of = {label: position for position, label in enumerate(labels)}
        omega = np.array([2.0 * math.pi / period for period in periods], dtype=float)

        excitation = np.zeros((len(periods), len(headings), len(labels)), dtype=np.complex128)
        period_index = {period: position for position, period in enumerate(periods)}
        heading_index = {beta: position for position, beta in enumerate(headings)}
        for period, beta, dof, mag, phase_deg in rows:
            label = DOF_ORDER[dof - 1]
            m = wamit_m_exponent(label)
            norm = float(rho) * float(g) * float(wave_amplitude) * float(length_scale) ** m
            angle = math.radians(phase_deg)
            if conjugate:
                angle = -angle
            value = norm * mag * complex(math.cos(angle), math.sin(angle))
            excitation[period_index[round(period, 12)], heading_index[round(beta, 10)], index_of[label]] = value

        return cls(
            omega=omega,
            added_mass=np.zeros((len(periods), len(labels), len(labels)), dtype=float),
            damping=np.zeros((len(periods), len(labels), len(labels)), dtype=float),
            dof_labels=labels,
            excitation=excitation,
            headings_deg=np.array(headings, dtype=float),
            rho=rho,
            g=g,
            length_scale=length_scale,
            provenance=dict(provenance or {}),
        )

    # -- Nemoh Tecplot dialect ---------------------------------------------
    def _tec_header(self, columns: Sequence[str], title: str) -> list[str]:
        quoted = " ".join(json.dumps(name) for name in columns)
        return [f'TITLE = {json.dumps(title)}', f"VARIABLES = {quoted}", 'ZONE T = "TensorBEM"']

    def write_nemoh_tec(self, path: str | Path, *, title: str = "Radiation coefficients") -> Path:
        """Write a Nemoh ``RadiationCoefficients.tec`` (omega + A_ij + B_ij rows)."""

        columns = ["omega"]
        columns += [f"A{i + 1}{j + 1}" for i in range(self.ndof) for j in range(self.ndof)]
        columns += [f"B{i + 1}{j + 1}" for i in range(self.ndof) for j in range(self.ndof)]
        lines = self._tec_header(columns, title)
        for f_index, value in enumerate(self.omega):
            cells = [f"{float(value):.9e}"]
            cells += [f"{self.added_mass[f_index, i, j]:.9e}" for i in range(self.ndof) for j in range(self.ndof)]
            cells += [f"{self.damping[f_index, i, j]:.9e}" for i in range(self.ndof) for j in range(self.ndof)]
            lines.append("  " + "  ".join(cells))
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return target

    @classmethod
    def read_nemoh_tec(
        cls,
        path: str | Path,
        *,
        rho: float = 1025.0,
        g: float = 9.80665,
        length_scale: float = 1.0,
        ndof: int | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> "HydrodynamicCoefficients":
        """Parse a Nemoh-style ``RadiationCoefficients.tec`` file.

        Column names are honoured when the ``VARIABLES`` header is present (so a
        file whose columns are reordered still parses); without a header the
        layout is assumed to be ``omega, A11..Ann, B11..Bnn`` with ``i`` the outer
        index, and ``ndof`` must then be supplied.
        """

        text = Path(path).read_text(encoding="utf-8")
        columns: list[str] | None = None
        table: list[list[float]] = []
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            upper = line.upper()
            if upper.startswith("TITLE"):
                continue
            if upper.startswith("VARIABLES"):
                body = line.split("=", 1)[1] if "=" in line else line[len("VARIABLES"):]
                columns = [token.strip().strip('"').strip("'") for token in body.replace(",", " ").split()]
                continue
            if upper.startswith("ZONE"):
                continue
            fields = line.replace(",", " ").split()
            try:
                table.append([float(token) for token in fields])
            except ValueError as exc:
                raise ValueError(f"{path}: cannot parse data row {raw!r}") from exc
        if not table:
            raise ValueError(f"{path}: no data rows found")
        widths = {len(row) for row in table}
        if len(widths) != 1:
            raise ValueError(f"{path}: ragged data rows (widths {sorted(widths)})")
        width = widths.pop()

        if columns is None:
            if ndof is None:
                solve = (width - 1) / 2.0
                root = int(round(math.sqrt(solve))) if solve > 0 else 0
                if root * root != solve:
                    raise ValueError(
                        f"{path}: cannot infer DOF count from {width} columns; pass ndof="
                    )
                ndof = root
            columns = ["omega"]
            columns += [f"A{i + 1}{j + 1}" for i in range(ndof) for j in range(ndof)]
            columns += [f"B{i + 1}{j + 1}" for i in range(ndof) for j in range(ndof)]
        if len(columns) != width:
            raise ValueError(f"{path}: {len(columns)} column names but {width} data columns")

        table_array = np.asarray(table, dtype=float)
        if "omega" not in columns:
            raise ValueError(f"{path}: no 'omega' column in the VARIABLES header")
        omega = table_array[:, columns.index("omega")]

        a_names = [name for name in columns if name.upper().startswith("A") and name[1:].isdigit()]
        b_names = [name for name in columns if name.upper().startswith("B") and name[1:].isdigit()]
        if not a_names:
            raise ValueError(f"{path}: no Aij columns found")
        def _pair_index(name: str) -> int:
            digits = name[1:]
            if len(digits) != 2 or not digits.isdigit():
                raise ValueError(f"{path}: coefficient column {name!r} is not of the form Aij/Bij")
            return int(digits[0]) * 10 + int(digits[1])

        indices = sorted({_pair_index(name) for name in a_names} | {_pair_index(name) for name in b_names})
        pair_dofs = sorted({index // 10 for index in indices} | {index % 10 for index in indices})
        labels = tuple(DOF_ORDER[i - 1] for i in pair_dofs if 1 <= i <= len(DOF_ORDER))
        if len(labels) != len(pair_dofs):
            raise ValueError(f"{path}: DOF index out of range 1..6 in {a_names + b_names}")
        position_of = {label: position for position, label in enumerate(labels)}

        def _fill(prefix: str, names: Sequence[str], name_of: dict[str, str]) -> FloatArray:
            out = np.zeros((table_array.shape[0], len(labels), len(labels)), dtype=float)
            for name in names:
                i_digit, j_digit = int(name[1]), int(name[2])
                label_i = DOF_ORDER[i_digit - 1]
                label_j = DOF_ORDER[j_digit - 1]
                if label_i in position_of and label_j in position_of:
                    out[:, position_of[label_i], position_of[label_j]] = table_array[:, columns.index(name)]
            del prefix, name_of
            return out

        added_mass = _fill("A", a_names, {})
        damping = _fill("B", b_names, {})

        order = np.argsort(omega)
        return cls(
            omega=omega[order],
            added_mass=added_mass[order],
            damping=damping[order],
            dof_labels=labels,
            rho=rho,
            g=g,
            length_scale=length_scale,
            provenance=dict(provenance or {}),
        )

    def write_nemoh_excitation_tec(self, path: str | Path, *, title: str = "Excitation forces") -> Path:
        """Write a Nemoh ``ExcitationForce.tec`` (angle blocks, |F|/phase columns)."""

        if self.excitation is None or self.headings_deg is None:
            raise ValueError("container has no excitation; cannot write an excitation .tec file")
        columns = ["beta", "omega"]
        for d_index in range(self.ndof):
            columns += [f"ModF{d_index + 1}", f"PhiF{d_index + 1}"]
        lines = self._tec_header(columns, title)
        for b_index, beta in enumerate(self.headings_deg):
            lines.append(f"ZONE T = \"beta = {float(beta):g}\"")
            for f_index, omega in enumerate(self.omega):
                cells = [f"{float(beta):.9e}", f"{float(omega):.9e}"]
                for d_index in range(self.ndof):
                    value = complex(self.excitation[f_index, b_index, d_index])
                    cells += [f"{abs(value):.9e}", f"{math.degrees(math.atan2(value.imag, value.real)):.9e}"]
                lines.append("  " + "  ".join(cells))
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return target

    @classmethod
    def read_nemoh_excitation_tec(
        cls,
        path: str | Path,
        *,
        rho: float = 1025.0,
        g: float = 9.80665,
        length_scale: float = 1.0,
        provenance: Mapping[str, Any] | None = None,
    ) -> "HydrodynamicCoefficients":
        """Parse a Nemoh-style excitation ``.tec`` file (``ModFk``/``PhiFk`` pairs)."""

        text = Path(path).read_text(encoding="utf-8")
        columns: list[str] | None = None
        table: list[list[float]] = []
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            upper = line.upper()
            if upper.startswith("TITLE"):
                continue
            if upper.startswith("VARIABLES"):
                body = line.split("=", 1)[1] if "=" in line else line[len("VARIABLES"):]
                columns = [token.strip().strip('"').strip("'") for token in body.replace(",", " ").split()]
                continue
            if upper.startswith("ZONE"):
                continue
            table.append([float(token) for token in line.replace(",", " ").split()])
        if not table or columns is None:
            raise ValueError(f"{path}: expected a VARIABLES header and at least one data row")
        table_array = np.asarray(table, dtype=float)
        if table_array.shape[1] != len(columns):
            raise ValueError(f"{path}: {len(columns)} column names but {table_array.shape[1]} data columns")
        lower = [name.lower() for name in columns]
        if "omega" not in lower:
            raise ValueError(f"{path}: no 'omega' column")
        omega_all = table_array[:, lower.index("omega")]
        beta_all = table_array[:, lower.index("beta")] if "beta" in lower else np.zeros_like(omega_all)

        mod_names = [name for name in columns if name.lower().startswith("modf")]
        if not mod_names:
            raise ValueError(f"{path}: no ModFk columns")
        dofs = sorted(int(name[4:]) for name in mod_names if name[4:].isdigit())
        labels = tuple(DOF_ORDER[index - 1] for index in dofs)

        omegas = sorted(set(omega_all.tolist()))
        betas = sorted(set(beta_all.tolist()))
        excitation = np.zeros((len(omegas), len(betas), len(labels)), dtype=np.complex128)
        for row_index in range(table_array.shape[0]):
            f_index = omegas.index(omega_all[row_index])
            b_index = betas.index(beta_all[row_index])
            for d_index, dof in enumerate(dofs):
                mod = table_array[row_index, lower.index(f"modf{dof}")]
                phase_name = f"phif{dof}"
                phase = table_array[row_index, lower.index(phase_name)]
                excitation[f_index, b_index, d_index] = mod * complex(
                    math.cos(math.radians(phase)), math.sin(math.radians(phase))
                )
        return cls(
            omega=np.array(omegas, dtype=float),
            added_mass=np.zeros((len(omegas), len(labels), len(labels)), dtype=float),
            damping=np.zeros((len(omegas), len(labels), len(labels)), dtype=float),
            dof_labels=labels,
            excitation=excitation,
            headings_deg=np.array(betas, dtype=float),
            rho=rho,
            g=g,
            length_scale=length_scale,
            provenance=dict(provenance or {}),
        )


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def from_radiation_3d(
    result: Any,
    *,
    rho: float = 1025.0,
    g: float = 9.80665,
    length_scale: float = 1.0,
    water_depth: float | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> HydrodynamicCoefficients:
    """Wrap a :class:`tensorbem.free_surface_solver3d.Radiation3DResult` as a container.

    The result stores one frequency slice for the DOF subset it was solved for
    (``result.modes``); DOF labels are mapped through :data:`DOF_ORDER` so the
    container speaks the same language as every external dialect.
    """

    modes = np.asarray(result.modes, dtype=int).reshape(-1)
    if modes.size != np.asarray(result.added_mass).shape[0]:
        raise ValueError("result.modes and result.added_mass disagree on the DOF count")
    labels = tuple(DOF_ORDER[int(mode)] for mode in modes)
    return HydrodynamicCoefficients(
        omega=np.array([float(result.omega)], dtype=float),
        added_mass=np.asarray(result.added_mass, dtype=float)[None, :, :],
        damping=np.asarray(result.damping, dtype=float)[None, :, :],
        dof_labels=labels,
        rho=rho,
        g=g,
        length_scale=length_scale,
        water_depth=water_depth,
        provenance=dict(provenance or {}),
    )


def stack_frequencies(slices: Iterable[HydrodynamicCoefficients]) -> HydrodynamicCoefficients:
    """Merge single-frequency containers into one frequency sweep.

    Every slice must share ``dof_labels``, ``rho``, ``g``, ``length_scale`` and
    ``water_depth``; slices with excitation must also share ``headings_deg``.
    Duplicate frequencies raise instead of silently overwriting.
    """

    items = list(slices)
    if not items:
        raise ValueError("stack_frequencies needs at least one container")
    first = items[0]
    for other in items[1:]:
        for attribute in ("dof_labels", "rho", "g", "length_scale", "water_depth"):
            if getattr(other, attribute) != getattr(first, attribute):
                raise ValueError(
                    f"cannot stack containers with different {attribute}: "
                    f"{getattr(other, attribute)!r} != {getattr(first, attribute)!r}"
                )
        if (other.excitation is None) != (first.excitation is None):
            raise ValueError("cannot stack containers where only some carry excitation")
        if first.headings_deg is not None and not np.array_equal(other.headings_deg, first.headings_deg):
            raise ValueError("cannot stack containers with different heading grids")

    ordered = sorted(items, key=lambda item: float(item.omega[0]))
    omegas = np.array([float(item.omega[0]) for item in ordered], dtype=float)
    if np.any(np.diff(omegas) <= 0.0):
        raise ValueError("duplicate frequencies in stack_frequencies input")

    provenance: dict[str, Any] = dict(first.provenance)
    merged_sources = [item.provenance.get("fingerprint") for item in ordered]
    if any(merged_sources):
        provenance.setdefault("slices", [source for source in merged_sources if source])

    return replace(
        first,
        omega=omegas,
        added_mass=np.concatenate([item.added_mass for item in ordered], axis=0),
        damping=np.concatenate([item.damping for item in ordered], axis=0),
        excitation=(
            None
            if first.excitation is None
            else np.concatenate([item.excitation for item in ordered], axis=0)
        ),
        provenance=provenance,
    )


#: Backwards-friendly alias: "combine frequency slices" is what it does.
combine_slices = stack_frequencies


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _summarise(container: HydrodynamicCoefficients, path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "dof_labels": list(container.dof_labels),
        "frequencies": int(container.omega.size),
        "omega_min": float(container.omega.min()),
        "omega_max": float(container.omega.max()),
        "has_excitation": container.excitation is not None,
        "headings_deg": None if container.headings_deg is None else container.headings_deg.tolist(),
        "added_mass_max_abs": float(np.abs(container.added_mass).max()),
        "damping_max_abs": float(np.abs(container.damping).max()),
        "fingerprint": container.fingerprint(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+", type=Path, help="coefficient container to inspect")
    parser.add_argument("--no-verify", action="store_true", help="skip JSON fingerprint checking")
    args = parser.parse_args(argv)

    status = 0
    for path in args.paths:
        try:
            container = HydrodynamicCoefficients.read_json(path, verify=not args.no_verify)
        except (CoefficientIntegrityError, ValueError) as exc:
            print(f"{path}: FAILED -- {exc}", file=sys.stderr)
            status = 1
            continue
        print(json.dumps(_summarise(container, path), indent=2))
    return status


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())