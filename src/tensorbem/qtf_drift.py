"""Second-order difference-frequency QTF and slow-drift force spectra.

The module fills the gap between the *diagonal* second-order force (the mean
drift force in regular waves, :mod:`tensorbem.drift`) and the *full*
difference-frequency quadratic transfer function ``T(omega_i, omega_j)`` that a
slow-drift / mooring design study needs.  Two routes to the off-diagonal part
are implemented and arbitrated against each other:

Direct QTF ("quadratic-pressure" part, all non-diagonal terms)
    For a body held **fixed** the second-order force at the difference frequency
    ``mu = omega_i - omega_j`` is, in the frequency domain,

        T_ij = (1/4) rho * Int_S (grad psi_i . grad psi_j*) n dS
               - (1/4) rho g * Int_WL zeta_i zeta_j* n dl ,

    with ``psi`` the first-order *unit-amplitude* diffraction potential and
    ``zeta = i omega psi / g`` the first-order elevation on the waterline.  Both
    prefactors reduce to the verified mean-drift assembly of
    :mod:`tensorbem.drift` on the diagonal (``i = j``), which is asserted to
    machine precision in the tests; that identity is the regression pin of the
    whole construction.  The force signal of a sea state is
    ``F(t) = Re sum_ij A_i A_j* T_ij exp(-i mu t)``.

Newman approximation
    ``T_ij^N = (T_ii + T_jj) / 2`` -- exact on the diagonal and everywhere in a
    zero-bandwidth (regular-wave) limit, increasingly in error as the sea
    state's relative bandwidth grows.  The module quantifies that error instead
    of assuming it away: :func:`newman_bandwidth_study` returns the relative
    error of the mean drift and of the low-frequency force RMS as functions of
    the spectral bandwidth.

Scope of the "direct" QTF (declared, not hidden)
    What is implemented is the complete quadratic *pressure* transfer function:
    the products of first-order potentials over the mean wetted surface plus the
    oscillating-waterline strip.  The remaining second-order contribution -- the
    force exerted by the second-order potential itself -- vanishes identically on
    the diagonal (the steady second-order potential of a fixed body is a
    constant) but is non-zero off the diagonal.  By Green's second identity it
    can be replaced by a free-surface integral ``Int_SF psi_j(mu) Q_ij dS``
    (Molins' theorem) that needs the first-order *radiation* potential of the
    body at every difference frequency ``mu``; that term is **not** implemented
    here.  ``docs/qtf_drift_oc4.md`` states the consequence and the acceptance
    gate it affects.  Mean drift, the small-bandwidth force RMS and the Newman
    arbitration are qualitatively unaffected; the off-diagonal QTF magnitude
    computed here is a lower bound.

Units are SI (m, s, kg, N); the time convention is ``exp(-i omega t)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Literal, Sequence

import numpy as np

from .drift import (
    _nodal_values,
    far_field_mean_drift_3d,
    near_field_mean_drift_3d,
    panel_surface_gradient,
    waterline_segments,
)
from .free_surface_solver3d import Diffraction3DResult, solve_diffraction_3d
from .geometry3d import SurfaceMesh3D, WettedHullMesh

FloatArray = np.ndarray

_DEFAULT_DENSITY = 1025.0
_DEFAULT_GRAVITY = 9.81


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _positive(name: str, value: float) -> float:
    out = float(value)
    if not math.isfinite(out) or out <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return out


def _resolve_normals(mesh: SurfaceMesh3D) -> FloatArray:
    """Unit normals pointing out of the body, independent of the input winding."""
    faces = np.asarray(mesh.faces, dtype=np.int64)
    v0 = mesh.vertices[faces[:, 0]]
    v1 = mesh.vertices[faces[:, 1]]
    v2 = mesh.vertices[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    lengths = np.linalg.norm(normals, axis=1)
    normals = normals / np.where(lengths == 0.0, 1.0, lengths)[:, None]
    # divergence-theorem volume of the (waterline-closed) hull; the closing lid at
    # z = 0 contributes nothing because its centroid lies on the plane
    volume = float(
        np.sum(mesh.areas * np.einsum("ij,ij->i", mesh.centroids, normals)) / 3.0
    )
    if volume < 0.0:
        normals = -normals
    return normals


def waterline_elevations(
    mesh: SurfaceMesh3D,
    potential: np.ndarray,
    *,
    wavenumber: float,
    gravity: float = _DEFAULT_GRAVITY,
) -> dict:
    """Complex elevation amplitude on every waterline edge of the mesh.

    ``zeta_hat = i omega phi_hat / g`` at the waterline panel nodes, averaged over
    the two nodes of each boundary edge -- the same reduction
    :func:`tensorbem.drift.near_field_mean_drift_3d` uses, so the diagonal of the
    QTF collapses onto that verified assembly.
    """
    phi = np.asarray(potential, dtype=np.complex128)
    wl = waterline_segments(mesh)
    omega = math.sqrt(float(gravity) * float(wavenumber))
    if wl["lengths"].size == 0:
        return {
            "lengths": np.zeros(0),
            "normals": np.zeros((0, 3)),
            "elevation": np.zeros(0, dtype=np.complex128),
            "count": 0,
        }
    nodal_phi = _nodal_values(mesh, phi)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    edge_vertices = []
    for face_idx in wl["adjacent_faces"]:
        face = faces[face_idx]
        zs = np.abs(np.asarray(mesh.vertices)[face, 2])
        edge_vertices.append(face[zs <= 1.0e-9 * max(float(mesh.h_max), 1e-12)])
    phi_edge = np.array(
        [np.mean(nodal_phi[v]) if v.size else 0.0 + 0.0j for v in edge_vertices],
        dtype=np.complex128,
    )
    return {
        "lengths": np.asarray(wl["lengths"], dtype=float),
        "normals": np.asarray(wl["normals"], dtype=float),
        "elevation": 1.0j * omega * phi_edge / float(gravity),
        "count": int(wl["lengths"].size),
    }


# ---------------------------------------------------------------------------
# QTF container
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DriftQTF:
    """Difference-frequency QTF on a frequency set, per wave-amplitude squared.

    ``matrix[i, j]`` is the complex QTF for the pair ``(omega_i, omega_j)``,
    normalised so that a regular wave of amplitude ``A`` at ``omega_i`` carries
    the mean drift force ``A**2 * Re(matrix[i, i])``.  ``diagonal`` holds the real
    mean-drift coefficients.
    """

    omegas: FloatArray
    wavenumbers: FloatArray
    matrix: np.ndarray = field(repr=False)
    diagonal: FloatArray
    route: str = "direct"
    density: float = _DEFAULT_DENSITY
    gravity: float = _DEFAULT_GRAVITY
    wave_angle: float = 0.0
    metadata: dict = field(default_factory=dict)

    @property
    def frequency_count(self) -> int:
        return int(self.matrix.shape[0])

    def difference_frequencies(self) -> FloatArray:
        return np.abs(self.omegas[:, None] - self.omegas[None, :])


def newman_qtf(diagonal: np.ndarray) -> np.ndarray:
    """Newman approximation ``T_ij = (T_ii + T_jj) / 2`` from the diagonal.

    ``diagonal`` has shape ``(N, M)`` (M force components); the result has shape
    ``(N, N, M)`` and is real symmetric.
    """
    diag = np.asarray(diagonal, dtype=float)
    if diag.ndim == 1:
        diag = diag[:, None]
    if diag.ndim != 2:
        raise ValueError("diagonal must have shape (N, M) or (N,)")
    return 0.5 * (diag[:, None, :] + diag[None, :, :])


def qtf_from_diagonal(
    diagonal: np.ndarray,
    omegas: Iterable[float],
    *,
    density: float = _DEFAULT_DENSITY,
    gravity: float = _DEFAULT_GRAVITY,
    wave_angle: float = 0.0,
    metadata: dict | None = None,
) -> DriftQTF:
    """Build a Newman-approximation :class:`DriftQTF` from mean-drift data."""
    om = np.asarray(list(omegas), dtype=float)
    diag = np.asarray(diagonal, dtype=float)
    if diag.ndim == 1:
        diag = diag[:, None]
    if diag.shape[0] != om.size:
        raise ValueError("diagonal and omegas must have the same length")
    matrix = newman_qtf(diag).astype(np.complex128)
    return DriftQTF(
        omegas=om,
        wavenumbers=om**2 / float(gravity),
        matrix=matrix,
        diagonal=np.array(diag, dtype=float),
        route="newman",
        density=float(density),
        gravity=float(gravity),
        wave_angle=float(wave_angle),
        metadata=dict(metadata or {}),
    )


# ---------------------------------------------------------------------------
# direct (quadratic-pressure) QTF
# ---------------------------------------------------------------------------
def diagonal_drift_coefficients(
    mesh: SurfaceMesh3D,
    diffraction_results: Sequence[Diffraction3DResult],
    *,
    density: float = _DEFAULT_DENSITY,
    gravity: float = _DEFAULT_GRAVITY,
) -> FloatArray:
    """Mean-drift coefficient ``F_bar / A**2`` for each regular wave."""
    rows = []
    for res in diffraction_results:
        near = near_field_mean_drift_3d(mesh, res, density=density, gravity=gravity)
        rows.append(np.asarray(near.mean_force, dtype=float) / float(res.amplitude) ** 2)
    return np.asarray(rows, dtype=float)


def quadratic_qtf(
    mesh: SurfaceMesh3D,
    wavenumbers: Sequence[float],
    *,
    wave_angle: float = 0.0,
    density: float = _DEFAULT_DENSITY,
    gravity: float = _DEFAULT_GRAVITY,
    diffraction_results: Sequence[Diffraction3DResult] | None = None,
    modes: Literal["horizontal", "all"] = "horizontal",
    solver_kwargs: dict | None = None,
) -> DriftQTF:
    """Full non-diagonal difference-frequency QTF (quadratic-pressure part).

    Every off-diagonal pair ``(i, j)`` is evaluated from the two first-order
    diffraction potentials of that pair; nothing is interpolated and no
    narrow-band, fore-aft or symmetric-body assumption is made.  The diagonal
    reproduces :func:`tensorbem.drift.near_field_mean_drift_3d` exactly (same
    panels, same gradient recovery, same waterline reduction).
    """
    ks = np.asarray([_positive("wavenumber", k) for k in wavenumbers], dtype=float)
    if ks.size == 0:
        raise ValueError("at least one wavenumber is required")
    rho = _positive("density", density)
    g = _positive("gravity", gravity)
    solver_kwargs = dict(solver_kwargs or {})
    normals = _resolve_normals(mesh)
    areas = np.asarray(mesh.areas, dtype=float)

    if diffraction_results is None:
        results = [
            solve_diffraction_3d(
                mesh,
                float(k),
                wave_angle=wave_angle,
                density=rho,
                gravity=g,
                **solver_kwargs,
            )
            for k in ks
        ]
    else:
        results = list(diffraction_results)
        if len(results) != ks.size:
            raise ValueError("one diffraction result is required per wavenumber")

    grads = []
    elevations = []
    for res in results:
        grad = panel_surface_gradient(
            mesh, np.asarray(res.total_potential, dtype=np.complex128)
        )
        # a fixed body has zero normal derivative; strip the numerical residual so
        # it cannot inflate the quadratic product
        grad = grad - np.einsum("pi,pi->p", grad, normals)[:, None] * normals
        grads.append(grad)
        elevations.append(
            waterline_elevations(
                mesh,
                res.total_potential,
                wavenumber=float(res.wavenumber),
                gravity=g,
            )
        )
    grads = np.asarray(grads)
    n_f = ks.size

    pair_grad = np.einsum("ipk,jpk->pij", grads, np.conj(grads))
    pressure = (0.25 * rho) * np.einsum(
        "pij,pk->ijk", pair_grad * areas[:, None, None], normals
    )

    lengths = elevations[0]["lengths"]
    wl_normals = elevations[0]["normals"]
    if lengths.size:
        zeta = np.asarray([e["elevation"] for e in elevations])
        pair_zeta = zeta[:, None, :] * np.conj(zeta)[None, :, :]
        waterline = -(0.25 * rho * g) * np.einsum(
            "ije,ek->ijk", pair_zeta * lengths[None, None, :], wl_normals
        )
    else:
        waterline = np.zeros((n_f, n_f, 3), dtype=np.complex128)

    matrix = pressure + waterline
    if modes == "horizontal":
        matrix = matrix[..., :2]
    diag = np.array(np.real(matrix[np.arange(n_f), np.arange(n_f)]), dtype=float)
    return DriftQTF(
        omegas=np.sqrt(g * ks),
        wavenumbers=ks,
        matrix=matrix,
        diagonal=diag,
        route="direct",
        density=rho,
        gravity=g,
        wave_angle=float(wave_angle),
        metadata={
            "panels": int(mesh.panel_count),
            "waterline_edges": int(lengths.size),
            "relative_residual_max": float(
                max(abs(float(r.relative_residual)) for r in results)
            ),
        },
    )


def diagonal_route_arbitration(
    mesh: SurfaceMesh3D,
    wavenumbers: Sequence[float],
    *,
    wave_angle: float = 0.0,
    density: float = _DEFAULT_DENSITY,
    gravity: float = _DEFAULT_GRAVITY,
    n_azimuth: int = 240,
    solver_kwargs: dict | None = None,
    diffraction_results: Sequence[Diffraction3DResult] | None = None,
) -> dict:
    """Near-field (Pinkster) versus far-field (Maruo) mean drift per frequency.

    The two routes share no numerical machinery: the near field integrates the
    quadratic pressure and the waterline strip over the body, the far field
    integrates the Kochin function around the azimuth circle.
    """
    solver_kwargs = dict(solver_kwargs or {})
    ks = [float(k) for k in wavenumbers]
    results = (
        list(diffraction_results)
        if diffraction_results is not None
        else [
            solve_diffraction_3d(
                mesh,
                k,
                wave_angle=wave_angle,
                density=density,
                gravity=gravity,
                **solver_kwargs,
            )
            for k in ks
        ]
    )
    rows = []
    for res in results:
        near = near_field_mean_drift_3d(mesh, res, density=density, gravity=gravity)
        far = far_field_mean_drift_3d(mesh, res, density=density, n_azimuth=n_azimuth)
        near_force = np.asarray(near.mean_force, dtype=float)
        far_force = np.asarray(far["force"], dtype=float)
        # the far-field route returns the two horizontal components only, so the
        # arbitration is done on the horizontal drift force
        near_horizontal = near_force[: far_force.size]
        denom = max(float(np.linalg.norm(near_horizontal)), 1.0e-30)
        rows.append(
            {
                "wavenumber": float(res.wavenumber),
                "omega": float(res.omega),
                "near_field": near_force.tolist(),
                "far_field": far_force.tolist(),
                "relative_gap": float(np.linalg.norm(near_horizontal - far_force) / denom),
                "relative_residual": float(res.relative_residual),
                "waterline_edges": int(near.waterline_count),
            }
        )
    gaps = [r["relative_gap"] for r in rows]
    return {
        "wave_angle": float(wave_angle),
        "panel_count": int(mesh.panel_count),
        "points": rows,
        "max_relative_gap": float(max(gaps)) if gaps else float("nan"),
        "median_relative_gap": float(np.median(gaps)) if gaps else float("nan"),
    }


# ---------------------------------------------------------------------------
# sea states
# ---------------------------------------------------------------------------
def sea_state_drift_force(
    qtf: DriftQTF,
    amplitudes: Sequence[float],
    *,
    bin_width: float | None = None,
) -> dict:
    """Phase-averaged mean drift, force variance and low-frequency force spectrum.

    ``F(t) = Re sum_ij A_i A_j* T_ij exp(-i mu_ij t)``; with random component
    phases the cross terms carry the variance ``sum_{i!=j} |F_ij|^2 / 2`` while
    the time mean collapses onto ``sum_i A_i^2 Re T_ii``.  The one-sided spectrum
    ``S_F(mu)`` is binned so that ``Int S_F dmu`` equals that variance.
    """
    amp = np.asarray(amplitudes, dtype=float)
    if amp.ndim != 1:
        raise ValueError("amplitudes must be one-dimensional")
    n = qtf.frequency_count
    if amp.size != n:
        raise ValueError("one amplitude per QTF frequency is required")
    t = np.asarray(qtf.matrix)  # (N, N, M)
    if amp.size == 1:
        pair = (amp[0] ** 2) * t
    else:
        pair = amp[:, None, None] * amp[None, :, None] * t
    mu = qtf.omegas[:, None] - qtf.omegas[None, :]
    abs_mu = np.abs(mu)

    diag_matrix = np.real(t[np.arange(n), np.arange(n)])  # (N, M)
    mean = amp**2 @ diag_matrix
    diag_mask = np.eye(n, dtype=bool)
    off = ~diag_mask
    variance = float(np.sum(0.5 * np.abs(pair[off]) ** 2)) if n > 1 else 0.0
    # (pair[off] has shape (N^2 - N, M); the sum over components is intended)

    pos = abs_mu[off]
    # total power over the force components: the vector force variance is the sum
    # of the per-component variances, so the spectrum reported here is the norm
    power = np.sum(np.abs(pair[off]) ** 2, axis=-1)
    if bin_width is None:
        unique = np.unique(np.round(pos, 12))
        step = float(np.min(np.diff(unique))) if unique.size > 1 else float(pos.max() or 1.0)
        bin_width = max(step, 1.0e-9)
    if pos.size:
        top = float(pos.max()) + 0.5 * float(bin_width)
        edges = np.arange(0.0, top + float(bin_width), float(bin_width))
        counts, edges = np.histogram(pos, bins=edges, weights=power)
        centres = 0.5 * (edges[:-1] + edges[1:])
        spectrum = 0.5 * counts / float(bin_width)  # energy per unit frequency
    else:
        centres = np.zeros(0)
        spectrum = np.zeros(0)
    return {
        "mean_force": np.asarray(mean, dtype=float),
        "force_variance": variance,
        "force_rms": math.sqrt(variance),
        "mu": centres,
        "spectrum": spectrum,
        "spectrum_integral": float(np.sum(spectrum) * float(bin_width)) if pos.size else 0.0,
        "bin_width": float(bin_width),
    }


def newman_bandwidth_study(
    qtf_direct: DriftQTF,
    sea_states: Sequence[dict],
) -> list[dict]:
    """Arbitrate Newton-Newman against the direct QTF over a bandwidth sweep.

    For every sea state the direct non-diagonal QTF and the Newman matrix built
    from the same diagonal are pushed through :func:`sea_state_drift_force`; the
    entries report how far the Newman approximation moves the mean drift (in
    principle zero), the low-frequency force RMS, the spectrum integral and the
    spectrum shape (Pearson correlation on the common bins).
    """
    diag = np.asarray(qtf_direct.diagonal, dtype=float)
    newman = qtf_from_diagonal(
        diag,
        qtf_direct.omegas,
        density=qtf_direct.density,
        gravity=qtf_direct.gravity,
        wave_angle=qtf_direct.wave_angle,
    )
    rows = []
    for sea in sea_states:
        amp = np.asarray(sea["amplitude"], dtype=float)
        if amp.size != qtf_direct.frequency_count:
            raise ValueError("sea state and QTF frequency sets must match")
        direct = sea_state_drift_force(qtf_direct, amp)
        approx = sea_state_drift_force(newman, amp)
        mean_exact = float(np.linalg.norm(direct["mean_force"]))
        denom = max(float(approx["force_rms"]), 1.0e-30)
        row = {
            "spectral_width": float(sea["spectral_width"]),
            "relative_bandwidth": float(sea["relative_bandwidth"]),
            "hs": float(sea.get("hs", sea.get("hs_discrete", float("nan")))),
            "tp": float(sea["tp"]),
            "mean_force_direct": direct["mean_force"].tolist(),
            "mean_force_newman": approx["mean_force"].tolist(),
            "mean_relative_error": float(
                np.linalg.norm(approx["mean_force"] - direct["mean_force"])
                / max(mean_exact, 1.0e-30)
            ),
            "rms_direct": float(direct["force_rms"]),
            "rms_newman": float(approx["force_rms"]),
            "rms_relative_error": float(
                abs(approx["force_rms"] - direct["force_rms"]) / denom
            ),
            "spectrum_integral_direct": float(direct["spectrum_integral"]),
            "spectrum_integral_newman": float(approx["spectrum_integral"]),
            "spectrum_correlation": _spectrum_correlation(direct, approx),
        }
        rows.append(row)
    return rows


def _spectrum_correlation(a: dict, b: dict) -> float:
    """Pearson correlation of two spectra on their common bins.

    A bin centre that exists in one spectrum but not the other counts as zero
    energy on both sides, so a missing low-frequency shoulder is penalised rather
    than ignored.
    """
    grid = np.union1d(a["mu"], b["mu"])
    if grid.size < 3:
        return float("nan")

    def project(d: dict) -> FloatArray:
        out = np.zeros(grid.size)
        if d["mu"].size:
            idx = np.searchsorted(grid, d["mu"])
            out[idx] = d["spectrum"]
        return out

    x, y = project(a), project(b)
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def slow_drift_response_spectrum(
    spectrum: dict,
    *,
    mass: float,
    added_mass: float,
    damping: float,
    stiffness: float,
) -> dict:
    """Slow-drift surge response of a single-degree-of-freedom moored system.

    ``S_x(mu) = S_F(mu) / |K - (M + A) mu^2 + i B mu|^2`` with ``mu`` the
    difference frequency; the returned ``rao`` is ``|H(mu)|`` itself, i.e. the
    response per unit drift-force amplitude, which is what a mooring study plots
    against the measured slow-drift RAO.
    """
    m_total = _positive("mass + added_mass", float(mass) + float(added_mass))
    k = _positive("stiffness", stiffness)
    b = _positive("damping", damping) if damping > 0 else 0.0
    mu = np.asarray(spectrum["mu"], dtype=float)
    s_f = np.asarray(spectrum["spectrum"], dtype=float)
    denom = (k - m_total * mu**2) ** 2 + (b * mu) ** 2
    transfer = 1.0 / np.sqrt(np.maximum(denom, 1.0e-300))
    s_x = s_f * transfer**2
    width = float(spectrum["bin_width"])
    return {
        "mu": mu,
        "rao": transfer,
        "spectrum": s_x,
        "std_motion": float(math.sqrt(max(np.sum(s_x) * width, 0.0))),
        "natural_frequency": float(math.sqrt(k / m_total)),
        "damping_ratio": float(b / (2.0 * math.sqrt(k * m_total))),
    }


# ---------------------------------------------------------------------------
# geometry: OC4-class four-column semi-submersible
# ---------------------------------------------------------------------------
SEMISUB_MAIN_COLUMN_RADIUS = 3.25  # 6.5 m diameter central column
SEMISUB_OFFSET_COLUMN_RADIUS = 6.0  # 12 m diameter upper columns
SEMISUB_BASE_COLUMN_RADIUS = 12.0  # 24 m diameter base / heave plate
SEMISUB_OFFSET_RADIUS = 28.8675  # offset columns at 50 m centre spacing
SEMISUB_DRAFT = 20.0
SEMISUB_BASE_COLUMN_HEIGHT = 6.0
SEMISUB_PONTOON_RADIUS = 0.8  # 1.6 m diameter pontoons
SEMISUB_PONTOON_Z = -17.0
SEMISUB_PUBLISHED_DISPLACEMENT = 13473.0  # m^3, DeepCwind/OC4 semi-submersible


def semisub_analytic_displacement(
    *,
    main_radius: float = SEMISUB_MAIN_COLUMN_RADIUS,
    offset_radius: float = SEMISUB_OFFSET_COLUMN_RADIUS,
    base_radius: float = SEMISUB_BASE_COLUMN_RADIUS,
    offset_centre_radius: float = SEMISUB_OFFSET_RADIUS,
    draft: float = SEMISUB_DRAFT,
    base_height: float = SEMISUB_BASE_COLUMN_HEIGHT,
    pontoon_radius: float = SEMISUB_PONTOON_RADIUS,
    n_theta: int | None = None,
    pontoon_theta: int | None = None,
) -> dict:
    """Analytic displaced volume and waterplane area of the four-column layout.

    The columns are disjoint, so the volume is a plain sum; the pontoons count
    only over the free span between the base-column surfaces (the part buried
    inside a base column is not wetted).  This closed form is the geometry
    validation target for :func:`four_column_semisub_mesh`.
    """
    def section(radius: float, facets: int | None) -> float:
        """Cross-section area: exact circle, or the inscribed n-gon of the mesh."""
        if facets is None:
            return math.pi * radius**2
        if facets < 3:
            raise ValueError("facets must be >= 3")
        return 0.5 * facets * radius**2 * math.sin(2.0 * math.pi / facets)

    main_volume = section(main_radius, n_theta) * draft
    upper_height = draft - base_height
    offset_volume = 3.0 * section(offset_radius, n_theta) * upper_height
    base_volume = 3.0 * section(base_radius, n_theta) * base_height
    span = offset_centre_radius * math.sqrt(3.0) - 2.0 * base_radius
    pontoon_volume = 3.0 * section(pontoon_radius, pontoon_theta) * max(span, 0.0)
    waterplane = section(main_radius, n_theta) + 3.0 * section(offset_radius, n_theta)
    return {
        "main_column_volume": float(main_volume),
        "offset_column_volume": float(offset_volume),
        "base_column_volume": float(base_volume),
        "pontoon_volume": float(pontoon_volume),
        "displacement": float(
            main_volume + offset_volume + base_volume + pontoon_volume
        ),
        "waterplane_area": float(waterplane),
        "pontoon_free_span": float(span),
    }


def _graded_levels(depth: float, n_z: int, growth: float) -> FloatArray:
    dz0 = depth * (growth - 1.0) / (growth**n_z - 1.0)
    dz = dz0 * growth ** np.arange(n_z, dtype=float)
    return np.concatenate(([0.0], -np.cumsum(dz)))


def _signed_volume(vertices: FloatArray, faces: np.ndarray) -> float:
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    return float(np.sum(np.einsum("ij,ij->i", v0, np.cross(v1, v2))) / 6.0)


def _column_component(
    centre: tuple[float, float],
    radius: float,
    z_bottom: float,
    *,
    n_theta: int,
    n_z: int,
    growth: float = 1.35,
    top: float = 0.0,
    bottom_cap: bool = True,
) -> tuple[FloatArray, list]:
    """Surface-piercing cylinder wall (open at the waterline) plus optional cap."""
    levels = top + _graded_levels(top - z_bottom, n_z, growth)
    theta = 2.0 * math.pi * np.arange(n_theta, dtype=float) / n_theta
    cx, cy = centre
    ring = np.stack((cx + radius * np.cos(theta), cy + radius * np.sin(theta)), axis=1)
    side = np.array([[x, y, z] for z in levels for (x, y) in ring], dtype=float)
    faces: list[tuple[int, int, int]] = []
    for i in range(n_z):
        for j in range(n_theta):
            jn = (j + 1) % n_theta
            a = i * n_theta + j
            b = i * n_theta + jn
            c = (i + 1) * n_theta + jn
            d = (i + 1) * n_theta + j
            if (i + j) % 2 == 0:
                faces.append((a, c, b))
                faces.append((a, d, c))
            else:
                faces.append((a, d, b))
                faces.append((b, d, c))
    if bottom_cap:
        centre_index = side.shape[0]
        side = np.vstack((side, [[cx, cy, levels[-1]]]))
        base = n_z * n_theta
        for j in range(n_theta):
            jn = (j + 1) % n_theta
            faces.append((centre_index, base + jn, base + j))
    return side, faces


def _annulus_component(
    centre: tuple[float, float],
    r_inner: float,
    r_outer: float,
    z: float,
    *,
    n_theta: int,
    facing_up: bool = True,
) -> tuple[FloatArray, list]:
    """Horizontal annular face (e.g. the wetted top of a base column)."""
    if not r_outer > r_inner > 0.0:
        raise ValueError("r_outer > r_inner > 0 is required")
    theta = 2.0 * math.pi * np.arange(n_theta, dtype=float) / n_theta
    cx, cy = centre
    inner = np.stack((cx + r_inner * np.cos(theta), cy + r_inner * np.sin(theta)), axis=1)
    outer = np.stack((cx + r_outer * np.cos(theta), cy + r_outer * np.sin(theta)), axis=1)
    vertices = np.vstack(
        (
            np.column_stack((inner, np.full(n_theta, z))),
            np.column_stack((outer, np.full(n_theta, z))),
        )
    )
    faces: list[tuple[int, int, int]] = []
    for j in range(n_theta):
        jn = (j + 1) % n_theta
        i0, i1 = j, jn
        o0, o1 = n_theta + j, n_theta + jn
        if facing_up:
            faces.append((i0, o0, o1))
            faces.append((i0, o1, i1))
        else:
            faces.append((i0, o1, o0))
            faces.append((i0, i1, o1))
    return vertices, faces


def _tube_component(
    p0: Sequence[float],
    p1: Sequence[float],
    radius: float,
    *,
    n_theta: int,
    n_axial: int,
) -> tuple[FloatArray, list]:
    """Closed cylinder (both ends capped) between two points."""
    a = np.asarray(p0, dtype=float)
    b = np.asarray(p1, dtype=float)
    axis = b - a
    length = float(np.linalg.norm(axis))
    if length <= 0.0:
        raise ValueError("p0 and p1 must differ")
    d = axis / length
    helper = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(d, helper)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(d, e1)  # e1 x e2 = -d, the convention the wall winding expects
    theta = 2.0 * math.pi * np.arange(n_theta, dtype=float) / n_theta
    circle = np.outer(np.cos(theta), e1) + np.outer(np.sin(theta), e2)
    stations = np.linspace(0.0, 1.0, n_axial + 1)
    vertices = np.array(
        [a + t * length * d + radius * c for t in stations for c in circle],
        dtype=float,
    )
    faces: list[tuple[int, int, int]] = []
    for i in range(n_axial):
        for j in range(n_theta):
            jn = (j + 1) % n_theta
            p = i * n_theta + j
            q = i * n_theta + jn
            r = (i + 1) * n_theta + jn
            s = (i + 1) * n_theta + j
            faces.append((p, r, q))
            faces.append((p, s, r))
    start_centre = vertices.shape[0]
    vertices = np.vstack((vertices, a[None, :], b[None, :]))
    for j in range(n_theta):
        jn = (j + 1) % n_theta
        faces.append((start_centre, j, jn))
        faces.append((start_centre + 1, n_axial * n_theta + jn, n_axial * n_theta + j))
    volume = _signed_volume(vertices, np.asarray(faces, dtype=np.int64))
    if volume < 0.0:
        faces = [(f[0], f[2], f[1]) for f in faces]
    return vertices, faces


def four_column_semisub_mesh(
    *,
    panels_per_column: int = 12,
    panels_vertical: int = 4,
    pontoon_theta: int = 10,
    pontoon_axial: int = 3,
    offset_angles: Sequence[float] = (0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0),
    with_pontoons: bool = True,
) -> WettedHullMesh:
    """OC4-class four-column semi-submersible wetted hull.

    Principal dimensions follow the DeepCwind/OC4 deep-draft semi-submersible
    (Robertson et al. 2014): a 6.5 m central column, three 12 m upper offset
    columns on a 28.87 m radius, 24 m base columns/heave plates of 6 m height,
    20 m draft and 1.6 m pontoons at z = -17 m.  The union is *not* meshed: every
    component is a separate closed-or-waterline surface, and the pontoon length
    covers only the free span between base columns, which is what
    :func:`semisub_analytic_displacement` counts.  Validated against its own
    closed form in the tests and against the published 13 473 m^3 displacement.
    """
    n_theta = int(panels_per_column)
    n_z = int(panels_vertical)
    if n_theta < 6 or n_z < 2:
        raise ValueError("panels_per_column >= 6 and panels_vertical >= 2 required")

    components: list[tuple[FloatArray, list]] = []
    components.append(
        _column_component(
            (0.0, 0.0),
            SEMISUB_MAIN_COLUMN_RADIUS,
            -SEMISUB_DRAFT,
            n_theta=n_theta,
            n_z=n_z,
        )
    )
    upper_height = SEMISUB_DRAFT - SEMISUB_BASE_COLUMN_HEIGHT
    offset_centres = []
    for angle in offset_angles:
        centre = (
            SEMISUB_OFFSET_RADIUS * math.cos(angle),
            SEMISUB_OFFSET_RADIUS * math.sin(angle),
        )
        offset_centres.append(centre)
        components.append(
            _column_component(
                centre,
                SEMISUB_OFFSET_COLUMN_RADIUS,
                -upper_height,
                n_theta=n_theta,
                n_z=n_z,
                bottom_cap=False,
            )
        )
        components.append(
            _annulus_component(
                centre,
                SEMISUB_OFFSET_COLUMN_RADIUS,
                SEMISUB_BASE_COLUMN_RADIUS,
                -upper_height,
                n_theta=n_theta,
                facing_up=True,
            )
        )
        components.append(
            _column_component(
                centre,
                SEMISUB_BASE_COLUMN_RADIUS,
                -SEMISUB_DRAFT,
                n_theta=n_theta,
                n_z=2,
                top=-upper_height,
            )
        )
    if with_pontoons:
        for i in range(len(offset_centres)):
            c0 = np.asarray(offset_centres[i], dtype=float)
            c1 = np.asarray(offset_centres[(i + 1) % len(offset_centres)], dtype=float)
            direction = c1 - c0
            chord = float(np.linalg.norm(direction))
            unit = direction / chord
            half = 0.5 * (chord - 2.0 * SEMISUB_BASE_COLUMN_RADIUS)
            mid = 0.5 * (c0 + c1)
            p0 = mid - half * unit
            p1 = mid + half * unit
            components.append(
                _tube_component(
                    (p0[0], p0[1], SEMISUB_PONTOON_Z),
                    (p1[0], p1[1], SEMISUB_PONTOON_Z),
                    SEMISUB_PONTOON_RADIUS,
                    n_theta=int(pontoon_theta),
                    n_axial=int(pontoon_axial),
                )
            )

    vertices, faces = _weld(components)
    return WettedHullMesh(vertices, faces)


def _weld(components: Sequence[tuple[FloatArray, list]], *, digits: int = 9) -> tuple[FloatArray, np.ndarray]:
    """Merge component vertex lists, welding coincident nodes.

    Welding turns the stacked components (column wall -> annular step -> base
    column) into one manifold wetted surface whose only open edges are the
    waterline loops, which is what :class:`~tensorbem.geometry3d.WettedHullMesh`
    validates.
    """
    registry: dict[tuple[float, float, float], int] = {}
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    for verts, tris in components:
        local = []
        for x, y, z in np.asarray(verts, dtype=float):
            key = (round(float(x), digits), round(float(y), digits), round(float(z), digits))
            idx = registry.get(key)
            if idx is None:
                idx = len(vertices)
                registry[key] = idx
                vertices.append((float(x), float(y), float(z)))
            local.append(idx)
        faces.extend((local[a], local[b], local[c]) for (a, b, c) in tris)
    return np.asarray(vertices, dtype=float), np.asarray(faces, dtype=np.int64)


def semisub_mesh_displacement(mesh: SurfaceMesh3D, *, panels: bool = True) -> float:
    """Displaced volume of a waterline-truncated hull (closing lid contributes 0)."""
    normals = np.asarray(mesh.normals, dtype=float)
    centroids = np.asarray(mesh.centroids, dtype=float)
    areas = np.asarray(mesh.areas, dtype=float)
    return float(np.sum(areas * np.einsum("ij,ij->i", centroids, normals)) / 3.0)
