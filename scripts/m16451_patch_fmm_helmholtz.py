"""Replace the Helmholtz section of tensorbem/fmm.py with the complex-kernel version."""
from __future__ import annotations

import pathlib

P = pathlib.Path("src/tensorbem/fmm.py")
SRC = P.read_text()

START = "# ---------------------------------------------------------------------------\n# Helmholtz kernel support (fail-closed)\n"
END = "def fmm_degenerate_matvec("
i = SRC.index(START)
j = SRC.index(END)

NEW = '''# ---------------------------------------------------------------------------
# Helmholtz kernel support (fail-closed, full complex kernel)
# ---------------------------------------------------------------------------

#: Mode tags returned by :func:`fmm_helmholtz_matvec`.  Every tag is a truthful
#: statement about *what was summed*, never about speed alone (m16451).
HELMHOLTZ_MODES = ("complex_direct", "laplace_approx", "not_supported")


def helmholtz_supported(ka: float, ka_max: float = 10.0) -> bool:
    """Whether the calibrated large-argument band is open for ``ka``.

    Outside ``ka <= ka_max`` the operator still returns the *exact* kernel sum,
    but tags the result ``"not_supported"`` so that an unaccelerated O(NM)
    product is never mistaken for an accelerated one.
    """
    return ka <= ka_max


def source_extent(sources: FloatArray) -> float:
    """Largest axis extent of the source cloud (``0.0`` for < 2 points)."""
    s = np.asarray(sources, dtype=float)
    if s.ndim != 2 or s.shape[1] != 3:
        raise ValueError("sources must have shape (M, 3)")
    if s.shape[0] < 2:
        return 0.0
    return float(np.max(np.ptp(s, axis=0)))


def _validate_helmholtz_inputs(
    targets: FloatArray, sources: FloatArray, charges: FloatArray
) -> tuple[FloatArray, FloatArray, NDArray[np.complex128]]:
    """Validate shapes and *preserve* complex charges (no silent cast)."""
    targets = np.asarray(targets)
    sources = np.asarray(sources)
    charges = np.asarray(charges)
    if targets.ndim != 2 or targets.shape[1] != 3:
        raise ValueError("targets must have shape (N, 3)")
    if sources.ndim != 2 or sources.shape[1] != 3:
        raise ValueError("sources must have shape (M, 3)")
    if charges.shape != (sources.shape[0],):
        raise ValueError("charges must have shape (M,)")
    if not np.issubdtype(targets.dtype, np.number) or not np.issubdtype(
        sources.dtype, np.number
    ):
        raise ValueError("targets and sources must be numeric")
    # complex charges carry the second (quadrature) degree of freedom of a
    # time-harmonic source; casting them to float silently deleted it (m16451).
    q = np.ascontiguousarray(charges, dtype=np.complex128)
    return (
        np.ascontiguousarray(targets, dtype=float),
        np.ascontiguousarray(sources, dtype=float),
        q,
    )


def _helmholtz_direct(
    targets: FloatArray,
    sources: FloatArray,
    charges: FloatArray,
    k: float,
    max_pairs: int = 2_000_000,
) -> NDArray[np.complex128]:
    """Exact complex single-layer Helmholtz matvec, blocked over targets.

    Returns ``sum_j q_j exp(i k r_ij) / (4 pi r_ij)`` keeping **both**
    quadratures.  Before m16451 only the real part was returned, i.e. the
    kernel ``cos(k r)/r`` -- a standing-wave mixture of outgoing and incoming
    waves whose distance to the outgoing solution is 28 %--99 % (evidence:
    ``docs/fmm_helmholtz_kernel/evidence.json``).
    """
    targets, sources, q = _validate_helmholtz_inputs(targets, sources, charges)
    n_t = targets.shape[0]
    n_s = sources.shape[0]
    result = np.zeros(n_t, dtype=np.complex128)
    if n_t == 0 or n_s == 0:
        return result

    step = int(max(1, max_pairs // n_s))
    for start in range(0, n_t, step):
        block = targets[start:start + step]
        diff = block[:, None, :] - sources[None, :, :]
        r = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
        r = np.maximum(r, 1e-15)
        vals = q[None, :] * np.exp(1j * k * r) / (4.0 * math.pi * r)
        result[start:start + len(block)] = np.sum(vals, axis=1)
    return result


def fmm_helmholtz_matvec(
    targets: FloatArray,
    sources: FloatArray,
    charges: FloatArray,
    k: float,
    p: int = 4,
    ka_max: float = 10.0,
    low_ka_asymptotic: bool = False,
    max_pairs: int = 2_000_000,
) -> tuple[NDArray[np.complex128], Literal["complex_direct", "laplace_approx", "not_supported"]]:
    """Complex single-layer Helmholtz matvec ``sum_j q_j e^{ikr}/(4 pi r)``.

    Returns ``(potential, mode)`` where ``potential`` is ``complex128`` -- the
    full field, both quadratures, so that ``Re(Phi e^{-i w t})`` is available to
    the caller -- and ``mode`` states exactly what was computed:

    ``"complex_direct"``
        Exact kernel sum for ``ka <= ka_max`` (no acceleration).
    ``"not_supported"``
        Exact kernel sum for ``ka > ka_max``: the accelerated band is closed,
        the value is still exact.  Fail-closed in the *cost* sense only.
    ``"laplace_approx"``
        Only when the caller explicitly asks for the small-``ka`` asymptotic
        (``low_ka_asymptotic=True`` and ``ka < 0.1``): the Laplace kernel sum is
        returned with a zero imaginary part.  This is an approximation with a
        measured error floor of tens of percent near ``ka = 0.1`` (see the
        evidence file), so it is **not** on the default path (m16451: it used to
        be the default and was labelled ``"fmm"``).

    Complex ``charges`` are accepted and preserved; real ``charges`` give the
    same ``exp(i k r)`` kernel with a real source strength.
    """
    tgt, src, q = _validate_helmholtz_inputs(targets, sources, charges)
    ka = float(k) * source_extent(src)

    if low_ka_asymptotic and ka < 0.1:
        if np.iscomplexobj(np.asarray(charges)) and np.any(np.asarray(charges).imag != 0.0):
            raise ValueError(
                "low_ka_asymptotic=True drops the imaginary quadrature and is "
                "undefined for genuinely complex charges; use the exact path"
            )
        pot = np.asarray(
            fmm_laplace_matvec(tgt, src, np.asarray(q.real, dtype=float), p=p),
            dtype=np.complex128,
        )
        return pot, "laplace_approx"

    if not helmholtz_supported(ka, ka_max):
        return _helmholtz_direct(tgt, src, q, k, max_pairs=max_pairs), "not_supported"

    return _helmholtz_direct(tgt, src, q, k, max_pairs=max_pairs), "complex_direct"


'''

P.write_text(SRC[:i] + NEW + SRC[j:])

# ---- module docstring: fail-closed paragraph -------------------------------
OLD_DOC = '''Fail-closed contract
--------------------
Helmholtz kernels (``e^{ikr}/r``) with ``ka > ka_max`` return
``NOT_SUPPORTED`` and fall back to direct summation.  The Laplace kernel
is always supported.
'''
NEW_DOC = '''Fail-closed contract
--------------------
Helmholtz kernels (``e^{ikr}/r``) are summed **exactly and complexly** by
:func:`fmm_helmholtz_matvec`: ``ka > ka_max`` closes the accelerated band and
tags the result ``"not_supported"``, but the value stays exact.  There is no
Helmholtz FMM in this module (see GAP-BEM-FMM-HELMHOLTZ-KERNEL).  The Laplace
kernel is always supported.
'''
text = P.read_text()
if OLD_DOC not in text:
    raise SystemExit("module docstring anchor not found")
P.write_text(text.replace(OLD_DOC, NEW_DOC, 1))
print("patched", P)