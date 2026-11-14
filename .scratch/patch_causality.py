from __future__ import annotations

import pathlib

PATH = pathlib.Path("src/tensorbem/causality.py")
text = PATH.read_text(encoding="utf-8")

EXPORT_OLD = '    "kk_added_mass_closure",\n    "profile_added_mass_inf",\n]'
EXPORT_NEW = (
    '    "kk_added_mass_closure",\n'
    '    "profile_added_mass_inf",\n'
    '    "HighBandBudget",\n'
    '    "high_band_truncation_budget",\n'
    '    "local_tail_exponent",\n'
    ']'
)
assert EXPORT_OLD in text
text = text.replace(EXPORT_OLD, EXPORT_NEW)

ANCHOR = '@dataclass(frozen=True)\nclass KKRClosure:'
assert ANCHOR in text

BLOCK = '''def local_tail_exponent(
    omegas: Sequence[float] | FloatArray,
    damping: Sequence[float] | FloatArray,
    *,
    window: int = 3,
) -> tuple[float, float]:
    """Log-log slope of the last ``window`` damping samples: ``(exponent, coefficient)``.

    Returns ``(-inf, 0.0)`` when the window cannot support a fit (non-positive
    damping, fewer than three usable samples).  No validity verdict is attached
    here on purpose: :func:`high_band_truncation_budget` turns the *measured*
    slope into a bound, and :func:`fit_damping_power_tail` is the guarded
    variant that decides whether the slope may be called an algebraic tail.
    """
    grid, data = _as_sweep(omegas, damping, "damping")
    count = max(3, int(window))
    if grid.size < 3:
        return (-math.inf, 0.0)
    count = min(count, grid.size)
    tail_grid = grid[-count:]
    tail_data = data[-count:]
    if np.any(tail_data <= 0.0) or not np.all(np.isfinite(tail_data)):
        return (-math.inf, 0.0)
    slope, intercept = np.polyfit(np.log(tail_grid), np.log(tail_data), 1)
    return (float(-slope), float(math.exp(intercept)))


@dataclass(frozen=True)
class HighBandBudget:
    """Upper bound on the Kramers-Kronig error left by the un-solved band.

    A band-limited sweep tabulates ``B`` on ``[s_0, s_N]`` only, so the closure
    of eq. (2) is missing

        ``delta A(omega) = (2/pi) integral_{s_N}^inf B(s)/(s^2-omega^2) ds``,

    which is **positive** for every ``omega < s_N`` and grows without bound as
    ``omega -> s_N`` (``integral ds/(s^2-omega^2)`` diverges logarithmically at
    ``s = omega``, so a sample sitting on the band top is not decidable from
    band-limited data at any tolerance).  Any KK residual measured at a sample
    frequency close to the band top therefore partly measures that omitted
    integral, not the causality of the data.

    This dataclass turns the omission into a number: it continues ``B`` above
    ``s_N`` with the *measured* local log-log slope ``p`` anchored at the last
    tabulated sample and integrates

        ``budget(omega) = (2/pi) PV integral_{s_N}^inf C s^-p/(s^2-omega^2) ds``

    in closed form (:func:`power_tail_principal_value`).  ``budget`` is then a
    conservative estimate of the truncation error, and
    ``|residual| <= tolerance*scale`` can only be read as a causality test where
    ``budget`` is small compared with the tolerance.  Samples where it is not are
    reported as undecidable rather than as a violation.

    If the measured exponent is not ``> 2`` the tail integral of eq. (2) does not
    converge at all; the budget is then ``inf`` (``admissible = False``) and the
    upper band edge is formally undecidable.
    """

    sample_omegas: FloatArray
    budget: FloatArray
    exponent: float
    coefficient: float
    start: float
    admissible: bool
    reason: str

    def relative(self, scale: float) -> FloatArray:
        """Budget divided by the added-mass scale used for the residual."""
        floor = max(float(scale), 1e-300)
        return np.asarray(self.budget, dtype=float) / floor

    @property
    def max_budget(self) -> float:
        finite = self.budget[np.isfinite(self.budget)]
        return float(np.max(finite)) if finite.size else math.inf


def high_band_truncation_budget(
    omegas: Sequence[float] | FloatArray,
    damping: Sequence[float] | FloatArray,
    sample_omegas: Sequence[float] | FloatArray | None = None,
    *,
    window: int = 3,
    min_exponent: float = 2.05,
) -> HighBandBudget:
    """Closed-form bound on the KK integral omitted above the tabulated band."""
    grid, data = _as_sweep(omegas, damping, "damping")
    samples = (
        _default_samples(grid)
        if sample_omegas is None
        else np.asarray(list(sample_omegas), dtype=float).reshape(-1)
    )
    exponent, _ = local_tail_exponent(grid, data, window=window)
    start = float(grid[-1])
    if not (math.isfinite(exponent) and exponent > float(min_exponent)):
        return HighBandBudget(
            sample_omegas=samples,
            budget=np.full(samples.shape, math.inf, dtype=float),
            exponent=exponent,
            coefficient=0.0,
            start=start,
            admissible=False,
            reason=(
                f"measured tail exponent {exponent:.3g} is not > {float(min_exponent):g}: "
                "the damping has not entered an algebraically decaying band, so the "
                "omitted integral above the band top is unbounded"
            ),
        )
    # ``C`` is rescaled so that the continuation passes exactly through the last
    # tabulated sample: no free parameter is introduced.
    anchored = float(data[-1]) * start**exponent
    budget = np.empty(samples.shape, dtype=float)
    for index, sample in enumerate(samples):
        value = float(sample)
        if not value < start:
            budget[index] = math.inf
            continue
        budget[index] = 2.0 / math.pi * power_tail_principal_value(
            value, anchored, exponent, start
        )
    return HighBandBudget(
        sample_omegas=samples,
        budget=budget,
        exponent=exponent,
        coefficient=anchored,
        start=start,
        admissible=True,
        reason=f"continuation B = C s^-{exponent:.3g} anchored at s_max = {start:g}",
    )


'''

text = text.replace(ANCHOR, BLOCK + ANCHOR, 1)
PATH.write_text(text, encoding="utf-8")
print("patched", PATH)