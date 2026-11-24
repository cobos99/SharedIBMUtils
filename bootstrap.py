import numpy.typing as npt
import numpy as np
import scipy as sp

def bootstrap_bits_mean(bits: npt.NDArray[np.bool_], confidence: float = 0.95, n_resamples: int = 1000, return_std: bool = False) -> npt.NDArray | tuple[npt.NDArray, npt.NDArray]:
    # BCa bootstrap of bootstrap_sample_dict_mean, vectorized over observables and any leading axes. bits has shape
    # (..., shots, observables), with True for a -1 outcome and False for a +1 outcome. Returns the confidence intervals with
    # shape (..., observables, 2) and, with return_std, the bootstrap stds with shape (..., observables)
    if not 0 < confidence < 1:
        raise ValueError("Confidence must be strictly between 0 and 1")
    if n_resamples < 1:
        raise ValueError("Number of resamples must be at least 1")
    bits = np.asarray(bits)
    if bits.dtype != np.bool_:
        raise ValueError("bits must be a boolean array")
    if bits.ndim < 2:
        raise ValueError("bits must be at least a 2D array")
    n_shots = bits.shape[-2]
    if n_shots < 1:
        raise ValueError("bits must contain at least one shot")

    # einsum sums along the shots axis faster than np.count_nonzero
    n_low = np.einsum("...ij->...j", bits.view(np.uint8), dtype=np.intp)
    n_high = n_shots - n_low
    original_expectation_values = (n_high - n_low) / n_shots

    # Bootstrap resampling. Every resample draws n_shots shots from the two-point empirical distribution, so its number of
    # +1 outcomes is binomial. Everything below works on these counts, and resample k has mean (2k - n_shots)/n_shots. The
    # counts are stored in the smallest integer type that holds n_shots, which makes the sort several times faster
    resampled_n_high = np.random.binomial(n_shots, n_high[..., None] / n_shots, size=(*n_high.shape, n_resamples)).astype(np.min_scalar_type(n_shots))

    # Correct for bias. The mean is increasing in the count, so comparing counts is comparing means
    n_high_rep = n_high[..., None]
    percentile = (np.count_nonzero(resampled_n_high < n_high_rep, axis=-1) + np.count_nonzero(resampled_n_high <= n_high_rep, axis=-1)) / 2 / n_resamples
    degenerate = (percentile == 0) | (percentile == 1)
    z0_hat = sp.special.ndtri(np.where(degenerate, 0.5, percentile))

    # Correct for acceleration. For a two-valued statistic the jackknife influence of a shot is its value minus the mean,
    # which reduces a_hat to (n_low - n_high)/(6 sqrt(n_shots n_low n_high)), and to 0 when either count is 0
    a_hat = np.divide(n_low - n_high, 6*np.sqrt(float(n_shots)*n_low*n_high), out=np.zeros(n_low.shape), where=(n_low > 0) & (n_high > 0))

    z_alpha = sp.special.ndtri((1 - confidence)/2)
    num1 = z0_hat + z_alpha
    alpha1 = sp.special.ndtr(z0_hat + num1 / (1 - a_hat*num1))
    num2 = z0_hat - z_alpha
    alpha2 = sp.special.ndtr(z0_hat + num2 / (1 - a_hat*num2))

    # Percentiles with linear interpolation, as np.percentile, but with a different percentile per observable
    sorted_n_high = np.sort(resampled_n_high, axis=-1)
    virtual_indices = (n_resamples - 1) * np.stack([alpha1, alpha2], axis=-1)
    lower_indices = np.minimum(np.floor(virtual_indices).astype(np.intp), n_resamples - 1)
    upper_indices = np.minimum(lower_indices + 1, n_resamples - 1)
    lower_n_high = np.take_along_axis(sorted_n_high, lower_indices, axis=-1)
    upper_n_high = np.take_along_axis(sorted_n_high, upper_indices, axis=-1)
    ci_n_high = lower_n_high + (virtual_indices - lower_indices) * (upper_n_high - lower_n_high)
    confidence_intervals = np.where(degenerate[..., None], original_expectation_values[..., None], (2*ci_n_high - n_shots) / n_shots)

    if return_std:
        return confidence_intervals, np.std(resampled_n_high, axis=-1) * (2 / n_shots)
    return confidence_intervals
