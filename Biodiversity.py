# Validations
import numpy as np
from scipy.optimize import minimize_scalar

'''
Compute the image histogram, and return a vector with the number of occurrences of each gray level
'''


def histogram(data):
    row, column = data.shape
    hist = np.zeros((256))

    for i in range(row):
        for j in range(column):
            pos = data[i, j]
            hist[pos] = hist[pos] + 1
    return hist


'''
Counts the number of species existing on the image date, 
and receives the histogram vector as a parameter
'''


def count_species(histogram):
    species = 0
    for i in range(len(histogram)):
        if (histogram[i] > 0):
            species += 1
    return species


def _validate_counts_vector(counts, suppress_cast=False):
    """
    Validate and convert input to an acceptable counts vector type.
    Note: may not always return a copy of `counts`!
    """
    counts = np.asarray(counts)
    if not suppress_cast:
        counts = counts.astype(int, casting='safe', copy=False)
    if counts.ndim != 1:
        raise ValueError("Only 1-D vectors are supported.")
    elif (counts < 0).any():
        raise ValueError("Counts vector cannot contain negative values.")
    return counts


def observed_gray_levels(counts):
    """
	Compute the number of distinct gray levels.
    Parameters: 1-D array_like, int Vector of counts.
    Returns: distinct gray levels count.
    """
    counts = _validate_counts_vector(counts)
    return (counts != 0).sum()


# Margalef diversity index
def dMg(counts):
    """
    Parameters: 1-D array_like, int Vector of counts.
    Returns: double
    """
    counts = _validate_counts_vector(counts)
    return (observed_gray_levels(counts) - 1) / np.log(counts.sum())


# Menhinick diversity index
def dMn(counts):
    """
    Parameters: 1-D array_like, int Vector of counts.
    Returns: double
    """
    counts = _validate_counts_vector(counts)
    return observed_gray_levels(counts) / np.sqrt(counts.sum())


# Berger Parker dominance
def dBP(counts):
    """
    Parameters: 1-D array_like, int Vector of counts.
    Returns: double
    """
    counts = _validate_counts_vector(counts)
    return counts.max() / counts.sum()


# Fisher's alpha, a metric of diversity
def dF(counts):
    """
    Parameters : 1-D array_like, int Vector of counts.
    Returns: double
    """
    counts = _validate_counts_vector(counts)
    n = counts.sum()
    s = observed_gray_levels(counts)

    def f(alpha):
        return (alpha * np.log(1 + (n / alpha)) - s) ** 2

    orig_settings = np.seterr(divide='ignore', invalid='ignore')
    try:
        alpha = minimize_scalar(f).x
    finally:
        np.seterr(**orig_settings)
    if f(alpha) > 1.0:
        raise RuntimeError("Optimizer failed to converge (error > 1.0), so "
                           "could not compute Fisher's alpha.")
    return alpha


# Kempton-Taylor Q index of alpha diversity
def dKT(counts, lower_quantile=0.25, upper_quantile=0.75):
    """
    Parameters: 1-D array_like, int Vector of counts.
		lower_quantile : float, optional
        Lower bound of the interquantile range. Defaults to lower quartile.
		upper_quantile : float, optional
        Upper bound of the interquantile range. Defaults to upper quartile.
    Returns: double
    """
    counts = _validate_counts_vector(counts)
    n = len(counts)
    lower = int(np.ceil(n * lower_quantile))
    upper = int(n * upper_quantile)
    sorted_counts = np.sort(counts)
    return (upper - lower) / np.log(sorted_counts[upper] /
                                    sorted_counts[lower])


# McIntosh's evenness measure
def eM(counts):
    """
    Parameters: 1-D array_like, int Vector of counts.
    Returns: double
    """
    counts = _validate_counts_vector(counts)
    numerator = np.sqrt((counts * counts).sum())
    n = counts.sum()
    s = observed_gray_levels(counts)
    denominator = np.sqrt((n - s + 1) ** 2 + s - 1)
    return numerator / denominator


# Shannon-Wiener diversity index
def dSW(counts, base=2):
    """
    Parameters: 1-D array_like, int Vector of counts.
		base : scalar, optional
        Logarithm base to use in the calculations.
    Returns: double
    """
    counts = _validate_counts_vector(counts)
    freqs = counts / counts.sum()
    nonzero_freqs = freqs[freqs.nonzero()]
    return -(nonzero_freqs * np.log(nonzero_freqs)).sum() / np.log(base)


# Biodiversity features
def biodiversity(data):
    data = np.array(data)
    dBP_index = np.float32(dBP(data.flatten()))
    dF_index = np.float32(dF(data.flatten()))
    dKT_index = np.float32(dKT(data.flatten()))
    dMg_index = np.float32(dMg(data.flatten()))
    eM_ndex = np.float32(eM(data.flatten()))
    dMn_index = np.float32(dMn((data.flatten())))
    dSW_index = np.float32(dSW((data.flatten())))
    return [dBP_index, dF_index, dKT_index, dMg_index, eM_ndex, dMn_index, dSW_index]
