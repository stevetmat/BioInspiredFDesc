import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
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

'''
Excludes species that have no individuals
'''
def remove_species(histogram):
    new_histogram = []

    for i in range(len(histogram)):
        if (histogram[i] > 0):
            new_histogram.append(histogram[i])
    return new_histogram

def validate_species_vector(counts, suppress_cast=False):
    """
    Validate and convert input to an acceptable counts vector type.
    """
    counts = np.asarray(counts)
    if not suppress_cast:
        counts = counts.astype(int, casting='safe', copy=False)
    if counts.ndim != 1:
        raise ValueError("Only 1-D vectors are supported.")
    elif (counts < 0).any():
        raise ValueError("Counts vector cannot contain negative values.")
    return counts

def _validate_counts_vector(counts, suppress_cast=False):
    """Validate and convert input to an acceptable counts vector type."""
    counts = np.asarray(counts)
    if not np.all(np.isreal(counts)):
        raise ValueError("Counts vector must contain real-valued entries.")
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
    counts = validate_species_vector(counts)
    return (counts != 0).sum()
def observed_species(counts):
    counts = _validate_counts_vector(counts)
    return (counts != 0).sum()
'''
Computes distance and taxonomic diversity index
'''
def taxo_diversity(data):
    hist = histogram(data)
    new_histogram = remove_species(hist)
    # accumulates the sum between the distances and abundance of species i and j
    summation = 0
    # receives the total of species
    total_species = count_species(new_histogram)
    for i in range(len(new_histogram)):
        for j in range(1, len(new_histogram)):
            if i == 0 and j == 1:
                distance = j - i + 1
            else:
                distance = j - i + 2

            product = distance * new_histogram[i] * new_histogram[j]
            summation += product
    index = summation / ((total_species * (total_species - 1)) / 2)
    return index

'''
Computes the distance and the taxonomic distinctness index
'''
def taxo_distinctiveness(data):
    hist = histogram(data)
    new_histogram = remove_species(hist)
    # accumulates the sum between the distances and abundance of species i and j
    summation1 = 0
    # accumulates the sum between the abundance of species i and j
    summation2 = 0
    # receives the total of species
    total_species = count_species(new_histogram)
    for i in range(len(new_histogram)):
        for j in range(1, len(new_histogram)):
            if i == 0 and j == 1:
                distance = j - i + 1
            else:
                distance = j - i + 2
            product = distance * new_histogram[i] * new_histogram[j]
            summation1 += product
            summation2 += new_histogram[i] * new_histogram[j]
    index = summation1 / summation2
    return index

'''
Computes the intensive quadratic entropy
'''
def eIQ(data):
    hist = histogram(data)
    new_histogram = remove_species(hist)
    # accumulates the sum between distances i, j
    summation = 0
    # receives the total of species
    total_species = count_species(new_histogram)
    for i in range(len(new_histogram)):
        for j in range(1, len(new_histogram)):
            if i == 0 and j == 1:
                distance = j - i + 1
            else:
                distance = j - i + 2
            summation += distance
    index = summation / (total_species * total_species)
    return index

'''
Computes extensive quadratic entropy
'''
def eEQ(data):
    hist = histogram(data)
    new_histogram = remove_species(hist)
    # accumulates the sum between distances i, j
    summation = 0
    # receives the total of species
    total_species = count_species(new_histogram)
    for i in range(len(new_histogram)):
        for j in range(1, len(new_histogram)):
            if i == 0 and j == 1:
                distance = j - i + 1
            else:
                distance = j - i + 2
            summation += distance
    return summation

'''
Computes the average taxonomic distinctness index
'''
def dNN(data):
    hist = histogram(data)
    new_histogram = remove_species(hist)
    # accumulates the sum between distances i, j
    summation = 0
    # receives the total of species
    total_species = count_species(new_histogram)

    for i in range(len(new_histogram)):
        for j in range(1, len(new_histogram)):
            if i == 0 and j == 1:
                distance = j - i + 1
            else:
                distance = j - i + 2
            summation += distance
    index = summation / ((total_species * (total_species - 1)) / 2)
    return index

'''
Computes the total taxonomic distinctness index
'''
def dTT(data):
    hist = histogram(data)
    new_histogram = remove_species(hist)

    # acumula a soma entre as distances de speciess i e j
    summation = 0

    # recebe o total de species
    total_species = count_species(new_histogram)

    for i in range(len(new_histogram)):
        for j in range(1, len(new_histogram)):
            if i == 0 and j == 1:
                distance = j - i + 1
            else:
                distance = j - i + 2
            summation += distance
    index = summation / (total_species - 1)
    return index

# Margalef diversity index
def dMg(species):
    """
    Parameters: 1-D array_like, int Vector of species.
    Returns: double
    """
    species = validate_species_vector(species)
    return (observed_gray_levels(species) - 1) / np.log(species.sum())


# Menhinick diversity index
def dMn(species):
    """
    Parameters: 1-D array_like, int Vector of species.
    Returns: double
    """
    species = validate_species_vector(species)
    return observed_gray_levels(species) / np.sqrt(species.sum())


# Berger Parker dominance
def dBP(species):
    """
    Parameters: 1-D array_like, int Vector of species.
    Returns: double
    """
    species = validate_species_vector(species)
    return species.max() / species.sum()


# Fisher's alpha, a metric of diversity
def dF(species):
    """
    Parameters : 1-D array_like, int Vector of species.
    Returns: double
    """
    species = validate_species_vector(species)
    n = species.sum()
    s = observed_gray_levels(species)

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
def dKT(species, lower_quantile=0.25, upper_quantile=0.75):
    """
    Parameters: 1-D array_like, int Vector of species.
		lower_quantile : float, optional
        Lower bound of the interquantile range. Defaults to lower quartile.
		upper_quantile : float, optional
        Upper bound of the interquantile range. Defaults to upper quartile.
    Returns: double
    """
    species = validate_species_vector(species)
    n = len(species)
    lower = int(np.ceil(n * lower_quantile))
    upper = int(n * upper_quantile)
    sorted_species = np.sort(species)
    return (upper - lower) / np.log(sorted_species[upper] /
                                    sorted_species[lower])


# McIntosh's evenness measure
def eM(species):
    r"""Calculate McIntosh's evenness measure E.
    Parameters
    ----------
    species : 1-D array_like, int
        Vector of counts.
    """
    species = _validate_counts_vector(species)
    numerator = np.sqrt((species * species).sum())
    n = species.sum()
    s = observed_species(species)
    denominator = np.sqrt((n - s + 1) ** 2 + s - 1)
    return numerator / denominator


# Shannon-Wiener diversity index
def dSW(species, base=2):
    """
    Parameters: 1-D array_like, int Vector of species.
		base : scalar, optional
        Logarithm base to use in the calculations.
    Returns: double
    """
    species = validate_species_vector(species)
    freqs = species / species.sum()
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

def taxonomy(data):
    diversity = np.float32(taxo_diversity(data))
    distinctness = np.float32(taxo_distinctiveness(data))
    eIQ_index = np.float32(eIQ(data))
    eEQ_index = np.float32(eEQ(data))
    dNN_index = np.float32(dNN(data))
    dTT_index = np.float32(dTT(data))
    return [diversity, distinctness, eIQ_index, eEQ_index, dNN_index, dTT_index]
