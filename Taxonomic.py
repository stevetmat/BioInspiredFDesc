#
import numpy as np

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

def taxonomy(data):
    diversity = np.float32(taxo_diversity(data))
    distinctness = np.float32(taxo_distinctiveness(data))
    eIQ_index = np.float32(eIQ(data))
    eEQ_index = np.float32(eEQ(data))
    dNN_index = np.float32(dNN(data))
    dTT_index = np.float32(dTT(data))
    return [diversity, distinctness, eIQ_index, eEQ_index, dNN_index, dTT_index]
