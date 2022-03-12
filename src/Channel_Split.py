# RGB - (R - G - B) split
import cv2
from PIL import Image, ImageFilter
import numpy as np
# Normalize data (length of 1)
from sklearn.preprocessing import Normalizer


# Crimmins function
def crimmins(data) :
    new_image = data.copy()
    nrow = len(data)
    ncol = len(data[0])

    # Dark pixel adjustment

    # First Step
    # N-S
    for i in range(1, nrow) :
        for j in range(ncol) :
            if data[i - 1, j] >= (data[i, j] + 2) :
                new_image[i, j] += 1
    data = new_image
    # E-W
    for i in range(nrow) :
        for j in range(ncol - 1) :
            if data[i, j + 1] >= (data[i, j] + 2) :
                new_image[i, j] += 1
    data = new_image
    # NW-SE
    for i in range(1, nrow) :
        for j in range(1, ncol) :
            if data[i - 1, j - 1] >= (data[i, j] + 2) :
                new_image[i, j] += 1
    data = new_image
    # NE-SW
    for i in range(1, nrow) :
        for j in range(ncol - 1) :
            if data[i - 1, j + 1] >= (data[i, j] + 2) :
                new_image[i, j] += 1
    data = new_image
    # Second Step
    # N-S
    for i in range(1, nrow - 1) :
        for j in range(ncol) :
            if (data[i - 1, j] > data[i, j]) and (data[i, j] <= data[i + 1, j]) :
                new_image[i, j] += 1
    data = new_image
    # E-W
    for i in range(nrow) :
        for j in range(1, ncol - 1) :
            if (data[i, j + 1] > data[i, j]) and (data[i, j] <= data[i, j - 1]) :
                new_image[i, j] += 1
    data = new_image
    # NW-SE
    for i in range(1, nrow - 1) :
        for j in range(1, ncol - 1) :
            if (data[i - 1, j - 1] > data[i, j]) and (data[i, j] <= data[i + 1, j + 1]) :
                new_image[i, j] += 1
    data = new_image
    # NE-SW
    for i in range(1, nrow - 1) :
        for j in range(1, ncol - 1) :
            if (data[i - 1, j + 1] > data[i, j]) and (data[i, j] <= data[i + 1, j - 1]) :
                new_image[i, j] += 1
    data = new_image
    # Third Step
    # N-S
    for i in range(1, nrow - 1) :
        for j in range(ncol) :
            if (data[i + 1, j] > data[i, j]) and (data[i, j] <= data[i - 1, j]) :
                new_image[i, j] += 1
    data = new_image
    # E-W
    for i in range(nrow) :
        for j in range(1, ncol - 1) :
            if (data[i, j - 1] > data[i, j]) and (data[i, j] <= data[i, j + 1]) :
                new_image[i, j] += 1
    data = new_image
    # NW-SE
    for i in range(1, nrow - 1) :
        for j in range(1, ncol - 1) :
            if (data[i + 1, j + 1] > data[i, j]) and (data[i, j] <= data[i - 1, j - 1]) :
                new_image[i, j] += 1
    data = new_image
    # NE-SW
    for i in range(1, nrow - 1) :
        for j in range(1, ncol - 1) :
            if (data[i + 1, j - 1] > data[i, j]) and (data[i, j] <= data[i - 1, j + 1]) :
                new_image[i, j] += 1
    data = new_image
    # Fourth Step
    # N-S
    for i in range(nrow - 1) :
        for j in range(ncol) :
            if (data[i + 1, j] >= (data[i, j] + 2)) :
                new_image[i, j] += 1
    data = new_image
    # E-W
    for i in range(nrow) :
        for j in range(1, ncol) :
            if (data[i, j - 1] >= (data[i, j] + 2)) :
                new_image[i, j] += 1
    data = new_image
    # NW-SE
    for i in range(nrow - 1) :
        for j in range(ncol - 1) :
            if (data[i + 1, j + 1] >= (data[i, j] + 2)) :
                new_image[i, j] += 1
    data = new_image
    # NE-SW
    for i in range(nrow - 1) :
        for j in range(1, ncol) :
            if (data[i + 1, j - 1] >= (data[i, j] + 2)) :
                new_image[i, j] += 1
    data = new_image

    # Light pixel adjustment

    # First Step
    # N-S
    for i in range(1, nrow) :
        for j in range(ncol) :
            if (data[i - 1, j] <= (data[i, j] - 2)) :
                new_image[i, j] -= 1
    data = new_image
    # E-W
    for i in range(nrow) :
        for j in range(ncol - 1) :
            if (data[i, j + 1] <= (data[i, j] - 2)) :
                new_image[i, j] -= 1
    data = new_image
    # NW-SE
    for i in range(1, nrow) :
        for j in range(1, ncol) :
            if (data[i - 1, j - 1] <= (data[i, j] - 2)) :
                new_image[i, j] -= 1
    data = new_image
    # NE-SW
    for i in range(1, nrow) :
        for j in range(ncol - 1) :
            if (data[i - 1, j + 1] <= (data[i, j] - 2)) :
                new_image[i, j] -= 1
    data = new_image
    # Second Step
    # N-S
    for i in range(1, nrow - 1) :
        for j in range(ncol) :
            if (data[i - 1, j] < data[i, j]) and (data[i, j] >= data[i + 1, j]) :
                new_image[i, j] -= 1
    data = new_image
    # E-W
    for i in range(nrow) :
        for j in range(1, ncol - 1) :
            if (data[i, j + 1] < data[i, j]) and (data[i, j] >= data[i, j - 1]) :
                new_image[i, j] -= 1
    data = new_image
    # NW-SE
    for i in range(1, nrow - 1) :
        for j in range(1, ncol - 1) :
            if (data[i - 1, j - 1] < data[i, j]) and (data[i, j] >= data[i + 1, j + 1]) :
                new_image[i, j] -= 1
    data = new_image
    # NE-SW
    for i in range(1, nrow - 1) :
        for j in range(1, ncol - 1) :
            if (data[i - 1, j + 1] < data[i, j]) and (data[i, j] >= data[i + 1, j - 1]) :
                new_image[i, j] -= 1
    data = new_image
    # Third Step
    # N-S
    for i in range(1, nrow - 1) :
        for j in range(ncol) :
            if (data[i + 1, j] < data[i, j]) and (data[i, j] >= data[i - 1, j]) :
                new_image[i, j] -= 1
    data = new_image
    # E-W
    for i in range(nrow) :
        for j in range(1, ncol - 1) :
            if (data[i, j - 1] < data[i, j]) and (data[i, j] >= data[i, j + 1]) :
                new_image[i, j] -= 1
    data = new_image
    # NW-SE
    for i in range(1, nrow - 1) :
        for j in range(1, ncol - 1) :
            if (data[i + 1, j + 1] < data[i, j]) and (data[i, j] >= data[i - 1, j - 1]) :
                new_image[i, j] -= 1
    data = new_image
    # NE-SW
    for i in range(1, nrow - 1) :
        for j in range(1, ncol - 1) :
            if (data[i + 1, j - 1] < data[i, j]) and (data[i, j] >= data[i - 1, j + 1]) :
                new_image[i, j] -= 1
    data = new_image
    # Fourth Step
    # N-S
    for i in range(nrow - 1) :
        for j in range(ncol) :
            if (data[i + 1, j] <= (data[i, j] - 2)) :
                new_image[i, j] -= 1
    data = new_image
    # E-W
    for i in range(nrow) :
        for j in range(1, ncol) :
            if (data[i, j - 1] <= (data[i, j] - 2)) :
                new_image[i, j] -= 1
    data = new_image
    # NW-SE
    for i in range(nrow - 1) :
        for j in range(ncol - 1) :
            if (data[i + 1, j + 1] <= (data[i, j] - 2)) :
                new_image[i, j] -= 1
    data = new_image
    # NE-SW
    for i in range(nrow - 1) :
        for j in range(1, ncol) :
            if (data[i + 1, j - 1] <= (data[i, j] - 2)) :
                new_image[i, j] -= 1
    data = new_image
    return new_image.copy()


# Split channels
def image_split(image):
    # Split channels
    b, g, r = cv2.split(image)
    # Process r
    r = cv2.cvtColor(r, cv2.COLOR_BGR2RGB)
    r = cv2.cvtColor(r, cv2.COLOR_RGB2GRAY)
    r = Image.fromarray(r.astype('uint8'))
    # Process g
    g = cv2.cvtColor(g, cv2.COLOR_BGR2RGB)
    g = cv2.cvtColor(g, cv2.COLOR_RGB2GRAY)
    g = Image.fromarray(g.astype('uint8'))
    # Process b
    b = cv2.cvtColor(b, cv2.COLOR_BGR2RGB)
    b = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
    b = Image.fromarray(b.astype('uint8'))
    return [r, g, b]

# Filter
def unsharpFilter_r_g_b(b, g, r):
    r = r.filter(ImageFilter.UnsharpMask(radius=2, percent=150))
    g = g.filter(ImageFilter.UnsharpMask(radius=2, percent=150))
    b = b.filter(ImageFilter.UnsharpMask(radius=2, percent=150))
    return [r, g, b]


# This function splits channel R, G, B and returns R, G, B, and RGB
# The UnsharpMask is applied over R, G, and B and Crimmins on RGB
def rgb_split_case_1(path):
    image = cv2.imread(path)
    image_ = cv2.imread(path)
    rgb = cv2.cvtColor(image_, cv2.COLOR_BGR2RGB)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    # Crimmins on RGB
    rgb = crimmins(rgb)
    rgb = Image.fromarray(rgb.astype('uint8'))
    # Split channels
    b, g, r = image_split(image)
    # Filter r, g, b
    # Apply unsharpMask
    r, g, b = unsharpFilter_r_g_b(b, g, r)
    return [r, g, b, rgb]


# This function splits channel R, G, B and returns R, G, B, and RGB
# The UnsharpMask is applied over R, G, and B and Crimmins on RGB not applied
def rgb_split_case_2(path):
    image = cv2.imread(path)
    image_ = cv2.imread(path)
    rgb = cv2.cvtColor(image_, cv2.COLOR_BGR2RGB)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    rgb = Image.fromarray(rgb.astype('uint8'))
    # Split channels
    b, g, r = image_split(image)
    # Filter r, g, b
    # Apply unsharpMask
    r, g, b = unsharpFilter_r_g_b(b, g, r)
    return [r, g, b, rgb]


# This function splits channel R, G, B and returns R, G, B, and RGB
# The UnsharpMask is not applied over R, G, and B and Crimmins is applied on RGB
def rgb_split_case_3(path):
    image = cv2.imread(path)
    image_ = cv2.imread(path)
    rgb = cv2.cvtColor(image_, cv2.COLOR_BGR2RGB)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    # Crimmins on RGB
    rgb = crimmins(rgb)
    rgb = Image.fromarray(rgb.astype('uint8'))
    # Split channels
    b, g, r = image_split(image)
    return [r, g, b, rgb]


# This function splits channel R, G, B and returns R, G, B, and RGB
# The UnsharpMask is not applied over R, G, and B and Crimmins on RGB is not applied
def rgb_split_case_4(path):
    image = cv2.imread(path)
    image_ = cv2.imread(path)
    rgb = cv2.cvtColor(image_, cv2.COLOR_BGR2RGB)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    rgb = Image.fromarray(rgb.astype('uint8'))
    # Split channels
    b, g, r = image_split(image)
    return [r, g, b, rgb]


'''
Function concat_features_extract_measures 
Parameters: receives a descriptorX and array of arrays (channels R, G, B, and RGB)
Return: array of concatenated features extracted by means of descriptorX from R, G, B, and RGB  
'''


def concat_features_extract_measures(func, list) :
    arrays_of_features = [(func(np.array(x))) for x in list]
    concatenated_features = [item for sublist in arrays_of_features for item in sublist]
    return concatenated_features


def concat_features_extract_measures_normalized(func, list) :
    arrays_of_features = [(func(np.array(x))) for x in list]
    concatenated_features = [item for sublist in arrays_of_features for item in sublist]
    features = [float(i) / max(concatenated_features) for i in
                concatenated_features]  # Normalize features by max values
    return features
