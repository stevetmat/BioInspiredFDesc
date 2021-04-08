import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np
from BioInspiredFDesc.src.BiT import taxonomy as taxo
from BioInspiredFDesc.src.BiT import biodiversity as bio
from BioInspiredFDesc.src.Channel_Split import concat_features_extract_measures as extract
from BioInspiredFDesc.src.Channel_Split import concat_features_extract_measures_normalized as normalized_extract
from BioInspiredFDesc.src.Channel_Split import rgb_split_case_1 as split1
from BioInspiredFDesc.src.Channel_Split import rgb_split_case_2 as split2
from BioInspiredFDesc.src.Channel_Split import rgb_split_case_3 as split3
from BioInspiredFDesc.src.Channel_Split import rgb_split_case_4 as split4

'''Functions to evaluate each possibility of parameters combination '''
''' 
Block I
In this block we extract both biodiversity indices and Taxonomic indices, that is, bfeat=True, tfeat=True

'''

# Case 1: bfeat=True, tfeat=True, unsharpfilter=True, crimminsfilter=True, normalization=True
def bit_case1(image) :
    # List for concatenating features
    features = list()
    # Split channels R, G, B, and RGB
    r_g_b_rgb = split1(image)
    # Extract Biodiversity features
    bio_features = extract(bio, r_g_b_rgb)
    # Extract Biodiversity features (normalized)
    # bio_features_norm = normalized_extract(bio, r_g_b_rgb)
    features.append(bio_features)
    # Extract Taxonomic features (not normalized)
    taxo_features = extract(taxo, r_g_b_rgb)
    # Extract Taxonomic features (normalized)
    # taxo_features_norm = normalized_extract(taxo, r_g_b_rgb)
    features.append(taxo_features)
    feature_vector = [item for sublist in features for item in sublist]
    feature_vector = np.nan_to_num(feature_vector, nan=0.0)
    # normalize the array with linear algebra
    norm = np.linalg.norm(feature_vector)
    feature_vector = feature_vector / norm
    return feature_vector


# Case 2: bfeat=True, tfeat=True, unsharpfilter=True, crimminsfilter=True, normalization=False
def bit_case2(image) :
    features = list()
    # Split channels R, G, B, and RGB
    r_g_b_rgb = split1(image)
    # Extract Biodiversity features
    bio_features = extract(bio, r_g_b_rgb)
    # Extract Biodiversity features (normalized)
    # bio_features_norm = normalized_extract(bio, r_g_b_rgb)
    features.append(bio_features)
    # Extract Taxonomic features (not normalized)
    taxo_features = extract(taxo, r_g_b_rgb)
    # Extract Taxonomic features (normalized)
    # taxo_features_norm = normalized_extract(taxo, r_g_b_rgb)
    features.append(taxo_features)
    feature_vector = [item for sublist in features for item in sublist]
    feature_vector = np.nan_to_num(feature_vector, nan=0.0)
    return feature_vector


# Case 3: bfeat=True, tfeat=True, unsharpfilter=True, crimminsfilter=False, normalization=True
def bit_case3(image) :
    # List for concatenating features
    features = list()
    # Split channels R, G, B, and RGB
    r_g_b_rgb = split2(image)
    # Extract Biodiversity features
    bio_features = extract(bio, r_g_b_rgb)
    # Extract Biodiversity features (normalized)
    # bio_features_norm = normalized_extract(bio, r_g_b_rgb)
    features.append(bio_features)
    # Extract Taxonomic features (not normalized)
    taxo_features = extract(taxo, r_g_b_rgb)
    # Extract Taxonomic features (normalized)
    # taxo_features_norm = normalized_extract(taxo, r_g_b_rgb)
    features.append(taxo_features)
    feature_vector = [item for sublist in features for item in sublist]
    feature_vector = np.nan_to_num(feature_vector, nan=0.0)
    # normalize the array with linear algebra
    norm = np.linalg.norm(feature_vector)
    feature_vector = feature_vector / norm
    return feature_vector


# Case 4: bfeat=True, tfeat=True, unsharpfilter=True, crimminsfilter=False, normalization=False
def bit_case4(image) :
    features = list()
    # Split channels R, G, B, and RGB
    r_g_b_rgb = split2(image)
    # Extract Biodiversity features
    bio_features = extract(bio, r_g_b_rgb)
    # Extract Biodiversity features (normalized)
    # bio_features_norm = normalized_extract(bio, r_g_b_rgb)
    features.append(bio_features)
    # Extract Taxonomic features (not normalized)
    taxo_features = extract(taxo, r_g_b_rgb)
    # Extract Taxonomic features (normalized)
    # taxo_features_norm = normalized_extract(taxo, r_g_b_rgb)
    features.append(taxo_features)
    feature_vector = [item for sublist in features for item in sublist]
    feature_vector = np.nan_to_num(feature_vector, nan=0.0)
    return feature_vector


# Case 5: bfeat=True, tfeat=True, unsharpfilter=False, crimminsfilter=True, normalization=True
def bit_case5(image) :
    features = list()
    # Split channels R, G, B, and RGB
    r_g_b_rgb = split3(image)
    # Extract Biodiversity features
    bio_features = extract(bio, r_g_b_rgb)
    # Extract Biodiversity features (normalized)
    # bio_features_norm = normalized_extract(bio, r_g_b_rgb)
    features.append(bio_features)
    # Extract Taxonomic features (not normalized)
    taxo_features = extract(taxo, r_g_b_rgb)
    # Extract Taxonomic features (normalized)
    # taxo_features_norm = normalized_extract(taxo, r_g_b_rgb)
    features.append(taxo_features)
    feature_vector = [item for sublist in features for item in sublist]
    feature_vector = np.nan_to_num(feature_vector, nan=0.0)
    # normalize the array with linear algebra
    norm = np.linalg.norm(feature_vector)
    feature_vector = feature_vector / norm
    return feature_vector


# Case 6: bfeat=True, tfeat=True, unsharpfilter=False, crimminsfilter=True, normalization=False
def bit_case6(image) :
    features = list()
    # Split channels R, G, B, and RGB
    r_g_b_rgb = split3(image)
    # Extract Biodiversity features
    bio_features = extract(bio, r_g_b_rgb)
    # Extract Biodiversity features (normalized)
    # bio_features_norm = normalized_extract(bio, r_g_b_rgb)
    features.append(bio_features)
    # Extract Taxonomic features (not normalized)
    taxo_features = extract(taxo, r_g_b_rgb)
    # Extract Taxonomic features (normalized)
    # taxo_features_norm = normalized_extract(taxo, r_g_b_rgb)
    features.append(taxo_features)
    feature_vector = [item for sublist in features for item in sublist]
    feature_vector = np.nan_to_num(feature_vector, nan=0.0)
    return feature_vector

# Case 7: bfeat=True, tfeat=True, unsharpfilter=False, crimminsfilter=False, normalization=True
def bit_case7(image) :
    features = list()
    # Split channels R, G, B, and RGB
    r_g_b_rgb = split4(image)
    # Extract Biodiversity features
    bio_features = extract(bio, r_g_b_rgb)
    # Extract Biodiversity features (normalized)
    # bio_features_norm = normalized_extract(bio, r_g_b_rgb)
    features.append(bio_features)
    # Extract Taxonomic features (not normalized)
    taxo_features = extract(taxo, r_g_b_rgb)
    # Extract Taxonomic features (normalized)
    # taxo_features_norm = normalized_extract(taxo, r_g_b_rgb)
    features.append(taxo_features)
    feature_vector = [item for sublist in features for item in sublist]
    feature_vector = np.nan_to_num(feature_vector, nan=0.0)
    # normalize the array with linear algebra
    norm = np.linalg.norm(feature_vector)
    feature_vector = feature_vector / norm
    return feature_vector


# Case 8: bfeat=True, tfeat=True, unsharpfilter=False, crimminsfilter=False, normalization=False
def bit_case8(image) :
    features = list()
    # Split channels R, G, B, and RGB
    r_g_b_rgb = split4(image)
    # Extract Biodiversity features
    bio_features = extract(bio, r_g_b_rgb)
    # Extract Biodiversity features (normalized)
    # bio_features_norm = normalized_extract(bio, r_g_b_rgb)
    features.append(bio_features)
    # Extract Taxonomic features (not normalized)
    taxo_features = extract(taxo, r_g_b_rgb)
    # Extract Taxonomic features (normalized)
    # taxo_features_norm = normalized_extract(taxo, r_g_b_rgb)
    features.append(taxo_features)
    feature_vector = [item for sublist in features for item in sublist]
    feature_vector = np.nan_to_num(feature_vector, nan=0.0)
    return feature_vector

''' 
Block II
In this block we extract only biodiversity indices, that is, bfeat=True, tfeat=False

'''

# Case 9: bfeat=True, tfeat=False, unsharpfilter=True, crimminsfilter=True, normalization=True
def bit_case9(image) :
    # List for concatenating features
    features = list()
    # Split channels R, G, B, and RGB
    r_g_b_rgb = split1(image)
    # Extract Biodiversity features
    bio_features = extract(bio, r_g_b_rgb)
    # Extract Biodiversity features (normalized)
    # bio_features_norm = normalized_extract(bio, r_g_b_rgb)
    features.append(bio_features)
    feature_vector = [item for sublist in features for item in sublist]
    feature_vector = np.nan_to_num(feature_vector, nan=0.0)
    # normalize the array with linear algebra
    norm = np.linalg.norm(feature_vector)
    feature_vector = feature_vector / norm
    return feature_vector


# Case 10: bfeat=True, tfeat=False, unsharpfilter=True, crimminsfilter=True, normalization=False
def bit_case10(image) :
    features = list()
    # Split channels R, G, B, and RGB
    r_g_b_rgb = split1(image)
    # Extract Biodiversity features
    bio_features = extract(bio, r_g_b_rgb)
    # Extract Biodiversity features (normalized)
    # bio_features_norm = normalized_extract(bio, r_g_b_rgb)
    features.append(bio_features)
    feature_vector = [item for sublist in features for item in sublist]
    feature_vector = np.nan_to_num(feature_vector, nan=0.0)
    return feature_vector


# Case 11: bfeat=True, tfeat=False, unsharpfilter=True, crimminsfilter=False, normalization=True
def bit_case11(image) :
    # List for concatenating features
    features = list()
    # Split channels R, G, B, and RGB
    r_g_b_rgb = split2(image)
    # Extract Biodiversity features
    bio_features = extract(bio, r_g_b_rgb)
    # Extract Biodiversity features (normalized)
    # bio_features_norm = normalized_extract(bio, r_g_b_rgb)
    features.append(bio_features)
    feature_vector = [item for sublist in features for item in sublist]
    feature_vector = np.nan_to_num(feature_vector, nan=0.0)
    # normalize the array with linear algebra
    norm = np.linalg.norm(feature_vector)
    feature_vector = feature_vector / norm
    return feature_vector


# Case 12: bfeat=True, tfeat=False, unsharpfilter=True, crimminsfilter=False, normalization=False
def bit_case12(image) :
    features = list()
    # Split channels R, G, B, and RGB
    r_g_b_rgb = split2(image)
    # Extract Biodiversity features
    bio_features = extract(bio, r_g_b_rgb)
    # Extract Biodiversity features (normalized)
    # bio_features_norm = normalized_extract(bio, r_g_b_rgb)
    features.append(bio_features)
    feature_vector = [item for sublist in features for item in sublist]
    feature_vector = np.nan_to_num(feature_vector, nan=0.0)
    return feature_vector


# Case 13: bfeat=True, tfeat=False, unsharpfilter=False, crimminsfilter=True, normalization=True
def bit_case13(image) :
    features = list()
    # Split channels R, G, B, and RGB
    r_g_b_rgb = split3(image)
    # Extract Biodiversity features
    bio_features = extract(bio, r_g_b_rgb)
    # Extract Biodiversity features (normalized)
    # bio_features_norm = normalized_extract(bio, r_g_b_rgb)
    features.append(bio_features)
    feature_vector = [item for sublist in features for item in sublist]
    feature_vector = np.nan_to_num(feature_vector, nan=0.0)
    # normalize the array with linear algebra
    norm = np.linalg.norm(feature_vector)
    feature_vector = feature_vector / norm
    return feature_vector


# Case 14: bfeat=True, tfeat=False, unsharpfilter=False, crimminsfilter=True, normalization=False
def bit_case14(image) :
    features = list()
    # Split channels R, G, B, and RGB
    r_g_b_rgb = split3(image)
    # Extract Biodiversity features
    bio_features = extract(bio, r_g_b_rgb)
    # Extract Biodiversity features (normalized)
    # bio_features_norm = normalized_extract(bio, r_g_b_rgb)
    features.append(bio_features)
    feature_vector = [item for sublist in features for item in sublist]
    feature_vector = np.nan_to_num(feature_vector, nan=0.0)
    return feature_vector

# Case 15: bfeat=True, tfeat=False, unsharpfilter=False, crimminsfilter=False, normalization=True
def bit_case15(image) :
    features = list()
    # Split channels R, G, B, and RGB
    r_g_b_rgb = split4(image)
    # Extract Biodiversity features
    bio_features = extract(bio, r_g_b_rgb)
    # Extract Biodiversity features (normalized)
    # bio_features_norm = normalized_extract(bio, r_g_b_rgb)
    features.append(bio_features)
    feature_vector = [item for sublist in features for item in sublist]
    feature_vector = np.nan_to_num(feature_vector, nan=0.0)
    # normalize the array with linear algebra
    norm = np.linalg.norm(feature_vector)
    feature_vector = feature_vector / norm
    return feature_vector


# Case 16: bfeat=True, tfeat=False, unsharpfilter=False, crimminsfilter=False, normalization=False
def bit_case16(image) :
    features = list()
    # Split channels R, G, B, and RGB
    r_g_b_rgb = split4(image)
    # Extract Biodiversity features
    bio_features = extract(bio, r_g_b_rgb)
    # Extract Biodiversity features (normalized)
    # bio_features_norm = normalized_extract(bio, r_g_b_rgb)
    features.append(bio_features)
    feature_vector = [item for sublist in features for item in sublist]
    feature_vector = np.nan_to_num(feature_vector, nan=0.0)
    return feature_vector

''' 
Block III
In this block we extract only taxonomic indices, that is, bfeat=False, tfeat=True

'''

# Case 17: bfeat=False, tfeat=True, unsharpfilter=True, crimminsfilter=True, normalization=True
def bit_case17(image) :
    # List for concatenating features
    features = list()
    # Split channels R, G, B, and RGB
    r_g_b_rgb = split1(image)
    # Extract Taxonomic features (not normalized)
    taxo_features = extract(taxo, r_g_b_rgb)
    # Extract Taxonomic features (normalized)
    # taxo_features_norm = normalized_extract(taxo, r_g_b_rgb)
    features.append(taxo_features)
    feature_vector = [item for sublist in features for item in sublist]
    feature_vector = np.nan_to_num(feature_vector, nan=0.0)
    # normalize the array with linear algebra
    norm = np.linalg.norm(feature_vector)
    feature_vector = feature_vector / norm
    return feature_vector


# Case 18: bfeat=False, tfeat=True, unsharpfilter=True, crimminsfilter=True, normalization=False
def bit_case18(image) :
    features = list()
    # Split channels R, G, B, and RGB
    r_g_b_rgb = split1(image)
    # Extract Taxonomic features (not normalized)
    taxo_features = extract(taxo, r_g_b_rgb)
    # Extract Taxonomic features (normalized)
    # taxo_features_norm = normalized_extract(taxo, r_g_b_rgb)
    features.append(taxo_features)
    feature_vector = [item for sublist in features for item in sublist]
    feature_vector = np.nan_to_num(feature_vector, nan=0.0)
    return feature_vector


# Case 19: bfeat=False, tfeat=True, unsharpfilter=True, crimminsfilter=False, normalization=True
def bit_case19(image) :
    # List for concatenating features
    features = list()
    # Split channels R, G, B, and RGB
    r_g_b_rgb = split2(image)
    # Extract Taxonomic features (not normalized)
    taxo_features = extract(taxo, r_g_b_rgb)
    # Extract Taxonomic features (normalized)
    # taxo_features_norm = normalized_extract(taxo, r_g_b_rgb)
    features.append(taxo_features)
    feature_vector = [item for sublist in features for item in sublist]
    feature_vector = np.nan_to_num(feature_vector, nan=0.0)
    # normalize the array with linear algebra
    norm = np.linalg.norm(feature_vector)
    feature_vector = feature_vector / norm
    return feature_vector


# Case 20: bfeat=False, tfeat=True, unsharpfilter=True, crimminsfilter=False, normalization=False
def bit_case20(image) :
    features = list()
    # Split channels R, G, B, and RGB
    r_g_b_rgb = split2(image)
    # Extract Taxonomic features (not normalized)
    taxo_features = extract(taxo, r_g_b_rgb)
    # Extract Taxonomic features (normalized)
    # taxo_features_norm = normalized_extract(taxo, r_g_b_rgb)
    features.append(taxo_features)
    feature_vector = [item for sublist in features for item in sublist]
    feature_vector = np.nan_to_num(feature_vector, nan=0.0)
    return feature_vector


# Case 21: bfeat=False, tfeat=True, unsharpfilter=False, crimminsfilter=True, normalization=True
def bit_case21(image) :
    features = list()
    # Split channels R, G, B, and RGB
    r_g_b_rgb = split3(image)
    # Extract Taxonomic features (not normalized)
    taxo_features = extract(taxo, r_g_b_rgb)
    # Extract Taxonomic features (normalized)
    # taxo_features_norm = normalized_extract(taxo, r_g_b_rgb)
    features.append(taxo_features)
    feature_vector = [item for sublist in features for item in sublist]
    feature_vector = np.nan_to_num(feature_vector, nan=0.0)
    # normalize the array with linear algebra
    norm = np.linalg.norm(feature_vector)
    feature_vector = feature_vector / norm
    return feature_vector


# Case 22: bfeat=False, tfeat=True, unsharpfilter=False, crimminsfilter=True, normalization=False
def bit_case22(image) :
    features = list()
    # Split channels R, G, B, and RGB
    r_g_b_rgb = split3(image)
    # Extract Taxonomic features (not normalized)
    taxo_features = extract(taxo, r_g_b_rgb)
    # Extract Taxonomic features (normalized)
    # taxo_features_norm = normalized_extract(taxo, r_g_b_rgb)
    features.append(taxo_features)
    feature_vector = [item for sublist in features for item in sublist]
    feature_vector = np.nan_to_num(feature_vector, nan=0.0)
    return feature_vector

# Case 23: bfeat=False, tfeat=True, unsharpfilter=False, crimminsfilter=False, normalization=True
def bit_case23(image) :
    features = list()
    # Split channels R, G, B, and RGB
    r_g_b_rgb = split4(image)
    # Extract Taxonomic features (not normalized)
    taxo_features = extract(taxo, r_g_b_rgb)
    # Extract Taxonomic features (normalized)
    # taxo_features_norm = normalized_extract(taxo, r_g_b_rgb)
    features.append(taxo_features)
    feature_vector = [item for sublist in features for item in sublist]
    feature_vector = np.nan_to_num(feature_vector, nan=0.0)
    # normalize the array with linear algebra
    norm = np.linalg.norm(feature_vector)
    feature_vector = feature_vector / norm
    return feature_vector


# Case 24: bfeat=False, tfeat=True, unsharpfilter=False, crimminsfilter=False, normalization=False
def bit_case24(image) :
    features = list()
    # Split channels R, G, B, and RGB
    r_g_b_rgb = split4(image)
    # Extract Taxonomic features (not normalized)
    taxo_features = extract(taxo, r_g_b_rgb)
    # Extract Taxonomic features (normalized)
    # taxo_features_norm = normalized_extract(taxo, r_g_b_rgb)
    features.append(taxo_features)
    feature_vector = [item for sublist in features for item in sublist]
    feature_vector = np.nan_to_num(feature_vector, nan=0.0)
    return feature_vector

# BiT Class
class BiT :

    # Constructor
    def __init__(self, image, bfeat=True, tfeat=True, unsharpfilter=True, crimminsfilter=True,
                 normalization=True) :
        self.image = image
        self.bfeat = bfeat
        self.tfeat = tfeat
        self.unsharpfilter = unsharpfilter
        self.crimminsfilter = crimminsfilter
        self.normalization = normalization

    # Conditional function
    def features(self):
        # Block I - bfeat=True, tfeat=True
        if self.bfeat and self.tfeat and self.unsharpfilter and self.crimminsfilter and self.normalization:
            feature_vector = bit_case1(self.image)
            return feature_vector
        elif self.bfeat and self.tfeat and self.unsharpfilter and self.crimminsfilter and (not self.normalization):
            feature_vector = bit_case2(self.image)
            return feature_vector
        elif self.bfeat and self.tfeat and self.unsharpfilter and (not self.crimminsfilter) and self.normalization:
            feature_vector = bit_case3(self.image)
            return feature_vector
        elif self.bfeat and self.tfeat and self.unsharpfilter and (not self.crimminsfilter) and (
        not self.normalization):
            feature_vector = bit_case4(self.image)
            return feature_vector
        elif self.bfeat and self.tfeat and (not self.unsharpfilter) and self.crimminsfilter and self.normalization:
            feature_vector = bit_case5(self.image)
            return feature_vector
        elif self.bfeat and self.tfeat and (not self.unsharpfilter) and self.crimminsfilter and (
        not self.normalization):
            feature_vector = bit_case6(self.image)
            return feature_vector
        elif self.bfeat and self.tfeat and (not self.unsharpfilter) and (not self.crimminsfilter) and self.normalization:
            feature_vector = bit_case7(self.image)
            return feature_vector
        elif self.bfeat and self.tfeat and (not self.unsharpfilter) and (not self.crimminsfilter) and (
        not self.normalization):
            feature_vector = bit_case8(self.image)
            return feature_vector
        # Block II - bfeat=True, tfeat=False
        elif self.bfeat and (not self.tfeat) and self.unsharpfilter and self.crimminsfilter and self.normalization:
            feature_vector = bit_case9(self.image)
            return feature_vector
        elif self.bfeat and (not self.tfeat) and self.unsharpfilter and self.crimminsfilter and (not self.normalization):
            feature_vector = bit_case10(self.image)
            return feature_vector
        elif self.bfeat and (not self.tfeat) and self.unsharpfilter and (not self.crimminsfilter) and self.normalization:
            feature_vector = bit_case11(self.image)
            return feature_vector
        elif self.bfeat and (not self.tfeat) and self.unsharpfilter and (not self.crimminsfilter) and (
        not self.normalization):
            feature_vector = bit_case12(self.image)
            return feature_vector
        elif self.bfeat and (not self.tfeat) and (not self.unsharpfilter) and self.crimminsfilter and self.normalization:
            feature_vector = bit_case13(self.image)
            return feature_vector
        elif self.bfeat and (not self.tfeat) and (not self.unsharpfilter) and self.crimminsfilter and (
        not self.normalization):
            feature_vector = bit_case14(self.image)
            return feature_vector
        elif self.bfeat and (not self.tfeat) and (not self.unsharpfilter) and (not self.crimminsfilter) and self.normalization:
            feature_vector = bit_case15(self.image)
            return feature_vector
        elif self.bfeat and (not self.tfeat) and (not self.unsharpfilter) and (not self.crimminsfilter) and (
        not self.normalization):
            feature_vector = bit_case16(self.image)
            return feature_vector
        # Block III - bfeat=False, tfeat=True
        if (not self.bfeat) and self.tfeat and self.unsharpfilter and self.crimminsfilter and self.normalization:
            feature_vector = bit_case17(self.image)
            return feature_vector
        elif (not self.bfeat) and self.tfeat and self.unsharpfilter and self.crimminsfilter and (not self.normalization):
            feature_vector = bit_case18(self.image)
            return feature_vector
        elif (not self.bfeat) and self.tfeat and self.unsharpfilter and (not self.crimminsfilter) and self.normalization:
            feature_vector = bit_case19(self.image)
            return feature_vector
        elif (not self.bfeat) and self.tfeat and self.unsharpfilter and (not self.crimminsfilter) and (
        not self.normalization):
            feature_vector = bit_case20(self.image)
            return feature_vector
        elif (not self.bfeat) and self.tfeat and (not self.unsharpfilter) and self.crimminsfilter and self.normalization:
            feature_vector = bit_case21(self.image)
            return feature_vector
        elif (not self.bfeat) and self.tfeat and (not self.unsharpfilter) and self.crimminsfilter and (
        not self.normalization):
            feature_vector = bit_case22(self.image)
            return feature_vector
        elif (not self.bfeat) and self.tfeat and (not self.unsharpfilter) and (not self.crimminsfilter) and self.normalization:
            feature_vector = bit_case23(self.image)
            return feature_vector
        elif (not self.bfeat) and self.tfeat and (not self.unsharpfilter) and (not self.crimminsfilter) and (
        not self.normalization):
            feature_vector = bit_case24(self.image)
            return feature_vector
        else :
            raise Exception("At least one descriptor should be true (either bfeat = True or tfeat = True; or bfeat = True and "
                  "tfeat = True)")

