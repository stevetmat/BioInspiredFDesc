import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np
import cv2
from BiT import taxonomy as taxo
from BiT import biodiversity as bio
from Channel_Split import concat_features_extract_measures as extract
from Channel_Split import concat_features_extract_measures_normalized as normalized_extract
from Channel_Split import rgb_split as split


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
    def features(self) :
        if self.bfeat and self.tfeat and self.unsharpfilter and self.crimminsfilter and self.normalization:
            # print("bfeat - tfeat - unsharpfilter - criminsfilter - normalization - are true")
            # List for concatenating features
            features = list()
            # Split channels R, G, B, and RGB
            r_g_b_rgb = split(self.image)
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
            feature_vector = feature_vector/norm
            return feature_vector
        elif self.bfeat and self.tfeat and self.unsharpfilter and self.crimminsfilter and (not self.normalization):
            print(self.bfeat)
            print(self.normalization)
        else :
            print("At least one is False")
