import cv2
from BiT import taxonomy as taxo
from BiT import biodiversity as bio
from Channel_Split import concat_features_extract_measures as extract
from Channel_Split import concat_features_extract_measures_normalized as normalized_extract
from Channel_Split import rgb_split as split
import numpy as np
from PIL import Image, ImageFilter


# Example
path = 'test.png'
features = list()
'''
Split channels R, G, B, and RGB
'''
r_g_b_rgb = split(path)
'''
Extract Taxonomic features (not normalized)
'''
taxo_features = extract(taxo, r_g_b_rgb)
print("Taxonomic Indices for R, G, B, and RGB\n--------------------------------------")
print(taxo_features)
print("\n")
'''
Extract Taxonomic features (normalized)
'''
taxo_features = normalized_extract(taxo, r_g_b_rgb)
print("Taxonomic Indices for R, G, B, and RGB\n--------------------------------------")
print(taxo_features)
print("\n")
'''
Extract Biodiversity features
'''
bio_features = extract(bio, r_g_b_rgb)
print("Biodiversity Indices for R, G, B, and RGB\n--------------------------------------")
print(bio_features)
print("\n")
'''
Extract Biodiversity features (normalized)
'''
bio_features = normalized_extract(bio, r_g_b_rgb)
print("Biodiversity Indices for R, G, B, and RGB\n--------------------------------------")
print(bio_features)


