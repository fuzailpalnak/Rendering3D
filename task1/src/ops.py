import cv2
import numpy as np


def find_features(img: np.ndarray):
    sift = cv2.ORB_create()
    key_points, descriptors = sift.detectAndCompute(img, None)
    return key_points, descriptors


def match_features(desc1, desc2):
    index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=2)
    search_params = {}
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc1, desc2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.85 * n.distance:
            good_matches.append(m)

    # matches = cv2.BFMatcher(cv2.NORM_L2, True).match(desc1, desc2)
    return good_matches
