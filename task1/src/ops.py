import cv2
import numpy as np


def find_features_with_sift(img: np.ndarray):
    sift = cv2.SIFT_create()
    key_points, descriptors = sift.detectAndCompute(img, None)
    return key_points, descriptors


def match_features_with_sift(desc1, desc2):
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = {}
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc1, desc2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # good_matches = cv2.BFMatcher(cv2.NORM_L2, True).match(desc1, desc2)
    return good_matches


#
# def find_features_with_sift(img: np.ndarray):
#     orb = cv2.ORB_create()
#     key_points, descriptors = orb.detectAndCompute(img, None)
#     return key_points, descriptors
#
#
# def match_features_with_sift(desc1, desc2):
#     FLANN_INDEX_LSH = 6
#     index_params = dict(algorithm=FLANN_INDEX_LSH,
#                         table_number=6,  # 12
#                         key_size=12,  # 20
#                         multi_probe_level=1)  # 2
#
#     # FLANN_INDEX_KDTREE = 0
#     # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
#     search_params = dict(checks=50)
#     flann = cv2.FlannBasedMatcher(index_params, search_params)
#     matches = flann.knnMatch(desc1, desc2, k=2)
#
#     good_matches = []
#     for m_n in matches:
#         if len(m_n) != 2:
#             continue
#         (m, n) = m_n
#         if m.distance < 0.75 * n.distance:
#             good_matches.append(m)
#
#     # good_matches = []
#     # for m, n in matches:
#     #     if m.distance < 0.75 * n.distance:
#     #         good_matches.append(m)
#
#     # good_matches = cv2.BFMatcher(cv2.NORM_L2, True).match(desc1, desc2)
#     return good_matches
