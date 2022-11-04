import os.path

import cv2
import numpy as np


THRESHOLD = 0.6
NUM_ITERATIONS = 1000


def compute_homography(pairs):
    A = []
    for x1, y1, x2, y2 in pairs:
        A.append([x1, y1, 1, 0, 0, 0, -x2 * x1, -x2 * y1, -x2])
        A.append([0, 0, 0, x1, y1, 1, -y2 * x1, -y2 * y1, -y2])
    A = np.array(A)

    # Singular Value Decomposition (SVD)
    U, S, V = np.linalg.svd(A)

    # V has shape (9, 9) for any number of input pairs. V[-1] is the eigenvector
    # of (A^T)A with the smalles eigenvalue. Reshape into 3x3 matrix.
    H = np.reshape(V[-1], (3, 3))

    # Normalization
    H = (1 / H.item(8)) * H
    return H


def error(pair, H):
    p1 = np.array([pair[0], pair[1], 1])
    p2 = np.array([pair[2], pair[3], 1])

    p2_estimate = np.dot(H, np.transpose(p1))
    p2_estimate = (1 / p2_estimate[2]) * p2_estimate

    return np.linalg.norm(np.transpose(p2) - p2_estimate)


def ransac(point_map, threshold=THRESHOLD):
    print(f"Running RANSAC with {len(point_map)} points...")
    max_inliers = set()
    homography = None
    for i in range(NUM_ITERATIONS):
        # randomly choose 4 points from the matrix to compute the homography
        pairs = [point_map[i] for i in np.random.choice(len(point_map), 4)]

        H = compute_homography(pairs)
        inliers = {(c[0], c[1], c[2], c[3]) for c in point_map if error(c, H) < 200}

        print(
            f"\x1b[2K\r└──> iteration {i + 1}/{NUM_ITERATIONS} "
            + f"\t{len(inliers)} inlier"
            + ("s " if len(inliers) != 1 else " ")
            + f"\tbest: {len(max_inliers)}",
            end="",
        )

        if len(inliers) > len(max_inliers):
            max_inliers = inliers
            homography = H

        if len(max_inliers) > (len(point_map) * threshold):
            break

    print(f"\nNum matches: {len(point_map)}")
    print(f"Num inliers: {len(max_inliers)}")
    print(f"Min inliers: {len(point_map) * threshold}")

    return homography, max_inliers


def find_features(img: np.ndarray):
    sift = cv2.SIFT_create()
    key_points, descriptors = sift.detectAndCompute(img, None)
    return key_points, descriptors


def match_features(desc1, desc2):
    matches = cv2.BFMatcher(cv2.NORM_L2, True).match(desc1, desc2)
    return matches
