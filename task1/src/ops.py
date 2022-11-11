import os.path

import cv2
import numpy as np


THRESHOLD = 0.6
NUM_ITERATIONS = 1000


def dlt(A):
    U, D, V = np.linalg.svd(A)
    P = V[11, :]
    P = np.reshape(P, (3, 4))
    ### P is the projection matrix
    P = P / P[2, 3]
    return P


def compute_homography(pairs):
    A = []
    B = []
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


# Finding Projection Matrix using DLT
def projection_matrix_estimation(img_pts, world_pts):
    n = world_pts.shape[0]
    A = np.zeros((2 * n, 12))
    for i in range(n):
        A[i * 2, 0:4] = -1 * world_pts[i, :]
        A[i * 2, 8:12] = img_pts[i, 0] * world_pts[i, :]
        A[i * 2 + 1, 4:8] = -1 * world_pts[i, :]
        A[i * 2 + 1, 8:12] = img_pts[i, 1] * world_pts[i, :]

    U, D, V = np.linalg.svd(A)
    P = V[11, :]
    P = np.reshape(P, (3, 4))
    P = P / P[2, 3]
    return P


def ransac(point_map, threshold=THRESHOLD):
    print(f"Running RANSAC with {len(point_map)} points...")
    best_pairs = set()
    homography = None
    for i in range(NUM_ITERATIONS):
        # randomly choose 4 points from the matrix to compute the homography
        pairs = [point_map[i] for i in np.random.choice(len(point_map), 6)]

        H = compute_homography(pairs)
        matched_pair = {
            (c[0], c[1], c[2], c[3]) for c in point_map if error(c, H) < 200
        }

        print(
            f"\x1b[2K\r└──> iteration {i + 1}/{NUM_ITERATIONS} "
            + f"\t{len(matched_pair)} inlier"
            + ("s " if len(matched_pair) != 1 else " ")
            + f"\tbest: {len(best_pairs)}",
            end="",
        )

        if len(matched_pair) > len(best_pairs):
            best_pairs = matched_pair
            homography = H

        if len(best_pairs) > (len(point_map) * threshold):
            break

    print(f"\nNum matches: {len(point_map)}")
    print(f"Num inliers: {len(best_pairs)}")
    print(f"Min inliers: {len(point_map) * threshold}")

    return homography, best_pairs


def find_features(img: np.ndarray):
    sift = cv2.ORB_create()
    key_points, descriptors = sift.detectAndCompute(img, None)
    return key_points, descriptors


def match_features(desc1, desc2):
    index_params = dict(algorithm=6,
                        table_number=6,
                        key_size=12,
                        multi_probe_level=2)
    search_params = {}
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc1, desc2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.85 * n.distance:
            good_matches.append(m)

    # matches = cv2.BFMatcher(cv2.NORM_L2, True).match(desc1, desc2)
    return good_matches
