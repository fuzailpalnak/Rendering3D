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


def dist(pair, H):
    p1 = np.array([pair[0], pair[1], 1])
    p2 = np.array([pair[2], pair[3], 1])

    p2_estimate = np.dot(H, np.transpose(p1))
    p2_estimate = (1 / p2_estimate[2]) * p2_estimate

    return np.linalg.norm(np.transpose(p2) - p2_estimate)


def ransac(point_map, threshold=THRESHOLD):
    print(f"Running RANSAC with {len(point_map)} points...")
    best_points_matches = set()
    homography = None
    for i in range(NUM_ITERATIONS):
        # randomly choose 4 points from the matrix to compute the homography
        pairs = [point_map[i] for i in np.random.choice(len(point_map), 4)]

        H = compute_homography(pairs)
        inliers = {(c[0], c[1], c[2], c[3]) for c in point_map if dist(c, H) < 500}

        print(
            f"\x1b[2K\r└──> iteration {i + 1}/{NUM_ITERATIONS} "
            + f"\t{len(inliers)} inlier"
            + ("s " if len(inliers) != 1 else " ")
            + f"\tbest: {len(best_points_matches)}",
            end="",
        )

        if len(inliers) > len(best_points_matches):
            best_points_matches = inliers
            homography = H
            if len(best_points_matches) > (len(point_map) * threshold):
                break

    print(f"\nNum matches: {len(point_map)}")
    print(f"Num inliers: {len(best_points_matches)}")
    print(f"Min inliers: {len(point_map) * threshold}")

    return homography, best_points_matches


def create_point_map(image1, image2):
    """
    Creates a point map of shape (n, 4) where n is the number of matches
    between the two images. Each row contains (x1, y1, x2, y2), where (x1, y1)
    in image1 maps to (x2, y2) in image2.

    sift.detectAndCompute returns
        keypoints: a list of keypoints
        descriptors: a numpy array of shape (num keypoints, 128)
    """
    print("Finding key-points and descriptors for both images...")
    sift = cv2.SIFT_create()
    kp1, desc1 = sift.detectAndCompute(image1, None)
    kp2, desc2 = sift.detectAndCompute(image2, None)

    # cv2.imwrite(util.OUTPUT_PATH + 'keypoints-1.png',
    #             cv2.drawKeypoints(image1, kp1, image1))
    # cv2.imwrite(util.OUTPUT_PATH + 'keypoints-2.png',
    #             cv2.drawKeypoints(image2, kp2, image2))

    print("Determining matches...")
    matches = cv2.BFMatcher(cv2.NORM_L2, True).match(desc1, desc2)

    point_map = np.array(
        [
            [
                kp1[match.queryIdx].pt[0],
                kp1[match.queryIdx].pt[1],
                kp2[match.trainIdx].pt[0],
                kp2[match.trainIdx].pt[1],
            ]
            for match in matches
        ]
    )

    # cv2.imwrite(util.OUTPUT_PATH + 'matches.png',
    #             util.drawMatches(image1, image2, point_map))

    return point_map


def main(image1, image2, directory, verbose=True):
    """
    Info
    """

    point_map = create_point_map(image1, image2)
    homography, inliers = ransac(point_map)

    # cv2.imwrite(util.OUTPUT_PATH + 'inlier_matches.png',
    #             util.drawMatches(image1, image2, point_map, inliers))

    return point_map, inliers, homography
