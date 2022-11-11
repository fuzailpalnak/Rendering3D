import math
import os
from typing import Union

import cv2
import numpy as np

# from task1.src.dlt import RANSAC
from task1.src.util import draw_pairs, draw_key_points
from task1.src.ops import find_features, match_features

# MODEL_IMAGE = r"../data/wetransfer_2022-11-10-131213-jpg_2022-11-10_1222/s1.jpg"
MODEL_IMAGE = r"../data/surface_test.jpg"
# MODEL_IMAGE = r"../data/test_source.png"
MP = r"../output/matches"
KP = r"../output/keypoints"


if not os.path.exists(KP):
    os.makedirs(KP)

if not os.path.exists(MP):
    os.makedirs(MP)

DEBUG = True
NUM_ITERATIONS = 2000


WEBCAM_INTRINSIC = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])
WEBCAM_DST = np.array(
    [[-1.38550017e00, 3.99507333e00, -2.90393843e-03, 2.41582743e-02, -4.97242005e00]]
)

# WD = (17, 12.7)
WD = (9.9, 7)
# WD = (1, 1)


def normalization(nd, x):
    """
    Normalization of coordinates (centroid to the origin and mean distance of sqrt(2 or 3).

    Input
    -----
    nd: number of dimensions, 3 here
    x: the data to be normalized (directions at different columns and points at rows)
    Output
    ------
    Tr: the transformation matrix (translation plus scaling)
    x: the transformed data
    """

    x = np.asarray(x)
    m, s = np.mean(x, 0), np.std(x)
    if nd == 2:
        Tr = np.array([[s, 0, m[0]], [0, s, m[1]], [0, 0, 1]])
    else:
        Tr = np.array([[s, 0, 0, m[0]], [0, s, 0, m[1]], [0, 0, s, m[2]], [0, 0, 0, 1]])

    Tr = np.linalg.inv(Tr)
    x = np.dot(Tr, np.concatenate((x.T, np.ones((1, x.shape[0])))))
    x = x[0:nd, :].T

    return Tr, x


def decompose_dlt(P):
    temp = np.linalg.inv(P[0:3, 0:3])
    R, K = np.linalg.qr(temp)
    R = np.linalg.inv(R)
    K = np.linalg.inv(K)
    K = K / K[2, 2]
    T = -1 * np.matmul(temp, P[:, 3])
    return R, K, T


def project_wc_on_ic(projection_matrix, wc):
    projected_points = np.matmul(projection_matrix, np.transpose(wc))
    projected_points = (
        projected_points
        / projected_points[
            -1:,
        ]
    )
    return projected_points.T


def projection_error(projection_matrix, ic, wc):
    projected_points = project_wc_on_ic(projection_matrix, wc)
    return np.abs(projected_points[:, 0] - ic[:, 0]) + np.abs(
        projected_points[:, 1] - ic[:, 1]
    )
    # return np.linalg.norm(ic - projected_points, axis=-1)


def projection_matrix_estimation(img_pts, world_pts):

    # Txyz, world_pts1 = normalization(3, world_pts)
    # Tuv, img_pts1 = normalization(2, img_pts)

    world_pts1 = world_pts
    img_pts1 = img_pts

    n = world_pts.shape[0]
    A = list()
    for i in range(n):
        x, y, z = world_pts1[i, 0], world_pts1[i, 1], world_pts1[i, 2]
        u, v = img_pts1[i, 0], img_pts1[i, 1]
        A.append([-x, -y, -z, -1, 0, 0, 0, 0, u * x, u * y, u * z, u])
        A.append([0, 0, 0, 0, -x, -y, -z, -1, v * x, v * y, v * z, v])

    U, D, V = np.linalg.svd(np.asarray(A))

    P = (
        V[10, :]
        if np.all(
            V[11, :]
            == np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        )
        else V[11:]
    )
    # P = V[11, :]
    P = np.reshape(P, (3, 4))
    P = P / P[2, 3]

    # H = np.dot(np.dot(np.linalg.pinv(Tuv), P), Txyz)
    # print(H)
    # H = H / H[-1, -1]
    # print(H)
    # L = H.flatten(0)
    # print(L)

    return P


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        my_batch = set(iterable[ndx : min(ndx + n, l)])
        yield list(my_batch), list(set(iterable) - my_batch)


def dlt_ransac(point_map, scale, threshold=0.6):
    best_pairs = set()
    best_projection = None
    best_error = np.inf
    points_range = list(range(len(point_map)))
    for i in range(NUM_ITERATIONS):
        # random_points = np.random.choice(len(point_map), 6)
        for random_points, remaining_points in batch(points_range, 6):
            # random_points = [0, 1, 2, 3, 5]
            # remaining_points = list(set(points_range) - set(random_points))

            random_pairs = point_map[random_points]
            approximation = projection_matrix_estimation(
                np.array(random_pairs)[:, 2:],
                np.c_[
                    np.array(random_pairs)[:, 0:2] * scale, np.zeros(len(random_pairs))
                ],
            )
            if np.all(np.isnan(approximation) == False):
                remaining_pairs = point_map[remaining_points]
                wc = np.c_[
                    np.array(remaining_pairs)[:, 0:2] * scale,
                    np.zeros(len(remaining_pairs)),
                    np.ones(len(remaining_pairs)),
                ]
                ic = np.c_[
                    np.array(remaining_pairs)[:, 2:], np.ones(len(remaining_pairs))
                ]

                pe1 = projection_error(approximation, ic, wc)
                matched_pair = np.hstack(
                    [
                        wc[np.where(pe1 < 10)][:, 0:2] / scale,
                        ic[np.where(pe1 < 10)][:, 0:2],
                    ]
                )

                if len(matched_pair) > len(best_pairs):
                    mp = np.vstack([np.array(list(matched_pair)), random_pairs])
                    bm = projection_matrix_estimation(
                        np.array(mp)[:, 2:],
                        np.c_[np.array(mp)[:, 0:2] * scale, np.zeros(len(mp))],
                    )
                    if np.all(np.isnan(bm) == False):

                        wc = np.c_[
                            np.array(mp)[:, 0:2] * scale,
                            np.zeros(len(mp)),
                            np.ones(len(mp)),
                        ]
                        ic = np.c_[np.array(mp)[:, 2:], np.ones(len(mp))]

                        te = projection_error(approximation, ic, wc)
                        te = np.array(te).mean()
                        if te < best_error:
                            best_pairs = matched_pair
                            best_projection = bm
                            best_error = te

                if len(best_pairs) > (len(point_map) * threshold):
                    break
    print(
        f"\x1b[2K\r└──> Best inliers {len(best_pairs)} ",
        end="",
    )
    return best_projection, best_pairs


def run(pth: Union[str, int] = 0):
    cap = cv2.VideoCapture(pth)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open THE provided")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # writer = cv2.VideoWriter(
    #     "pose_estimation.mp4", cv2.VideoWriter_fourcc(*"DIVX"), cap.get(cv2.CAP_PROP_FPS), (width, height)
    # )

    fc = 0
    model_image = cv2.imread(MODEL_IMAGE)
    model_image = cv2.cvtColor(model_image, cv2.COLOR_BGR2GRAY)
    model_image_key_points, model_image_desc = find_features(model_image)
    h, w = model_image.shape[0:2]
    scale_width = WD[0] / w
    scale_height = WD[1] / h

    # model_image_roi = model_image[234:219, 114:433]
    # cv2.imwrite("roi.png", model_image_roi)
    #
    # q = np.zeros(model_image.shape)
    # for i in (range(model_image_roi.shape[0])):
    #     for j in (range(model_image_roi.shape[1])):
    #         q[i+234][j+219] = model_image_roi[i][j]
    #
    # cv2.imwrite("../data/roi_q_12.png", q)

    while cap.isOpened():
        print(f"\x1b[2K\r└──> Frame {fc + 1}", end="")
        _, frame = cap.read()

        # frame = cv2.imread(MODEL_IMAGE)
        frame_rgb = frame.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        # INTEREST POINT DETECTION
        f_key_points, f_desc = find_features(frame)
        if DEBUG:
            cv2.imwrite(
                os.path.join(KP, f"{fc}_model_img_keypoints-1.png"),
                cv2.drawKeypoints(
                    model_image.copy(), model_image_key_points, model_image.copy()
                ),
            )
            cv2.imwrite(
                os.path.join(KP, f"{fc}_frame_keypoints-2.png"),
                cv2.drawKeypoints(frame.copy(), f_key_points, frame.copy()),
            )

        # FEATURE MATCHING
        matches = match_features(model_image_desc, f_desc)
        matches = sorted(matches, key=lambda x: x.distance)
        point_map = np.array(
            [
                [
                    model_image_key_points[match.queryIdx].pt[0],
                    model_image_key_points[match.queryIdx].pt[1],
                    f_key_points[match.trainIdx].pt[0],
                    f_key_points[match.trainIdx].pt[1],
                ]
                for match in matches
            ]
        )

        pm, matched_pairs = dlt_ransac(point_map, (scale_width, scale_height))
        # r, k, t = decompose_dlt(pm)
        if DEBUG and len(matched_pairs) > 0:
            pairs_img = draw_pairs(frame.copy(), list(matched_pairs), matched_pairs)
            cv2.imwrite(
                os.path.join(MP, f"{str(fc)}.png"),
                pairs_img,
            )

            mapping_img = draw_key_points(
                model_image.copy(),
                frame.copy(),
                list(matched_pairs),
                pairs=matched_pairs,
            )
            cv2.imwrite(
                os.path.join(KP, f"{fc}_mapping.png"),
                mapping_img,
            )

            mapping_img_1 = draw_key_points(
                model_image.copy(), frame.copy(), list(point_map), pairs=None
            )

            cv2.imwrite(
                os.path.join(KP, f"{fc}_point_map.png"),
                mapping_img_1,
            )
            # frame = mapping_img

        # ORIGIN
        origin_frame = frame_rgb.copy()
        origin_ic = project_wc_on_ic(
            wc=np.asarray([0, 0, 0, 1])[np.newaxis], projection_matrix=pm
        )[0]
        origin_frame[
            int(origin_ic[1]) : int(origin_ic[1]) + 15,
            int(origin_ic[0]) : int(origin_ic[0]) + 15,
            :,
        ] = [0, 255, 0]

        points = np.c_[
            np.array(point_map)[:, 0:2] * (scale_width, scale_height),
            np.zeros(len(point_map)),
            np.ones(len(point_map)),
        ]

        # points = np.float32(
        #     [
        #         [262 * scale_width, 115 * scale_height, 0],
        #         [262 * scale_width, 125 * scale_height, 0],
        #         [272 * scale_width, 125 * scale_height, 0],
        #         [272 * scale_width, 115 * scale_height, 0],
        #         [262 * scale_width, 115 * scale_height, -10],
        #         [262 * scale_width, 125 * scale_height, -10],
        #         [272 * scale_width, 125 * scale_height, -10],
        #         [272 * scale_width, 115 * scale_height, -10],
        #     ]
        # )
        # points = np.c_[np.array(points), np.ones(len(points))]

        ic_pts = project_wc_on_ic(projection_matrix=pm, wc=points)
        imgpts = []
        for aa in ic_pts:
            imgpts.append([int(aa[0]), int(aa[1])])
        # for pt in imgpts:
        #     frame_rgb[pt[1] - 1: pt[1] + 1, pt[0] - 1: pt[0] + 1, :] = [255, 0, 0]
        #
        # frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        for pt in imgpts:
            origin_frame[pt[1] - 2 : pt[1] + 2, pt[0] - 2 : pt[0] + 2, :] = [255, 0, 0]
        # cv2.imwrite("frame.png", frame)
        # imgpts = np.int32(np.array(imgpts)).reshape(-1, 2)

        # # dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        # imgpts = np.int32(dst).reshape(-1, 2)

        # # draw ground floor in green
        # img = cv2.drawContours(origin_frame, [imgpts[:4]], -1, (0, 255, 0), -3)
        # # draw pillars in blue color
        # for i, j in zip(range(4), range(4, 8)):
        #     img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255, 0, 0), 3)
        # # draw top layer in red color
        # img = cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)

        fc += 1

        cv2.imshow("img", origin_frame)
        if cv2.waitKey(1) == 27:
            break


run(r"../data/surface_demo.webm")
