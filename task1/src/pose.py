import math
import os
from typing import Union

import cv2
import numpy as np

# from task1.src.dlt import RANSAC
from task1.src.util import draw_pairs, draw_key_points
from task1.src.ops import ransac, find_features, match_features

MODEL_IMAGE = r"/home/palnak/Workspace/Studium/msc/sem3/assignment/AR/task1/data/wetransfer_img_3449-mov_2022-11-08_2048/image00001.jpeg"
MP = r"../output/matches"
KP = r"../output/keypoints"


if not os.path.exists(KP):
    os.makedirs(KP)

if not os.path.exists(MP):
    os.makedirs(MP)

DEBUG = True
NUM_ITERATIONS = 1000


WEBCAM_INTRINSIC = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])
WEBCAM_DST = np.array(
    [[-1.38550017e00, 3.99507333e00, -2.90393843e-03, 2.41582743e-02, -4.97242005e00]]
)


def decompose_dlt(P):
    temp = np.linalg.inv(P[0:3, 0:3])
    R, K = np.linalg.qr(temp)
    R = np.linalg.inv(R)
    K = np.linalg.inv(K)
    K = K / K[2, 2]
    T = -1 * np.matmul(temp, P[:, 3])
    return R, K, T


def project_wc_on_ic(projection_matrix, wc):
    projected_points = np.matmul(projection_matrix, np.transpose(wc[0, :]))
    projected_points = projected_points / projected_points[2]
    return projected_points


def projection_error(projection_matrix, ic, wc):
    projected_points = project_wc_on_ic(projection_matrix, wc)
    # return np.abs(projected_points[0] - ic[0, 0]) + np.abs(
    #     projected_points[1] - ic[0, 1]
    # )
    return np.linalg.norm(np.transpose(ic) - projected_points[..., np.newaxis])
    # return np.sqrt(np.mean(np.sum((projected_points[np.newaxis][0:2, :].T - ic) ** 2, 1)))


def projection_matrix_estimation(img_pts, world_pts):
    n = world_pts.shape[0]
    A = list()
    for i in range(n):
        x, y, z = world_pts[i, 0], world_pts[i, 1], world_pts[i, 2]
        u, v = img_pts[i, 0], img_pts[i, 1]
        A.append([x, y, z, 1, 0, 0, 0, 0, -u * x, -u * y, -u * z, -u])
        A.append([0, 0, 0, 0, x, y, z, 1, -v * x, -v * y, -v * z, -v])

    U, D, V = np.linalg.svd(np.asarray(A))
    P = V[11, :]
    P = np.reshape(P, (3, 4))
    P = P / P[2, 3]

    return P


def dlt_ransac(point_map, threshold=0.6):
    best_pairs = set()
    best_projection = None
    points_range = list(range(len(point_map)))
    for i in range(NUM_ITERATIONS):
        random_points = np.random.choice(len(point_map), 6)
        remaining_points = list(set(points_range) - set(random_points))

        pairs = point_map[random_points]

        pm = projection_matrix_estimation(
            np.array(pairs)[:, 2:], np.c_[np.array(pairs)[:, 0:2], np.zeros(len(pairs))]
        )
        if np.all(np.isnan(pm) == False):
            remaining_pairs = point_map[remaining_points]
            wc = np.c_[
                np.array(remaining_pairs)[:, 0:2],
                np.zeros(len(remaining_pairs)),
                np.ones(len(remaining_pairs)),
            ]
            ic = np.c_[np.array(remaining_pairs)[:, 2:], np.ones(len(remaining_pairs))]

            matched_pair = set()
            for it in range(len(remaining_pairs)):
                _wc = wc[it]
                _ic = ic[it]

                pe = projection_error(pm, _ic[np.newaxis], _wc[np.newaxis])
                if pe < 20:
                    matched_pair.add((_wc[0], _wc[1], _ic[0], _ic[1]))

            print(
                f"\x1b[2K\r└──> iteration {i + 1}/{NUM_ITERATIONS} "
                + f"\t{len(matched_pair)} inlier"
                + ("s " if len(matched_pair) != 1 else " ")
                + f"\tbest: {len(best_pairs)}",
                end="",
            )

            if len(matched_pair) > len(best_pairs):
                best_pairs = matched_pair
                best_projection = pm

            # if len(best_pairs) > (len(point_map) * threshold):
            #     break

            print(f"\nNum matches: {len(point_map)}")
            print(f"Num inliers: {len(best_pairs)}")
            print(f"Min inliers: {len(point_map) * threshold}")

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

    i = 0
    model_image = cv2.imread(MODEL_IMAGE)
    model_image = cv2.cvtColor(model_image, cv2.COLOR_BGR2GRAY)

    while cap.isOpened():
        print(f"\x1b[2K\r└──> Frame {i + 1}", end="")
        _, frame = cap.read()
        # cv2.imwrite("source_test_1.jpg", frame)
        frame_rgb = frame.copy()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        h, w = frame.shape[0:2]
        model_image = cv2.resize(model_image, (w, h), interpolation=cv2.INTER_AREA)
        model_image_key_points, model_image_desc = find_features(model_image)

        # INTEREST POINT DETECTION
        f_key_points, f_desc = find_features(frame)
        if DEBUG:
            cv2.imwrite(
                os.path.join(KP, f"{i}_keypoints-1.png"),
                cv2.drawKeypoints(model_image, model_image_key_points, model_image),
            )
            cv2.imwrite(
                os.path.join(KP, f"{i}_keypoints-2.png"),
                cv2.drawKeypoints(frame.copy(), f_key_points, frame.copy()),
            )

        # FEATURE MATCHING
        matches = match_features(model_image_desc, f_desc)
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

        # HOMOGRAPHY ESTIMATION
        pm, matched_pairs = dlt_ransac(point_map)

        if DEBUG:
            pairs_img = draw_pairs(frame.copy(), point_map, matched_pairs)
            cv2.imwrite(
                os.path.join(MP, f"{str(i)}.png"),
                pairs_img,
            )

            mapping_img = draw_key_points(
                model_image, frame.copy(), list(matched_pairs), pairs=matched_pairs
            )
            cv2.imwrite(
                os.path.join(KP, f"{i}_mapping.png"),
                mapping_img,
            )

            frame = mapping_img


        points = 20 * np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],
                   [0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3] ])
        # points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])

        pp = np.c_[points, np.ones(len(points))]
        # uv2 = np.dot(dlt, pp.T)
        # uv2 = uv2 / uv2[2, :]
        #
        projections = np.zeros((points.shape[0], 3))
        for i in range(points.shape[0]):
            projections[i, :] = np.matmul(pm, np.transpose(pp[i, :]))
            projections[i, :] = projections[i, :] / projections[i, 2]

        imgpts = []
        for aa in projections:
            imgpts.append([int(aa[0]), int(aa[1])])
        imgpts = np.int32(np.array(imgpts)).reshape(-1, 2)

        # dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        # imgpts = np.int32(dst).reshape(-1, 2)

        # draw ground floor in green
        img = cv2.drawContours(frame_rgb, [imgpts[:4]], -1, (0, 255, 0), -3)
        # draw pillars in blue color
        for i, j in zip(range(4), range(4, 8)):
            img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255, 0, 0), 3)
        # draw top layer in red color
        img = cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)

        cv2.imshow("img", frame_rgb)
        if cv2.waitKey(1) == 27:
            break


run(r"/home/palnak/Workspace/Studium/msc/sem3/assignment/AR/task1/data/wetransfer_img_3449-mov_2022-11-08_2048/IMG_3449.MOV")
