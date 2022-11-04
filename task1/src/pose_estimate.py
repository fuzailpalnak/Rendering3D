import os
from typing import Union

import cv2
import numpy as np

from task1.src.util import draw_pairs, draw_key_points
from task1.src.ops import ransac, find_features, match_features

MODEL_IMAGE = r""
MP = r"../output/matches"
KP = r"../output/keypoints"

if not os.path.exists(KP):
    os.makedirs(KP)

if not os.path.exists(MP):
    os.makedirs(MP)

DEBUG = True


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

    while cap.isOpened():
        print(f"\x1b[2K\r└──> Frame {i + 1}", end="")
        _, frame = cap.read()

        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        h, w = frame.shape[0:2]
        model_image = cv2.resize(model_image, (w, h), interpolation=cv2.INTER_AREA)
        model_image = cv2.rotate(model_image, cv2.ROTATE_90_CLOCKWISE)
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
        homography, inliers = ransac(point_map)
        if DEBUG:
            matches_sorted = sorted(matches, key=lambda x: x.distance)
            src_pts = np.float32(
                [model_image_key_points[m.queryIdx].pt for m in matches_sorted]
            ).reshape(-1, 1, 2)
            dst_pts = np.float32(
                [f_key_points[m.trainIdx].pt for m in matches_sorted]
            ).reshape(-1, 1, 2)
            homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            print("└──>homography by OPENCV")
            print(f"\n{homography}")

            cv2.imwrite(
                os.path.join(MP, f"{str(i)}.png"),
                draw_pairs(frame.copy(), point_map, inliers),
            )

            cv2.imwrite(
                os.path.join(KP, f"{i}_mapping.png"),
                draw_key_points(model_image, frame.copy(), point_map, pairs=inliers),
            )

        # if homography is not None:
        #     h, w = model_image.shape
        #     pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(
        #         -1, 1, 2
        #     )
        #     # project corners into frame
        #     dst = cv2.perspectiveTransform(pts, homography)
        #     # connect them with lines
        #     frame = cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
        #     cv2.imwrite(
        #         os.path.join(OUTPUT_PATH, f"{str(i)}_polylines.png"),
        #         frame,
        #     )
        # cv2.imshow("img", img)
        # cv2.imwrite("her.png", img)
        # writer.write(img)

        i += 1


run(r"")
