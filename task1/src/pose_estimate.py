import cv2
import numpy as np

from task1.src.ops import create_point_map, ransac

MODEL_IMAGE = cv2.imread(r"/home/palnak/Workspace/Studium/msc/sem3/assignment/AR/task1/src/IMG_3435.jpg")


def run():
    cap = cv2.VideoCapture(0)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        "pose_estimation.mp4", cv2.VideoWriter_fourcc(*"DIVX"), fps, (width, height)
    )

    loop = True
    while loop:
        print("ACCESS THROUGH VIDEO FEED")
        _, img = cap.read()

        point_map = create_point_map(MODEL_IMAGE, img)
        homography, inliers = ransac(point_map)
        if homography is not None:
            h, w = MODEL_IMAGE.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            # project corners into frame
            dst = cv2.perspectiveTransform(pts, homography)
            # connect them with lines
            frame = cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

run()