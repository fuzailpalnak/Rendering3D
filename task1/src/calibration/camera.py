import os

import cv2
import numpy as np

__reference__ = [
    "https://www.ostirion.net/post/webcam-calibration-with-opencv-directly-from-the-stream",
    "https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html",
]

# Define the dimensions of checkerboard
CHECKERBOARD = (7, 7)
MIN_POINTS = 50
RECORD = True
DATA = r"/home/palnak/Workspace/Studium/msc/sem3/assignment/AR/task1/data/imagesForCalibration"


def obj3d():
    object_p3d = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)

    object_p3d[0, :, :2] = np.mgrid[0 : CHECKERBOARD[0], 0 : CHECKERBOARD[1]].T.reshape(
        -1, 2
    )
    return object_p3d


def generate_parameters_for_calibration():
    # Stop the iteration when specified
    # accuracy, epsilon, is reached or
    # specified number of iterations are completed.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Vector for the 3D points:
    points_3d = []

    # Vector for 2D points:
    points_2d = []

    object_p3d = obj3d()

    for file in os.listdir(DATA):
        print(f"PROCESSING {file}")

        fp = os.path.join(DATA, file)
        image = cv2.imread(fp)
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        # if desired number of corners are
        # found in the image then ret = true:
        ret, corners = cv2.findChessboardCorners(
            img_gray,
            CHECKERBOARD,
            cv2.CALIB_CB_ADAPTIVE_THRESH
            + cv2.CALIB_CB_FAST_CHECK
            + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )

        # If desired number of corners can be detected then,
        # refine the pixel coordinates and display
        # them on the images of checker board
        if ret:
            points_3d.append(object_p3d)

            # Refining pixel coordinates
            # for given 2d points.
            corners2 = cv2.cornerSubPix(
                img_gray, corners, CHECKERBOARD, (-1, -1), criteria
            )

            points_2d.append(corners2)

            # Draw and display the corners:
            image = cv2.drawChessboardCorners(image, CHECKERBOARD, corners2, ret)
            cv2.imshow("img", image)
            cv2.waitKey(500)

        return image, img_gray, points_2d, points_3d


def run_calibration():
    image, img_gray, points_2d, points_3d = generate_parameters_for_calibration()

    # Perform camera calibration by
    # passing the value of above found out 3D points (threedpoints)
    # and its corresponding pixel coordinates of the
    # detected corners (twodpoints):
    print("CALIBRATION IN PROGRESS")
    ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(
        points_3d, points_2d, img_gray.shape[::-1], None, None
    )

    # Displaying required output
    print(" Camera matrix:")
    print(matrix)

    print("\n Distortion coefficient:")
    print(distortion)

    print("\n Rotation Vectors:")
    print(r_vecs)

    print("\n Translation Vectors:")
    print(t_vecs)


if __name__ == "__main__":
    """
    iphone 14 pr :
        [[6.23649154e+02 0.00000000e+00 1.45491457e+03]
        [0.00000000e+00 6.24325606e+02 1.94913052e+03]
        [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
    """
    run_calibration()
