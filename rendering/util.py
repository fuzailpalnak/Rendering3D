import time
from functools import wraps

import cv2
import numpy as np

BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)


def execution_time(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f"└──> FUNC {func.__name__}, EXECUTION TIME {total_time:.4f} seconds")
        return result

    return timeit_wrapper


def draw_pairs(image, point_map, pairs):
    rows, cols = image.shape[0:2]
    ret_image = np.zeros((rows, cols, 3), dtype="uint8")
    ret_image[:, :, :] = image if image.ndim == 3 else np.dstack([image] * 3)

    # Draw circles on top of the lines
    for x1, y1, x2, y2 in point_map:
        point = (int(x2), int(y2))
        color = GREEN if (x1, y1, x2, y2) in pairs else RED
        cv2.circle(ret_image, point, 4, color, 1)

    return ret_image


def draw_key_points(image1, image2, point_map, pairs=None, max_points=1000):
    rows1, cols1 = image1.shape[0:2]
    rows2, cols2 = image2.shape[0:2]

    match_image = np.zeros((max(rows1, rows2), cols1 + cols2, 3), dtype="uint8")
    match_image[:rows1, :cols1, :] = (
        image1 if image1.ndim == 3 else np.dstack([image1] * 3)
    )
    match_image[:rows2, cols1 : cols1 + cols2, :] = (
        image2 if image2.ndim == 3 else np.dstack([image2] * 3)
    )

    small_point_map = [
        point_map[i] for i in np.random.choice(len(point_map), max_points)
    ]

    # draw lines
    for x1, y1, x2, y2 in small_point_map:
        point1 = (int(x1), int(y1))
        point2 = (int(x2 + image1.shape[1]), int(y2))
        color = BLUE if pairs is None else (GREEN if (x1, y1, x2, y2) in pairs else RED)

        cv2.line(match_image, point1, point2, color, 1)

    # Draw circles on top of the lines
    for x1, y1, x2, y2 in small_point_map:
        point1 = (int(x1), int(y1))
        point2 = (int(x2 + image1.shape[1]), int(y2))
        cv2.circle(match_image, point1, 5, BLUE, 1)
        cv2.circle(match_image, point2, 5, BLUE, 1)

    return match_image


def draw_origin(frame, origin_ic):
    cv2.circle(
        frame, (abs(int(origin_ic[0])), abs(int(origin_ic[1]))), 5, (255, 0, 0), -1
    )
    return frame


def draw_projected_pts(frame, pts):
    img_pts = []
    for aa in pts:
        img_pts.append([int(aa[0]), int(aa[1])])

    for pt in img_pts:
        frame[pt[1] - 3 : pt[1] + 3, pt[0] - 3 : pt[0] + 3, :] = [0, 0, 255]
    return frame


def draw_cube(frame, pts):
    img_pts = []
    for aa in pts:
        img_pts.append([int(aa[0]), int(aa[1])])

    img_pts = np.int32(np.array(img_pts)).reshape(-1, 2)
    # draw ground floor in green
    frame = cv2.drawContours(frame, [img_pts[:4]], -1, (0, 255, 0), -3)
    # draw pillars in blue color
    for i, j in zip(range(4), range(4, 8)):
        frame = cv2.line(frame, tuple(img_pts[i]), tuple(img_pts[j]), (255, 0, 0), 3)
    # draw top layer in red color
    frame = cv2.drawContours(frame, [img_pts[4:]], -1, (0, 0, 255), 3)
    return frame
