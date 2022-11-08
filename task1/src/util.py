import cv2
import numpy as np

BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)


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
