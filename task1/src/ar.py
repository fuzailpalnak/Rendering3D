import os.path
import shutil
from typing import Union

import cv2
import numpy as np

from task1.src.model.camera import DLT, Homography
from task1.src.model.reference import Reference3DCylindrical, Reference2D
from task1.src.util import (
    draw_key_points,
    draw_origin,
    draw_projected_pts,
    draw_cube,
)

REF_IMG_PTH_3D = r"../data/3d/reference.jpg"
REF_IMG_PTH_2D = r""

DISPLAY = True

OUTPUT = r"../output/"
if not os.path.exists(OUTPUT):
    os.makedirs(OUTPUT)


def render_with_3d_world_coordinates(image_plane: np.ndarray, dlt_model: DLT):
    image_plane_rgb = image_plane.copy()
    image_plane = cv2.cvtColor(image_plane, cv2.COLOR_BGR2GRAY)

    _pose = dlt_model.run(image_plane)

    # RENDER ORIGIN AND CUBE
    origin_ic = dlt_model.project_origin(projection_matrix=_pose.projection_matrix)
    rendered_frame = draw_origin(image_plane_rgb.copy(), origin_ic)

    m_wc, _ = dlt_model.get_wc_ic_from_map(_pose.matched_pairs)
    m_wc_on_ic = dlt_model.project_matching_points(
        m_wc,
        projection_matrix=_pose.projection_matrix,
    )
    rendered_frame = draw_projected_pts(rendered_frame, m_wc_on_ic)

    cube_3d = dlt_model.project_cube(
        projection_matrix=_pose.projection_matrix,
    )
    rendered_frame = draw_cube(rendered_frame, cube_3d)

    rendered_frame = draw_key_points(
        dlt_model.reference.rgb.copy(),
        rendered_frame.copy(),
        list(_pose.matched_pairs_px),
        pairs=_pose.matched_pairs_px,
    )

    return rendered_frame


def render_with_2d_world_coordinates(
    image_plane: np.ndarray, homography_model: Homography, camera_parameters: np.ndarray
):

    frame_rgb = image_plane.copy()
    # image_plane = cv2.cvtColor(image_plane, cv2.COLOR_BGR2GRAY)

    _pose = homography_model.run(image_plane, camera_parameters=camera_parameters)

    # RENDER ORIGIN AND CUBE
    origin_ic = homography_model.project_origin(
        projection_matrix=_pose.projection_matrix
    )
    rendered_frame = draw_origin(frame_rgb, origin_ic)

    m_wc, _ = homography_model.get_wc_ic_from_map(_pose.matched_pairs)
    m_wc_on_ic = homography_model.project_matching_points(
        np.c_[m_wc, np.zeros(len(m_wc))],
        projection_matrix=_pose.projection_matrix,
    )
    rendered_frame = draw_projected_pts(rendered_frame, m_wc_on_ic)

    cube_3d = homography_model.project_cube(
        projection_matrix=_pose.projection_matrix,
    )
    rendered_frame = draw_cube(rendered_frame, cube_3d)

    rendered_frame = draw_key_points(
        homography_model.reference.rgb.copy(),
        rendered_frame.copy(),
        list(_pose.matched_pairs_px),
        pairs=_pose.matched_pairs_px,
    )

    return rendered_frame


def stream_homography(pth: Union[str, int], camera_parameters: np.ndarray):
    cap = cv2.VideoCapture(pth)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open THE provided")

    _homography = Homography(Reference2D(img_pth=REF_IMG_PTH_2D))
    fc = 1
    while cap.isOpened():
        print(f"└──>FRAME IN PROGRESS {fc+1}")

        _, image_plane = cap.read()
        if image_plane is None:
            continue

        _rendered_frame = render_with_2d_world_coordinates(
            image_plane=image_plane,
            homography_model=_homography,
            camera_parameters=camera_parameters,
        )
        if DISPLAY:
            cv2.imshow("img", _rendered_frame)

        cv2.imwrite(os.path.join(OUTPUT, f"{fc}.png"), _rendered_frame)
        if cv2.waitKey(1) == 27:
            print("EXIT")
            cap.release()
            cv2.destroyAllWindows()
        fc += 1


def stream_dlt(pth: Union[str, int] = 0):
    cap = cv2.VideoCapture(pth)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open THE provided")

    _dlt = DLT(Reference3DCylindrical(img_pth=REF_IMG_PTH_3D))
    fc = 1
    while cap.isOpened():
        print(f"└──>FRAME IN PROGRESS {fc}")

        _, image_plane = cap.read()
        if image_plane is None:
            fc += 1
            continue
        # image_plane = cv2.rotate(image_plane, cv2.ROTATE_90_CLOCKWISE)
        # image_plane = cv2.rotate(image_plane, cv2.ROTATE_90_CLOCKWISE)

        _rendered_frame = render_with_3d_world_coordinates(
            image_plane=image_plane, dlt_model=_dlt
        )
        if DISPLAY:
            cv2.imshow("img", _rendered_frame)

        cv2.imwrite(os.path.join(OUTPUT, f"{fc}.png"), _rendered_frame)

        if cv2.waitKey(1) == 27:
            print("EXIT")
            cap.release()
            cv2.destroyAllWindows()
        fc += 1


if __name__ == "__main__":
    stream_dlt(r"../data/3d/demo-1_Trim-1.mp4")
