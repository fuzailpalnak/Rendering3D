import cv2
import numpy as np

from task1.src.models.camera import DLT, Homography
from task1.src.models.reference import Reference3DCylindrical, Reference2D
from task1.src.util import draw_key_points, draw_origin, draw_projected_pts, draw_cube

REF_IMG_PTH = (
    r"../data/wetransfer_image00001-jpeg_2022-11-15_0719/ezgif-frame-005-crop.jpg"
)
REF_IMG_PTH_2D = r"C:\Users\Fuzail.Palnak\UHD\openSource\AR\task1\data\surface_test.jpg"
WEBCAM_INTRINSIC = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])


def with_3d_world_coordinates(image_plane: np.ndarray, dlt_model: DLT):
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


def with_2d_world_coordinates(image_plane: np.ndarray, homography_model: Homography):

    frame_rgb = image_plane.copy()
    image_plane = cv2.cvtColor(image_plane, cv2.COLOR_BGR2GRAY)

    _pose = homography_model.run(image_plane, camera_parameters=WEBCAM_INTRINSIC)

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


# on_image(r"../data/wetransfer_image00001-jpeg_2022-11-15_0719/ezgif-frame-005.jpg", 100)

# _dlt = DLT(Reference3DCylindrical(img_pth=REF_IMG_PTH))
# frame = cv2.imread(r"C:\Users\Fuzail.Palnak\UHD\openSource\AR\task1\data\base_1.jpg")
# for i in range(100):
#     _frame = with_3d_world_coordinates(image_plane=frame, dlt_model=_dlt)
#     cv2.imshow("img", _frame)
#
#     # writer.write(mapping_img)
#     if cv2.waitKey(3000) == 27:
#         print("EXIT")
#         cv2.destroyAllWindows()

_homography = Homography(Reference2D(img_pth=REF_IMG_PTH_2D))
frame = cv2.imread(
    r"C:\Users\Fuzail.Palnak\UHD\openSource\AR\task1\data\surface_test.jpg"
)
for i in range(100):
    _frame = with_2d_world_coordinates(image_plane=frame, homography_model=_homography)
    cv2.imshow("img", _frame)

    # writer.write(mapping_img)
    if cv2.waitKey(3000) == 27:
        print("EXIT")
        cv2.destroyAllWindows()

# on_2d_image(
#     r"C:\Users\Fuzail.Palnak\UHD\openSource\AR\task1\data\surface_test.jpg", 100
# )
# st.stream(r"../data/wetransfer_image00001-jpeg_2022-11-15_0719/IMG_3411.MOV")

# run(r"../data/wetransfer_2022-11-10-131213-jpg_2022-11-10_1222/s1.webm")
# run(0)
# run(r"/home/palnak/2022-11-10-132141.webm")
# run(r"/home/palnak/2022-11-10-132141.webm")
# run(r"/home/palnak/swde1.webm")
# run(r"/home/palnak/2022-11-10-132141.webm")
# print(np.log(1-0.99) / (np.log(1 - ((1-0.50) ** 6))) * 10)

# x1 = np.array([20, 30, 40, 50, 60, 30, 20, 40])
# y1 =  np.array([12, 34, 56, 78, 89, 45, 90, 29])
# x = np.column_stack((x1,y1))
# centroid = np.mean( np.transpose( x ) , axis=-1)
# dist = [ np.sqrt( np.sum( np.square( v - centroid ) ) ) for v in x ]
# centroid = np.mean( np.transpose( x ) , axis=-1)
#
# for v in x:
#     q = np.sqrt( np.sum( np.square( v - centroid ) ) )
#     print(q)
# print(dist)
