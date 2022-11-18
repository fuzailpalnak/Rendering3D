import math
import os
from typing import Union

import cv2
import numpy as np

# from task1.src.dlt import RANSAC
from task1.src.util import draw_pairs, draw_key_points
from task1.src.ops import find_features, match_features

# MODEL_IMAGE = r"/home/palnak/base.jpg"
# MODEL_IMAGE = r"/home/palnak/sw1.jpg"
MODEL_IMAGE = r"../data/surface_test.jpg"
# MODEL_IMAGE = r"/home/palnak/2022-11-10-131656.jpg"
# MODEL_IMAGE = r"../data/wetransfer_2022-11-10-131213-jpg_2022-11-10_1222/s1-crop.jpg"
# MODEL_IMAGE = r"/home/palnak/2022-11-10-131656.jpg"
MODEL_IMAGE = (
    r"../data/wetransfer_image00001-jpeg_2022-11-15_0719/ezgif-frame-005-crop.jpg"
)
MP = r"../output/matches"
KP = r"../output/keypoints"


if not os.path.exists(KP):
    os.makedirs(KP)

if not os.path.exists(MP):
    os.makedirs(MP)

DEBUG = True
NUM_ITERATIONS = 2924


WEBCAM_INTRINSIC = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])
WEBCAM_DST = np.array(
    [[-1.38550017e00, 3.99507333e00, -2.90393843e-03, 2.41582743e-02, -4.97242005e00]]
)

# WD = (17, 12.7)
# WD = (9.9, 7)
# WD = (1, 1)
# WD = (20.8, 13.6)
WD = (12.2, 19.5)
WC_Z = 1


def get_projection_matrix(camera_parameters, homography):
    homography = homography * (-1)
    rot_and_transl = np.dot(np.linalg.inv(camera_parameters), homography)
    col_1 = rot_and_transl[:, 0]
    col_2 = rot_and_transl[:, 1]
    col_3 = rot_and_transl[:, 2]

    # normalise vectors
    l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
    rot_1 = col_1 / l
    rot_2 = col_2 / l
    translation = col_3 / l

    # compute the orthonormal basis
    c = rot_1 + rot_2
    p = np.cross(rot_1, rot_2)
    d = np.cross(c, p)
    rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_3 = np.cross(rot_1, rot_2)

    # finally, compute the 3D projection matrix from the model to the current frame
    projection = np.stack((rot_1, rot_2, rot_3, translation)).T

    return np.dot(camera_parameters, projection)


def calm_before_the_storm(x):
    d = x.shape[-1]
    m = x.mean(0)
    s = 1 / (x.std() * (1 / np.sqrt(2)))

    if d == 2:
        tr = np.array([[s, 0, -s * m[0]], [0, s, -s * m[1]], [0, 0, 1]])
    else:
        tr = np.array([[s, 0, 0, -s * m[0]], [0, s, 0, -s * m[1]], [0, 0, s, -s * m[2]], [0, 0, 0, 1]])

    return tr


def calm_before_the_storm_1(x):
    d = x.shape[-1]
    if d == 3:
        center = x.mean(0)
        x_c = x[:, 0:1] - center[0]
        y_c = x[:, 1:2] - center[1]
        z_c = x[:, 2:3] - center[2]

        dist = np.sqrt(np.power(x_c, 2) + np.power(y_c, 2) + np.power(z_c, 2))
        scale = np.sqrt(2) / dist.mean()
        tr = np.array([
            [scale, 0, 0, -scale * center[0]],
            [0, scale, 0, -scale * center[1]],
            [0, 0, scale, -scale * center[2]],
            [0, 0, 0, 1]
        ])
    else:
        center = x.mean(0)
        x_c = x[:, 0:1] - center[0]
        y_c = x[:, 1:2] - center[1]

        dist = np.sqrt(np.power(x_c, 2) + np.power(y_c, 2))
        scale = np.sqrt(2) / dist.mean()
        tr = np.array([
            [scale, 0, -scale * center[0]],
            [0, scale, -scale * center[1]],
            [0, 0, 1]
        ])

    return tr


def calm_before_the_storm_2(x):
    d = x.shape[-1]
    s = np.sqrt(2) / np.sqrt((abs(x - np.mean(x, axis=0)) ** 2)).sum(axis=-1).mean()
    m = np.mean(x, 0)
    if d == 3:
        tr = np.array([[s, 0, 0, m[0]], [0, s, 0, m[1]], [0, 0, s, m[2]], [0, 0, 0, 1]])
    else:
        tr = np.array([[s, 0, m[0]], [0, s, m[1]], [0, 0, 1]])
    return tr


def scale_and_translate_wc(pts):
    center = pts.mean(0)
    x_c = pts[:, 0:1] - center[0]
    y_c = pts[:, 1:2] - center[1]
    z_c = pts[:, 2:3] - center[2]

    dist = np.sqrt(np.power(x_c, 2) + np.power(y_c, 2) + np.power(z_c, 2))
    scale = np.sqrt(2) / dist.mean()
    norm4d = np.array([
        [scale, 0, 0, -scale * center[0]],
        [0, scale, 0, -scale * center[1]],
        [0, 0, scale, -scale * center[2]],
        [0, 0, 0, 1]
    ])
    return np.dot(norm4d, pts.T).T, norm4d


def scale_and_translate_ic(pts):
    center = pts.mean(0)
    x_c = pts[:, 0:1] - center[0]
    y_c = pts[:, 1:2] - center[1]

    dist = np.sqrt(np.power(x_c, 2) + np.power(y_c, 2))
    scale = np.sqrt(2) / dist.mean()
    norm3d = np.array([
        [scale, 0, -scale * center[0]],
        [0, scale, -scale * center[1]],
        [0, 0, 1]
    ])
    return np.dot(norm3d, pts.T).T, norm3d


def normalize_2d(x):
    s = np.sqrt(2) / np.sqrt((abs(x - np.mean(x, axis=0)) ** 2)).sum(axis=-1).mean()
    m = np.mean(x, 0)
    normalised_points = np.zeros((len(x), 3))
    Tr = np.array([[s, 0, m[0]], [0, s, m[1]], [0, 0, 1]])
    for i in range(x.shape[0]):
        normalised_points[i][0] = s * x[i][0] + (m[0])
        normalised_points[i][1] = s * x[i][1] + (m[1])
        normalised_points[i][2] = 1
    return Tr, normalised_points


def normalize_3d(x):
    s = np.sqrt(2) / np.sqrt((abs(x - np.mean(x, axis=0)) ** 2)).sum(axis=-1).mean()
    m = np.mean(x, 0)
    normalised_points = np.zeros((len(x), 4))
    Tr = np.array([[s, 0, 0, m[0]], [0, s, 0, m[1]], [0, 0, s, m[2]], [0, 0, 0, 1]])

    for i in range(x.shape[0]):
        normalised_points[i][0] = s * x[i][0] + (m[0])
        normalised_points[i][1] = s * x[i][1] + (m[1])
        normalised_points[i][2] = s * x[i][2] + (m[2])
        normalised_points[i][3] = 1

    return Tr, normalised_points


def normalization(nd, x):
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
    projected_points = projection_matrix @ wc.T
    projected_points = (
        projected_points
        / projected_points[
            -1:,
        ]
    )
    return projected_points.T


def project_ic_on_wc(projection_matrix, ic):
    projected_points = np.linalg.pinv(projection_matrix) @ ic.T
    projected_points = (
        projected_points
        / projected_points[
            -1:,
        ]
    )
    return projected_points.T


def projection_error(projection_matrix, ic, wc):
    projected_points = project_wc_on_ic(projection_matrix, wc)
    # return np.abs(projected_points[:, 0] - ic[:, 0]) + np.abs(
    #     projected_points[:, 1] - ic[:, 1]
    # )
    # return np.linalg.norm(ic - projected_points, ord=1, axis=-1)
    return np.linalg.norm(ic - projected_points, axis=-1)


def homography_estimate(img_pts, world_pts):
    n = world_pts.shape[0]
    A = list()
    for i in range(n):
        x, y, z = world_pts[i, 0], world_pts[i, 1], world_pts[i, 2]
        u, v = img_pts[i, 0], img_pts[i, 1]
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])

    try:
        U, D, V = np.linalg.svd(np.asarray(A))
    except np.linalg.LinAlgError:
        print("here")
    # v_simplified = V[D != 0]
    H = np.reshape(V[-1], (3, 3))
    # Normalization
    H = (1 / H.item(8)) * H

    return H


def homography_ransac(point_map, threshold=0.7):
    best_pairs = set()
    best_random_pairs = None
    best_projection = None

    if len(point_map) >= 4:
        points_range = list(range(len(point_map)))
        for i in range(NUM_ITERATIONS):
            random_points = np.random.choice(len(point_map), 4)
            remaining_points = list(set(points_range) - set(random_points))

            random_pairs = point_map[random_points]
            remaining_pairs = point_map[remaining_points]

            ic = np.array(random_pairs)[:, 2:]
            wc = np.array(random_pairs)[:, 0:2]

            # t_wc, normalized_wc = normalization(3, add_z_for_wc(wc * scale))
            # t_ic, normalized_ic = normalization(2, ic)
            # normalized_wc, t_wc = scale_and_translate_wc(np.c_[make_wc_with_ones(wc * scale), np.ones(len(wc))])
            # normalized_ic, t_ic = scale_and_translate_ic(np.c_[
            #         ic, np.ones(len(ic))
            #     ])

            # tt_wc = calm_before_the_storm(add_z_for_wc(wc * scale))
            # tt_ic = calm_before_the_storm(ic)
            #
            # tt_wc = calm_before_the_storm_1(add_z_for_wc(wc * scale))
            # tt_ic = calm_before_the_storm_1(ic)

            tt_wc = calm_before_the_storm(wc)
            tt_ic = calm_before_the_storm(ic)

            normalized_wcc = (tt_wc@np.c_[wc, np.ones(len(wc))].T).T
            normalized_icc = (tt_ic@np.c_[ic, np.ones(len(ic))].T).T
            if np.isinf(np.sum(normalized_icc)):
                print("here")
            # t_wc, normalized_wc = normalize_3d(add_z_for_wc(wc * scale))
            # t_ic, normalized_ic = normalize_2d(ic)
            approximation_normalized = homography_estimate(
                normalized_icc,
                normalized_wcc,
            )

            # approximation = approximation_normalized
            approximation = (np.linalg.inv(tt_ic)@approximation_normalized)@tt_wc

            approximation = approximation / approximation[-1, -1]

            if np.all(np.isnan(approximation) == False):
                wc = np.c_[
                    np.array(remaining_pairs)[:, 0:2],
                    np.ones(len(remaining_pairs)),
                ]
                ic = np.c_[
                    np.array(remaining_pairs)[:, 2:], np.ones(len(remaining_pairs))
                ]

                pe1 = projection_error(approximation, ic, wc)
                matched_pair = np.hstack(
                    [
                        wc[np.where(pe1 < 3)][:, 0:2],
                        ic[np.where(pe1 < 3)][:, 0:2],
                    ]
                )
                if len(matched_pair) > len(best_pairs):
                    best_pairs = matched_pair
                    best_projection = approximation
                    best_random_pairs = random_pairs
                    print(f"\t\t└──>ITERATION {i+1}, ERROR {np.mean(pe1)}")

                if len(best_pairs) > (len(point_map) * threshold):
                    break

    if best_pairs is not None and len(best_pairs) > 4:
        # bp =  np.vstack([np.array(list(best_pairs)), best_random_pairs])
        # t_wc, normalized_wc = normalize_3d(make_wc_with_ones(np.array(bp)[:, 0:2] * scale))
        # t_ic, normalised_ic = normalize_2d(np.array(bp)[:, 2:])
        # best_projection = projection_matrix_estimation(
        #     normalised_ic,
        #     normalized_wc,
        # )
        #
        # best_projection = np.dot(
        #     np.dot(np.linalg.pinv(t_ic), best_projection), t_wc
        # )
        # best_projection = best_projection / best_projection[-1, -1]

        bp = np.array(list(point_map))
        total_error = projection_error(
            projection_matrix=best_projection,
            ic=np.c_[np.array(bp)[:, 2:], np.ones(len(bp))],
            wc=np.c_[
                np.array(bp)[:, 0:2],
                np.ones(len(bp)),
            ],
        )
        print(f"\t└──>BEST INLIERS {len(best_pairs)}")
        print(f"\t\t└──>REFINED ERROR {np.mean(total_error)}")
    return best_projection, best_pairs


def project_origin(origin_frame, projection_matrix):
    origin_ic = project_wc_on_ic(
        wc=np.asarray([0, 0, 0, 1])[np.newaxis], projection_matrix=projection_matrix
    )[0]
    origin_frame[
        int(origin_ic[1]) : int(origin_ic[1]) + 15,
        int(origin_ic[0]) : int(origin_ic[0]) + 15,
        :,
    ] = [0, 255, 0]
    print(f"\t└──>ORIGIN FOUND AT {origin_ic}")
    return origin_frame


def project_matching_points(origin_frame, point_map, projection_matrix):
    print(f"\t└──>RE-PROJECTING MATCHING POINTS")
    wc = np.array(point_map)[:, 0:2]
    points = np.c_[
        np.c_[wc, np.zeros(len(wc))],
        np.ones(len(wc)),
    ]
    ic_pts = project_wc_on_ic(projection_matrix=projection_matrix, wc=points)
    img_pts = []
    for aa in ic_pts:
        img_pts.append([int(aa[0]), int(aa[1])])

    for pt in img_pts:
        origin_frame[pt[1] - 3 : pt[1] + 3, pt[0] - 3 : pt[0] + 3, :] = [0, 0, 255]

    return origin_frame


def project_cube(origin_frame, projection_matrix):
    print(f"\t└──>PROJECTING CUBE")

    points = (
        np.float32(
            [
                [0, 0, 0],
                [0, 60, 0],
                [60, 60, 0],
                [60, 0, 0],
                [0, 0, -6],
                [0, 60, -6],
                [60, 60, -6],
                [60, 0, -6],
            ]
        )
        + [180, 114, 0]
    )
    points = np.c_[np.array(points), np.ones(len(points))]

    ic_pts = project_wc_on_ic(projection_matrix=projection_matrix, wc=points)
    img_pts = []
    for aa in ic_pts:
        img_pts.append([int(aa[0]), int(aa[1])])

    img_pts = np.int32(np.array(img_pts)).reshape(-1, 2)
    print(f"\t\t└──>DRAWING CUBE")

    # draw ground floor in green
    origin_frame = cv2.drawContours(origin_frame, [img_pts[:4]], -1, (0, 255, 0), -3)
    # draw pillars in blue color
    for i, j in zip(range(4), range(4, 8)):
        origin_frame = cv2.line(
            origin_frame, tuple(img_pts[i]), tuple(img_pts[j]), (255, 0, 0), 3
        )
    # draw top layer in red color
    origin_frame = cv2.drawContours(origin_frame, [img_pts[4:]], -1, (0, 0, 255), 3)

    return origin_frame


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        my_batch = set(iterable[ndx : min(ndx + n, l)])
        yield list(my_batch), list(set(iterable) - my_batch)


def estimate_projection_matrix(
    model_image_key_points,
    model_image_desc,
    f_key_points,
    f_desc,
):
    # cv2.imwrite(r"C:\Users\Fuzail.Palnak\UHD\openSource\AR\task1\data\base_1.jpg", frame)
    # frame = cv2.imread(r"../data/base_1.jpg")
    # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    # FEATURE MATCHING
    matches = match_features(model_image_desc, f_desc)
    # matches = sorted(matches, key=lambda x: x.distance)
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

    # DLT
    pm, matched_pairs = homography_ransac(point_map)

    return pm, matched_pairs


def get_data_from_model_image():
    model_image = cv2.imread(MODEL_IMAGE)
    model_image = cv2.cvtColor(model_image, cv2.COLOR_BGR2GRAY)
    model_image_key_points, model_image_desc = find_features(model_image)
    return (
        model_image,
        model_image_key_points,
        model_image_desc,
    )


def render(img, matched_pairs, projection_matrix):
    rendered_frame = project_origin(img.copy(), projection_matrix=projection_matrix)
    rendered_frame = project_matching_points(
        rendered_frame, matched_pairs, projection_matrix
    )
    # rendered_frame = project_cube(
    #     rendered_frame,
    #     projection_matrix=projection_matrix
    # )

    return rendered_frame


def stream(pth: Union[str, int] = 0):
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
    (
        model_image,
        model_image_key_points,
        model_image_desc,
    ) = get_data_from_model_image()

    while cap.isOpened():
        print(f"└──>FRAME IN PROGRESS {fc+1}")

        _, frame = cap.read()
        # frame = cv2.imread(r"../data/test_source.png")
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        frame_rgb = frame.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # INTEREST POINT DETECTION
        f_key_points, f_desc = find_features(frame)

        # GET PROJECTION MATRIX
        homography_matrix, matched_pairs = estimate_projection_matrix(
            model_image_key_points,
            model_image_desc,
            f_key_points,
            f_desc,
        )
        pm = get_projection_matrix(WEBCAM_INTRINSIC, homography_matrix)
        # RENDER ORIGIN AND CUBE
        rendered_frame = (
            render(
                frame_rgb, matched_pairs, pm
            )
            if pm is not None
            else frame_rgb
        )

        h, w = model_image.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(
            -1, 1, 2
        )
        # project corners into frame
        dst = cv2.perspectiveTransform(pts, homography_matrix)
        rendered_frame = cv2.polylines(rendered_frame.copy(), [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

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

            pairs_img = draw_pairs(frame.copy(), list(matched_pairs), matched_pairs)
            cv2.imwrite(
                os.path.join(MP, f"{str(fc)}.png"),
                pairs_img,
            )

            rendered_frame = draw_key_points(
                model_image.copy(),
                rendered_frame.copy(),
                list(matched_pairs),
                pairs=matched_pairs,
            )
            cv2.imwrite(
                os.path.join(KP, f"{fc}_mapping.png"),
                rendered_frame,
            )
        fc += 1
        # imS = cv2.resize(origin_frame, (960, 540))
        cv2.imshow("img", rendered_frame)

        # writer.write(mapping_img)
        if cv2.waitKey(1) == 27:
            print("EXIT")
            cap.release()
            # writer.release()
            cv2.destroyAllWindows()


# run(0)
stream(r"../data/wetransfer_image00001-jpeg_2022-11-15_0719/IMG_3411.MOV")
# stream(r"../data/surface_demo.webm")
# stream(r"../data/wetransfer_2022-11-10-131213-jpg_2022-11-10_1222/s1.webm")
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