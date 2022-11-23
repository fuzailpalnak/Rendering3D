import math
import os
from typing import Union

import cv2
import numpy as np

# from task1.src.dlt import RANSAC
from matplotlib import pyplot as plt

from task1.src.util import draw_pairs, draw_key_points
from task1.src.ops import find_features, match_features

# MODEL_IMAGE = r"/home/palnak/base.jpg"
# MODEL_IMAGE = r"/home/palnak/sw1.jpg"
# MODEL_IMAGE = r"../data/surface_test.jpg"
# MODEL_IMAGE = r"/home/palnak/2022-11-10-131656.jpg"
# MODEL_IMAGE = r"../data/wetransfer_2022-11-10-131213-jpg_2022-11-10_1222/s1.jpg"
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


def calm_before_the_storm(x):
    d = x.shape[-1]
    m = x.mean(0)
    s = 1 / (x.std() * (1 / np.sqrt(2)))

    if d == 2:
        tr = np.array([[s, 0, -s * m[0]], [0, s, -s * m[1]], [0, 0, 1]])
    else:
        tr = np.array(
            [
                [s, 0, 0, -s * m[0]],
                [0, s, 0, -s * m[1]],
                [0, 0, s, -s * m[2]],
                [0, 0, 0, 1],
            ]
        )

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
        tr = np.array(
            [
                [scale, 0, 0, -scale * center[0]],
                [0, scale, 0, -scale * center[1]],
                [0, 0, scale, -scale * center[2]],
                [0, 0, 0, 1],
            ]
        )
    else:
        center = x.mean(0)
        x_c = x[:, 0:1] - center[0]
        y_c = x[:, 1:2] - center[1]

        dist = np.sqrt(np.power(x_c, 2) + np.power(y_c, 2))
        scale = np.sqrt(2) / dist.mean()
        tr = np.array(
            [[scale, 0, -scale * center[0]], [0, scale, -scale * center[1]], [0, 0, 1]]
        )

    return tr


def project_wc_on_ic(projection_matrix, wc):
    projected_points = projection_matrix @ wc.T
    projected_points = (
        projected_points
        / projected_points[
            -1:,
        ]
    )
    return projected_points.T


class DLT:
    def estimate(self, wc, ic):
        raise NotImplementedError

    def solve(self, wc, ic):
        raise NotImplementedError

    @staticmethod
    def projection_error(projection_matrix, ic, wc):
        projected_points = project_wc_on_ic(projection_matrix, wc)
        # return np.abs(projected_points[:, 0] - ic[:, 0]) + np.abs(
        #     projected_points[:, 1] - ic[:, 1]
        # )
        return np.linalg.norm(ic - np.abs(projected_points), axis=-1)


class Reference:
    def __init__(self):
        self._rgb = cv2.imread(MODEL_IMAGE)
        self._image = cv2.cvtColor(self._rgb, cv2.COLOR_BGR2GRAY)

        self._x_cm = 0.122 * 100  # m # 12.2 # cm  # width
        self._y_cm = 0.195 * 100  # m # 19.5  # cm # height

        self._cylinder_radius = 0.045 * 100  # m  # 4.5  # cm

    @property
    def origin(self):
        return np.array([0, 0, 0, 1])[np.newaxis]

    @property
    def image(self):
        return self._image

    @property
    def rgb(self):
        return self._rgb

    @property
    def px_dim(self):
        h, w = self._image.shape[0:2]
        return w, h

    @property
    def wc_dim(self):
        return self._x_cm, self._y_cm

    @property
    def px_to_wc_scale(self):
        scale = np.array(self.wc_dim) / np.array(self.px_dim)
        return scale

    @property
    def cube(self):
        print(f"\t└──>CUBE COORDINATES")

        points_ic = (
            np.float32(
                [
                    [0, 0, 0],
                    [0, 3, 0],
                    [3, 3, 0],
                    [3, 0, 0],
                    [0, 0, -3],
                    [0, 3, -3],
                    [3, 3, -3],
                    [3, 0, -3],
                ]
            )
            + [25, 10, 0]
        )

        # points_with_z = self.model_cylindrical_z(points_ic) * [1, 1, -1]
        # points_with_z_zero = np.c_[points_with_z[:, 0:2], np.zeros(4)]
        #
        # points = np.vstack([points_with_z_zero, points_with_z])
        points = points_ic
        return points

    def xy_from_px_to_wc(self, wc_px: np.ndarray):
        return wc_px * self.px_to_wc_scale

    def xy_from_wc_to_px(self, wc: np.ndarray):
        return wc / self.px_to_wc_scale

    def model_cylindrical_z(self, wc_px: np.ndarray):
        _wc_xy = self.xy_from_px_to_wc(wc_px)

        _x = _wc_xy[:, 0:1]
        _y = _wc_xy[:, 1:2]

        _theta_x = _x / self._cylinder_radius
        _cx = np.sin(-_theta_x) * (-self._cylinder_radius)
        _cy = _y
        _cz = np.cos(-_theta_x) * (-self._cylinder_radius) + self._cylinder_radius

        return np.hstack([_cx, _cy, _cz])

    def features(self):
        _key_points, _desc = find_features(self._image)
        return _key_points, _desc


class Pose:
    def __init__(self, model_image: Reference):
        self._reference = model_image
        self._fc = 1

        self._m_key_points, self._m_desc = self._reference.features()

    def projection_error(self, projection_matrix, ic, wc):
        projected_points = self.project_wc_on_ic(projection_matrix, wc)
        # return np.abs(projected_points[:, 0] - ic[:, 0]) + np.abs(
        #     projected_points[:, 1] - ic[:, 1]
        # )
        return np.linalg.norm(ic - projected_points, axis=-1)

    @staticmethod
    def get_match_coordinates(matches, m_key_points, f_key_points):
        point_map = np.array(
            [
                [
                    m_key_points[match.queryIdx].pt[0],
                    m_key_points[match.queryIdx].pt[1],
                    f_key_points[match.trainIdx].pt[0],
                    f_key_points[match.trainIdx].pt[1],
                ]
                for match in matches
            ]
        )
        return point_map[:, 0:2], point_map[:, 2:]

    @staticmethod
    def get_matches(m_desc, f_desc):
        # FEATURE MATCHING
        matches = match_features(m_desc, f_desc)
        # matches = sorted(matches, key=lambda x: x.distance)
        return matches

    def map_reference_frame_to_wxc_in_point_map(self, wc):
        wc = self._reference.xy_from_px_to_wc(wc)
        return wc

    @staticmethod
    def get_point_map(wc, ic):
        return np.hstack([wc, ic])

    @staticmethod
    def get_wc_ic_from_point_map(point_map):
        assert point_map.shape[-1] == 5, "MUST HAVE 5 dim"
        return point_map[:, 0:3], point_map[:, 3:]

    def map_reference_frame_to_ic_in_point_map(self, wc):
        wc = self._reference.xy_from_wc_to_px(wc)
        return wc

    @staticmethod
    def project_wc_on_ic(projection_matrix, wc):
        projected_points = projection_matrix @ wc.T
        projected_points = (
            projected_points
            / projected_points[
                -1:,
            ]
        )
        return projected_points.T

    @staticmethod
    def dlt_estimate(img_pts, world_pts):
        n = world_pts.shape[0]
        A = list()
        for i in range(n):
            x, y, z = world_pts[i, 0], world_pts[i, 1], world_pts[i, 2]
            u, v = img_pts[i, 0], img_pts[i, 1]
            A.append([x, y, z, 1, 0, 0, 0, 0, -u * x, -u * y, -u * z, -u])
            A.append([0, 0, 0, 0, x, y, z, 1, -v * x, -v * y, -v * z, -v])

        U, D, V = np.linalg.svd(np.asarray(A))
        v_simplified = V[D != 0]
        P = v_simplified[-1, :]
        P = np.reshape(P, (3, 4))
        P = P / P[2, 3]

        return P

    def project_origin(self, frame, projection_matrix):
        origin_ic = self.project_wc_on_ic(
            wc=self._reference.origin, projection_matrix=projection_matrix
        )[0]

        frame[
            int(origin_ic[0]) : int(origin_ic[0]) + 15,
            int(origin_ic[1]) : int(origin_ic[1]) + 15,
            :,
        ] = [0, 255, 0]
        print(f"\t└──>ORIGIN FOUND AT {origin_ic}")
        return frame, origin_ic

    def project_matching_points(self, frame, point_map, projection_matrix):
        print(f"\t└──>RE-PROJECTING MATCHING POINTS")

        wc, _ = self.get_wc_ic_from_point_map(point_map)

        points = np.c_[
            wc,
            np.ones(len(point_map)),
        ]
        ic_pts = self.project_wc_on_ic(projection_matrix=projection_matrix, wc=points)
        img_pts = []
        for aa in ic_pts:
            img_pts.append([int(aa[0]), int(aa[1])])

        for pt in img_pts:
            frame[pt[1] - 3 : pt[1] + 3, pt[0] - 3 : pt[0] + 3, :] = [0, 0, 255]

        return frame, ic_pts

    def project_cube(self, frame, projection_matrix):
        print(f"\t└──>PROJECTING CUBE")

        wc_cube = self._reference.cube
        points = np.c_[np.array(wc_cube), np.ones(len(wc_cube))]

        ic_pts = self.project_wc_on_ic(projection_matrix=projection_matrix, wc=points)
        img_pts = []
        for aa in ic_pts:
            img_pts.append([int(aa[0]), int(aa[1])])

        img_pts = np.int32(np.array(img_pts)).reshape(-1, 2)
        print(f"\t\t└──>DRAWING CUBE")

        # draw ground floor in green
        frame = cv2.drawContours(frame, [img_pts[:4]], -1, (0, 255, 0), -3)
        # draw pillars in blue color
        for i, j in zip(range(4), range(4, 8)):
            frame = cv2.line(
                frame, tuple(img_pts[i]), tuple(img_pts[j]), (255, 0, 0), 3
            )
        # draw top layer in red color
        frame = cv2.drawContours(frame, [img_pts[4:]], -1, (0, 0, 255), 3)

        return frame, ic_pts

    def dlt_ransac(self, point_map, threshold=0.7):
        best_pairs = set()
        best_projection = None
        best_random_pairs = set()

        if len(point_map) >= 6:
            points_range = list(range(len(point_map)))
            for i in range(NUM_ITERATIONS):
                random_points = np.random.choice(len(point_map), 6)
                remaining_points = list(set(points_range) - set(random_points))

                random_pairs = point_map[random_points]
                remaining_pairs = point_map[remaining_points]

                wc, ic = self.get_wc_ic_from_point_map(random_pairs)

                # t_wc, normalized_wc = normalization(3, add_z_for_wc(wc * scale))
                # t_ic, normalized_ic = normalization(2, ic)
                # normalized_wc, t_wc = scale_and_translate_wc(np.c_[make_wc_with_ones(wc * scale), np.ones(len(wc))])
                # normalized_ic, t_ic = scale_and_translate_ic(np.c_[
                #         ic, np.ones(len(ic))
                #     ])

                # tt_wc = calm_before_the_storm(wc)
                # tt_ic = calm_before_the_storm(ic)
                # #
                # # tt_wc = calm_before_the_storm_1(wc)
                # # tt_ic = calm_before_the_storm_1(ic)
                #
                # # tt_wc = calm_before_the_storm_2(wc)
                # # tt_ic = calm_before_the_storm_2(ic)
                # #
                # normalized_wcc = (tt_wc @ np.c_[wc, np.ones(len(wc))].T).T
                # normalized_icc = (tt_ic @ np.c_[ic, np.ones(len(ic))].T).T
                #
                # # t_wc, normalized_wc = normalize_3d(add_z_for_wc(wc * scale))
                # # t_ic, normalized_ic = normalize_2d(ic)
                # approximation_normalized = self.dlt_estimate(
                #     normalized_icc,
                #     normalized_wcc,
                # )
                #
                # # approximation = approximation_normalized
                # approximation = (np.linalg.inv(tt_ic) @ approximation_normalized) @ tt_wc
                # approximation = approximation / approximation[-1, -1]

                approximation = self.dlt_estimate(
                    ic,
                    wc,
                )

                if np.all(np.isnan(approximation) == False):
                    wc, ic = self.get_wc_ic_from_point_map(np.array(remaining_pairs))

                    wc = np.c_[
                        wc,
                        np.ones(len(remaining_pairs)),
                    ]
                    ic = np.c_[ic, np.ones(len(remaining_pairs))]

                    pe1 = self.projection_error(approximation, ic, wc)
                    matched_pair = np.hstack(
                        [
                            wc[np.where(pe1 < 5)][:, 0:3],
                            ic[np.where(pe1 < 5)][:, 0:2],
                        ]
                    )
                    if len(matched_pair) > len(best_pairs):
                        best_pairs = matched_pair
                        best_projection = approximation
                        best_random_pairs = random_pairs
                        print(f"\t\t└──>ITERATION {i + 1}, ERROR {np.mean(pe1)}")

                    if len(best_pairs) > (len(point_map) * threshold):
                        break

        if best_pairs is not None and len(best_pairs) > 6:
            # bp = np.vstack([np.array(list(best_pairs)), best_random_pairs])
            #
            # wc, ic = self.get_wc_ic_from_point_map(bp)
            #
            # tt_wc = calm_before_the_storm(wc)
            # tt_ic = calm_before_the_storm(ic)
            # #
            # # tt_wc = calm_before_the_storm_1(wc)
            # # tt_ic = calm_before_the_storm_1(ic)
            #
            # # tt_wc = calm_before_the_storm_2(wc)
            # # tt_ic = calm_before_the_storm_2(ic)
            # #
            # normalized_wcc = (tt_wc @ np.c_[wc, np.ones(len(wc))].T).T
            # normalized_icc = (tt_ic @ np.c_[ic, np.ones(len(ic))].T).T
            #
            # # t_wc, normalized_wc = normalize_3d(add_z_for_wc(wc * scale))
            # # t_ic, normalized_ic = normalize_2d(ic)
            # approximation_normalized = projection_matrix_estimation(
            #     normalized_icc,
            #     normalized_wcc,
            # )
            #
            # # approximation = approximation_normalized
            # approximation = (np.linalg.inv(tt_ic) @ approximation_normalized) @ tt_wc
            # best_projection = approximation / approximation[-1, -1]

            bp = np.array(list(point_map))
            wc, ic = self.get_wc_ic_from_point_map(np.array(bp))

            total_error = self.projection_error(
                projection_matrix=best_projection,
                ic=np.c_[ic, np.ones(len(bp))],
                wc=np.c_[
                    wc,
                    np.ones(len(bp)),
                ],
            )
            print(f"\t└──>BEST INLIERS {len(best_pairs)}")
            print(f"\t\t└──>REFINED ERROR {np.mean(total_error)}")
        return best_projection, best_pairs

    def find_pose(self, frame):
        # INTEREST POINT DETECTION
        f_key_points, f_desc = find_features(frame)

        # Matches
        matches = self.get_matches(m_desc=self._m_desc, f_desc=f_desc)

        # GET MAP COORDINATES
        wc, ic = self.get_match_coordinates(
            matches, m_key_points=self._m_key_points, f_key_points=f_key_points
        )

        # MAP TO WC
        # wc = self.map_reference_frame_to_wxc_in_point_map(wc)
        wc = self._reference.model_cylindrical_z(wc)

        # POINT MAP
        point_map = self.get_point_map(wc, ic)

        # DLT
        pm, matched_pairs = self.dlt_ransac(point_map)

        return pm, matched_pairs


class Stream(Pose):
    def __init__(self, model_image: Reference):
        super().__init__(model_image)
        self._reference = model_image
        self._fc = 1

    def on_image(self, pth: str, t_it: int = 1):

        frame = cv2.imread(pth)
        frame_rgb = frame.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        for _ in range(t_it):
            pm, matched_pairs = self.find_pose(frame)

            # RENDER ORIGIN AND CUBE
            rendered_frame, oc = self.project_origin(
                frame_rgb.copy(), projection_matrix=pm
            )
            rendered_frame, mc = self.project_matching_points(
                rendered_frame, matched_pairs, projection_matrix=pm
            )
            rendered_frame, cc = self.project_cube(
                rendered_frame,
                projection_matrix=pm,
            )

            # wc, ic = self.get_wc_ic_from_point_map(matched_pairs)
            # wc = self.map_reference_frame_to_ic_in_point_map(wc)
            # matched_pairs = self.get_point_map(wc, ic)
            #
            # rendered_frame = draw_key_points(
            #     self._reference.image.copy(),
            #     rendered_frame.copy(),
            #     list(matched_pairs),
            #     pairs=matched_pairs,
            # )

            pp = np.hstack([mc[:, 0:2], matched_pairs[:, 3:]])
            rendered_frame = draw_key_points(
                self._reference.rgb.copy(),
                rendered_frame.copy(),
                list(pp),
                pairs=pp,
            )

            self._fc += 1
            # imS = cv2.resize(origin_frame, (960, 540))
            cv2.imshow("img", rendered_frame)

            # writer.write(mapping_img)
            if cv2.waitKey(3000) == 27:
                print("EXIT")
                cv2.destroyAllWindows()

    def stream(self, pth: Union[str, int]):
        m_key_points, m_desc = self._reference.features()

        cap = cv2.VideoCapture(pth)

        # Check if the webcam is opened correctly
        if not cap.isOpened():
            raise IOError("Cannot open THE provided")

        while cap.isOpened():
            print(f"└──>FRAME IN PROGRESS {self._fc}")

            _, frame = cap.read()
            # frame = cv2.imread(r"../data/base_1.jpg")
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            frame_rgb = frame.copy()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            pm, matched_pairs = self.find_pose(frame)

            # RENDER ORIGIN AND CUBE
            rendered_frame, _ = self.project_origin(
                frame_rgb.copy(), projection_matrix=pm
            )
            rendered_frame, ic = self.project_matching_points(
                rendered_frame, matched_pairs, projection_matrix=pm
            )
            rendered_frame, _ = self.project_cube(
                rendered_frame,
                projection_matrix=pm,
            )

            # wc, ic = self.get_wc_ic_from_point_map(matched_pairs)
            # wc = self.map_reference_frame_to_ic_in_point_map(wc)
            # matched_pairs = self.get_point_map(wc, ic)
            #
            # rendered_frame = draw_key_points(
            #     self._reference.image.copy(),
            #     rendered_frame.copy(),
            #     list(matched_pairs),
            #     pairs=matched_pairs,
            # )

            self._fc += 1
            # imS = cv2.resize(origin_frame, (960, 540))
            cv2.imshow("img", rendered_frame)

            # writer.write(mapping_img)
            if cv2.waitKey(1) == 27:
                print("EXIT")
                cap.release()
                # writer.release()
                cv2.destroyAllWindows()


def calm_before_the_storm_2(x):
    d = x.shape[-1]
    s = np.sqrt(2) / np.sqrt((abs(x - np.mean(x, axis=0)) ** 2)).sum(axis=-1).mean()
    m = np.mean(x, 0)
    if d == 3:
        tr = np.array([[s, 0, 0, m[0]], [0, s, 0, m[1]], [0, 0, s, m[2]], [0, 0, 0, 1]])
    else:
        tr = np.array([[s, 0, m[0]], [0, s, m[1]], [0, 0, 1]])
    return tr


st = Stream(Reference())

# run(0)
st.on_image(MODEL_IMAGE, 1000)
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
