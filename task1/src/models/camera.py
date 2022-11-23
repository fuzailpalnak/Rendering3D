import math
from dataclasses import dataclass

import cv2
import numpy as np

from task1.src.ops import find_features, match_features

NUM_ITERATIONS = 2924


@dataclass
class PoseOutput:
    projection_matrix: np.ndarray
    matched_pairs: np.ndarray
    matched_pairs_px: np.ndarray = None


class Pose:
    def __init__(self, reference):
        self._reference = reference
        self._m_key_points, self._m_desc = self._reference.features()

    @property
    def reference(self):
        return self._reference

    def projection_error(self, projection_matrix, ic, wc):
        raise NotImplementedError

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

    def map_reference_frame_ic_xy_to_wc_xy(self, wc_in_px):
        raise NotImplementedError

    @staticmethod
    def get_point_map(wc, ic):
        return np.hstack([wc, ic])

    def get_wc_ic_from_map(self, point_map):
        raise NotImplementedError

    def map_reference_frame_wc_xy_to_ic_xy(self, wc):
        raise NotImplementedError

    @staticmethod
    def to_homogenous(pts):
        return np.c_[
            pts,
            np.ones(len(pts)),
        ]

    @staticmethod
    def _project_wc_on_ic(projection_matrix, wc):
        projected_points = projection_matrix @ wc.T
        projected_points = (
            projected_points
            / projected_points[
                -1:,
            ]
        )
        return projected_points.T

    def project_origin(self, projection_matrix):
        origin_ic = self._project_wc_on_ic(
            wc=self._reference.origin, projection_matrix=projection_matrix
        )[0]
        print(f"\t└──>ORIGIN FOUND AT {abs(origin_ic)}")
        return origin_ic

    def project_matching_points(self, wc, projection_matrix):
        print(f"\t└──>RE-PROJECTING MATCHING POINTS")
        points = self.to_homogenous(wc)
        ic_pts = self._project_wc_on_ic(projection_matrix=projection_matrix, wc=points)
        return ic_pts

    def project_cube(self, projection_matrix):
        print(f"\t└──>PROJECTING CUBE")

        wc_cube = self._reference.cube
        points = self.to_homogenous(wc_cube)

        ic_pts = self._project_wc_on_ic(projection_matrix=projection_matrix, wc=points)
        return ic_pts

    def ransac(self, wc, ic, threshold=0.7):
        raise NotImplementedError

    def pose(self, wc, ic, **kwargs):
        raise NotImplementedError

    def create_linear_eqn(self, wc, ic):
        raise NotImplementedError

    def solve(self, a):
        raise NotImplementedError

    def run(self, frame, **kwargs) -> PoseOutput:
        print(f"└──>ESTIMATING POSE")
        f_key_points, f_desc = find_features(frame)

        # Matches
        matches = self.get_matches(m_desc=self._m_desc, f_desc=f_desc)

        # GET MAP COORDINATES
        wc, ic = self.get_match_coordinates(
            matches, m_key_points=self._m_key_points, f_key_points=f_key_points
        )

        return self.pose(wc=wc, ic=ic, **kwargs)


class DLT(Pose):
    def __init__(self, reference):
        super().__init__(reference)

    def map_reference_frame_wc_xy_to_ic_xy(self, wc):
        pass

    @staticmethod
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

    def map_reference_frame_ic_xy_to_wc_xy(self, wc_in_px):
        wc = self._reference.xy_from_px_to_wc(wc_in_px)
        return wc

    def get_wc_ic_from_map(self, point_map):
        assert point_map.shape[-1] == 5, "MUST HAVE 5 dim"
        return point_map[:, 0:3], point_map[:, 3:]

    def projection_error(self, projection_matrix, ic, wc):
        projected_points = self._project_wc_on_ic(projection_matrix, wc)
        # return np.abs(projected_points[:, 0] - ic[:, 0]) + np.abs(
        #     projected_points[:, 1] - ic[:, 1]
        # )
        return np.linalg.norm(ic - np.abs(projected_points), axis=-1)

    def ransac(self, wc, ic, threshold=0.7):
        best_pairs = set()
        best_projection = None

        point_map = self.get_point_map(wc, ic)

        if len(point_map) >= 6:
            points_range = list(range(len(point_map)))
            for i in range(NUM_ITERATIONS):
                random_points = np.random.choice(len(point_map), 6)
                remaining_points = list(set(points_range) - set(random_points))

                random_pairs = point_map[random_points]
                remaining_pairs = point_map[remaining_points]

                wc, ic = self.get_wc_ic_from_map(random_pairs)

                tt_wc = self.calm_before_the_storm(wc)
                tt_ic = self.calm_before_the_storm(ic)

                # #
                # # tt_wc = calm_before_the_storm_1(wc)
                # # tt_ic = calm_before_the_storm_1(ic)
                #
                # # tt_wc = calm_before_the_storm_2(wc)
                # # tt_ic = calm_before_the_storm_2(ic)
                # #
                normalized_wcc = (tt_wc @ np.c_[wc, np.ones(len(wc))].T).T
                normalized_icc = (tt_ic @ np.c_[ic, np.ones(len(ic))].T).T
                #
                # # t_wc, normalized_wc = normalize_3d(add_z_for_wc(wc * scale))
                # # t_ic, normalized_ic = normalize_2d(ic)

                approximation_normalized = self.solve(
                    self.create_linear_eqn(
                        wc=normalized_wcc,
                        ic=normalized_icc,
                    )
                )
                #
                approximation = (
                    np.linalg.inv(tt_ic) @ approximation_normalized
                ) @ tt_wc
                approximation = approximation / approximation[-1, -1]

                # approximation = self.estimate(
                #     wc,
                #     ic,
                # )

                if np.all(np.isnan(approximation) == False):
                    wc, ic = self.get_wc_ic_from_map(np.array(remaining_pairs))

                    wc = self.to_homogenous(wc)
                    ic = self.to_homogenous(ic)

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
                        print(f"\t\t└──>ITERATION {i + 1}, ERROR {np.mean(pe1)}")

                    if len(best_pairs) > (len(point_map) * threshold):
                        break

        if best_pairs is not None and len(best_pairs) > 6:
            bp = np.array(list(point_map))
            wc, ic = self.get_wc_ic_from_map(np.array(bp))

            total_error = self.projection_error(
                projection_matrix=best_projection,
                ic=self.to_homogenous(ic),
                wc=self.to_homogenous(wc),
            )
            print(f"\t└──>BEST INLIERS {len(best_pairs)}")
            print(f"\t\t└──>REFINED ERROR {np.mean(total_error)}")

        return best_projection, best_pairs

    def pose(self, wc, ic, **kwargs):
        # MAP TO WC
        wcz = self._reference.model_z_coordinate(wc)

        # DLT
        pm, matched_pairs = self.ransac(wc=wcz, ic=ic)

        _wcz, _ = self.get_wc_ic_from_map(matched_pairs)
        _matched_wx_px_idx = list()
        _matched_ic_px_idx = list()
        for i in _wcz:
            xyz = np.where(i[np.newaxis] == wcz)
            x = xyz[0][0:2]
            y = xyz[1][0:2]
            _matched_wx_px_idx.append(wc[x, y])
            _matched_ic_px_idx.append(ic[x, y])

        return PoseOutput(
            projection_matrix=pm,
            matched_pairs=matched_pairs,
            matched_pairs_px=np.hstack([_matched_wx_px_idx, _matched_ic_px_idx]),
        )

    def create_linear_eqn(self, wc, ic):
        n = wc.shape[0]
        A = list()
        for i in range(n):
            x, y, z = wc[i, 0], wc[i, 1], wc[i, 2]
            u, v = ic[i, 0], ic[i, 1]
            A.append([x, y, z, 1, 0, 0, 0, 0, -u * x, -u * y, -u * z, -u])
            A.append([0, 0, 0, 0, x, y, z, 1, -v * x, -v * y, -v * z, -v])

        return np.asarray(A)

    def solve(self, a):
        U, D, V = np.linalg.svd(a)
        v_simplified = V[D != 0]
        P = v_simplified[-1, :]
        P = np.reshape(P, (3, 4))
        P = P / P[2, 3]

        return P


class Homography(Pose):
    def __init__(self, reference):
        super().__init__(reference)

    @staticmethod
    def estimate_projection_matrix(camera_parameters, homography):
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
        rot_1 = np.dot(
            c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2)
        )
        rot_2 = np.dot(
            c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2)
        )
        rot_3 = np.cross(rot_1, rot_2)

        # finally, compute the 3D projection matrix from the model to the current frame
        projection = np.stack((rot_1, rot_2, rot_3, translation)).T

        return np.dot(camera_parameters, projection)

    def map_reference_frame_wc_xy_to_ic_xy(self, wc):
        pass

    @staticmethod
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

    def map_reference_frame_ic_xy_to_wc_xy(self, wc_in_px):
        wc = self._reference.xy_from_px_to_wc(wc_in_px)
        return wc

    def get_wc_ic_from_map(self, point_map):
        assert point_map.shape[-1] == 4, "MUST HAVE 4 dim"
        return point_map[:, 0:2], point_map[:, 2:]

    def projection_error(self, projection_matrix, ic, wc):
        projected_points = self._project_wc_on_ic(projection_matrix, wc)
        # return np.abs(projected_points[:, 0] - ic[:, 0]) + np.abs(
        #     projected_points[:, 1] - ic[:, 1]
        # )
        return np.linalg.norm(ic - np.abs(projected_points), axis=-1)

    def ransac(self, wc, ic, threshold=0.7):
        best_pairs = set()
        best_projection = None

        point_map = self.get_point_map(wc, ic)

        if len(point_map) >= 4:
            points_range = list(range(len(point_map)))
            for i in range(NUM_ITERATIONS):
                random_points = np.random.choice(len(point_map), 4)
                remaining_points = list(set(points_range) - set(random_points))

                random_pairs = point_map[random_points]
                remaining_pairs = point_map[remaining_points]

                wc, ic = self.get_wc_ic_from_map(random_pairs)

                tt_wc = self.calm_before_the_storm(wc)
                tt_ic = self.calm_before_the_storm(ic)

                # #
                # # tt_wc = calm_before_the_storm_1(wc)
                # # tt_ic = calm_before_the_storm_1(ic)
                #
                # # tt_wc = calm_before_the_storm_2(wc)
                # # tt_ic = calm_before_the_storm_2(ic)
                # #
                normalized_wcc = (tt_wc @ np.c_[wc, np.ones(len(wc))].T).T
                normalized_icc = (tt_ic @ np.c_[ic, np.ones(len(ic))].T).T
                #
                # # t_wc, normalized_wc = normalize_3d(add_z_for_wc(wc * scale))
                # # t_ic, normalized_ic = normalize_2d(ic)

                approximation_normalized = self.solve(
                    self.create_linear_eqn(
                        wc=normalized_wcc,
                        ic=normalized_icc,
                    )
                )
                #
                approximation = (
                    np.linalg.inv(tt_ic) @ approximation_normalized
                ) @ tt_wc
                approximation = approximation / approximation[-1, -1]

                # approximation = self.estimate(
                #     wc,
                #     ic,
                # )

                if np.all(np.isnan(approximation) == False):
                    wc, ic = self.get_wc_ic_from_map(np.array(remaining_pairs))

                    wc = self.to_homogenous(wc)
                    ic = self.to_homogenous(ic)

                    pe1 = self.projection_error(approximation, ic, wc)
                    matched_pair = np.hstack(
                        [
                            wc[np.where(pe1 < 5)][:, 0:2],
                            ic[np.where(pe1 < 5)][:, 0:2],
                        ]
                    )
                    if len(matched_pair) > len(best_pairs):
                        best_pairs = matched_pair
                        best_projection = approximation
                        print(f"\t\t└──>ITERATION {i + 1}, ERROR {np.mean(pe1)}")

                    if len(best_pairs) > (len(point_map) * threshold):
                        break

        if best_pairs is not None and len(best_pairs) > 4:
            bp = np.array(list(point_map))
            wc, ic = self.get_wc_ic_from_map(np.array(bp))

            total_error = self.projection_error(
                projection_matrix=best_projection,
                ic=self.to_homogenous(ic),
                wc=self.to_homogenous(wc),
            )
            print(f"\t└──>BEST INLIERS {len(best_pairs)}")
            print(f"\t\t└──>REFINED ERROR {np.mean(total_error)}")

        return best_projection, best_pairs

    def pose(self, wc, ic, **kwargs):
        # MAP TO WC
        wcz = self._reference.model_z_coordinate(wc)

        # DLT
        homography_estimate, matched_pairs = self.ransac(wc=wcz, ic=ic)
        pm = self.estimate_projection_matrix(
            kwargs["camera_parameters"], homography_estimate
        )

        return PoseOutput(
            projection_matrix=pm,
            matched_pairs=matched_pairs,
            matched_pairs_px=matched_pairs,
        )

    def create_linear_eqn(self, wc, ic):
        n = wc.shape[0]
        A = list()
        for i in range(n):
            x, y = wc[i, 0], wc[i, 1]
            u, v = ic[i, 0], ic[i, 1]
            A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
            A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])

        return np.asarray(A)

    def solve(self, a):
        U, D, V = np.linalg.svd(a)
        P = V[-1, :]
        P = np.reshape(P, (3, 3))
        P = (1 / P.item(8)) * P

        return P
