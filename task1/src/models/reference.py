import cv2
import numpy as np
from task1.src.ops import find_features


class Reference:
    def __init__(self, img_pth):
        self._rgb = cv2.imread(img_pth)
        self._image = cv2.cvtColor(self._rgb, cv2.COLOR_BGR2GRAY)

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
        raise NotImplementedError

    @property
    def px_to_wc_scale(self):
        scale = np.array(self.wc_dim) / np.array(self.px_dim)
        return scale

    @property
    def cube(self):
        raise NotImplementedError

    def xy_from_px_to_wc(self, wc_px: np.ndarray):
        return wc_px * self.px_to_wc_scale

    def xy_from_wc_to_px(self, wc: np.ndarray):
        return wc / self.px_to_wc_scale

    def model_z_coordinate(self, wc_px: np.ndarray):
        raise NotImplementedError

    def features(self):
        _key_points, _desc = find_features(self._image)
        return _key_points, _desc


class Reference3DCylindrical(Reference):
    def __init__(self, img_pth):
        super().__init__(img_pth)
        self._rgb = cv2.imread(img_pth)
        self._image = cv2.cvtColor(self._rgb, cv2.COLOR_BGR2GRAY)

        self._x_cm = 0.122 * 100  # m # 12.2 # cm  # width
        self._y_cm = 0.196 * 100  # m # 19.5  # cm # height

        self._cylinder_radius = 0.035 * 100  # m  # 4.5  # cm

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
        points_ic = (
            np.float32(
                [
                    [0, 0, 0],
                    [0, 3, 0],
                    [3, 3, 0],
                    [3, 0, 0],
                    [0, 0, 3],
                    [0, 3, 3],
                    [3, 3, 3],
                    [3, 0, 3],
                ]
            )
            + [5, 5, 5]
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

    def model_z_coordinate(self, wc_px: np.ndarray):
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


class Reference2D(Reference):
    def __init__(self, img_pth):
        super().__init__(img_pth)
        self._rgb = cv2.imread(img_pth)
        self._image = cv2.cvtColor(self._rgb, cv2.COLOR_BGR2GRAY)

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
        return None

    @property
    def px_to_wc_scale(self):
        return None

    @property
    def cube(self):
        points_ic = (
            np.float32(
                [
                    [0, 0, 0],
                    [0, 3, 0],
                    [3, 3, 0],
                    [3, 0, 0],
                    [0, 0, 3],
                    [0, 3, 3],
                    [3, 3, 3],
                    [3, 0, 3],
                ]
            )
            + [5, 5, 5]
        )

        # points_with_z = self.model_cylindrical_z(points_ic) * [1, 1, -1]
        # points_with_z_zero = np.c_[points_with_z[:, 0:2], np.zeros(4)]
        #
        # points = np.vstack([points_with_z_zero, points_with_z])
        points = points_ic
        return points

    def xy_from_px_to_wc(self, wc_px: np.ndarray):
        return wc_px

    def xy_from_wc_to_px(self, wc: np.ndarray):
        return wc

    def model_z_coordinate(self, wc_px: np.ndarray):
        return wc_px

    def features(self):
        _key_points, _desc = find_features(self._image)
        return _key_points, _desc
