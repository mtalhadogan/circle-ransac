from __future__ import division
import abc
import numpy as np
import scipy.linalg as la
import scipy.spatial.distance as sd


class Feature(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def min_points(self):
        pass

    @abc.abstractmethod
    def points_distance(self, points):
        pass

    @abc.abstractmethod
    def print_feature(self, num_points):
        pass


class Circle(Feature):
    min_points = 3

    def __init__(self, points):
        r, cx, cy = self._algebraic_solve(points)
        self.radius = r
        self.xc = cx
        self.yc = cy

    def _algebraic_solve(self, pts):
        n = len(pts)
        one_col = np.ones((n, 1))
        xy = np.asarray(pts)
        lhs = np.hstack([xy, one_col])
        rhs_vec = -np.sum(xy * xy, axis=1)
        coef, _, _, _ = la.lstsq(lhs, rhs_vec, cond=None)
        a, b, c = coef
        center_x = -a / 2.0
        center_y = -b / 2.0
        rad_squared = center_x * center_x + center_y * center_y - c
        if rad_squared <= 0:
            raise RuntimeError("Circle fit produced non-positive radius squared.")
        return np.sqrt(rad_squared), center_x, center_y

    def points_distance(self, points):
        C = np.array([self.xc, self.yc])
        to_center = sd.cdist(points, C.reshape(1, -2), metric="euclidean").flatten()
        return np.abs(to_center - self.radius)

    def print_feature(self, num_points):
        t = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        out_x = self.xc + self.radius * np.cos(t)
        out_y = self.yc + self.radius * np.sin(t)
        return np.vstack((out_x, out_y))


class RansacFeature(object):
    def __init__(self, model_class, max_it=100, inliers_percent=0.6, threshold=100, inlier_threshold=10, seed=None):
        self.model_class = model_class
        self.max_it = max_it
        self.inliers_percent = inliers_percent
        self.threshold = threshold
        self.inlier_threshold = inlier_threshold
        self.seed = seed

    def _random_sample_indices(self, n_population, n_draw, gen):
        if n_draw <= n_population:
            return gen.choice(n_population, size=n_draw, replace=False)
        return gen.integers(0, n_population, size=n_draw)

    def _fraction_agreeing(self, model, pts):
        d = model.points_distance(pts)
        return np.sum(d <= self.inlier_threshold) / pts.shape[0]

    def _consensus_set(self, model, pts):
        d = model.points_distance(pts)
        return pts[d.ravel() <= self.inlier_threshold]

    def _run_consensus_loop(self, pts):
        gen = np.random.default_rng(self.seed)
        N = pts.shape[0]
        k = self.model_class.min_points
        winning = None
        winning_frac = 0.0

        for _ in range(self.max_it):
            sel = self._random_sample_indices(N, k, gen)
            batch = pts[sel]
            try:
                trial = self.model_class(batch)
            except RuntimeError:
                continue
            frac = self._fraction_agreeing(trial, pts)
            if frac > winning_frac:
                winning_frac = frac
                winning = trial
            if winning_frac >= self.inliers_percent:
                break

        if winning is not None and winning_frac > 0:
            agreeing = self._consensus_set(winning, pts)
            if agreeing.shape[0] >= k:
                try:
                    winning = self.model_class(agreeing)
                except RuntimeError:
                    pass
        return winning, winning_frac

    def detect_feature(self, points):
        return self._run_consensus_loop(points)

    def image_search(self, image):
        nonzero = np.argwhere(image > 0)
        if nonzero.size == 0:
            raise ValueError(
                "Image has no non-zero pixels. Check threshold or use an edge image."
            )
        pts = nonzero[:, :2]
        return self._run_consensus_loop(pts)
