"""Circle model and least-squares fit. See docs/ALGORITHM.md."""

from __future__ import division
import abc
import numpy as np
import scipy.linalg as linalg
import scipy.spatial.distance as dist


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
    """(x - xc)² + (y - yc)² = r². Fit from n≥3 points (exact if n=3)."""

    min_points = 3

    def __init__(self, points):
        self.radius, self.xc, self.yc = self._fit(points)

    def _fit(self, points):
        A = np.array([[x, y, 1] for x, y in points])
        rhs = np.array([-(x * x + y * y) for x, y in points])
        try:
            D, E, F = linalg.lstsq(A, rhs, cond=None)[0]
        except linalg.LinAlgError:
            raise RuntimeError(
                "Circle fit failed (singular or degenerate configuration)."
            )
        xc = -D / 2
        yc = -E / 2
        r_sq = xc * xc + yc * yc - F
        if r_sq <= 0:
            raise RuntimeError("Circle fit produced non-positive radius squared.")
        r = np.sqrt(r_sq)
        return (r, xc, yc)

    def points_distance(self, points):
        center = np.array([[self.xc, self.yc]])
        d = np.abs(dist.cdist(points, center, "euclidean") - self.radius)
        return d

    def print_feature(self, num_points):
        theta = np.linspace(0, 2 * np.pi, num_points)
        x = self.xc + self.radius * np.cos(theta)
        y = self.yc + self.radius * np.sin(theta)
        return np.vstack((x, y))
