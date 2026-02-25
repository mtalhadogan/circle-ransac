from __future__ import division
import numpy as np


class RansacFeature(object):
    def __init__(self, model_class, max_it=100, inliers_percent=0.6, threshold=100, inlier_threshold=10, seed=None):
        self.model_class = model_class
        self.max_it = max_it
        self.inliers_percent = inliers_percent
        self.threshold = threshold
        self.inlier_threshold = inlier_threshold
        self.seed = seed

    def _fit_from_points(self, points):
        n_total = points.shape[0]
        sample_size = self.model_class.min_points
        rng = np.random.default_rng(self.seed)

        best_model = None
        best_ratio = 0.0

        for _ in range(self.max_it):
            if sample_size <= n_total:
                indices = rng.choice(n_total, size=sample_size, replace=False)
            else:
                indices = rng.integers(0, n_total, size=sample_size)
            sample = points[indices]

            try:
                candidate = self.model_class(sample)
            except RuntimeError:
                continue

            dist = candidate.points_distance(points)
            mask = dist.ravel() <= self.inlier_threshold
            ratio = mask.sum() / n_total

            if ratio > best_ratio:
                best_ratio = ratio
                best_model = candidate

            if best_ratio > self.inliers_percent:
                break

        if best_model is not None and best_ratio > 0:
            best_model = self._refit_on_inliers(best_model, points)

        return best_model, best_ratio

    def _refit_on_inliers(self, model, points):
        dist = model.points_distance(points)
        mask = dist.ravel() <= self.inlier_threshold
        inliers = points[mask]
        if inliers.shape[0] < self.model_class.min_points:
            return model
        try:
            return self.model_class(inliers)
        except RuntimeError:
            return model

    def detect_feature(self, points):
        """points: (n, 2) each row (row, col). Returns (model, inlier_fraction)."""
        return self._fit_from_points(points)

    def image_search(self, image):
        """Uses non-zero pixels as points. Returns (model, inlier_fraction)."""
        rows, cols = np.where(image > 0)
        if rows.size == 0:
            raise ValueError(
                "Image has no non-zero pixels. Check threshold or use an edge image."
            )
        points = np.column_stack((rows, cols))
        return self._fit_from_points(points)
