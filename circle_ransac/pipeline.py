from __future__ import division
import os
import glob
from collections import namedtuple

import cv2
import numpy as np

from circle_ransac.ransac import RansacFeature, Circle
from circle_ransac import config as cfg

PipelineResult = namedtuple(
    "PipelineResult",
    [
        "input_path",
        "image",
        "edges",
        "image_with_circle",
        "circle",
        "inlier_percent",
        "center",
        "radius_px",
        "radius_mm",
        "diameter_mm",
    ],
)

IMAGE_EXTENSIONS = ("*.bmp", "*.jpeg", "*.jpg", "*.png")


def _resolve_input_path(given_path=None, fallback_dir="input/backlight"):
    if given_path and os.path.exists(given_path):
        return given_path
    found = []
    for ext in IMAGE_EXTENSIONS:
        found.extend(glob.glob(os.path.join(fallback_dir, ext)))
    found.sort()
    if not found:
        raise FileNotFoundError(
            f"No image found at {given_path!r} or in folder {fallback_dir!r}"
        )
    return found[0]


def _draw_circle_on_image(bgr_image, circle, n_contour_points=2000):
    out = bgr_image.copy()
    row_coords, col_coords = circle.print_feature(n_contour_points)
    contour = np.column_stack((col_coords, row_coords)).astype(np.int32)
    cv2.polylines(out, [contour], isClosed=True, color=(0, 0, 255), thickness=2)
    cv2.circle(out, (int(circle.yc), int(circle.xc)), 8, (0, 255, 0), -1)
    return out


def _default(value, config_name):
    return value if value is not None else getattr(cfg, config_name)


def run_pipeline(
    image_path=None,
    folder_fallback="input/backlight",
    canny_low=None,
    canny_high=None,
    canny_aperture=None,
    max_it=None,
    inliers_percent=None,
    dst=None,
    min_points=None,
    pixels_to_mm=None,
    draw_points=None,
    output_dir=None,
    save_edges=False,
):
    canny_low = _default(canny_low, "CANNY_LOW")
    canny_high = _default(canny_high, "CANNY_HIGH")
    canny_aperture = _default(canny_aperture, "CANNY_APERTURE")
    max_it = _default(max_it, "RANSAC_MAX_ITERATIONS")
    inliers_percent = _default(inliers_percent, "RANSAC_INLIER_PERCENT")
    dst = _default(dst, "RANSAC_DISTANCE_THRESHOLD")
    draw_points = _default(draw_points, "CIRCLE_DRAW_POINTS")
    pixels_to_mm = _default(pixels_to_mm, "PIXELS_TO_MM")
    seed = getattr(cfg, "RANSAC_SEED", None)

    Circle.min_points = min_points if min_points is not None else cfg.RANSAC_MIN_POINTS

    input_path = _resolve_input_path(image_path, folder_fallback)
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError(f"Could not read image: {input_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, canny_low, canny_high, apertureSize=canny_aperture)

    detector = RansacFeature(Circle, max_it=max_it, inliers_percent=inliers_percent, inlier_threshold=dst, seed=seed)
    circle, inlier_percent = detector.image_search(edges)

    image_with_circle = _draw_circle_on_image(image, circle, draw_points)
    center = (circle.xc, circle.yc)
    radius_px = circle.radius
    if pixels_to_mm is not None:
        radius_mm = radius_px * pixels_to_mm
        diameter_mm = 2 * radius_mm
    else:
        radius_mm = None
        diameter_mm = None

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(input_path))[0]
        cv2.imwrite(os.path.join(output_dir, f"{stem}_result.bmp"), image_with_circle)
        if save_edges:
            cv2.imwrite(os.path.join(output_dir, f"{stem}_edges.bmp"), edges)

    return PipelineResult(
        input_path=input_path,
        image=image,
        edges=edges,
        image_with_circle=image_with_circle,
        circle=circle,
        inlier_percent=inlier_percent,
        center=center,
        radius_px=radius_px,
        radius_mm=radius_mm,
        diameter_mm=diameter_mm,
    )
