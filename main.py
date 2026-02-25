#!/usr/bin/env python
"""CircleRansac CLI — RANSAC circle detection. See README.md and docs/ALGORITHM.md."""

from __future__ import division
import argparse
import sys

from circle_ransac.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Detect a circle in an image using RANSAC (Random Sample Consensus).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "image",
        nargs="?",
        default=None,
        help="Path to input image. If omitted, first image in --input-dir is used.",
    )
    parser.add_argument(
        "--input-dir",
        default="input/backlight",
        help="Fallback directory to search for images if IMAGE is not provided.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="output",
        help="Directory for result and optional edges image.",
    )
    parser.add_argument(
        "--save-edges",
        action="store_true",
        help="Save the Canny edges image to output directory.",
    )
    parser.add_argument(
        "--pixels-to-mm",
        type=float,
        default=None,
        help="Scale: diameter_mm = 2 * radius_px * PIXELS_TO_MM (e.g. 170.66/8192).",
    )
    args = parser.parse_args()

    try:
        result = run_pipeline(
            image_path=args.image,
            folder_fallback=args.input_dir,
            pixels_to_mm=args.pixels_to_mm,
            output_dir=args.output_dir,
            save_edges=args.save_edges,
        )
    except (FileNotFoundError, ValueError) as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    print("Input:", result.input_path)
    print("RANSAC inlier fraction: {:.4f}".format(result.inlier_percent))
    print("Center (xc, yc): ({:.2f}, {:.2f})".format(*result.center))
    print("Radius (px): {:.2f}".format(result.radius_px))
    if result.radius_mm is not None:
        print("Radius (mm): {:.4f}".format(result.radius_mm))
        print("Diameter (mm): {:.4f}".format(result.diameter_mm))

    return 0


if __name__ == "__main__":
    sys.exit(main())
