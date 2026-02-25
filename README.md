# CircleRansac

**RANSAC-based circle detection** for images: robustly fit a circle to edge pixels and optionally convert to real-world units. This project explains and implements the **RANSAC circle fitting** method in detail.

---

## What is RANSAC?

**RANSAC** (Random Sample Consensus) is an algorithm for estimating a mathematical model from data that contain **outliers**—points that do not follow the model. Instead of fitting to all points (which can be ruined by noise or wrong data), RANSAC repeatedly:

1. **Samples** a subset of points at random.
2. **Fits** the model to that subset.
3. **Counts** how many of the *entire* dataset agree with the model (inliers).
4. **Keeps** the model with the most inliers.

So the solution is the model that best explains the data while ignoring outliers.

---

## Why use RANSAC for circle fitting?

In images, edge detectors (e.g. Canny) give many points along the circle but also:

- **Noise** and false edges,
- **Gaps** or partial occlusions,
- **Other contours** (other circles or lines).

A simple least-squares fit to *all* edge pixels can be pulled away by these outliers. RANSAC instead searches for the circle that has the **largest number of edge pixels close to it**, which is exactly the visible circle we want.

---

## How RANSAC circle fitting works here

### 1. Input: edge image

We start from a **grayscale image**, run **Canny edge detection**, and take all **non-zero pixels** as 2D points. These are the candidates the circle must explain.

### 2. Iteration loop

In each RANSAC iteration:

- **Sample** a **minimum** number of edge points at random (e.g. 3, without replacement — the minimum to define a circle). Each hypothesis uses distinct points; outliers do not affect the fit. This is the “minimum number of points the circle must agree with” in one trial.
- **Fit a circle** to these points (exact for 3 points; see [Algorithm details](docs/ALGORITHM.md)): we get center and radius.
- **Score**: For **all** edge pixels, compute the **distance to the circle**. Pixels within a small distance (e.g. 2 px) are **inliers**.
- **Update**: If the inlier fraction (inliers / total edge pixels) is better than before, keep this circle.

We stop when the inlier fraction reaches a target (e.g. 40%) or after a maximum number of iterations. The best model is then **refined** by fitting the circle again to all its inliers (least-squares) for better accuracy.

### 3. Output

- **Best circle**: center in (row, col) image coordinates, radius in pixels. When drawing, we use (col, row) for OpenCV.
- Optionally **diameter in mm** if a pixel-to-mm scale is given.

So the method’s “solution” is: **the circle that has the most edge pixels lying on (or very near) its circumference**, which is robust to outliers and clutter.

---

## Installation

```bash
git clone https://github.com/mtalhadogan/circle-ransac.git
cd circle-ransac
pip install -r requirements.txt
```

Requirements: Python 3.6+, NumPy, SciPy, OpenCV.

---

## Usage

### Command line

```bash
# Use an image file (or first image in input/backlight if not found)
python main.py image.bmp

# Save to a specific directory
python main.py image.bmp --output-dir output

# Save edges image too; set scale for diameter in mm
python main.py image.bmp -o output --save-edges --pixels-to-mm 0.02083
```

Run from the project root directory so `circle_ransac` can be imported.

### From Python

Run from the project root so `circle_ransac` is on the path:

```python
from circle_ransac.pipeline import run_pipeline

result = run_pipeline(
    image_path="image.bmp",
    output_dir="output",
    pixels_to_mm=170.66 / 8192,  # optional
)

print("Center:", result.center)
print("Radius (px):", result.radius_px)
print("Diameter (mm):", result.diameter_mm)
print("Inlier fraction:", result.inlier_percent)
# result.image_with_circle: BGR image with circle drawn
# result.edges: Canny edge image
```

---

## Project structure

```
CircleRansac/
├── circle_ransac/       # Main package
│   ├── __init__.py
│   ├── config.py        # Default parameters (Canny, RANSAC, scale)
│   ├── ransac.py        # RANSAC loop + Circle model (least-squares fit)
│   └── pipeline.py      # Load image → Canny → RANSAC → draw/save
├── main.py              # CLI entry point
├── docs/
│   └── ALGORITHM.md     # Detailed math and algorithm
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Algorithm details

For the **algebraic circle equation**, **least-squares fitting** from \(n\) points, and **distance from a point to the circle**, see:

**[docs/ALGORITHM.md](docs/ALGORITHM.md)** — step-by-step derivation and pseudocode.

---

## Configuration

Defaults are in `circle_ransac/config.py`. You can override when calling `run_pipeline()` or by editing the config:

- **Canny**: `CANNY_LOW`, `CANNY_HIGH`, `CANNY_APERTURE`
- **RANSAC**: `RANSAC_MAX_ITERATIONS`, `RANSAC_INLIER_PERCENT`, `RANSAC_DISTANCE_THRESHOLD`, `RANSAC_MIN_POINTS`
- **Scale**: `PIXELS_TO_MM` (optional, for diameter in mm)

---

## License

MIT — see [LICENSE](LICENSE).
