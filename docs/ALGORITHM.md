# RANSAC Circle Fitting — Algorithm

Step-by-step: how the circle is defined, fitted from points, and how RANSAC picks the best circle.

---

## Overview

```
  Input image          Edge pixels           RANSAC loop              Output
  ┌──────────┐         ┌──────────┐          ┌──────────────┐         ┌──────────┐
  │          │  Canny  │  · · ·   │  sample  │ 3 pts → circle│  best  │ center,  │
  │  (gray)  │ ──────► │  · · ·   │ ───────► │ score inliers │ ─────► │ radius   │
  │          │         │  · · ·   │  repeat  │ refine on in. │        │ (px/mm)  │
  └──────────┘         └──────────┘          └──────────────┘         └──────────┘
```

---

## 1. Circle equation

A circle is all points \((x, y)\) with

\[
(x - x_c)^2 + (y - y_c)^2 = r^2
\]

with center \((x_c, y_c)\) and radius \(r\). In code we use the **algebraic form**:

\[
D\,x + E\,y + F = -(x^2 + y^2)
\]

with \(D = -2x_c\), \(E = -2y_c\), \(F = x_c^2 + y_c^2 - r^2\). So:

\[
x_c = -\frac{D}{2},\quad y_c = -\frac{E}{2},\quad
r = \sqrt{x_c^2 + y_c^2 - F}.
\]

---

## 2. Fitting a circle from points

Given \(n\) points \((x_i, y_i)\), we want \(D, E, F\) so that each point (ideally) satisfies \(D x_i + E y_i + F = -(x_i^2 + y_i^2)\). In matrix form:

\[
A\,w = b,\quad
A = \begin{bmatrix} x_1 & y_1 & 1 \\ \vdots & \vdots & \vdots \\ x_n & y_n & 1 \end{bmatrix},\quad
w = \begin{bmatrix} D \\ E \\ F \end{bmatrix},\quad
b = -\begin{bmatrix} x_1^2+y_1^2 \\ \vdots \\ x_n^2+y_n^2 \end{bmatrix}.
\]

- **n = 3** (and not collinear): unique circle through the 3 points.
- **n > 3**: we solve in **least-squares** sense (\(\min \|Aw - b\|^2\)).

So: build \(A\) and \(b\), solve with `scipy.linalg.lstsq`, then get \(x_c, y_c, r\) from \(D, E, F\).

---

## 3. Inliers: distance to the circle

A point is an **inlier** if it is close to the circle. The distance from point \(P = (x_i, y_i)\) to the circumference is:

\[
d = \left| \sqrt{(x_i - x_c)^2 + (y_i - y_c)^2} - r \right|.
\]

If \(d \le \tau\) (e.g. 2 pixels), the point counts as inlier.

---

## 4. RANSAC loop

| Step | Action |
|------|--------|
| 1 | Get all edge pixels (row, col) from the image. |
| 2 | **Repeat** (e.g. 3000 times or until good enough): |
| 2a | **Sample** 3 distinct points (without replacement). |
| 2b | **Fit** a circle through these 3 points. |
| 2c | **Score**: for all edge pixels, compute distance to this circle; count inliers. |
| 2d | If inlier fraction is best so far, **save** this circle. |
| 3 | **Refine**: take all inliers of the best circle, refit circle with least-squares. |
| 4 | **Output**: refined center and radius. |

So the result is the circle that **maximizes the number of edge pixels near it**, and is robust to noise and other edges.

---

## 5. Coordinates

- **Internal**: points and circle in **(row, col)** (image coordinates).
- **Drawing (OpenCV)**: use **(x, y) = (col, row)** for `cv2.circle` and polylines.

---

## 6. References

- Fischler & Bolles (1981), *Random sample consensus*, CACM.
- Algebraic circle fit: \(D, E, F\) least-squares formulation (standard).
