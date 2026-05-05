# Homography

Homography is a transformation matrix that maps points from one plane to another. In this project: **pixel coordinates → real-world court coordinates**.

---

## The Math

A point in the image $(x, y)$ maps to a real-world point $(X, Y)$ via a 3×3 matrix $H$:

$$\begin{pmatrix} wX \\ wY \\ w \end{pmatrix} = H \begin{pmatrix} x \\ y \\ 1 \end{pmatrix}$$

Where $H$ is:

$$H = \begin{pmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & h_{33} \end{pmatrix}$$

To get actual coordinates, divide by $w$:

$$X = \frac{h_{11}x + h_{12}y + h_{13}}{h_{31}x + h_{32}y + h_{33}}, \quad Y = \frac{h_{21}x + h_{22}y + h_{23}}{h_{31}x + h_{32}y + h_{33}}$$

---

## How H is Computed

You give OpenCV 4+ point correspondences — pixel positions of court keypoints paired with their known real-world positions on a standard court:

```python
# pixel positions of court keypoints (from annotations)
src = np.array([[px1,py1], [px2,py2], [px3,py3], [px4,py4]], dtype=np.float32)

# known real-world positions in meters on a standard court
dst = np.array([[0,0], [6.1,0], [6.1,13.4], [0,13.4]], dtype=np.float32)

H, _ = cv2.findHomography(src, dst)
```

OpenCV solves for the 9 values of $H$ (8 degrees of freedom since it's scale-invariant) using those 4+ correspondences.

---

## Where the Real-World Coordinates Come From

You don't measure the `dst` values — you look them up. A badminton court is a standardized object, so every regulation court has the same line positions in meters. The real-world coordinates are known by definition.

### Recipe

1. **Pick an origin.** Any consistent point works — commonly the back-left doubles corner, with X across the court (sideline-to-sideline) and Y down its length (baseline-to-baseline).
2. **Look up BWF court dimensions.** Width 6.1 m (doubles), length 13.4 m, short service line 1.98 m from net, long service line 0.76 m from baseline, etc.
3. **For each labeled keypoint, record its (X, Y) from your origin.**

| Landmark | Real-world (m) |
|----------|----------------|
| Back-left doubles corner | (0, 0) |
| Back-right doubles corner | (6.1, 0) |
| Front-right doubles corner | (6.1, 13.4) |
| Front-left doubles corner | (0, 13.4) |
| Net × left sideline | (0, 6.7) |
| Center × short service line (back) | (3.05, 4.72) |

So `dst` is a hard-coded constant — the BWF spec baked into your code. Only `src` (the detected pixel positions) changes per video.

### Keypoint identity matters

The detector must tell *which* keypoint each detection is — "back-left corner" vs "back-right corner" — otherwise you can't pair pixel positions with the right real-world coordinates. Two approaches:

- **Per-class keypoints:** one class per landmark (`kp_back_left`, `kp_back_right`, …). The detector itself disambiguates. Cleaner but more classes.
- **Single class + spatial sorting:** one `court_keypoint` class, disambiguate by image position (top-leftmost detection = back-left corner, etc.). Fewer classes, fragile if camera angle varies.

---

## At Inference Time

Once you have $H$, any detected pixel position (shuttle, player) gets transformed instantly:

```python
real_world_pt = cv2.perspectiveTransform(pixel_pt, H)
```

---

## Why It Works

A camera viewing a flat plane (the court) from any fixed angle is exactly the case where homography holds perfectly. The camera introduces perspective distortion (far end looks smaller), and $H$ undoes that distortion — that's why the far baseline looks compressed in the image but the math still gives correct meter distances.

This breaks if the camera moves or zooms mid-video, which is why fixed-camera broadcast footage is essential.

---

## Linear Algebra Concepts Used

### 1. Homogeneous Coordinates

Regular 2D points are $(x, y)$. To do perspective transforms with matrix multiplication, we add a third coordinate and write them as $(x, y, 1)$. The extra $1$ is a trick that lets translation and perspective be expressed as matrix multiplication instead of separate addition.

After multiplying by $H$ you get $(wX, wY, w)$. Dividing by $w$ recovers the real 2D point. When $w=1$ it's simple; when $w \neq 1$ (which happens with perspective) the division is what "undoes" the distortion.

**Important:** the 3-vector form is *not* a real 3D point. Both input $(x, y, 1)$ and output $(wX, wY, w)$ are 2D points wearing a "scale tag" — $(2, 3, 1)$, $(4, 6, 2)$, and $(10, 15, 5)$ all represent the same 2D point $(2, 3)$. Homography is a **plane → plane** mapping (court plane to image plane); there is no real 3D intermediate. The third coordinate is bookkeeping for the perspective divide, not a trip through 3D space.

**Why the divide is what makes perspective work:** if $H$ were affine (rotation/scale/translation only), the bottom row would be $(0, 0, 1)$ and $w$ would always equal $1$ — no division needed. But for perspective, $w = h_{31}x + h_{32}y + h_{33}$ depends on the input pixel, so different pixels get divided by different amounts. That input-dependent division is precisely what produces non-linear effects (far baseline compressed, parallel lines converging). Linear matrix multiplication alone can't do that.

### 2. Matrix-Vector Multiplication

$$H \begin{pmatrix} x \\ y \\ 1 \end{pmatrix}$$

This is a $3 \times 3$ matrix times a $3 \times 1$ column vector, producing a $3 \times 1$ result. Each output row is the dot product of that row of $H$ with the input vector:

$$\text{row 1} \cdot \mathbf{v} = h_{11}x + h_{12}y + h_{13}$$

That's where the numerator/denominator expressions come from.

### 3. Degrees of Freedom

$H$ has 9 numbers but only 8 degrees of freedom. Homogeneous coordinates are scale-invariant — multiplying all of $H$ by any constant gives the same transformation (the $w$ in the denominator cancels it out). So one value is redundant; by convention $h_{33} = 1$, leaving 8 free parameters.

Each point correspondence gives you 2 equations ($X$ and $Y$). So you need at least **4 points** to solve for 8 unknowns. OpenCV uses more than 4 (overdetermined system) and solves it with least squares — minimizing the total error across all correspondences.

---

## The Full Pipeline

| Step | Tool | Frequency |
|------|------|-----------|
| Detect court keypoints | YOLOv8 (trained model) | Per frame, or once per video |
| Compute $H$ | `cv2.findHomography(src, dst)` | Once per video |
| Detect shuttle | YOLOv8 | Per frame |
| Map shuttle to court coordinates | `cv2.perspectiveTransform` | Per frame |
| Speed from positions + timestamps | Plain math | Per shot |

YOLO finds things in pixel space. OpenCV does the geometry. The labeled training set teaches YOLO what court keypoints look like — without those labels, there's no way to find the `src` points needed to compute $H$.

$H$ is computed **once per video**, not per frame, because the court doesn't move. A moving or zooming camera invalidates $H$ and forces continuous recomputation — which is why fixed-camera broadcast footage is essential.

### 4. Solving for H — SVD

With 4+ correspondences, OpenCV constructs a system $A\mathbf{h} = 0$ where $\mathbf{h}$ is the 9 flattened values of $H$ and $A$ encodes the point pairs. It solves this using **SVD (Singular Value Decomposition)** — the solution is the eigenvector of $A^T A$ corresponding to the smallest eigenvalue. This finds the $H$ that best fits all the correspondences in a least-squares sense.
