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

### 2. Matrix-Vector Multiplication

$$H \begin{pmatrix} x \\ y \\ 1 \end{pmatrix}$$

This is a $3 \times 3$ matrix times a $3 \times 1$ column vector, producing a $3 \times 1$ result. Each output row is the dot product of that row of $H$ with the input vector:

$$\text{row 1} \cdot \mathbf{v} = h_{11}x + h_{12}y + h_{13}$$

That's where the numerator/denominator expressions come from.

### 3. Degrees of Freedom

$H$ has 9 numbers but only 8 degrees of freedom. Homogeneous coordinates are scale-invariant — multiplying all of $H$ by any constant gives the same transformation (the $w$ in the denominator cancels it out). So one value is redundant; by convention $h_{33} = 1$, leaving 8 free parameters.

Each point correspondence gives you 2 equations ($X$ and $Y$). So you need at least **4 points** to solve for 8 unknowns. OpenCV uses more than 4 (overdetermined system) and solves it with least squares — minimizing the total error across all correspondences.

### 4. Solving for H — SVD

With 4+ correspondences, OpenCV constructs a system $A\mathbf{h} = 0$ where $\mathbf{h}$ is the 9 flattened values of $H$ and $A$ encodes the point pairs. It solves this using **SVD (Singular Value Decomposition)** — the solution is the eigenvector of $A^T A$ corresponding to the smallest eigenvalue. This finds the $H$ that best fits all the correspondences in a least-squares sense.
