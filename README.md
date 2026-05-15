# ShuttleVision — Project Plan

Real-time badminton analysis: detect players and shuttlecock using YOLOv8, map the court to real-world
coordinates, and compute shot speed and trajectory angle from video footage.

**Total estimate:** ~5 weeks at 3 hours/day, 6 days/week (~90 hours)

---

## Phase 1 — Data Collection & Annotation
*Goal: produce a labeled image dataset ready for model training.*

---

### Step 1 — Collect badminton footage
**Time: 3 hours | Difficulty: Easy**

Download 5–10 badminton match videos from YouTube using `yt-dlp`. Aim for variety:
- At least 2 different camera angles (high side-angle broadcast, end-court angle)
- Indoor and outdoor lighting if possible
- Professional matches (cleaner footage) and amateur matches (harder conditions)
- Each video should be 5–15 minutes long

Diversity matters more than quantity. A model trained on one angle and one lighting
condition will fail on anything different.

Tools: `yt-dlp` (command-line YouTube downloader)

```
yt-dlp "https://youtube.com/watch?v=..." -o "data/raw/%(title)s.%(ext)s"
```

---

### Step 2 — Extract frames from videos
**Time: 2 hours | Difficulty: Easy**

A video is just thousands of images played in sequence. You need to split each video
into individual image files so you can annotate them. You don't need every frame —
extract one frame every 0.5 seconds (every 15 frames at 30fps). This avoids having
thousands of nearly identical images.

Tool: OpenCV (`pip install opencv-python`)

The extraction script reads a video file and saves every Nth frame as a .jpg into
`data/frames/`. Target: 500–800 total frames across all videos.

Edge case to handle: some frames will be blurry (fast shuttlecock motion). Keep them —
the model needs to learn to detect a blurry shuttlecock too.

---

### Step 3 — Set up Roboflow project and annotate
**Time: 15–20 hours | Difficulty: Easy (tedious)**

Roboflow is a web tool for labeling images. Create a free account at roboflow.com and
start a new Object Detection project.

**Three classes to annotate:**
- `player` — draw a bounding box around each player's full body
- `shuttlecock` — draw a tight box around the shuttlecock (it's small, zoom in)
- `court_keypoint` — draw a small box on each visible court line intersection

**Court keypoints explained:** A badminton court has many line intersections (corners,
service line crossings, doubles/singles sideline crossings). You don't need all of them
— mark every one that's clearly visible. These are used later to compute the
court-to-camera mapping (homography).

**Annotation tips:**
- Use Roboflow's "Smart Polygon" or auto-label feature for players — it saves time
- Shuttlecock annotation is slow. If the shuttlecock is not visible in a frame, skip it
- Aim for at least 300 shuttlecock annotations total — it's the hardest class to detect
- Court keypoints: mark 6–10 per frame where the court is visible

**Why this takes so long:** There is no shortcut. Bad labels produce a bad model.
This is the most important phase of the entire project.

---

### Step 4 — Preprocess and export dataset
**Time: 1 hour | Difficulty: Easy**

In Roboflow, generate a dataset version with:

**Preprocessing** (applied to all images):
- Auto-Orient: ON (fixes EXIF rotation issues)
- Resize: Stretch to 640×640 (YOLOv8's default training input size)

**Augmentation:** skip in Roboflow. YOLOv8 applies its own augmentation pipeline
at training time — HSV jitter, scale, translation, horizontal flip, and mosaic
(which stitches 4 images together to dramatically help small-object detection
like shuttlecock). Static Roboflow augmentation on top of this is redundant and
can drift augmented-of-augmented images too far from realistic data.

**Split:** 70% train / 20% validation / 10% test. The validation set is
non-optional — YOLOv8 uses it for early stopping and overfitting detection.

Export in **YOLOv8 format**. Roboflow gives you a download link or a Python
snippet (with API key) that pulls the dataset directly into your training script.

---

## Phase 2 — Model Training
*Goal: a YOLOv8 model that reliably detects players, shuttlecocks, and court keypoints.*

---

### Approach: single pose model vs two-model split

The Roboflow project exports as **YOLOv8 Pose** format with multi-class support
(`nc=3`, `kpt_shape=[22, 3]`). This unlocks two viable training approaches:

**Approach A — Single YOLOv8 Pose model (currently trying):**
One `yolov8m-pose.pt` model handles all three classes simultaneously:
- `badminton_court` — bbox + 22 visible keypoints
- `Player` — bbox only (keypoint slots stored as zeros, visibility=0)
- `Birdie` — bbox only (keypoint slots stored as zeros, visibility=0)

Keypoint loss is automatically masked when visibility=0, so `Player`/`Birdie`
train as standard object detection. One model, one training run, one inference
pass per frame.

**Approach B — Two separate models (fallback):**
- Model 1: YOLOv8 Detect for `Player` + `Birdie` (2 classes, no keypoints)
- Model 2: YOLOv8 Pose for `badminton_court` only (1 class, 22 keypoints)

Requires a small Python script to split the YOLOv8-pose export into two
datasets. Slightly more inference cost, though the court model only needs to
run once per video since the camera is fixed.

Use Approach A unless training is unstable. Signs to fall back to B:
- Loss diverges or produces NaNs partway through training
- `pose_loss` is 10×+ the `box_loss` / `cls_loss` terms in the per-epoch log
- `badminton_court` mAP is fine but `Player` / `Birdie` mAP is much lower than
  expected for those simple classes
- Validation images saved under `runs/pose/train/` show garbage keypoint
  predictions even on training data
- Multi-class pose proves fragile in this Ultralytics version

**Fallback procedure (if Approach A fails):**

1. **Split the dataset locally.** Write a small Python script that walks the
   YOLOv8-pose `labels/*.txt` files and produces two parallel datasets:
   - `dataset_detect/` — keep only rows for class 0 (`Birdie`) and class 1
     (`Player`). Strip the 66 keypoint columns from each row, leaving only
     `class_id cx cy w h`.
   - `dataset_pose/` — keep only rows for class 2 (`badminton_court`). Remap
     the class_id from 2 → 0 (single-class pose). Keep all keypoint columns.

   Copy the corresponding images alongside each label set. Reuse the same
   train/valid/test split assignments from the original export.

2. **Write two `data.yaml` files**, one per dataset:
   - Detect yaml: `nc: 2`, `names: ['Birdie', 'Player']`, no `kpt_shape`
   - Pose yaml: `nc: 1`, `names: ['badminton_court']`, `kpt_shape: [22, 3]`,
     `flip_idx: [0, 1, 2, ..., 21]`

3. **Train both models:**
   - Detect: `yolo train model=yolov8m.pt data=dataset_detect/data.yaml ...` →
     produces `players.pt`
   - Pose: `yolo train model=yolov8m-pose.pt data=dataset_pose/data.yaml ...` →
     produces `court.pt`

4. **Update inference glue.** Load both checkpoints at startup. Run `court.pt`
   **once per video** on frame 1 to extract 22 keypoint pixel positions and
   compute the homography matrix. Run `players.pt` **on every frame** for
   `Player` and `Birdie` detection. Transform detections to court coordinates
   using the precomputed homography.

The split script is ~30 lines of Python; the inference glue adds ~20 lines on
top of the single-model path. The downstream homography math, shot detection,
and visualization (Phases 3–5) are unchanged.

---

### Step 5 — Set up training environment
**Time: 2 hours | Difficulty: Easy–Medium**

**Hardware:** local training on NVIDIA RTX 5070 Ti Laptop GPU (Blackwell, 12GB
VRAM). Sufficient for YOLOv8m-pose at 640×640 with batch=16. No cloud GPU
needed for the MVP — keeps the iteration loop tight.

Install dependencies:
```
pip install ultralytics torch torchvision opencv-python
```

The 5070 Ti is a Blackwell-generation GPU, so install a **recent** PyTorch
build with CUDA 12.4+ support. From the PyTorch website, pick the latest
stable wheel for CUDA 12.4 or higher.

Verify CUDA is available and the GPU is recognized:
```python
import torch
print(torch.cuda.is_available())           # must print True
print(torch.cuda.get_device_name(0))       # should print "NVIDIA GeForce RTX 5070 Ti Laptop GPU"
```

If `cuda.is_available()` is False or the device name is wrong, reinstall
PyTorch with the correct CUDA build before continuing.

---

### Step 6 — Train YOLOv8 on your dataset
**Time: 3 hours active + 1–3 hours compute | Difficulty: Medium**

Start from a **pose** checkpoint (Approach A):
- `yolov8s-pose.pt` for a fast baseline (~45 min on 5070 Ti)
- `yolov8m-pose.pt` for the final run (~1.5–3 hours on 5070 Ti)

If falling back to Approach B, start from `yolov8s.pt` / `yolov8m.pt` (detect)
for the player/birdie model, and `yolov8s-pose.pt` / `yolov8m-pose.pt` for the
court model.

The training script specifies: which model to start from, where your dataset
is, how many epochs, image size, and batch size. YOLOv8 saves the
best-performing checkpoint automatically under `runs/pose/train*/weights/`.

Key parameters:
- `epochs=100` — number of full passes through the dataset
- `imgsz=640` — image size (larger = slower but better for small objects like shuttlecock)
- `batch=16` — images processed simultaneously (reduce if GPU runs out of memory)
- `device=0` — use the first CUDA GPU

---

### Step 7 — Evaluate and iterate
**Time: 5–8 hours over 2–3 days | Difficulty: Medium**

After training, check the mAP (mean Average Precision) score for each class separately.
A score above 0.80 per class is acceptable for this project.

Common problems and fixes:
- **Shuttlecock mAP is low (<0.6):** Go back to Roboflow, annotate 100–200 more
  shuttlecock frames, especially blurry and distant ones. Retrain.
- **Court keypoint mAP is low:** Make sure you're annotating keypoints consistently
  (same points across frames). Inconsistent labeling confuses the model.
- **Player mAP is low (<0.75):** Add images with partial player occlusion and unusual
  court angles.

Do not chase a perfect score. Once all three classes are above 0.80, move on.
Diminishing returns kick in hard after that point.

---

## Phase 3 — Court Mapping
*Goal: transform any pixel position on the court into real-world coordinates in meters.*

---

### Step 8 — Define real-world court coordinates
**Time: 1 hour | Difficulty: Easy**

A standard badminton court is exactly 13.4m long and 6.1m wide. Define a coordinate
system where one back corner is (0, 0):

```
(0, 0)    = back-left corner
(6.1, 0)  = back-right corner
(0, 13.4) = front-left corner
(6.1, 13.4) = front-right corner
net at y = 6.7 (center)
```

Write out the real-world (x, y) coordinates in meters for every court keypoint you
annotated. These are fixed and the same for every video — it's a standard court.

---

### Step 9 — Compute the homography matrix
**Time: 4–6 hours | Difficulty: Hard**

The homography is a 3x3 matrix that maps pixel coordinates to real-world coordinates.
You compute it by giving OpenCV a list of corresponding point pairs:
- Where each court keypoint appears in the image (in pixels) — from your detector
- Where that same point is in real-world coordinates (in meters) — from Step 8

OpenCV's `cv2.findHomography(pixel_points, real_world_points)` returns the matrix H.

To convert any pixel (px, py) to real-world (rx, ry):
```python
point = np.array([px, py, 1])
result = H @ point          # matrix multiply
rx = result[0] / result[2]
ry = result[1] / result[2]
```

The division by `result[2]` (called the homogeneous coordinate) is what makes the
perspective math work correctly. This step is easy to get wrong — the most common
bugs are: wrong coordinate order (x,y vs y,x), forgetting the division, or mixing
up which points are source vs destination.

**Important limitation:** The homography maps the camera view onto the court floor
plane (height = 0). It is accurate for objects on or near the court surface. For
a shuttlecock high in the air, it will give a slightly wrong ground position — we
accept this tradeoff since we are not computing height.

**Important strength:** The homography is computed fresh for every video from that
video's own court keypoints. Different camera angles in test videos are handled
automatically — the homography adapts to each camera.

---

### Step 10 — Validate the court mapping
**Time: 2–3 hours | Difficulty: Medium**

Draw the top-down court diagram and project your detected court keypoints onto it.
They should land on the correct court lines. If they don't, debug in this order:
1. Are the pixel points and real-world points in the same order?
2. Is the coordinate system consistent (which corner is 0,0)?
3. Are you dividing by the homogeneous coordinate?

Also test: take a known court line (e.g. the baseline) in the image, convert both
endpoints, and verify the distance between them is close to 6.1m.

---

## Phase 4 — Analytics Engine
*Goal: compute shuttlecock speed and trajectory angle from real-world positions.*

---

### Step 11 — Implement shot detection
**Time: 2–3 hours | Difficulty: Medium**

You don't want to calculate speed continuously — a shuttlecock sitting still at the
start of a rally would show speed = 0, and random jitter gives false readings.
Calculate speed only at the moment of a shot.

Detect a shot by watching the shuttlecock's Y position in the image over time.
When the Y position changes direction (was moving down, now moving up — or vice
versa), a shot has just been hit. This is the same technique the tennis_analysis
reference project uses.

Store the frame index of each detected shot. Speed is calculated between consecutive
shot frames.

Edge case: the shuttlecock is not detected in every frame (it's fast and small).
Keep only frames where detection confidence is above 0.5. Interpolate the position
linearly for missing frames between two confident detections.

---

### Step 12 — Calculate shot speed
**Time: 2 hours | Difficulty: Easy**

Between two shot frames (frame A and frame B):

1. Get shuttlecock pixel position in both frames
2. Convert both to real-world coordinates using homography (Step 9)
3. Calculate distance in meters: `sqrt((x2-x1)² + (y2-y1)²)`
4. Calculate time elapsed: `(frame_B - frame_A) / FPS` seconds
5. Speed = distance / time → m/s → multiply by 3.6 for km/h

**Note on what this speed measures:** This is court-plane speed — the speed of
horizontal movement across the court surface. It does not include vertical movement
(rising or falling). For flat shots and smashes, this is close to the true speed.
For high lobs, it underestimates the true speed because significant motion is vertical.
This limitation should be documented.

---

### Step 13 — Calculate trajectory angle
**Time: 2 hours | Difficulty: Easy**

Between two positions in real-world court coordinates:

```python
dx = real_x2 - real_x1   # movement across court width
dy = real_y2 - real_y1   # movement along court length
angle_from_net = abs(math.degrees(math.atan2(dx, dy)))
```

This gives the angle of travel relative to the net (perpendicular to net = 0°,
parallel to net = 90°). A cross-court shot is roughly 30–45°, a down-the-line
shot is close to 0°.

**Note on what this angle measures:** This is the horizontal trajectory angle on
the court plane. It does not include the elevation angle (how steeply the
shuttlecock is rising or falling). Elevation requires height data which we are
not computing.

---

## Phase 5 — Visualization
*Goal: produce an output video with analytics overlaid.*

---

### Step 14 — Draw detection overlays
**Time: 3–4 hours | Difficulty: Easy**

Using OpenCV drawing functions on each frame:
- Bounding boxes around players (different color per player)
- Bounding box around shuttlecock
- Shuttlecock trail: draw a line connecting the last 10 shuttlecock positions

All drawing happens on a copy of the frame — never modify the original.

---

### Step 15 — Draw bird's-eye court view
**Time: 3–4 hours | Difficulty: Medium**

Create a small top-down court diagram in the corner of the output video.
Draw the court lines using the known court dimensions scaled to the diagram size.
Project each player and the shuttlecock onto the diagram using the homography.

This requires an inverse step: you already have homography from pixels → real-world.
For the diagram you need real-world → diagram pixels. Compute a second homography
or just apply a simple scale factor.

---

### Step 16 — Draw stats HUD
**Time: 1–2 hours | Difficulty: Easy**

Overlay text on the video frame showing:
- Last shot speed in km/h
- Shot trajectory angle in degrees
- Shot count

Use OpenCV's `cv2.putText`. Position the stats in a consistent corner that doesn't
overlap with the court.

---

## Phase 6 — Optimization
*Goal: reach ≤ 0.2 seconds per frame (5 FPS minimum).*

---

### Step 17 — Benchmark current speed
**Time: 1 hour | Difficulty: Easy**

Time each stage of the pipeline separately:
- Detection (YOLOv8 inference)
- Homography transform
- Analytics calculation
- Drawing / visualization

The detection step will be the bottleneck by far.

---

### Step 18 — Enable CUDA inference
**Time: 2–3 hours | Difficulty: Medium**

Pass `device=0` to YOLOv8 inference to run on GPU. Verify GPU utilization with
`nvidia-smi` while the pipeline runs — it should show >50% GPU usage.

If already using GPU, check batch size. Processing multiple frames simultaneously
is faster than one at a time.

---

### Step 19 — Vectorize analytics with NumPy
**Time: 2–3 hours | Difficulty: Medium**

Replace any Python `for` loops over frame data with NumPy array operations.
For example, instead of computing distance for each frame pair in a loop,
compute all distances at once using `np.linalg.norm` on an array of position deltas.

This matters less than the CUDA step but contributes to hitting the 0.2s/frame target.

---

## Phase 7 — AWS Deployment
*Goal: the pipeline runs on a cloud GPU instance and accepts video input.*

---

### Step 20 — Dockerize the pipeline
**Time: 4–5 hours | Difficulty: Medium**

Docker packages your code and all its dependencies into a container that runs
identically on any machine. Write a Dockerfile that:
1. Starts from a CUDA base image (nvidia/cuda)
2. Installs Python dependencies (ultralytics, opencv, numpy, etc.)
3. Copies your code into the container
4. Defines the entry point command

Test locally: `docker build` then `docker run` on a local video file.

---

### Step 21 — Set up AWS EC2 GPU instance
**Time: 3–4 hours | Difficulty: Medium**

Create an AWS account if you don't have one. Launch a `g4dn.xlarge` instance:
- 1x NVIDIA T4 GPU, 16GB VRAM
- ~$0.53/hour (stop the instance when not using it)

Steps: EC2 console → Launch instance → Select Deep Learning AMI (Ubuntu) →
Choose g4dn.xlarge → Configure storage (50GB minimum) → Launch.

The Deep Learning AMI comes with CUDA and Docker pre-installed, which saves
significant setup time.

---

### Step 22 — Deploy and test on EC2
**Time: 3–4 hours | Difficulty: Medium**

Push your Docker image to ECR (AWS's container registry) or pull it from Docker Hub.
Run the container on EC2 with a test video. Verify:
- It runs without errors
- GPU is being used (nvidia-smi inside the container)
- Output video is produced correctly
- Processing speed meets the 0.2s/frame target

---

## Hours Summary

| Phase | Steps | Estimated Hours |
|-------|-------|----------------|
| 1. Data & Annotation | 1–4 | 21–26 h |
| 2. Model Training | 5–7 | 10–13 h active + compute |
| 3. Court Mapping | 8–10 | 7–10 h |
| 4. Analytics | 11–13 | 6–8 h |
| 5. Visualization | 14–16 | 7–10 h |
| 6. Optimization | 17–19 | 5–7 h |
| 7. AWS Deployment | 20–22 | 10–13 h |
| **Total** | | **~66–87 active hours** |

At 3 hours/day, 6 days/week: **~4–5 weeks**

---

## Critical Path

Steps that block everything else if delayed:
1. **Annotation quality (Step 3)** — bad labels cannot be fixed by better code
2. **Homography correctness (Step 9)** — all speed and angle math is wrong if this is wrong
3. **Shot detection (Step 11)** — speed calculation depends on knowing when shots occur

---

## Minimum Viable Demo (for resume)

Complete Steps 1–16 first. This produces a working annotated output video showing
player tracking, shuttlecock tracking, shot speed, and trajectory angle. This is
sufficient to demonstrate the project on a resume.

Steps 17–22 (optimization and AWS) add the performance and deployment bullets
but are not needed for an initial demo.
