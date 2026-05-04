# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

ShuttleVision is a Python computer vision pipeline for real-time badminton analysis: detect players and shuttlecocks from video, map court coordinates via homography, and compute shot speed and trajectory angle.

**Tech stack:** YOLOv8 (ultralytics), OpenCV, PyTorch, Roboflow (annotation), yt-dlp (data collection), AWS EC2 GPU (deployment).

## Setup

```bash
pip install ultralytics roboflow torch torchvision opencv-python yt-dlp
```

No `requirements.txt` exists yet — create one when adding dependencies.

## Pipeline Architecture

The project is organized into 7 sequential phases. Each phase feeds the next; the critical path items are annotation quality, homography correctness, and shot detection.

| Phase | Goal |
|-------|------|
| 1. Data Collection & Annotation | Labeled dataset (player, shuttlecock, court keypoints) via Roboflow |
| 2. Model Training | YOLOv8m trained 100 epochs, target mAP >0.80 per class |
| 3. Court Mapping | OpenCV homography: pixel coords → real-world court coords (13.4m × 6.1m) |
| 4. Analytics Engine | Shot speed (km/h) and trajectory angle (degrees) from homography-mapped positions |
| 5. Visualization | Bounding boxes, shuttle trail, top-down court diagram, stats HUD overlaid on output video |
| 6. Optimization | ≤0.2s/frame (5 FPS minimum) |
| 7. AWS Deployment | Docker container on g4dn.xlarge with CUDA |

## Key Design Decisions

- **Three detection classes:** `player`, `shuttlecock`, `court_keypoint`. The keypoints drive the homography matrix, so their annotation accuracy is the single biggest quality lever.
- **Homography-based coordinate transform:** All downstream speed/angle math lives in court-plane coordinates, not pixel space. Don't compute distances or angles in pixel space.
- **Shot detection trigger:** A shot is detected when the shuttlecock's Y-position (in image space) changes direction. Speed is then derived from the homography-transformed positions + frame timestamps.
- **Speed measurement is 2D (court plane only):** Height/elevation is not computed.
- **MVP milestone:** Steps 1–16 (Phases 1–5) constitute a resume-ready demo. Optimization and AWS deployment are post-MVP.

## Critical Path

1. **Annotation quality (Phase 1):** Bad labels cannot be recovered by better code downstream.
2. **Homography correctness (Phase 3):** All speed and angle calculations depend on this being right.
3. **Shot detection logic (Phase 4):** Speed calculation requires knowing when shots occur.
