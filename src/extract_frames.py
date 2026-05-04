import cv2
import sys
from pathlib import Path

INTERVAL_SECONDS = 0.5
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}


def extract_frames(video_path: Path, output_dir: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval_frames = int(fps * INTERVAL_SECONDS)

    frame_idx = 0
    saved = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval_frames == 0:
            filename = output_dir / f"{video_path.stem}_{saved:06d}.jpg"
            cv2.imwrite(str(filename), frame)
            saved += 1
        frame_idx += 1

    cap.release()
    return saved - 1


def main():
    if len(sys.argv) != 2:
        print("Usage: python src/extract_frames.py data/raw/video/")
        sys.exit(1)

    input_dir = Path(sys.argv[1])
    if not input_dir.is_dir():
        print(f"Not a directory: {input_dir}")
        sys.exit(1)

    videos = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in VIDEO_EXTENSIONS)
    if not videos:
        print(f"No video files found in {input_dir}")
        sys.exit(1)

    for video in videos:
        output_dir = Path("data/frames") / video.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Processing {video.name}...")
        count = extract_frames(video, output_dir)
        print(f"  → {count} frames saved to {output_dir}")


if __name__ == "__main__":
    main()
