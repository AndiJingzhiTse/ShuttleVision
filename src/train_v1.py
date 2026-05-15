from pathlib import Path
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = REPO_ROOT / "data" / "v1" / "data.yaml"


def main():
    model = YOLO("yolov8m-pose.pt")

    model.train(
        data=str(DATA_YAML),
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        project=str(REPO_ROOT / "runs" / "pose"),
        name="v1",
        patience=20,
        seed=0,
        # fliplr=0 because data.yaml flip_idx is identity; mirroring would
        # produce labels where left-side court keypoints land on the right.
        fliplr=0.0,
        flipud=0.0,
    )


if __name__ == "__main__":
    main()
