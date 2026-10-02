"""Train YOLO26-cls on the phenology dataset."""
import argparse
from pathlib import Path

from ultralytics import YOLO

LEGACY_ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(LEGACY_ROOT / "yolo26s-cls.pt"))
    p.add_argument("--data", default=str(LEGACY_ROOT / "data/dataset"))
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="mps")
    p.add_argument("--name", default="cls")
    p.add_argument("--patience", type=int, default=20)
    a = p.parse_args()

    YOLO(a.model).train(
        data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
        device=a.device, project=str(LEGACY_ROOT / "runs/classify"), name=a.name,
        patience=a.patience, degrees=10, fliplr=0.5,
    )


if __name__ == "__main__":
    main()
