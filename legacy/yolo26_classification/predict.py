"""Predict per-organ phenology probabilities for images."""
import argparse, csv, sys
from pathlib import Path

from ultralytics import YOLO

from labels import ORGANS, marginals

LEGACY_ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("source", nargs="+", help="image files or directories")
    p.add_argument("--weights", default=str(LEGACY_ROOT / "runs/classify/cls640/weights/best.pt"))
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="mps")
    p.add_argument("--out", help="write CSV here instead of stdout")
    a = p.parse_args()

    model = YOLO(a.weights)
    files = [f for s in a.source for f in
             (sorted(Path(s).glob("*.jpg")) if Path(s).is_dir() else [Path(s)])]

    w = csv.writer(open(a.out, "w", newline="") if a.out else sys.stdout)
    w.writerow(["image", "class", *ORGANS])
    for f in files:
        r = model(str(f), imgsz=a.imgsz, device=a.device, verbose=False)[0]
        m = marginals([r.names[i] for i in range(len(r.names))], r.probs.data.tolist())
        w.writerow([f.name, r.names[r.probs.top1], *(f"{m[o]:.3f}" for o in ORGANS)])


if __name__ == "__main__":
    main()
