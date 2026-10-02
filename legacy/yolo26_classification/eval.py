"""Per-organ metrics on the val split, via softmax marginalization."""
import argparse
from pathlib import Path

from ultralytics import YOLO

from labels import ORGANS, decode, marginals

LEGACY_ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default=str(LEGACY_ROOT / "runs/classify/cls640/weights/best.pt"))
    p.add_argument("--split", default=str(LEGACY_ROOT / "data/dataset/val"))
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="mps")
    p.add_argument("--thresh", type=float, default=0.5)
    p.add_argument("--batch", type=int, default=32)
    a = p.parse_args()

    model = YOLO(a.weights)
    files = sorted(Path(a.split).glob("*/*.jpg"))
    names = [model.names[i] for i in range(len(model.names))]

    stats = {o: [0, 0, 0, 0] for o in ORGANS}  # tp, fp, fn, correct
    exact = 0
    for i in range(0, len(files), a.batch):
        chunk = files[i:i + a.batch]
        results = model([str(f) for f in chunk], imgsz=a.imgsz, device=a.device, verbose=False)
        for f, r in zip(chunk, results):
            truth = decode(f.parent.name)
            m = marginals(names, r.probs.data.tolist())
            pred = tuple(int(m[o] >= a.thresh) for o in ORGANS)
            exact += pred == truth
            for o, y, yh in zip(ORGANS, truth, pred):
                s = stats[o]
                s[0] += y and yh
                s[1] += yh and not y
                s[2] += y and not yh
                s[3] += y == yh

    n = len(files)
    print(f"{'organ':8} {'acc':>6} {'prec':>6} {'rec':>6} {'F1':>6}")
    for o in ORGANS:
        tp, fp, fn, ok = stats[o]
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
        print(f"{o:8} {ok/n:6.3f} {pr:6.3f} {rc:6.3f} {f1:6.3f}")
    print(f"\nexact match (all 3 organs right): {exact/n:.3f}  n={n}")


if __name__ == "__main__":
    main()
