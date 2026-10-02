"""Run LeafMachine2's herbarium-native detector and aggregate phenology evidence.

The detector emits YOLO boxes. This wrapper reports the maximum object
confidence and number of detections for bud, flower, and fruit on each sheet.
Those maxima are detector scores, not calibrated sheet-level probabilities.
"""
import argparse
import csv
import os
import shutil
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LM2_COMPONENT_DETECTOR = (
    ROOT / "third_party" / "LeafMachine2" / "leafmachine2" / "component_detector"
)
DEFAULT_WEIGHTS = ROOT / "third_party" / "LeafMachine2" / "checkpoints" / "best.pt"
PLANT_DATA = LM2_COMPONENT_DETECTOR / "data" / "PLANT_Full.yaml"
IMAGE_SUFFIXES = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp"}
PHENOLOGY_CLASSES = {
    "bud": {7},
    "flower": {5, 6},       # flower_one, flower_many
    "fruit": {3, 4},        # seed_fruit_one, seed_fruit_many
}


def image_files(sources):
    files = []
    for source in sources:
        path = Path(source)
        if path.is_dir():
            files.extend(
                p for p in sorted(path.rglob("*"))
                if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
            )
        elif path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            files.append(path)
        else:
            raise FileNotFoundError(f"No supported image source: {path}")

    stems = [path.stem for path in files]
    if len(stems) != len(set(stems)):
        raise ValueError("LeafMachine2 keys outputs by filename stem; duplicate stems are ambiguous")
    if not files:
        raise ValueError("No supported images found")
    # Keep symlink paths so a classification dataset's parent directory still
    # carries the ground-truth class. LeafMachine2 resolves them internally.
    return [path.absolute() for path in files]


def read_evidence(label_path):
    detections = []
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            fields = line.split()
            if len(fields) == 6:
                detections.append((int(float(fields[0])), float(fields[5])))

    row = {}
    for organ, class_ids in PHENOLOGY_CLASSES.items():
        scores = [confidence for class_id, confidence in detections if class_id in class_ids]
        row[f"{organ}_max_conf"] = max(scores, default=0.0)
        row[f"{organ}_count"] = len(scores)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", nargs="+", help="image files or directories (searched recursively)")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.25, help="minimum object confidence")
    parser.add_argument("--device", default="cpu", help="cpu or CUDA device; old YOLOv5 lacks MPS support")
    parser.add_argument("--out", type=Path, default=Path("lm2_predictions.csv"))
    parser.add_argument(
        "--labels",
        type=Path,
        help="optionally retain raw YOLO prediction labels (class x y w h confidence)",
    )
    args = parser.parse_args()

    files = image_files(args.source)
    if not args.weights.is_file():
        raise FileNotFoundError(f"Missing LeafMachine2 checkpoint: {args.weights}")

    # The vendored YOLOv5 code imports top-level modules named models and utils.
    sys.path.insert(0, str(LM2_COMPONENT_DETECTOR))
    os.environ.setdefault("YOLOv5_AUTOINSTALL", "false")
    mpl_cache = Path(tempfile.gettempdir()) / "myrtavision-matplotlib"
    mpl_cache.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    from detect import run as detect  # pylint: disable=import-outside-toplevel

    with tempfile.TemporaryDirectory(prefix="myrtavision-lm2-") as temporary:
        project = Path(temporary)
        detect(
            weights=str(args.weights.resolve()),
            source=[str(path) for path in files],
            data=str(PLANT_DATA),
            anno_type="Plant_Detector",
            imgsz=(args.imgsz, args.imgsz),
            conf_thres=args.conf,
            device=args.device,
            save_txt=True,
            save_conf=True,
            nosave=True,
            project=project,
            name="detect",
            exist_ok=True,
        )

        labels = project / "detect" / "labels"
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "image",
            "bud_max_conf", "flower_max_conf", "fruit_max_conf",
            "bud_count", "flower_count", "fruit_count",
        ]
        with args.out.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for image in files:
                writer.writerow({"image": str(image), **read_evidence(labels / f"{image.stem}.txt")})

        if args.labels:
            args.labels.mkdir(parents=True, exist_ok=True)
            for label in labels.glob("*.txt"):
                shutil.copy2(label, args.labels / label.name)

    print(f"wrote {len(files)} rows to {args.out}")
    if args.labels:
        print(f"wrote raw prediction labels to {args.labels}")


if __name__ == "__main__":
    main()
