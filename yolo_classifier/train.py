"""Train ultralytics YOLO-cls phenology classifiers and export per-organ probabilities.

Two tasks share the datasets built by tools/build_datasets.py:

    combo   one model over the eight bud/flower/fruit combinations; an organ's
            probability is the summed softmax over combinations containing it
    binary  three present/absent models, one per organ

After training, val and test probabilities are written to
results/predictions/<name>/{val,test}.csv in the format every method shares
(image, bud_prob, flower_prob, fruit_prob), and the best weights and training
curves are copied to results/train/<name>/.
"""
import argparse
import csv
import shutil
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ORGANS = ("bud", "flower", "fruit")
SPLITS = ("val", "test")


def organ_flags(class_name):
    parts = class_name.split("_")
    return [int(organ in parts) for organ in ORGANS]


def predict(model, files, imgsz, device, batch):
    """Softmax probabilities for each file, keyed by class name."""
    names = [model.names[index] for index in range(len(model.names))]
    probabilities = []
    for start in range(0, len(files), batch):
        chunk = [str(path) for path in files[start : start + batch]]
        for result in model.predict(chunk, imgsz=imgsz, device=device, verbose=False):
            probabilities.append(dict(zip(names, result.probs.data.tolist())))
    return probabilities


def private_view(dataset, run_dir):
    """A per-run dataset folder whose splits link to the shared dataset.

    ultralytics writes <split>.cache next to each split folder (it resolves the
    dataset folder but not the split links), so concurrent runs on one shared
    dataset race to delete and rewrite the same cache file.
    """
    view = run_dir / "data"
    view.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        link = view / split
        if link.is_symlink():
            link.unlink()
        link.symlink_to((dataset / split).resolve(), target_is_directory=True)
    return view


def train_one(data, model, run_dir, args):
    """Train one classifier and return the reloaded best checkpoint."""
    YOLO(str(model)).train(
        data=str(private_view(data, run_dir)),
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        seed=args.seed,
        deterministic=True,
        degrees=10,
        # Absolute path: ultralytics nests a relative project under runs/<task>/.
        project=str(run_dir.parent.resolve()),
        name=run_dir.name,
        exist_ok=True,
    )
    return YOLO(str(run_dir / "weights" / "best.pt"))


def keep(run_dir, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in [run_dir / "weights" / "best.pt", run_dir / "args.yaml", run_dir / "results.csv", *run_dir.glob("*.png")]:
        if path.exists():
            shutil.copy2(path, out_dir / path.name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("combo", "binary"), required=True)
    parser.add_argument("--model", required=True, help="pretrained classifier, e.g. weights/yolo26s-cls.pt")
    parser.add_argument("--name", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="0")
    parser.add_argument("--datasets", type=Path, default=PROJECT_ROOT / "datasets")
    parser.add_argument("--runs", type=Path, default=PROJECT_ROOT / "runs" / "yolo")
    parser.add_argument("--results", type=Path, default=PROJECT_ROOT / "results")
    args = parser.parse_args()

    files = {split: sorted((args.datasets / "combo" / split).glob("*/*.jpg")) for split in SPLITS}
    probabilities = {split: [[0.0] * len(ORGANS) for _ in files[split]] for split in SPLITS}

    if args.task == "combo":
        run_dir = args.runs / args.name
        model = train_one(args.datasets / "combo", args.model, run_dir, args)
        keep(run_dir, args.results / "train" / args.name)
        for split in SPLITS:
            for row, scores in zip(probabilities[split], predict(model, files[split], args.imgsz, args.device, args.batch)):
                for class_name, probability in scores.items():
                    for index, flag in enumerate(organ_flags(class_name)):
                        row[index] += probability * flag
    else:
        for index, organ in enumerate(ORGANS):
            run_dir = args.runs / args.name / organ
            model = train_one(args.datasets / f"binary_{organ}", args.model, run_dir, args)
            keep(run_dir, args.results / "train" / args.name / organ)
            for split in SPLITS:
                for row, scores in zip(probabilities[split], predict(model, files[split], args.imgsz, args.device, args.batch)):
                    row[index] = scores["present"]

    out_dir = args.results / "predictions" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        with (out_dir / f"{split}.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["image", *[f"{organ}_prob" for organ in ORGANS]])
            for path, row in zip(files[split], probabilities[split]):
                writer.writerow([path.stem, *[f"{value:.6f}" for value in row]])
        print(f"wrote {len(files[split])} {split} predictions to {out_dir / f'{split}.csv'}")


if __name__ == "__main__":
    main()
