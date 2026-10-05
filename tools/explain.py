"""Grad-CAM evidence maps for the YOLO combo classifiers, one per organ.

A combo model's organ probability is the summed softmax over the combinations
containing that organ. Grad-CAM on the last convolution before pooling shows
where the evidence for that sum comes from; maps are averaged over the
ensemble members. Each figure shows the full sheet with the square the model
actually sees (ultralytics center-crops portrait sheets at predict time) and
the bud / flower / fruit maps inside that square.

Images are picked from the val split by outcome (confident hits, misses, and
false alarms per organ) using the ensemble's predictions and val thresholds.

    python tools/explain.py                       # val examples -> results/explain/
    python tools/explain.py --images data/images/X.jpg data/images/Y.jpg
"""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from ultralytics.data.augment import classify_transforms  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ORGANS = ("bud", "flower", "fruit")
Image.MAX_IMAGE_PIXELS = None


def load_sheet(image_id, path, maxpx):
    """The same 1536 px sheet the models trained on (derived cache, or rebuilt from the native image)."""
    cached = PROJECT_ROOT / "data" / "derived" / str(maxpx) / f"{image_id}.jpg"
    source = cached if cached.is_file() else path
    with Image.open(source) as image:
        image.draft("RGB", (maxpx, maxpx))
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((maxpx, maxpx), Image.Resampling.LANCZOS)
        return image.copy()


class OrganCAM:
    """Grad-CAM for organ marginals of one ultralytics classification model."""

    def __init__(self, weights, device):
        self.model = YOLO(str(weights)).model.float().to(device).eval()
        names = [self.model.names[index] for index in range(len(self.model.names))]
        self.members = torch.tensor(
            [[organ in name.split("_") for name in names] for organ in ORGANS], device=device
        )
        head = self.model.model[-1]
        head.conv.register_forward_hook(lambda _module, _inputs, output: setattr(self, "features", output))

    def __call__(self, batch):
        with torch.enable_grad():
            # ultralytics loads weights frozen; a grad-requiring input gives the features a graph
            output = self.model(batch.clone().requires_grad_(True))
            logits = output[1] if isinstance(output, tuple) else output
            log_probs = logits.log_softmax(1)[0]
            maps, probabilities = [], []
            for members in self.members:
                # log P(organ) = logsumexp of log-probabilities of combinations containing it
                score = log_probs[members].logsumexp(0)
                (grad,) = torch.autograd.grad(score, self.features, retain_graph=True)
                weights = grad.mean(dim=(2, 3), keepdim=True)
                maps.append(F.relu((weights * self.features).sum(1))[0].detach())
                probabilities.append(float(score.detach().exp()))
        return torch.stack(maps), probabilities


def pick_examples(predictions, thresholds, truth, per_group):
    """Confident hits, misses (false negatives) and false alarms (false positives) per organ."""
    picked = []
    for index, organ in enumerate(ORGANS):
        scored = [(image, probs[index], truth[image][index]) for image, probs in predictions.items()]
        threshold = thresholds[index]
        groups = {
            "hit": sorted((s for s in scored if s[2] and s[1] >= threshold), key=lambda s: -s[1]),
            "miss": sorted((s for s in scored if s[2] and s[1] < threshold), key=lambda s: s[1]),
            "false_alarm": sorted((s for s in scored if not s[2] and s[1] >= threshold), key=lambda s: -s[1]),
        }
        for group, items in groups.items():
            picked += [(image, f"{organ}_{group}") for image, _, _ in items[:per_group]]
    return picked


def figure(sheet, maps, probabilities, truth, thresholds, title, imgsz, out):
    width, height = sheet.size
    scale = imgsz / min(width, height)
    crop_w, crop_h = imgsz / scale, imgsz / scale  # crop square in sheet pixels
    left, top = (width - crop_w) / 2, (height - crop_h) / 2
    view = sheet.crop((round(left), round(top), round(left + crop_w), round(top + crop_h)))

    fig, axes = plt.subplots(1, 4, figsize=(18, 6), gridspec_kw={"width_ratios": [width / height, 1, 1, 1]})
    axes[0].imshow(sheet)
    axes[0].add_patch(Rectangle((left, top), crop_w, crop_h, fill=False, edgecolor="yellow", linewidth=2))
    for y0, y1 in ((0, top), (top + crop_h, height)):
        axes[0].axhspan(y0, y1, color="black", alpha=0.45)
    for x0, x1 in ((0, left), (left + crop_w, width)):
        axes[0].axvspan(x0, x1, color="black", alpha=0.45)
    axes[0].set_xlim(0, width)
    axes[0].set_ylim(height, 0)
    axes[0].set_title("full sheet (dimmed = never seen)")
    peak = float(maps.max()) or 1.0
    for axis, organ, evidence, probability, flag, threshold in zip(
        axes[1:], ORGANS, maps, probabilities, truth, thresholds
    ):
        heat = F.interpolate(evidence[None, None], size=view.size[::-1], mode="bilinear", align_corners=False)[0, 0]
        axis.imshow(view)
        axis.imshow(heat.cpu(), cmap="inferno", alpha=0.55, vmin=0, vmax=peak)
        called = "present" if probability >= threshold else "absent"
        verdict = "ok" if (probability >= threshold) == bool(flag) else "WRONG"
        axis.set_title(f"{organ}: p={probability:.2f} -> {called}\ntruth {'present' if flag else 'absent'} ({verdict})")
    for axis in axes:
        axis.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=Path,
                        default=sorted((PROJECT_ROOT / "results/train").glob("combo_y26l_896*/best.pt")))
    parser.add_argument("--ensemble", default="ens_y26l_896_x3", help="predictions/thresholds used to pick examples")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "splits/phenology_v2.csv")
    parser.add_argument("--images", nargs="*", type=Path, help="explain these images instead of picking val examples")
    parser.add_argument("--per-group", type=int, default=3, help="examples per organ x outcome")
    parser.add_argument("--imgsz", type=int, default=896)
    parser.add_argument("--maxpx", type=int, default=1536)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "results/explain")
    args = parser.parse_args()
    if not args.weights:
        raise SystemExit("No weights found; pass --weights")

    manifest = {row["image_id"]: row for row in csv.DictReader(args.manifest.open())}
    truth = {image: [int(row[organ]) for organ in ORGANS] for image, row in manifest.items()}
    compare = {row["name"]: row for row in csv.DictReader((PROJECT_ROOT / "results/compare_val.csv").open())
               if row["split"] == "val"}
    thresholds = [float(compare[args.ensemble][f"{organ}_thresh"]) for organ in ORGANS]

    rows = csv.DictReader((PROJECT_ROOT / "results/predictions" / args.ensemble / "val.csv").open())
    predictions = {Path(row["image"]).stem: [float(row[f"{organ}_prob"]) for organ in ORGANS] for row in rows}
    if args.images:
        picked = [(path.stem, "requested") for path in args.images]
    else:
        picked = pick_examples(predictions, thresholds, truth, args.per_group)

    cams = [OrganCAM(weights, args.device) for weights in args.weights]
    transform = classify_transforms(args.imgsz)
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"{len(cams)} models, thresholds {dict(zip(ORGANS, thresholds))}")
    done = set()
    for image_id, reason in picked:
        if image_id in done:
            continue
        done.add(image_id)
        row = manifest[image_id]
        sheet = load_sheet(image_id, PROJECT_ROOT / row["path"], args.maxpx)
        batch = transform(sheet)[None].to(args.device)
        results = [cam(batch) for cam in cams]
        maps = torch.stack([result[0] for result in results]).mean(0)
        probabilities = [sum(result[1][i] for result in results) / len(results) for i in range(len(ORGANS))]
        out = args.out / f"{reason}__{image_id}.png"
        figure(sheet, maps, probabilities, truth[image_id], thresholds, f"{image_id}  ({reason})", args.imgsz, out)
        # Recomputed probabilities should match the ensemble's saved ones if preprocessing matches.
        saved = predictions.get(image_id)
        check = "" if saved is None else "  saved " + " ".join(f"{p:.3f}" for p in saved)
        print(f"wrote {out.name}: p " + " ".join(f"{p:.3f}" for p in probabilities) + check)


if __name__ == "__main__":
    main()
