"""Learning curve: how accuracy scales with the number of labelled sheets."""
import argparse, random, shutil, subprocess, sys
from pathlib import Path

LEGACY_ROOT = Path(__file__).resolve().parent
DATASET = LEGACY_ROOT / "data" / "dataset"


def subset(frac, seed, out):
    """Stratified subset of train/, sharing the full val/ split."""
    shutil.rmtree(out, ignore_errors=True)
    (out / "val").mkdir(parents=True)
    for c in (DATASET / "val").iterdir():
        (out / "val" / c.name).symlink_to(c.resolve())
    rng = random.Random(seed)
    n = 0
    for c in (DATASET / "train").iterdir():
        files = sorted(c.glob("*.jpg"))
        rng.shuffle(files)
        keep = files[:max(1, round(len(files) * frac))]
        d = out / "train" / c.name
        d.mkdir(parents=True)
        for f in keep:
            (d / f.name).symlink_to(f.resolve())
        n += len(keep)
    return n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fracs", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--epochs", type=int, default=60,
                   help="epochs at frac=1.0; scaled by 1/frac to equalise gradient steps")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    for frac in a.fracs:
        out = LEGACY_ROOT / "data" / f"curve_{frac}"
        n = subset(frac, a.seed, out)
        name = f"curve{frac}"
        print(f"\n=== {n} train images ===", flush=True)
        epochs = round(a.epochs / frac)  # equal gradient steps at every size
        print(f"{epochs} epochs, ~{epochs * -(-n // a.batch)} steps", flush=True)
        subprocess.run([sys.executable, str(LEGACY_ROOT / "train.py"),
                        "--data", str(out), "--name", name,
                        "--imgsz", str(a.imgsz), "--batch", str(a.batch),
                        "--epochs", str(epochs), "--patience", "0"], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for w in ("best", "last"):
            print(w, end=" ", flush=True)
            subprocess.run([sys.executable, str(LEGACY_ROOT / "eval.py"),
                            "--imgsz", str(a.imgsz),
                            "--weights", str(LEGACY_ROOT / "runs" / "classify" / name / "weights" / f"{w}.pt")],
                           check=True)


if __name__ == "__main__":
    main()
