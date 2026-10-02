"""Eight-combination label encoding for the legacy YOLO26 classifier."""

ORGANS = ("bud", "flower", "fruit")
CLASSES = ("none", "bud", "flower", "fruit",
           "bud_flower", "bud_fruit", "flower_fruit", "bud_flower_fruit")


def encode(flags):
    """(bud, flower, fruit) -> class name."""
    return "_".join(o for o, f in zip(ORGANS, flags) if f) or "none"


def decode(name):
    """class name -> (bud, flower, fruit)."""
    parts = name.split("_")
    return tuple(int(o in parts) for o in ORGANS)


def marginals(names, probs):
    """Softmax over the model's class names -> per-organ probability."""
    return {o: sum(p for c, p in zip(names, probs) if decode(c)[i])
            for i, o in enumerate(ORGANS)}
