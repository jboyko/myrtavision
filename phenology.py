"""Shared phenology target definitions for the current LeafMachine approach."""

ORGANS = ("bud", "flower", "fruit")


def decode_combination(name):
    """Decode a legacy combination-directory name into three binary targets."""
    parts = name.split("_")
    return tuple(int(organ in parts) for organ in ORGANS)
