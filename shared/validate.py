import argparse
import numpy as np
import torch
from pathlib import Path

def bounded_int(minimum: int, maximum: int):
    """Create an argparse validator for bounded integers."""

    def validate(value: str) -> int:
        try:
            number = int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "must be an integer"
            ) from exc

        if not minimum <= number <= maximum:
            raise argparse.ArgumentTypeError(
                f"must be between {minimum} and {maximum}"
            )

        return number

    return validate


def bounded_float(
    minimum: float,
    maximum: float,
    *,
    min_inclusive: bool = True,
):
    """Create an argparse validator for bounded finite floats."""

    def validate(value: str) -> float:
        try:
            number = float(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "must be a number"
            ) from exc

        if not np.isfinite(number):
            raise argparse.ArgumentTypeError(
                "must be a finite number"
            )

        valid_min = (
            number >= minimum
            if min_inclusive
            else number > minimum
        )

        if not valid_min or number > maximum:
            left = "[" if min_inclusive else "("
            raise argparse.ArgumentTypeError(
                f"must be in the range {left}{minimum}, {maximum}]"
            )

        return number

    return validate


def resolve_path(path: str | Path) -> Path:
    """Resolve a path relative to the current working directory."""
    return Path(path).expanduser().resolve()


def validate_device(value: str) -> str:
    """Validate the requested PyTorch device."""

    try:
        device = torch.device(value)
    except (RuntimeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            f"invalid device: {value}"
        ) from exc

    if device.type == "cuda" and not torch.cuda.is_available():
        raise argparse.ArgumentTypeError(
            "CUDA was requested but is not available."
        )

    if device.type == "mps" and not torch.backends.mps.is_available():
        raise argparse.ArgumentTypeError(
            "MPS was requested but is not available."
        )

    if device.type not in {"cpu", "cuda", "mps"}:
        raise argparse.ArgumentTypeError(
            "device must be cpu, cuda, or mps"
        )

    return value

