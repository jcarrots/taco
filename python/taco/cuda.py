from __future__ import annotations

from . import _native


def is_available() -> bool:
    return bool(_native.cuda_is_available())


def device_count() -> int:
    return int(_native.cuda_device_count())


__all__ = ["device_count", "is_available"]

