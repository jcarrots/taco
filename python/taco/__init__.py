"""TACO Python bindings."""

from pathlib import Path

_pkg_dir = Path(__file__).resolve().parent
for _cfg in ("Release", "Debug"):
    _cfg_dir = _pkg_dir / _cfg
    if _cfg_dir.is_dir():
        __path__.append(str(_cfg_dir))

try:
    from . import _taco_native as _native
except Exception as exc:
    raise ImportError(
        "taco native extension not built. Configure with TACO_BUILD_PYTHON=ON."
    ) from exc


def version():
    return _native.version()


__all__ = ["_native", "version"]
