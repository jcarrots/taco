"""TACO Python bindings."""

import os
import sys
from pathlib import Path

_pkg_dir = Path(__file__).resolve().parent
for _cfg in ("Release", "Debug"):
    _cfg_dir = _pkg_dir / _cfg
    if _cfg_dir.is_dir():
        __path__.append(str(_cfg_dir))

_dll_dirs = []
if sys.platform == "win32":
    cuda_paths = []
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        cuda_paths.append(cuda_path)
    for key, value in os.environ.items():
        if key.startswith("CUDA_PATH_V") and value:
            cuda_paths.append(value)

    seen = set()
    for cuda_path in cuda_paths:
        if cuda_path in seen:
            continue
        seen.add(cuda_path)
        for _rel in ("bin", os.path.join("bin", "x64")):
            _candidate = Path(cuda_path) / _rel
            if _candidate.is_dir():
                _dll_dirs.append(os.add_dll_directory(str(_candidate)))

try:
    from . import _taco as _native
except Exception as exc:
    raise ImportError(
        "taco native extension not built. Configure with -DTACO_BUILD_PYTHON=ON."
    ) from exc


def version():
    return _native.version()

def build_info():
    return _native.build_info()

from . import cuda  # noqa: E402
from . import tcl  # noqa: E402

__all__ = ["_native", "build_info", "cuda", "tcl", "version"]
