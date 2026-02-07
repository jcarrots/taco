from __future__ import annotations

import sys
from pathlib import Path


def pytest_configure() -> None:
    # Prefer the in-repo package `python/taco` only when a locally-built native
    # extension is present (e.g. built via CMake with `-DTACO_BUILD_PYTHON=ON`).
    # Otherwise, fall back to the installed package (e.g. after `pip install .`).
    python_dir = Path(__file__).resolve().parents[1]
    pkg_dir = python_dir / "taco"

    has_local_ext = False
    for cfg in ("Release", "Debug"):
        cfg_dir = pkg_dir / cfg
        if cfg_dir.is_dir() and any(cfg_dir.glob("_taco*.pyd")):
            has_local_ext = True
            break
    if not has_local_ext and any(pkg_dir.glob("_taco*.so")):
        has_local_ext = True

    if has_local_ext:
        sys.path.insert(0, str(python_dir))
        return

    # No local extension detected, so tests will import installed package.
    # Fail fast if the installed wheel does not match this repo's expected API.
    try:
        import taco  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "No local native extension found under python/taco and failed to import installed `taco`. "
            "Build/install this repo first (for example: `pip install .`)."
        ) from exc

    has_expected_api = hasattr(taco, "tcl") and hasattr(taco.tcl, "e2e_cuda_compare_spin_boson")
    if not has_expected_api:
        installed_path = getattr(taco, "__file__", "<unknown>")
        raise RuntimeError(
            "Installed `taco` package is stale/incompatible for this test suite "
            f"(imported from: {installed_path}). Reinstall this repo package with `pip install --force-reinstall .`."
        )
