from __future__ import annotations

"""
High-level TCL simulation API.

This module is a thin, Python-friendly wrapper over the native extension (`taco._taco`).
It keeps the physics/numerics in the existing C++ CPU/CUDA implementations and only:
- validates/normalizes inputs (dtype/shape/contiguity),
- exposes small dataclasses for config/bath inputs,
- returns NumPy arrays (time grid + density matrix time series).
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from . import _native


def _require_complex128_square(name: str, x: np.ndarray) -> np.ndarray:
    # Validate a square complex128 matrix and return a C-contiguous view/copy.
    arr = np.asarray(x)
    if arr.dtype != np.complex128:
        raise TypeError(f"{name} must have dtype complex128 (got {arr.dtype}); cast with np.asarray(..., dtype=np.complex128)")
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D (got shape {arr.shape})")
    if arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name} must be square (got shape {arr.shape})")
    return np.ascontiguousarray(arr)


def _require_float64_1d(name: str, x: np.ndarray) -> np.ndarray:
    # Validate a 1D float64 array and return a C-contiguous view/copy.
    arr = np.asarray(x)
    if arr.dtype != np.float64:
        raise TypeError(f"{name} must have dtype float64 (got {arr.dtype}); cast with np.asarray(..., dtype=np.float64)")
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D (got shape {arr.shape})")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    return np.ascontiguousarray(arr)


@dataclass(frozen=True)
class BathTabulated:
    """
    Tabulated spectral density bath input.

    Parameters
    ----------
    temperature:
        Bath temperature T (units assumed consistent with the C++ backend; internally it uses beta = 1/T,
        with T=0 treated as beta=+inf).
    omega:
        Frequency grid `omega` (float64, strictly increasing, >= 0).
    J:
        Spectral density values `J(omega)` on the same grid as `omega` (float64).
    bcf_end_time:
        Memory horizon for the bath correlation C(t): for t > bcf_end_time we treat C(t) as 0.
    """
    temperature: float
    omega: np.ndarray
    J: np.ndarray
    bcf_end_time: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.temperature) or self.temperature < 0.0:
            raise ValueError("temperature must be finite and >= 0")
        if not np.isfinite(self.bcf_end_time) or self.bcf_end_time < 0.0:
            raise ValueError("bcf_end_time must be finite and >= 0")

        omega = _require_float64_1d("omega", self.omega)
        J = _require_float64_1d("J", self.J)
        if omega.shape != J.shape:
            raise ValueError("omega and J must have the same shape")
        if omega.size < 2:
            raise ValueError("omega and J must have length >= 2")
        if omega[0] < 0.0:
            raise ValueError("omega must be >= 0")
        if not np.all(omega[1:] > omega[:-1]):
            raise ValueError("omega must be strictly increasing")

        object.__setattr__(self, "omega", omega)
        object.__setattr__(self, "J", J)


@dataclass(frozen=True)
class SimConfig:
    """
    Simulation configuration.

    Parameters
    ----------
    dt:
        Fixed time step size.
    n_steps / t_end:
        Provide either:
        - n_steps: number of fixed steps, or
        - t_end: final time; internally we convert to n_steps = ceil(t_end / dt).
    save_stride:
        Save every `save_stride` steps (and always save step 0 and the final step).
    order:
        Generator order:
        - 0: unitary only (L0 = -i[H, rho])
        - 2: TCL2 (L0 + dissipator + Lamb shift)
        - 4: TCL2 + TCL4 correction series (calls existing C++ TCL4 CPU/CUDA path)
    """
    dt: float
    n_steps: int | None = None
    t_end: float | None = None
    save_stride: int = 1
    order: int = 4

    def __post_init__(self) -> None:
        if not np.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("dt must be finite and > 0")
        if self.n_steps is None and self.t_end is None:
            raise ValueError("either n_steps or t_end must be provided")
        if self.n_steps is not None and self.n_steps < 0:
            raise ValueError("n_steps must be >= 0")
        if self.t_end is not None and (not np.isfinite(self.t_end) or self.t_end < 0.0):
            raise ValueError("t_end must be finite and >= 0")
        if self.save_stride < 1:
            raise ValueError("save_stride must be >= 1")
        if self.order not in (0, 2, 4):
            raise ValueError("order must be 0, 2, or 4")

    def resolved_n_steps(self) -> int:
        if self.n_steps is not None:
            return int(self.n_steps)
        if self.t_end is None:
            raise ValueError("t_end is required when n_steps is None")
        if self.t_end == 0.0:
            return 0
        return int(np.ceil(self.t_end / self.dt))


@dataclass(frozen=True)
class SimResult:
    """Simulation outputs (saved times + density matrix trajectory in the lab basis)."""
    t: np.ndarray
    rho: np.ndarray


def precompute_bcf(bath: BathTabulated, dt: float) -> np.ndarray:
    """
    Precompute the bath correlation function C(t_k) from a tabulated spectral density J(omega).

    Returns
    -------
    bcf:
        complex128 array of length N_bcf+1 where N_bcf = ceil(bath.bcf_end_time / dt) and
        t_k = k * dt for k=0..N_bcf.
    """
    if not isinstance(bath, BathTabulated):
        raise TypeError("bath must be a BathTabulated")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and > 0")
    return _native.tcl_precompute_bcf(
        temperature=float(bath.temperature),
        dt=float(dt),
        bcf_end_time=float(bath.bcf_end_time),
        omega=bath.omega,
        J=bath.J,
    )


def simulate(
    H: np.ndarray,
    A: np.ndarray,
    bath: BathTabulated,
    cfg: SimConfig,
    rho0: np.ndarray,
    device: Literal["cpu", "cuda"] = "cpu",
    *,
    precision: Literal["fp64", "fp32"] = "fp64",
    gpu_id: int = 0,
    bcf: np.ndarray | None = None,
) -> SimResult:
    """
    Run a TCL simulation using the existing C++ CPU path, or the existing CUDA path if built.

    Parameters (main physics inputs)
    -------------------------------
    H:
        System Hamiltonian (N,N) complex128 in the lab basis.
    A:
        Coupling operator (N,N) complex128 in the lab basis (currently one channel).
    rho0:
        Initial density matrix (N,N) complex128 in the lab basis.
    bath:
        Bath model parameters; for now the Python API exposes a tabulated spectral density J(omega).

    Parameters (execution)
    ----------------------
    device:
        "cpu" or "cuda". If built without CUDA, "cuda" raises RuntimeError.
    precision:
        CUDA computation precision:
        - "fp64": double-precision (default)
        - "fp32": single-precision (casts inputs on upload and outputs on download)
        Only used when device="cuda".
    gpu_id:
        CUDA device index (only used when device="cuda").
    bcf:
        Optional precomputed correlation C(t_k) (complex128, 1D). If provided, we skip computing C(t)
        from (omega, J) and instead use this array (padded/truncated to the simulation length).

    Returns
    -------
    SimResult:
        - t: float64 array of saved times
        - rho: complex128 array (n_saved, N, N) in the lab basis
    """
    if not isinstance(bath, BathTabulated):
        raise TypeError("bath must be a BathTabulated")
    if not isinstance(cfg, SimConfig):
        raise TypeError("cfg must be a SimConfig")

    H = _require_complex128_square("H", H)
    A = _require_complex128_square("A", A)
    rho0 = _require_complex128_square("rho0", rho0)
    if H.shape != A.shape or H.shape != rho0.shape:
        raise ValueError("H, A, and rho0 must have the same (N,N) shape")

    n_steps = cfg.resolved_n_steps()
    precision = str(precision).lower()
    if precision not in ("fp64", "fp32"):
        raise ValueError("precision must be 'fp64' or 'fp32'")
    if device == "cpu" and precision != "fp64":
        raise ValueError("precision='fp32' requires device='cuda'")

    if bcf is None:
        t, rho = _native.tcl_simulate(
            H=H,
            A=A,
            rho0=rho0,
            temperature=float(bath.temperature),
            dt=float(cfg.dt),
            n_steps=int(n_steps),
            save_stride=int(cfg.save_stride),
            bcf_end_time=float(bath.bcf_end_time),
            omega=bath.omega,
            J=bath.J,
            device=str(device),
            precision=str(precision),
            order=int(cfg.order),
            gpu_id=int(gpu_id),
        )
    else:
        bcf = np.asarray(bcf)
        if bcf.dtype != np.complex128:
            raise TypeError(f"bcf must have dtype complex128 (got {bcf.dtype}); cast with np.asarray(..., dtype=np.complex128)")
        if bcf.ndim != 1:
            raise ValueError(f"bcf must be 1D (got shape {bcf.shape})")
        bcf = np.ascontiguousarray(bcf)

        t, rho = _native.tcl_simulate_from_bcf(
            H=H,
            A=A,
            rho0=rho0,
            dt=float(cfg.dt),
            n_steps=int(n_steps),
            save_stride=int(cfg.save_stride),
            bcf=bcf,
            device=str(device),
            precision=str(precision),
            order=int(cfg.order),
            gpu_id=int(gpu_id),
        )

    return SimResult(t=np.asarray(t, dtype=np.float64), rho=np.asarray(rho, dtype=np.complex128))


def e2e_cuda_compare_spin_boson(
    *,
    Nt_samples: int = 100000,
    dt: float = 6.25e-4,
    temperature: float = 2.0,
    omega_c: float = 10.0,
    tidx: str | list[int] | np.ndarray | None = None,
    threads: int = 0,
    gpu_id: int = 0,
    gpu_warmup: int = 1,
    rk4_steps: int = 50,
    rk4_order: int = 4,
    rk4_method: Literal["warp", "cublas"] = "warp",
    precision: Literal["fp64", "fp32"] = "fp64",
    check: bool = True,
) -> dict:
    """
    Run the spin-boson CPU vs CUDA end-to-end compare (the same workload as `tcl4_e2e_cuda_compare`).

    This is intended for benchmarking and validation (it returns timings + max CPU/GPU differences).
    For perf-only runs (especially FP32), pass `check=False` to avoid raising on mismatches.
    """
    if not isinstance(Nt_samples, int) or Nt_samples <= 0:
        raise ValueError("Nt_samples must be an int > 0")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and > 0")
    if not np.isfinite(temperature) or temperature < 0.0:
        raise ValueError("temperature must be finite and >= 0")
    if not np.isfinite(omega_c) or omega_c <= 0.0:
        raise ValueError("omega_c must be finite and > 0")
    if threads < 0:
        raise ValueError("threads must be >= 0")
    if gpu_id < 0:
        raise ValueError("gpu_id must be >= 0")
    if gpu_warmup < 0:
        raise ValueError("gpu_warmup must be >= 0")
    if rk4_steps < 0:
        raise ValueError("rk4_steps must be >= 0")
    if rk4_order not in (0, 2, 4):
        raise ValueError("rk4_order must be 0, 2, or 4")
    if rk4_method not in ("warp", "cublas"):
        raise ValueError("rk4_method must be 'warp' or 'cublas'")
    if precision not in ("fp64", "fp32"):
        raise ValueError("precision must be 'fp64' or 'fp32'")

    if isinstance(tidx, np.ndarray):
        tidx_arg = [int(x) for x in tidx.ravel().tolist()]
    else:
        tidx_arg = tidx

    return dict(
        _native.tcl4_e2e_cuda_compare_spin_boson(
            Nt_samples=int(Nt_samples),
            dt=float(dt),
            temperature=float(temperature),
            omega_c=float(omega_c),
            tidx=tidx_arg,
            threads=int(threads),
            gpu_id=int(gpu_id),
            gpu_warmup=int(gpu_warmup),
            rk4_steps=int(rk4_steps),
            rk4_order=int(rk4_order),
            rk4_method=str(rk4_method),
            precision=str(precision),
            check=bool(check),
        )
    )


__all__ = [
    "BathTabulated",
    "SimConfig",
    "SimResult",
    "e2e_cuda_compare_spin_boson",
    "precompute_bcf",
    "simulate",
]
