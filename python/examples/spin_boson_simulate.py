from __future__ import annotations

"""
Spin-boson TCL simulation example (Python).

This reproduces the same basic setup as `examples/TCL4_spin_boson_example.cpp`, but runs via:
  taco.tcl.simulate(...)

Run (from repo root, after `pip install .`):
  python python/examples/spin_boson_simulate.py --device cpu
  python python/examples/spin_boson_simulate.py --device cuda
  python python/examples/spin_boson_simulate.py --device cuda --precision fp32
"""

import argparse

import numpy as np

import taco


def sigma_x() -> np.ndarray:
    return np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)


def sigma_z() -> np.ndarray:
    return np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)


def rho_ground() -> np.ndarray:
    return np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)


def main() -> int:
    p = argparse.ArgumentParser()

    # Spin-boson system parameters (qubit)
    p.add_argument("--Delta", type=float, default=1.0, help="tunneling term Delta (in H = (Delta*sigma_x + epsilon*sigma_z)/2)")
    p.add_argument("--epsilon", type=float, default=0.0, help="bias term epsilon (in H = (Delta*sigma_x + epsilon*sigma_z)/2)")
    p.add_argument("--a-scale", type=float, default=0.5, help="coupling scale (L = a_scale*sigma_z)")

    # Bath / spectral density parameters (Ohmic with exponential cutoff)
    p.add_argument("--alpha", type=float, default=1.0, help="Ohmic prefactor in J(w) = alpha*w*exp(-w/omega_c)")
    p.add_argument("--omega-c", type=float, default=10.0, help="cutoff frequency omega_c")
    p.add_argument("--temperature", type=float, default=2.0, help="bath temperature T (beta = 1/T; use T=0 for zero temp)")
    p.add_argument("--beta", type=float, default=None, help="optional inverse temperature beta (overrides --temperature)")

    # Numerics
    p.add_argument("--dt", type=float, default=1e-2, help="time step")
    p.add_argument("--t-end", type=float, default=0.2, help="final time")
    p.add_argument("--save-stride", type=int, default=1, help="save every N steps (always includes step 0 and final step)")
    p.add_argument("--order", type=int, default=4, choices=(0, 2, 4), help="generator order: 0, 2, or 4")
    p.add_argument("--bcf-end-time", type=float, default=None, help="BCF memory horizon (defaults to t_end)")

    # Spectral density tabulation
    p.add_argument("--omega-max", type=float, default=40.0, help="max omega in the tabulated omega grid")
    p.add_argument("--n-omega", type=int, default=2048, help="number of omega grid points")

    # Execution
    p.add_argument("--device", type=str, default="cpu", choices=("cpu", "cuda"), help="execution device")
    p.add_argument("--precision", type=str, default="fp64", choices=("fp64", "fp32"), help="CUDA precision (if device=cuda)")
    p.add_argument("--gpu-id", type=int, default=0, help="CUDA device index (if device=cuda)")

    # Output
    p.add_argument("--print-series", action="store_true", help="print sz(t) each saved step")

    args = p.parse_args()

    if args.beta is not None:
        if not np.isfinite(args.beta) or args.beta < 0.0:
            raise ValueError("beta must be finite and >= 0")
        temperature = 0.0 if args.beta == np.inf else (0.0 if args.beta == 0.0 else 1.0 / args.beta)
    else:
        temperature = float(args.temperature)

    dt = float(args.dt)
    t_end = float(args.t_end)
    bcf_end_time = float(args.bcf_end_time) if args.bcf_end_time is not None else t_end

    # System: H = (Delta*sigma_x + epsilon*sigma_z)/2, A = a_scale*sigma_z
    H = 0.5 * float(args.Delta) * sigma_x() + 0.5 * float(args.epsilon) * sigma_z()
    A = float(args.a_scale) * sigma_z()
    rho0 = rho_ground()

    # Tabulated spectral density J(omega) on omega in [0, omega_max]
    omega_max = float(args.omega_max)
    omega = np.linspace(0.0, omega_max, int(args.n_omega), dtype=np.float64)
    J = float(args.alpha) * omega * np.exp(-omega / float(args.omega_c))

    nyquist = np.pi / dt
    if omega_max < nyquist:
        print(f"Warning: omega_max={omega_max:g} < Nyquist(pi/dt)={nyquist:g}; consider increasing --omega-max for accuracy.")

    bath = taco.tcl.BathTabulated(
        temperature=temperature,
        omega=omega,
        J=J,
        bcf_end_time=bcf_end_time,
    )
    cfg = taco.tcl.SimConfig(
        dt=dt,
        t_end=t_end,
        save_stride=int(args.save_stride),
        order=int(args.order),
    )

    print("taco.version():", taco.version())
    print("taco.build_info():", taco.build_info())

    res = taco.tcl.simulate(
        H, A, bath, cfg, rho0, device=args.device, precision=str(args.precision), gpu_id=int(args.gpu_id)
    )

    sz = np.einsum("tij,ji->t", res.rho, sigma_z()).real
    tr = np.trace(res.rho, axis1=1, axis2=2).real

    print("result shapes:", res.t.shape, res.rho.shape)
    print("final t:", float(res.t[-1]))
    print("final Tr(rho):", float(tr[-1]))
    print("final <sigma_z>:", float(sz[-1]))
    print("final rho:\n", res.rho[-1])

    if args.print_series:
        print("t,sz,Tr(rho)")
        for t, szi, tri in zip(res.t, sz, tr):
            print(f"{float(t):.12g},{float(szi):.12g},{float(tri):.12g}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
