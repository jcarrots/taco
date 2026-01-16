import numpy as np
import pytest


def _sigma_x() -> np.ndarray:
    return np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)


def _sigma_z() -> np.ndarray:
    return np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)


def _rho0() -> np.ndarray:
    return np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)


def _make_inputs():
    import taco

    dt = 1e-2
    n_steps = 10
    bcf_end_time = n_steps * dt

    H = 0.5 * _sigma_x()
    A = 0.5 * _sigma_z()
    rho0 = _rho0()

    omega = np.linspace(0.0, 20.0, 256, dtype=np.float64)
    J = omega * np.exp(-omega / 5.0)

    bath = taco.tcl.BathTabulated(
        temperature=2.0,
        omega=omega,
        J=J,
        bcf_end_time=bcf_end_time,
    )
    cfg = taco.tcl.SimConfig(dt=dt, n_steps=n_steps, save_stride=1, order=4)
    return taco, H, A, bath, cfg, rho0


def test_cpu_simulate_shapes_and_sanity():
    taco, H, A, bath, cfg, rho0 = _make_inputs()

    res = taco.tcl.simulate(H, A, bath, cfg, rho0, device="cpu")
    with pytest.raises(ValueError):
        taco.tcl.simulate(H, A, bath, cfg, rho0, device="cpu", precision="fp32")

    assert isinstance(taco.version(), str)
    assert isinstance(taco.build_info(), dict)

    assert res.t.dtype == np.float64
    assert res.rho.dtype == np.complex128

    assert res.t.shape == (cfg.n_steps + 1,)
    assert res.rho.shape == (cfg.n_steps + 1, 2, 2)

    assert np.allclose(res.t, np.arange(cfg.n_steps + 1, dtype=np.float64) * cfg.dt)

    tr = np.trace(res.rho, axis1=1, axis2=2)
    assert np.allclose(tr.real, 1.0, atol=1e-10)
    assert np.allclose(tr.imag, 0.0, atol=1e-10)

    herm_err = res.rho - np.conjugate(np.swapaxes(res.rho, -1, -2))
    assert np.max(np.abs(herm_err)) < 1e-10

    assert np.allclose(res.rho[0], rho0, atol=1e-10)


def test_cuda_matches_cpu_when_available():
    taco, H, A, bath, cfg, rho0 = _make_inputs()

    info = taco.build_info()
    if not info.get("cuda_enabled", False):
        pytest.skip("CUDA not enabled in build")
    if not taco.cuda.is_available():
        pytest.skip("No CUDA device available")

    cpu = taco.tcl.simulate(H, A, bath, cfg, rho0, device="cpu")
    gpu = taco.tcl.simulate(H, A, bath, cfg, rho0, device="cuda")

    assert cpu.t.shape == gpu.t.shape
    assert cpu.rho.shape == gpu.rho.shape
    assert np.allclose(cpu.t, gpu.t)
    assert np.allclose(cpu.rho, gpu.rho, atol=1e-5, rtol=1e-5)


def test_cuda_fp32_matches_cpu_when_available():
    taco, H, A, bath, cfg, rho0 = _make_inputs()

    info = taco.build_info()
    if not info.get("cuda_enabled", False):
        pytest.skip("CUDA not enabled in build")
    if not taco.cuda.is_available():
        pytest.skip("No CUDA device available")

    cpu = taco.tcl.simulate(H, A, bath, cfg, rho0, device="cpu")
    gpu = taco.tcl.simulate(H, A, bath, cfg, rho0, device="cuda", precision="fp32")

    assert cpu.t.shape == gpu.t.shape
    assert cpu.rho.shape == gpu.rho.shape
    assert np.allclose(cpu.t, gpu.t)
    assert np.allclose(cpu.rho, gpu.rho, atol=1e-4, rtol=1e-4)
