import numpy as np


def test_e2e_compare_spin_boson_returns_metrics():
    import taco

    res = taco.tcl.e2e_cuda_compare_spin_boson(
        Nt_samples=1024,
        dt=6.25e-4,
        temperature=2.0,
        omega_c=10.0,
        tidx="0:64:256",
        threads=0,
        gpu_warmup=0,
        rk4_steps=0,
        precision="fp64",
        check=False,
    )

    assert isinstance(res, dict)
    assert isinstance(res.get("cuda_enabled"), bool)
    assert isinstance(res.get("cuda_available"), bool)
    assert res.get("precision") in ("fp64", "fp32")
    assert isinstance(res.get("tidx"), list)

    l4 = res["l4"]
    assert isinstance(l4, dict)
    assert l4["cpu_fcr_ms"] >= 0.0
    assert l4["cpu_total_ms"] >= 0.0
    assert l4["cpu_avg_ms"] >= 0.0

    if res["cuda_enabled"] and res["cuda_available"]:
        assert l4["has_gpu"] is True
        assert l4["gpu_total_ms"] is not None
        assert l4["gpu_fcr_ms"] is not None


def test_e2e_compare_accepts_tidx_list():
    import taco

    res = taco.tcl.e2e_cuda_compare_spin_boson(
        Nt_samples=256,
        tidx=[0, 1, 2, 10],
        rk4_steps=0,
        gpu_warmup=0,
        check=False,
    )
    assert res["tidx"] == [0, 1, 2, 10]


def test_e2e_compare_accepts_tidx_numpy():
    import taco

    res = taco.tcl.e2e_cuda_compare_spin_boson(
        Nt_samples=256,
        tidx=np.array([10, 0, 2, 2], dtype=np.int64),
        rk4_steps=0,
        gpu_warmup=0,
        check=False,
    )
    # The binding sorts+uniques indices.
    assert res["tidx"] == [0, 2, 10]

