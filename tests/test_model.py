"""
test_model.py — Unit tests for AxSymDKIModel and AxSymDKIFit

Tests the full fitting pipeline by:
  - Generating synthetic signals with known ground-truth parameters
  - Fitting the model and checking recovered parameters are within tolerance
  - Verifying metric properties (FA, MD, MK, AK, RK) on synthetic data
  - Checking the model interface (masking, shape handling)

The synthetic signal approach (known-in, check-recovered-out) is the
standard validation strategy for quantitative MRI fitting code.
"""

import numpy as np
import pytest
from axdki.model import AxSymDKIModel, AxSymDKIFit
from axdki.signal import axsym_signal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_scheme(n_dirs=30, bvals_shells=(0, 1000, 2000)):
    """Generate a simple multi-shell gradient scheme for testing."""
    rng = np.random.default_rng(42)
    directions = rng.standard_normal((n_dirs, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    bvals = []
    bvecs = []
    for b in bvals_shells:
        if b == 0:
            bvals.extend([0] * 5)
            bvecs.extend([[0, 0, 0]] * 5)
        else:
            bvals.extend([b] * n_dirs)
            bvecs.extend(directions.tolist())

    return np.array(bvals, dtype=float), np.array(bvecs, dtype=float)


def synthetic_voxel(bvals, bvecs, D_par, D_perp, W_par, W_perp, axis,
                    S0=1000.0, snr=None, seed=0):
    """Generate synthetic signal with optional Rician noise."""
    S = axsym_signal(S0, bvals, bvecs, D_par, D_perp, W_par, W_perp, axis)
    if snr is not None:
        rng = np.random.default_rng(seed)
        sigma = S0 / snr
        noise_r = rng.normal(0, sigma, S.shape)
        noise_i = rng.normal(0, sigma, S.shape)
        S = np.sqrt((S + noise_r) ** 2 + noise_i ** 2)  # Rician magnitude
    return S


# ---------------------------------------------------------------------------
# Ground truth parameters
# ---------------------------------------------------------------------------

GT_PARAMS = dict(
    D_par=1.7e-3,
    D_perp=0.3e-3,
    W_par=0.5,
    W_perp=1.2,
    axis=np.array([1.0, 0.0, 0.0]),
    S0=1000.0,
)

# Tolerances: noiseless should be tight; with noise, looser
TOL_NOISELESS = dict(D_par=1e-4, D_perp=1e-4, W_par=0.05, W_perp=0.05)
TOL_NOISY     = dict(D_par=5e-4, D_perp=5e-4, W_par=0.3,  W_perp=0.3)


# ---------------------------------------------------------------------------
# Tests: model construction
# ---------------------------------------------------------------------------

def test_model_requires_b0():
    """Model must raise if no b=0 volumes are present."""
    bvals = np.array([1000, 1000, 2000], dtype=float)
    bvecs = np.eye(3, dtype=float)
    with pytest.raises(ValueError, match="b=0"):
        AxSymDKIModel(bvals, bvecs)


def test_model_requires_enough_directions():
    """Model must raise if fewer than 10 DW directions are present."""
    bvals = np.array([0, 1000, 1000, 1000], dtype=float)
    bvecs = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=float)
    with pytest.raises(ValueError, match="10"):
        AxSymDKIModel(bvals, bvecs)


# ---------------------------------------------------------------------------
# Tests: noiseless single voxel recovery
# ---------------------------------------------------------------------------

def test_noiseless_parameter_recovery():
    """Noiseless synthetic data should recover ground truth within tight tolerance."""
    bvals, bvecs = make_scheme()
    p = GT_PARAMS
    signal = synthetic_voxel(bvals, bvecs, **{k: p[k] for k in
                              ["D_par","D_perp","W_par","W_perp","axis","S0"]})

    model = AxSymDKIModel(bvals, bvecs)
    fit = model.fit(signal)

    assert fit.D_par  == pytest.approx(p["D_par"],  abs=TOL_NOISELESS["D_par"]),  "D_par recovery failed"
    assert fit.D_perp == pytest.approx(p["D_perp"], abs=TOL_NOISELESS["D_perp"]), "D_perp recovery failed"
    assert fit.W_par  == pytest.approx(p["W_par"],  abs=TOL_NOISELESS["W_par"]),  "W_par recovery failed"
    assert fit.W_perp == pytest.approx(p["W_perp"], abs=TOL_NOISELESS["W_perp"]), "W_perp recovery failed"


def test_noiseless_fa():
    """FA should be close to the analytical value for noiseless data."""
    bvals, bvecs = make_scheme()
    p = GT_PARAMS
    signal = synthetic_voxel(bvals, bvecs, **{k: p[k] for k in
                              ["D_par","D_perp","W_par","W_perp","axis","S0"]})

    model = AxSymDKIModel(bvals, bvecs)
    fit = model.fit(signal)

    # Analytical FA for axisymmetric tensor: λ1=D_par, λ2=λ3=D_perp
    MD = (p["D_par"] + 2 * p["D_perp"]) / 3
    FA_expected = (np.sqrt(2) *
                   np.sqrt((p["D_par"] - MD)**2 + 2*(p["D_perp"] - MD)**2) /
                   np.sqrt(p["D_par"]**2 + 2*p["D_perp"]**2))

    assert fit.fa == pytest.approx(FA_expected, abs=0.02), "FA recovery failed"


# ---------------------------------------------------------------------------
# Tests: noisy data
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("snr", [20, 50])
def test_noisy_parameter_recovery(snr):
    """Parameters should be recovered within looser tolerance under noise."""
    bvals, bvecs = make_scheme()
    p = GT_PARAMS
    signal = synthetic_voxel(bvals, bvecs, **{k: p[k] for k in
                              ["D_par","D_perp","W_par","W_perp","axis","S0"]},
                              snr=snr)

    model = AxSymDKIModel(bvals, bvecs)
    fit = model.fit(signal)

    assert abs(fit.D_par  - p["D_par"])  < TOL_NOISY["D_par"],  f"D_par failed at SNR={snr}"
    assert abs(fit.D_perp - p["D_perp"]) < TOL_NOISY["D_perp"], f"D_perp failed at SNR={snr}"


# ---------------------------------------------------------------------------
# Tests: output shapes and masking
# ---------------------------------------------------------------------------

def test_fit_shape_3d():
    """Fit output shapes must match input spatial dimensions."""
    bvals, bvecs = make_scheme()
    p = GT_PARAMS
    signal_1d = synthetic_voxel(bvals, bvecs, **{k: p[k] for k in
                                 ["D_par","D_perp","W_par","W_perp","axis","S0"]})

    # Tile into a small 3D volume (2x2x2)
    data_4d = np.tile(signal_1d, (2, 2, 2, 1))

    model = AxSymDKIModel(bvals, bvecs)
    fit = model.fit(data_4d)

    assert fit.fa.shape    == (2, 2, 2), "FA shape mismatch"
    assert fit.D_par.shape == (2, 2, 2), "D_par shape mismatch"
    assert fit.params.shape == (2, 2, 2, 5), "params shape mismatch"


def test_fit_with_mask():
    """Masked-out voxels should have zero parameters."""
    bvals, bvecs = make_scheme()
    p = GT_PARAMS
    signal_1d = synthetic_voxel(bvals, bvecs, **{k: p[k] for k in
                                 ["D_par","D_perp","W_par","W_perp","axis","S0"]})
    data_4d = np.tile(signal_1d, (2, 2, 1, 1))

    mask = np.ones((2, 2, 1), dtype=bool)
    mask[0, 0, 0] = False  # mask out one voxel

    model = AxSymDKIModel(bvals, bvecs)
    fit = model.fit(data_4d, mask=mask)

    assert fit.D_par[0, 0, 0] == 0.0, "Masked voxel D_par should be zero"
    assert fit.fa[0, 0, 0]    == 0.0, "Masked voxel FA should be zero"


# ---------------------------------------------------------------------------
# Tests: AxSymDKIFit properties
# ---------------------------------------------------------------------------

def test_fit_metric_properties():
    """All metric properties must return arrays of the correct shape."""
    bvals, bvecs = make_scheme()
    p = GT_PARAMS
    signal = synthetic_voxel(bvals, bvecs, **{k: p[k] for k in
                              ["D_par","D_perp","W_par","W_perp","axis","S0"]})

    model = AxSymDKIModel(bvals, bvecs)
    fit = model.fit(signal)

    for prop in ["fa", "md", "mk", "ak", "rk", "D_par", "D_perp", "W_par", "W_perp"]:
        val = getattr(fit, prop)
        assert val is not None, f"Property {prop!r} returned None"
        assert np.isfinite(val).all(), f"Property {prop!r} contains non-finite values"


def test_ak_equals_w_par():
    """AK should be identical to W_par (they are the same quantity)."""
    bvals, bvecs = make_scheme()
    p = GT_PARAMS
    signal = synthetic_voxel(bvals, bvecs, **{k: p[k] for k in
                              ["D_par","D_perp","W_par","W_perp","axis","S0"]})
    model = AxSymDKIModel(bvals, bvecs)
    fit = model.fit(signal)
    assert np.allclose(fit.ak, fit.W_par), "AK must equal W_par"


def test_rk_equals_w_perp():
    """RK should be identical to W_perp."""
    bvals, bvecs = make_scheme()
    p = GT_PARAMS
    signal = synthetic_voxel(bvals, bvecs, **{k: p[k] for k in
                              ["D_par","D_perp","W_par","W_perp","axis","S0"]})
    model = AxSymDKIModel(bvals, bvecs)
    fit = model.fit(signal)
    assert np.allclose(fit.rk, fit.W_perp), "RK must equal W_perp"
