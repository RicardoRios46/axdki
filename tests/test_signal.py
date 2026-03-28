"""
test_signal.py — Unit tests for the axisymmetric DKI signal model

Tests the forward model by checking:
  - Signal is S0 at b=0
  - Signal decreases monotonically with b-value
  - Isotropic case (D_par == D_perp) produces direction-independent signal
  - Symmetry: signal is identical for g and -g
"""

import numpy as np
import pytest
from axdki.signal import axsym_signal, compute_apparent_diffusivity


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def isotropic_params():
    """Parameters for an isotropic voxel (D_par == D_perp)."""
    return dict(
        S0=1000.0,
        D_par=1.0e-3,
        D_perp=1.0e-3,   # isotropic: equal eigenvalues
        W_par=0.5,
        W_perp=0.5,
        axis=np.array([0.0, 0.0, 1.0]),
    )


@pytest.fixture
def anisotropic_params():
    """Parameters for a white matter-like anisotropic voxel."""
    return dict(
        S0=1000.0,
        D_par=1.7e-3,
        D_perp=0.3e-3,
        W_par=0.5,
        W_perp=1.2,
        axis=np.array([1.0, 0.0, 0.0]),   # axis along x
    )


@pytest.fixture
def simple_scheme():
    """A minimal gradient scheme: one b=0 and six DW directions."""
    bvals = np.array([0, 1000, 1000, 1000, 2000, 2000, 2000], dtype=float)
    bvecs = np.array([
        [0, 0, 0],        # b=0
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ], dtype=float)
    return bvals, bvecs


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_signal_at_b0(simple_scheme, anisotropic_params):
    """Signal at b=0 should equal S0 regardless of direction."""
    bvals, bvecs = simple_scheme
    p = anisotropic_params
    S = axsym_signal(p["S0"], bvals, bvecs, p["D_par"], p["D_perp"],
                     p["W_par"], p["W_perp"], p["axis"])
    assert S[0] == pytest.approx(p["S0"], rel=1e-6), \
        "Signal at b=0 must equal S0"


def test_signal_decreases_with_b(anisotropic_params):
    """Signal should decrease monotonically as b-value increases."""
    p = anisotropic_params
    bvals = np.array([0, 500, 1000, 2000], dtype=float)
    bvecs = np.tile([1, 0, 0], (4, 1)).astype(float)  # same direction

    S = axsym_signal(p["S0"], bvals, bvecs, p["D_par"], p["D_perp"],
                     p["W_par"], p["W_perp"], p["axis"])
    assert np.all(np.diff(S) < 0), \
        "Signal must decrease monotonically with increasing b-value"


def test_isotropic_direction_independence(isotropic_params):
    """For isotropic parameters, signal must not depend on gradient direction."""
    p = isotropic_params
    bvals = np.full(6, 1000.0)
    # Six orthogonal and diagonal directions
    bvecs = np.array([
        [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [1, 1, 0], [1, 0, 1], [0, 1, 1],
    ], dtype=float)
    bvecs /= np.linalg.norm(bvecs, axis=1, keepdims=True)

    S = axsym_signal(p["S0"], bvals, bvecs, p["D_par"], p["D_perp"],
                     p["W_par"], p["W_perp"], p["axis"])
    assert np.allclose(S, S[0], rtol=1e-6), \
        "Isotropic signal must be the same in all gradient directions"


def test_signal_antipodal_symmetry(simple_scheme, anisotropic_params):
    """Signal must be identical for g and -g (diffusion is symmetric)."""
    bvals, bvecs = simple_scheme
    p = anisotropic_params
    S_pos = axsym_signal(p["S0"], bvals, bvecs, p["D_par"], p["D_perp"],
                         p["W_par"], p["W_perp"], p["axis"])
    S_neg = axsym_signal(p["S0"], bvals, -bvecs, p["D_par"], p["D_perp"],
                         p["W_par"], p["W_perp"], p["axis"])
    assert np.allclose(S_pos, S_neg, rtol=1e-10), \
        "Signal must be antipodally symmetric: S(g) == S(-g)"


def test_signal_positive(simple_scheme, anisotropic_params):
    """All predicted signals must be positive."""
    bvals, bvecs = simple_scheme
    p = anisotropic_params
    S = axsym_signal(p["S0"], bvals, bvecs, p["D_par"], p["D_perp"],
                     p["W_par"], p["W_perp"], p["axis"])
    assert np.all(S > 0), "All signal values must be positive"


def test_signal_parallel_vs_perpendicular(anisotropic_params):
    """Signal parallel to axis should be lower than perpendicular (D_par > D_perp)."""
    p = anisotropic_params
    bvals = np.array([1000.0, 1000.0])
    bvecs = np.array([
        p["axis"],                              # parallel to axis
        np.array([0.0, 1.0, 0.0]),             # perpendicular
    ])

    S = axsym_signal(p["S0"], bvals, bvecs, p["D_par"], p["D_perp"],
                     p["W_par"], p["W_perp"], p["axis"])
    assert S[0] < S[1], \
        "Signal parallel to principal axis must be lower (D_par > D_perp → more attenuation)"


def test_apparent_diffusivity_bounds(anisotropic_params):
    """Apparent diffusivity must be bounded by [D_perp, D_par]."""
    p = anisotropic_params
    n = 50
    bvecs = np.random.randn(n, 3)
    bvecs /= np.linalg.norm(bvecs, axis=1, keepdims=True)

    D_app = compute_apparent_diffusivity(
        p["D_par"], p["D_perp"], p["axis"], bvecs
    )
    assert np.all(D_app >= p["D_perp"] - 1e-12)
    assert np.all(D_app <= p["D_par"] + 1e-12)
