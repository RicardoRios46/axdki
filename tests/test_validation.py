"""
test_validation.py — Numerical validation against MATLAB reference outputs

These tests compare metric maps produced by AxSymDKIModel against
the reference outputs saved from running nii2kurt.m on Day 1.

HOW TO USE
----------
1. On Day 1, run nii2kurt.m on the CFIN dataset and save outputs:

    In MATLAB:
        out = nii2kurt('path/to/data.nii.gz', bvals, bvecs);
        save('reference/matlab_outputs.mat', 'out');

   Or save individual NIfTI maps:
        niftiwrite(out.FA,  'reference/FA_matlab.nii');
        niftiwrite(out.MD,  'reference/MD_matlab.nii');
        niftiwrite(out.MK,  'reference/MK_matlab.nii');
        niftiwrite(out.AK,  'reference/AK_matlab.nii');
        niftiwrite(out.RK,  'reference/RK_matlab.nii');

2. Set REFERENCE_DIR below (or set env var AXDKI_REFERENCE_DIR).
3. Run: pixi run test

All tests here are marked with @pytest.mark.validation and are skipped
automatically if reference files are not found. This way the test suite
still runs cleanly on CI or for team members without MATLAB access.
"""

import os
import numpy as np
import pytest

from axdki.model import AxSymDKIModel
from axdki.utils import load_nifti, load_bvals_bvecs

# ---------------------------------------------------------------------------
# Configuration — point this to your reference output directory
# ---------------------------------------------------------------------------

REFERENCE_DIR = os.environ.get(
    "AXDKI_REFERENCE_DIR",
    "reference"   # relative to repo root
)

DATA_PATH  = os.path.join(REFERENCE_DIR, "data.nii.gz")
BVAL_PATH  = os.path.join(REFERENCE_DIR, "data.bval")
BVEC_PATH  = os.path.join(REFERENCE_DIR, "data.bvec")
MASK_PATH  = os.path.join(REFERENCE_DIR, "mask.nii.gz")

MATLAB_FA  = os.path.join(REFERENCE_DIR, "FA_matlab.nii")
MATLAB_MD  = os.path.join(REFERENCE_DIR, "MD_matlab.nii")
MATLAB_MK  = os.path.join(REFERENCE_DIR, "MK_matlab.nii")
MATLAB_AK  = os.path.join(REFERENCE_DIR, "AK_matlab.nii")
MATLAB_RK  = os.path.join(REFERENCE_DIR, "RK_matlab.nii")

# Tolerance for metric map comparison (absolute difference within mask)
# These are intentionally loose for a first-pass comparison.
# Tighten as the implementation matures.
TOLERANCES = {
    "FA": 0.05,
    "MD": 5e-5,   # mm²/s
    "MK": 0.1,
    "AK": 0.15,
    "RK": 0.15,
}


# ---------------------------------------------------------------------------
# Skip condition: skip all validation tests if reference files are absent
# ---------------------------------------------------------------------------

reference_available = all(os.path.exists(p) for p in [
    DATA_PATH, BVAL_PATH, BVEC_PATH, MATLAB_FA
])

skip_if_no_reference = pytest.mark.skipif(
    not reference_available,
    reason=(
        "Reference MATLAB outputs not found. "
        f"Set AXDKI_REFERENCE_DIR or populate {REFERENCE_DIR}/. "
        "See test_validation.py docstring for instructions."
    )
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def python_fit():
    """Run AxSymDKIModel on the reference dataset (cached per test module)."""
    data, affine = load_nifti(DATA_PATH)
    bvals, bvecs  = load_bvals_bvecs(BVAL_PATH, BVEC_PATH)

    mask = None
    if os.path.exists(MASK_PATH):
        mask, _ = load_nifti(MASK_PATH)
        mask = mask.astype(bool)

    model = AxSymDKIModel(bvals, bvecs)
    return model.fit(data, mask=mask), mask


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

@skip_if_no_reference
def test_fa_vs_matlab(python_fit):
    """FA map must match MATLAB reference within tolerance."""
    fit, mask = python_fit
    fa_matlab, _ = load_nifti(MATLAB_FA)
    _compare_maps(fit.fa, fa_matlab, mask, "FA", TOLERANCES["FA"])


@skip_if_no_reference
def test_md_vs_matlab(python_fit):
    """MD map must match MATLAB reference within tolerance."""
    fit, mask = python_fit
    md_matlab, _ = load_nifti(MATLAB_MD)
    _compare_maps(fit.md, md_matlab, mask, "MD", TOLERANCES["MD"])


@skip_if_no_reference
def test_mk_vs_matlab(python_fit):
    """MK map must match MATLAB reference within tolerance."""
    fit, mask = python_fit
    mk_matlab, _ = load_nifti(MATLAB_MK)
    _compare_maps(fit.mk, mk_matlab, mask, "MK", TOLERANCES["MK"])


@skip_if_no_reference
def test_ak_vs_matlab(python_fit):
    """AK map must match MATLAB reference within tolerance."""
    fit, mask = python_fit
    ak_matlab, _ = load_nifti(MATLAB_AK)
    _compare_maps(fit.ak, ak_matlab, mask, "AK", TOLERANCES["AK"])


@skip_if_no_reference
def test_rk_vs_matlab(python_fit):
    """RK map must match MATLAB reference within tolerance."""
    fit, mask = python_fit
    rk_matlab, _ = load_nifti(MATLAB_RK)
    _compare_maps(fit.rk, rk_matlab, mask, "RK", TOLERANCES["RK"])


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _compare_maps(python_map, matlab_map, mask, metric_name, tol):
    """Compare two metric maps within a mask and report statistics."""
    if mask is not None:
        diff = np.abs(python_map[mask] - matlab_map[mask])
    else:
        diff = np.abs(python_map - matlab_map)

    mean_diff = diff.mean()
    max_diff  = diff.max()
    pct_within = 100 * (diff < tol).mean()

    print(f"\n{metric_name} comparison:")
    print(f"  Mean |diff| = {mean_diff:.6f}")
    print(f"  Max  |diff| = {max_diff:.6f}")
    print(f"  % within tol ({tol}) = {pct_within:.1f}%")

    assert mean_diff < tol, (
        f"{metric_name}: mean absolute difference {mean_diff:.6f} exceeds "
        f"tolerance {tol}. Max diff = {max_diff:.6f}."
    )
