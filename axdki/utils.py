"""
utils.py — Helper Functions for Axisymmetric DKI
=================================================

Utility functions for:
  - DTI tensor fitting (Step 1: estimate principal axis)
  - Axisymmetric tensor construction from fitted parameters
  - Metric map computation (FA, MD, MK, AK, RK, Wbar)
  - NIfTI I/O helpers

Note on DIPY integration
------------------------
The DTI fitting here is a minimal standalone implementation so the
package works without depending on DIPY. In the DIPY integration
(dipy/reconst/axdki.py), replace ``fit_dti_ols`` with:

    from dipy.reconst.dti import TensorModel
    dti_fit = TensorModel(gtab).fit(data)
    axis = dti_fit.evecs[..., 0]   # principal eigenvector
"""

import numpy as np


# ---------------------------------------------------------------------------
# Step 1 helpers: DTI fit to estimate the principal axis
# ---------------------------------------------------------------------------

def build_dti_design_matrix(bvals, bvecs):
    """Build the DTI log-linear design matrix.

    Constructs the standard 7-column DTI design matrix for the model:

        log(S) = log(S0) - b * [Dxx*gx² + 2Dxy*gx*gy + ... ]

    Parameters
    ----------
    bvals : ndarray, shape (N,)
        b-values in s/mm².
    bvecs : ndarray, shape (N, 3)
        Unit gradient directions (rows). Rows with bval=0 get zero rows.

    Returns
    -------
    A : ndarray, shape (N, 7)
        Design matrix. Columns: [log(S0), -b*gx², -2b*gx*gy, -2b*gx*gz,
        -b*gy², -2b*gy*gz, -b*gz²].
    """
    bvecs = np.atleast_2d(bvecs)
    gx, gy, gz = bvecs[:, 0], bvecs[:, 1], bvecs[:, 2]
    b = bvals

    A = np.column_stack([
        np.ones(len(b)),       # log(S0)
        -b * gx * gx,          # Dxx
        -2 * b * gx * gy,      # Dxy
        -2 * b * gx * gz,      # Dxz
        -b * gy * gy,          # Dyy
        -2 * b * gy * gz,      # Dyz
        -b * gz * gz,          # Dzz
    ])
    return A


def fit_dti_ols(bvals, bvecs, log_signal):
    """Fit the diffusion tensor using ordinary least squares (OLS).

    This is a minimal DTI fit used only to estimate the principal
    diffusion axis (n̂) for Step 1 of the axisymmetric fitting.

    In the DIPY integration, replace this with dipy.reconst.dti.TensorModel.

    Parameters
    ----------
    bvals : ndarray, shape (N,)
        b-values in s/mm².
    bvecs : ndarray, shape (N, 3)
        Unit gradient directions.
    log_signal : ndarray, shape (N,)
        Natural log of the measured signal (log(S)).

    Returns
    -------
    D : ndarray, shape (3, 3)
        Fitted diffusion tensor (symmetric).
    evecs : ndarray, shape (3, 3)
        Eigenvectors as columns, sorted by descending eigenvalue.
    evals : ndarray, shape (3,)
        Eigenvalues in descending order (λ1 ≥ λ2 ≥ λ3).
    principal_axis : ndarray, shape (3,)
        Unit principal eigenvector (n̂) corresponding to λ1.
    """
    A = build_dti_design_matrix(bvals, bvecs)
    params, _, _, _ = np.linalg.lstsq(A, log_signal, rcond=None)

    # Reconstruct symmetric tensor from 6 unique elements
    # params = [log_S0, Dxx, Dxy, Dxz, Dyy, Dyz, Dzz]
    D = np.array([
        [params[1], params[2], params[3]],
        [params[2], params[4], params[5]],
        [params[3], params[5], params[6]],
    ])

    evals, evecs = np.linalg.eigh(D)
    # eigh returns ascending order — reverse to descending
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    principal_axis = evecs[:, 0]
    return D, evecs, evals, principal_axis


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_fa(D_par, D_perp):
    """Compute Fractional Anisotropy from axisymmetric eigenvalues.

    For an axisymmetric tensor, λ1 = D_par and λ2 = λ3 = D_perp.

    Parameters
    ----------
    D_par : float or ndarray
        Parallel diffusivity D∥ (λ1).
    D_perp : float or ndarray
        Perpendicular diffusivity D⊥ (λ2 = λ3).

    Returns
    -------
    FA : float or ndarray
        Fractional Anisotropy in [0, 1].
    """
    D_par = np.asarray(D_par)
    D_perp = np.asarray(D_perp)
    MD = (D_par + 2 * D_perp) / 3.0
    numerator = np.sqrt(2) * np.sqrt((D_par - MD)**2 + 2 * (D_perp - MD)**2)
    denominator = np.sqrt(2) * np.sqrt(D_par**2 + 2 * D_perp**2)
    # Avoid division by zero in isotropic case
    FA = np.where(denominator > 0, numerator / denominator, 0.0)
    return FA


def compute_md(D_par, D_perp):
    """Compute Mean Diffusivity.

    Parameters
    ----------
    D_par : float or ndarray
        Parallel diffusivity D∥.
    D_perp : float or ndarray
        Perpendicular diffusivity D⊥.

    Returns
    -------
    MD : float or ndarray
        Mean Diffusivity = (D∥ + 2·D⊥) / 3.
    """
    return (np.asarray(D_par) + 2 * np.asarray(D_perp)) / 3.0


def compute_mk(W_par, W_perp):
    """Compute Mean Kurtosis.

    For the axisymmetric model, MK is the orientational average of the
    apparent kurtosis over a sphere.

    Parameters
    ----------
    W_par : float or ndarray
        Parallel kurtosis W∥.
    W_perp : float or ndarray
        Perpendicular kurtosis W⊥.

    Returns
    -------
    MK : float or ndarray
        Mean Kurtosis.

    Notes
    -----
    TODO: Verify the exact MK expression for the axisymmetric case.
          The expression below assumes a simple average — confirm from
          Hamilton et al. or Jensen & Helpern (2010).
    """
    # TODO: replace with exact analytical expression from the paper
    return (np.asarray(W_par) + 2 * np.asarray(W_perp)) / 3.0


def compute_wbar(W_par, W_perp, D_par, D_perp):
    """Compute the mean kurtosis tensor scalar W̄.

    W̄ is a rotationally invariant kurtosis measure defined in terms of
    the kurtosis tensor W_ijkl.

    Parameters
    ----------
    W_par : float or ndarray
        Parallel kurtosis W∥.
    W_perp : float or ndarray
        Perpendicular kurtosis W⊥.
    D_par : float or ndarray
        Parallel diffusivity D∥.
    D_perp : float or ndarray
        Perpendicular diffusivity D⊥.

    Returns
    -------
    Wbar : float or ndarray
        Mean kurtosis tensor scalar W̄.

    Notes
    -----
    TODO: Derive the exact expression for W̄ in the axisymmetric case
          from Hamilton et al. eq. (X). Placeholder below.
    """
    # TODO: implement correct expression
    raise NotImplementedError(
        "compute_wbar: derive from Hamilton et al. before implementing."
    )


def compute_all_metrics(D_par, D_perp, W_par, W_perp):
    """Compute all standard DKI metrics from axisymmetric parameters.

    Parameters
    ----------
    D_par : float or ndarray
        Parallel diffusivity D∥.
    D_perp : float or ndarray
        Perpendicular diffusivity D⊥.
    W_par : float or ndarray
        Parallel kurtosis W∥.
    W_perp : float or ndarray
        Perpendicular kurtosis W⊥.

    Returns
    -------
    metrics : dict
        Dictionary with keys: 'FA', 'MD', 'AK', 'RK', 'MK'.
        'Wbar' is omitted until compute_wbar is implemented.
    """
    return {
        "FA": compute_fa(D_par, D_perp),
        "MD": compute_md(D_par, D_perp),
        "AK": np.asarray(W_par),   # Axial Kurtosis = W∥
        "RK": np.asarray(W_perp),  # Radial Kurtosis = W⊥
        "MK": compute_mk(W_par, W_perp),
    }


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_nifti(path):
    """Load a NIfTI file and return (data, affine).

    Parameters
    ----------
    path : str or Path
        Path to .nii or .nii.gz file.

    Returns
    -------
    data : ndarray
        Image data array.
    affine : ndarray, shape (4, 4)
        Voxel-to-world affine matrix.
    """
    import nibabel as nib
    img = nib.load(str(path))
    return np.asarray(img.dataobj), img.affine


def save_nifti(data, affine, path):
    """Save an array as a NIfTI file.

    Parameters
    ----------
    data : ndarray
        Data to save.
    affine : ndarray, shape (4, 4)
        Voxel-to-world affine matrix.
    path : str or Path
        Output path (.nii or .nii.gz).
    """
    import nibabel as nib
    img = nib.Nifti1Image(data.astype(np.float32), affine)
    nib.save(img, str(path))


def load_bvals_bvecs(bval_path, bvec_path):
    """Load FSL-format bvals and bvecs files.

    Parameters
    ----------
    bval_path : str or Path
        Path to .bval file (space-separated values, single row).
    bvec_path : str or Path
        Path to .bvec file (3 rows × N columns).

    Returns
    -------
    bvals : ndarray, shape (N,)
        b-values in s/mm².
    bvecs : ndarray, shape (N, 3)
        Unit gradient directions (transposed from FSL convention).
    """
    bvals = np.loadtxt(str(bval_path))
    bvecs = np.loadtxt(str(bvec_path)).T  # FSL: (3, N) → (N, 3)
    return bvals, bvecs
