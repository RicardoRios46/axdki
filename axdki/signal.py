"""
signal.py — Axisymmetric DKI Forward Signal Model
==================================================

This module implements the forward signal model for axisymmetric DKI.

The axisymmetric model assumes rotational symmetry around the principal
diffusion axis (n̂), reducing the kurtosis tensor to two independent
parameters: W∥ (parallel) and W⊥ (perpendicular).

The two-step algorithm (mirroring nii2kurt.m) is:
  Step 1 — Estimate the principal diffusion axis n̂ from a DTI fit.
            In this standalone version we use a simple WLS tensor fit.
            In the DIPY integration, this will be replaced by
            dipy.reconst.dti.TensorModel.
  Step 2 — Fit the axisymmetric kurtosis model, constraining the
            symmetry axis to n̂ from Step 1.

Reference
---------
Hamilton et al. (2024), Section 2 (Theory).
"""

import numpy as np


def axsym_signal(S0, bvals, bvecs, D_par, D_perp, W_par, W_perp, axis):
    """Predict the dMRI signal under the axisymmetric DKI model.

    The signal equation is the standard DKI cumulant expansion:

        S(b, g) = S0 * exp(-b * D_app + (1/6) * b^2 * D_app^2 * W_app)

    where D_app and W_app are the apparent (direction-dependent)
    diffusivity and kurtosis computed from the axisymmetric parameters.

    Parameters
    ----------
    S0 : float
        Baseline signal (b=0).
    bvals : ndarray, shape (N,)
        b-values in s/mm².
    bvecs : ndarray, shape (N, 3)
        Unit gradient directions (rows). b=0 directions are ignored.
    D_par : float
        Parallel diffusivity D∥ (along n̂), in mm²/s.
    D_perp : float
        Perpendicular diffusivity D⊥ (transverse to n̂), in mm²/s.
    W_par : float
        Parallel kurtosis W∥.
    W_perp : float
        Perpendicular kurtosis W⊥.
    axis : ndarray, shape (3,)
        Unit vector n̂ — principal symmetry axis.

    Returns
    -------
    S : ndarray, shape (N,)
        Predicted signal for each (b, g) pair.

    Notes
    -----
    TODO: Verify the exact formula for W_app from Hamilton et al. eq. (X).
          The expression below is a placeholder — confirm from the paper
          before using for validation.
    """
    axis = axis / np.linalg.norm(axis)
    bvecs = np.atleast_2d(bvecs)

    # Cosine squared of angle between each gradient direction and n̂
    cos2 = np.dot(bvecs, axis) ** 2  # shape (N,)

    # Apparent diffusivity (direction-dependent)
    D_app = D_perp + (D_par - D_perp) * cos2  # shape (N,)

    # Apparent kurtosis (direction-dependent)
    # TODO: confirm exact W_app expression from Hamilton et al. eq. (X)
    W_app = W_perp + (W_par - W_perp) * cos2  # placeholder — verify

    # DKI signal model (2nd-order cumulant expansion)
    log_S = -bvals * D_app + (1.0 / 6.0) * bvals**2 * D_app**2 * W_app
    S = S0 * np.exp(log_S)

    return S


def axsym_design_matrix(bvals, bvecs, axis):
    """Build the linearised design matrix for axisymmetric DKI fitting.

    Expresses the log-signal as a linear combination of [D_par, D_perp,
    W_par, W_perp] given a fixed axis n̂. This enables a fast linear
    least-squares solve (OLS/WLS) as a first-pass estimator.

    Parameters
    ----------
    bvals : ndarray, shape (N,)
        b-values in s/mm².
    bvecs : ndarray, shape (N, 3)
        Unit gradient directions (rows).
    axis : ndarray, shape (3,)
        Fixed unit symmetry axis n̂ (estimated from DTI in Step 1).

    Returns
    -------
    A : ndarray, shape (N, 4)
        Design matrix. Columns correspond to [D_par, D_perp, W_par, W_perp].

    Notes
    -----
    TODO: Derive the exact column expressions from the signal model.
          This requires expanding D_app and W_app in terms of cos²(θ)
          and collecting terms by parameter. See Hamilton et al. eq. (X).
    """
    axis = axis / np.linalg.norm(axis)
    bvecs = np.atleast_2d(bvecs)

    cos2 = np.dot(bvecs, axis) ** 2  # shape (N,)

    # Placeholder columns — TODO: derive exact expressions
    # These should come from linearising log(S) ≈ -b*D_app + b²*D_app²*W_app/6
    col_D_par  = NotImplemented  # function of b, cos2
    col_D_perp = NotImplemented
    col_W_par  = NotImplemented
    col_W_perp = NotImplemented

    raise NotImplementedError(
        "axsym_design_matrix: derive column expressions from Hamilton et al. "
        "before implementing. See signal.py docstring for guidance."
    )


def compute_apparent_diffusivity(D_par, D_perp, axis, bvecs):
    """Compute apparent diffusivity for each gradient direction.

    Parameters
    ----------
    D_par : float
        Parallel diffusivity D∥.
    D_perp : float
        Perpendicular diffusivity D⊥.
    axis : ndarray, shape (3,)
        Unit symmetry axis n̂.
    bvecs : ndarray, shape (N, 3)
        Unit gradient directions.

    Returns
    -------
    D_app : ndarray, shape (N,)
        Apparent diffusivity for each gradient direction.
    """
    axis = axis / np.linalg.norm(axis)
    cos2 = np.dot(np.atleast_2d(bvecs), axis) ** 2
    return D_perp + (D_par - D_perp) * cos2
