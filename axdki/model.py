"""
model.py — Axisymmetric DKI Model and Fit
==========================================

Provides the two main classes:

  AxSymDKIModel — takes gradient table + data, runs the two-step fit
  AxSymDKIFit   — holds fitted parameters, exposes metric properties

This follows the same interface pattern as DIPY's ReconstModel/ReconstFit
so the eventual port to dipy/reconst/axdki.py is straightforward.

Two-step fitting algorithm
--------------------------
Step 1: Fit the diffusion tensor (DTI) using OLS on the log-signal to
        estimate the principal diffusion axis n̂.
        → In standalone mode: utils.fit_dti_ols
        → In DIPY integration: dipy.reconst.dti.TensorModel

Step 2: Fit the axisymmetric kurtosis model with n̂ fixed, estimating
        [D_par, D_perp, W_par, W_perp] using nonlinear least squares.
        → scipy.optimize.least_squares (CPU)
        → GPU backend: future work (CuPy / PyTorch)

Modular design
--------------
The ``method`` argument on AxSymDKIModel controls the fitting backend.
Currently only 'nlls' (nonlinear least squares) is implemented.
'wls' (weighted linear least squares via the design matrix) is stubbed
and should be added next for speed comparison.
"""

import numpy as np
from scipy.optimize import least_squares

from .signal import axsym_signal
from .utils import fit_dti_ols, compute_all_metrics


class AxSymDKIModel:
    """Axisymmetric Diffusion Kurtosis Imaging model.

    Fits the 5-parameter axisymmetric DKI model to multi-shell dMRI data
    using a two-step algorithm: DTI fit to find the principal axis, then
    constrained kurtosis fitting.

    Parameters
    ----------
    bvals : ndarray, shape (N,)
        b-values in s/mm².
    bvecs : ndarray, shape (N, 3)
        Unit gradient directions (one per row).
    method : str, optional
        Fitting method. Currently supported: 'nlls'.
        'wls' is planned but not yet implemented.
    b0_threshold : float, optional
        b-values below this threshold are treated as b=0 (default 50).

    Examples
    --------
    >>> model = AxSymDKIModel(bvals, bvecs)
    >>> fit = model.fit(data)           # data shape: (X, Y, Z, N)
    >>> fa_map = fit.fa                 # shape: (X, Y, Z)
    >>> mk_map = fit.mk
    """

    def __init__(self, bvals, bvecs, method="nlls", b0_threshold=50):
        self.bvals = np.asarray(bvals, dtype=float)
        self.bvecs = np.asarray(bvecs, dtype=float)
        self.method = method
        self.b0_threshold = b0_threshold

        # Masks for b=0 and diffusion-weighted volumes
        self._b0_mask = self.bvals < b0_threshold
        self._dw_mask = ~self._b0_mask

        if self._b0_mask.sum() == 0:
            raise ValueError("No b=0 volumes found. Check b0_threshold.")
        if self._dw_mask.sum() < 10:
            raise ValueError(
                f"Only {self._dw_mask.sum()} diffusion-weighted directions found. "
                "Axisymmetric DKI requires at least 10."
            )

    def fit(self, data, mask=None):
        """Fit the axisymmetric DKI model to dMRI data.

        Parameters
        ----------
        data : ndarray, shape (..., N)
            dMRI signal. Last dimension must match bvals/bvecs length.
            Can be a single voxel (shape (N,)) or a volume (X, Y, Z, N).
        mask : ndarray, shape (...), optional
            Boolean brain mask. Voxels where mask=False are skipped and
            their parameters are set to zero.

        Returns
        -------
        fit : AxSymDKIFit
            Object holding fitted parameters and metric properties.
        """
        data = np.asarray(data, dtype=float)
        original_shape = data.shape[:-1]  # spatial dimensions

        # Flatten spatial dimensions for voxel-wise loop
        data_2d = data.reshape(-1, data.shape[-1])  # (V, N)
        n_voxels = data_2d.shape[0]

        if mask is not None:
            mask_flat = np.asarray(mask, dtype=bool).reshape(-1)
        else:
            mask_flat = np.ones(n_voxels, dtype=bool)

        # Output arrays: [D_par, D_perp, W_par, W_perp, S0]
        params = np.zeros((n_voxels, 5))

        for v in range(n_voxels):
            if not mask_flat[v]:
                continue
            signal = data_2d[v]
            try:
                params[v] = self._fit_voxel(signal)
            except Exception:
                # Failed voxels are left as zero
                pass

        params = params.reshape(original_shape + (5,))
        return AxSymDKIFit(self, params)

    def _fit_voxel(self, signal):
        """Fit a single voxel. Returns [D_par, D_perp, W_par, W_perp, S0].

        Parameters
        ----------
        signal : ndarray, shape (N,)
            Raw signal for one voxel.

        Returns
        -------
        params : ndarray, shape (5,)
            [D_par, D_perp, W_par, W_perp, S0]
        """
        # --- Estimate S0 from mean of b=0 volumes ---
        S0 = np.mean(signal[self._b0_mask])
        if S0 <= 0:
            return np.zeros(5)

        # --- Step 1: DTI fit to estimate principal axis ---
        log_signal = np.log(np.maximum(signal, 1e-10))
        _, _, _, axis = fit_dti_ols(
            self.bvals, self.bvecs, log_signal
        )

        # --- Step 2: Axisymmetric kurtosis fit (NLLS) ---
        if self.method == "nlls":
            return self._fit_nlls(signal, S0, axis)
        elif self.method == "wls":
            raise NotImplementedError(
                "WLS fitting not yet implemented. Use method='nlls'."
            )
        else:
            raise ValueError(f"Unknown fitting method: {self.method!r}")

    def _fit_nlls(self, signal, S0, axis):
        """Nonlinear least squares fit for [D_par, D_perp, W_par, W_perp].

        Parameters
        ----------
        signal : ndarray, shape (N,)
            Raw signal.
        S0 : float
            Estimated baseline signal.
        axis : ndarray, shape (3,)
            Fixed principal diffusion axis from Step 1.

        Returns
        -------
        params : ndarray, shape (5,)
            [D_par, D_perp, W_par, W_perp, S0]
        """
        bvals_dw = self.bvals[self._dw_mask]
        bvecs_dw = self.bvecs[self._dw_mask]
        signal_dw = signal[self._dw_mask]

        def residuals(x):
            D_par, D_perp, W_par, W_perp = x
            S_pred = axsym_signal(
                S0, bvals_dw, bvecs_dw, D_par, D_perp, W_par, W_perp, axis
            )
            return signal_dw - S_pred

        # Initial guess: isotropic diffusion, zero kurtosis
        # Typical white matter values in SI units (mm²/s)
        x0 = [1.5e-3, 0.5e-3, 1.0, 1.0]

        # Parameter bounds: diffusivities > 0, kurtosis in [0, 3]
        bounds = (
            [0,    0,    0, 0],   # lower
            [4e-3, 4e-3, 3, 3],   # upper
        )

        result = least_squares(residuals, x0, bounds=bounds, method="trf")
        D_par, D_perp, W_par, W_perp = result.x
        return np.array([D_par, D_perp, W_par, W_perp, S0])


class AxSymDKIFit:
    """Holds the result of an axisymmetric DKI fit.

    Exposes fitted parameters and derived metric maps as properties,
    mirroring the interface of dipy.reconst.dki.DiffusionKurtosisFit.

    Parameters
    ----------
    model : AxSymDKIModel
        The model that produced this fit.
    params : ndarray, shape (..., 5)
        Fitted parameters per voxel: [D_par, D_perp, W_par, W_perp, S0].
    """

    def __init__(self, model, params):
        self.model = model
        self.params = params

    @property
    def D_par(self):
        """Parallel diffusivity D∥ map."""
        return self.params[..., 0]

    @property
    def D_perp(self):
        """Perpendicular diffusivity D⊥ map."""
        return self.params[..., 1]

    @property
    def W_par(self):
        """Parallel kurtosis W∥ (Axial Kurtosis) map."""
        return self.params[..., 2]

    @property
    def W_perp(self):
        """Perpendicular kurtosis W⊥ (Radial Kurtosis) map."""
        return self.params[..., 3]

    @property
    def S0(self):
        """Baseline signal S0 map."""
        return self.params[..., 4]

    @property
    def fa(self):
        """Fractional Anisotropy map."""
        metrics = compute_all_metrics(self.D_par, self.D_perp, self.W_par, self.W_perp)
        return metrics["FA"]

    @property
    def md(self):
        """Mean Diffusivity map."""
        metrics = compute_all_metrics(self.D_par, self.D_perp, self.W_par, self.W_perp)
        return metrics["MD"]

    @property
    def mk(self):
        """Mean Kurtosis map."""
        metrics = compute_all_metrics(self.D_par, self.D_perp, self.W_par, self.W_perp)
        return metrics["MK"]

    @property
    def ak(self):
        """Axial Kurtosis map (W∥)."""
        return self.W_par

    @property
    def rk(self):
        """Radial Kurtosis map (W⊥)."""
        return self.W_perp

    def predict(self, bvals, bvecs, axis):
        """Predict the signal from the fitted parameters.

        Parameters
        ----------
        bvals : ndarray, shape (N,)
            b-values to predict at.
        bvecs : ndarray, shape (N, 3)
            Gradient directions to predict at.
        axis : ndarray, shape (3,)
            Symmetry axis n̂.

        Returns
        -------
        S_pred : ndarray, shape (..., N)
            Predicted signal.

        Notes
        -----
        TODO: For volumetric data, this needs to be vectorised over voxels
              or applied per-voxel with the voxel's own fitted axis.
              Currently only reliable for single-voxel fits.
        """
        return axsym_signal(
            self.S0, bvals, bvecs,
            self.D_par, self.D_perp,
            self.W_par, self.W_perp,
            axis,
        )
