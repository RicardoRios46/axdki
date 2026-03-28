"""
axdki — Axisymmetric Diffusion Kurtosis Imaging
================================================

Standalone Python translation of the axisymmetric DKI fitting algorithm
from the CFMM MatMRI MATLAB toolbox (nii2kurt.m).

This package is a working prototype intended to be ported into the DIPY
framework (dipy/reconst/axdki.py) once validated.

Reference
---------
Hamilton et al. (2024). Axisymmetric diffusion kurtosis imaging with a
small number of diffusion encoding directions.
https://pmc.ncbi.nlm.nih.gov/articles/PMC12224416/

Modules
-------
signal  : Forward signal model (axisymmetric parameterisation)
model   : AxSymDKIModel and AxSymDKIFit — fitting pipeline
utils   : Helper functions (tensor ops, metric maps, I/O)
"""

from .model import AxSymDKIModel, AxSymDKIFit
from .signal import axsym_signal, axsym_design_matrix

__all__ = [
    "AxSymDKIModel",
    "AxSymDKIFit",
    "axsym_signal",
    "axsym_design_matrix",
]
