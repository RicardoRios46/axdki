"""
=====================================================================
Reconstruction of the diffusion signal with DTI (single tensor) model
=====================================================================

The diffusion tensor model is a model that describes the diffusion within a
voxel. First proposed by Basser and colleagues :footcite:p:`Basser1994a`, it has
been very influential in demonstrating the utility of diffusion MRI in
characterizing the micro-structure of white matter tissue and of the biophysical
properties of tissue, inferred from local diffusion properties and it is still
very commonly used.

The diffusion tensor models the diffusion signal as:

.. math::

    \\frac{S(\\mathbf{g}, b)}{S_0} = e^{-b\\mathbf{g}^T \\mathbf{D} \\mathbf{g}}

Where $\\mathbf{g}$ is a unit vector in 3 space indicating the direction of
measurement and b are the parameters of measurement, such as the strength and
duration of diffusion-weighting gradient. $S(\\mathbf{g}, b)$ is the
diffusion-weighted signal measured and $S_0$ is the signal conducted in a
measurement with no diffusion weighting. $\\mathbf{D}$ is a positive-definite
quadratic form, which contains six free parameters to be fit. These six
parameters are:

.. math::

    \\mathbf{D} = \\begin{pmatrix} D_{xx} & D_{xy} & D_{xz} \\\\
                       D_{yx} & D_{yy} & D_{yz} \\\\
                       D_{zx} & D_{zy} & D_{zz} \\\\ \\end{pmatrix}

This matrix is a variance/covariance matrix of the diffusivity along the three
spatial dimensions. Note that we can assume that diffusivity has antipodal
symmetry, so elements across the diagonal are equal. For example:
$D_{xy} = D_{yx}$. This is why there are only 6 free parameters to estimate
here.

In the following example we show how to reconstruct your diffusion datasets
using a single tensor model.

First import the necessary modules:

``numpy`` is for numerical computation

"""

import numpy as np

###############################################################################
# ``dipy.io.image`` is for loading / saving imaging datasets
# ``dipy.io.gradients`` is for loading / saving our bvals and bvecs


from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti, save_nifti

###############################################################################
# ``dipy.reconst`` is for the reconstruction algorithms which we use to create
# voxel models from the raw data.


import dipy.reconst.dti as dti

###############################################################################
# ``dipy.data`` is used for small datasets that we use in tests and examples.


from dipy.data import get_fnames

###############################################################################
# ``get_fnames`` will download the raw dMRI dataset of a single subject.
# The size of the dataset is 87 MBytes. You only need to fetch once. It
# will return the file names of our data.


hardi_fname, hardi_bval_fname, hardi_bvec_fname = get_fnames(name="stanford_hardi")
test_fname = '/nfs/khan/trainees/larcamon/baronproject/WIP/brainhack/axdki/data_sample/sub-01/dwi/sub-01_run-01_acq-lte_desc-preproc_dwi.nii.gz'
test_bval_fname = '/nfs/khan/trainees/larcamon/baronproject/WIP/brainhack/axdki/data_sample/sub-01/dwi/sub-01_run-01_acq-lte_desc-preproc_dwi.bval'
test_bvec_fname = '/nfs/khan/trainees/larcamon/baronproject/WIP/brainhack/axdki/data_sample/sub-01/dwi/sub-01_run-01_acq-lte_desc-preproc_dwi.bvec'

data, affine = load_nifti(test_fname)
bvals, bvecs = read_bvals_bvecs(test_bval_fname, test_bvec_fname)
gtab = gradient_table(bvals, bvecs=bvecs)

from dipy.segment.mask import bounding_box, crop, median_otsu

maskdata, mask = median_otsu(data, vol_idx=range(10, 49), median_radius=3, numpass=1, dilate=2)

mins, maxs = bounding_box(mask)

maskdata = crop(maskdata, mins, maxs)
mask = crop(mask, mins, maxs)
y
tenmodel = dti.TensorModel(gtab, fit_method="WLS")

###############################################################################
# The ``fit_method`` argument gives the method that will be used when fitting the
# data. Several options are available, such as weighted least squares ``WLS``
# (default), non-linear least squares ``NLLS``, as well as robust fitting methods
# such as ``RWLS`` and ``RNLLS`` as in :footcite:t:`Coveney2025`.

tenfit = tenmodel.fit(maskdata)

###############################################################################
# The fit method creates a ``TensorFit`` object which contains the fitting
# parameters and other attributes of the model. You can recover the 6 values
# of the triangular matrix representing the tensor D. By default, in DIPY, values
# are ordered as (Dxx, Dxy, Dyy, Dxz, Dyz, Dzz). The ``tensor_vals`` variable
# defined below is a 4D data with last dimension of size 6.


tensor_vals = dti.lower_triangular(tenfit.quadratic_form)

###############################################################################
# You can also recover other metrics from the model. For example we can generate
# fractional anisotropy (FA) from the eigen-values of the tensor. FA is used to
# characterize the degree to which the distribution of diffusion in a voxel is
# directional. That is, whether there is relatively unrestricted diffusion in one
# particular direction.
# 
# Mathematically, FA is defined as the normalized variance of the eigen-values of
# the tensor:
# 
# .. math::
# 
#         FA = \sqrt{\frac{1}{2} \cdot \frac{(\lambda_1-\lambda_2)^2 +
#             (\lambda_1-\lambda_3)^2 + (\lambda_2-\lambda_3)^2}
#             {\lambda_1^2 + \lambda_2^2 + \lambda_3^2}}
# 
# Where $\lambda_1$, $\lambda_2$ and $\lambda_3$ are the eigen-values of the
# tensor.
# 
# Note that FA should be interpreted carefully. It may be an indication of
# the density of packing of fibers in a voxel, and the amount of myelin wrapping
# these axons, but it is not always a measure of "tissue integrity". For example,
# FA may decrease in locations in which there is fanning of white matter fibers,
# or where more than one population of white matter fibers crosses.


print("Computing anisotropy measures (FA, MD, RGB)")
from dipy.reconst.dti import color_fa, fractional_anisotropy

FA = fractional_anisotropy(tenfit.evals)

FA[np.isnan(FA)] = 0

save_nifti("tensor_fa.nii.gz", FA.astype(np.float32), affine)
save_nifti("tensor_evecs.nii.gz", tenfit.evecs.astype(np.float32), affine)

MD1 = dti.mean_diffusivity(tenfit.evals)
save_nifti("tensors_md.nii.gz", MD1.astype(np.float32), affine)

MD2 = tenfit.md

FA = np.clip(FA, 0, 1)
RGB = color_fa(FA, tenfit.evecs)
save_nifti("tensor_rgb.nii.gz", np.array(255 * RGB, "uint8"), affine)

print("Computing tensor ellipsoids in a part of the splenium of the CC")

from dipy.data import get_sphere

sphere = get_sphere(name="repulsion724")

from dipy.viz import actor, window

# Enables/disables interactive visualization
interactive = False

scene = window.Scene()

evals = tenfit.evals[13:43, 44:74, 28:29]
evecs = tenfit.evecs[13:43, 44:74, 28:29]

cfa = RGB[13:43, 44:74, 28:29]
cfa /= cfa.max()

scene.add(
    actor.tensor_slicer(evals, evecs, scalar_colors=cfa, sphere=sphere, scale=0.3)
)

print("Saving illustration as tensor_ellipsoids.png")
window.record(
    scene=scene, n_frames=1, out_path="tensor_ellipsoids.png", size=(600, 600)
)
if interactive:
    window.show(scene)

scene.clear()

tensor_odfs = tenmodel.fit(data[20:50, 55:85, 38:39]).odf(sphere)

odf_actor = actor.odf_slicer(tensor_odfs, sphere=sphere, scale=0.5, colormap=None)
scene.add(odf_actor)
print("Saving illustration as tensor_odfs.png")
window.record(scene=scene, n_frames=1, out_path="tensor_odfs.png", size=(600, 600))
if interactive:
    window.show(scene)

