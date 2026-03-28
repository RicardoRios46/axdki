import numpy as np
from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti, save_nifti

import dipy.reconst.dti as dti
from dipy.data import get_fnames

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

tenmodel = dti.TensorModel(gtab, fit_method="WLS")

tenfit = tenmodel.fit(maskdata)

eigen_vec_1 = tenfit.evecs.astype(np.float32)[:,:,:,0] # extract first eigenvector
#save_nifti("tensor_evecs_1.nii.gz", tenfit.evecs.astype(np.float32), affine)

"""
NO NEED vectors are alreay normalized
# eigenvectors normalization 
v_norm = np.linalg.norm(eigen_vec_1, axis=-1, keepdims=True)
eigen_vec_1 = eigen_vec_1 / (v_norm + 1e-12)

# bvec normalization 
g_norm = np.linalg.norm(bvecs, axis=1, keepdims=True)
bvec = bvecs / (g_norm + 1e-12)
"""


#save_nifti("cos_theta_nonnorm.nii.gz", cos_theta, affine)


#### DKI PART

# calculus of cos(theta)
cos_theta = np.sum(
    eigen_vec_1[..., np.newaxis, :] * 
    bvecs[np.newaxis, np.newaxis, np.newaxis, :, :],
    axis=-1
)
