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


b = bvals[None, None, None, :]
c = cos_theta
c2 = c**2
c4 = c**4

A = np.empty(c.shape + (6,), dtype=c.dtype)

A[..., 0] = 1
A[..., 1] = -b * (1 - c2)
A[..., 2] = -b * c2
A[..., 3] = (b**2 / 6) * (5*c4 - 6*c2 + 1)
A[..., 4] = (b**2 / 6) * (0.5 * c2 * (5*c2 - 3))
A[..., 5] = (b**2 / 6) * (-15/2 * (c4 - c2))

A = A.reshape(-1, 6)
maskdata_log = np.log(maskdata)
maskdata_log = maskdata_log.reshape(-1)


nx, ny, nz, nt = maskdata.shape

S = np.log(np.clip(maskdata, 1e-6, None))
S = S.reshape(-1, nt)
A = A.reshape(-1, nt, 6)
mask = mask.reshape(-1)

X = np.zeros((S.shape[0], 6))

I = np.eye(6) * 1e-6

for v in range(S.shape[0]):
    if not mask[v]:
        continue
    Av = A[v]
    y = S[v]
    ATA = Av.T @ Av
    ATy = Av.T @ y
    X[v] = np.linalg.solve(ATA + I, ATy)

X = X.reshape(nx, ny, nz, 6)


save_nifti("/nfs/khan/trainees/larcamon/baronproject/WIP/brainhack/axdki/data_sample/test_lucas/S0.nii.gz", X[:,:,:,0], affine)
save_nifti("/nfs/khan/trainees/larcamon/baronproject/WIP/brainhack/axdki/data_sample/test_lucas/Dperp.nii.gz", X[:,:,:,1], affine)
save_nifti("/nfs/khan/trainees/larcamon/baronproject/WIP/brainhack/axdki/data_sample/test_lucas/Wmean.nii.gz", X[:,:,:,5], affine)
