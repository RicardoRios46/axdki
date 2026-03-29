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

#A = A.reshape(-1, 6)
#maskdata_log = np.log(maskdata)
#maskdata_log = maskdata_log.reshape(-1)


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

Dperp=X[:,:,:,1]
Dpara=X[:,:,:,2]
Wperp=X[:,:,:,3]
Wpara=X[:,:,:,4]
Wmean=X[:,:,:,5]

#save_nifti("/nfs/khan/trainees/larcamon/baronproject/WIP/brainhack/axdki/data_sample/test_lucas/logS0.nii.gz", X[:,:,:,0], affine)
#save_nifti("/nfs/khan/trainees/larcamon/baronproject/WIP/brainhack/axdki/data_sample/test_lucas/Dperp.nii.gz", X[:,:,:,1], affine)
#save_nifti("/nfs/khan/trainees/larcamon/baronproject/WIP/brainhack/axdki/data_sample/test_lucas/Dpara.nii.gz", X[:,:,:,2], affine)
#save_nifti("/nfs/khan/trainees/larcamon/baronproject/WIP/brainhack/axdki/data_sample/test_lucas/Wperp.nii.gz", X[:,:,:,3], affine)
#save_nifti("/nfs/khan/trainees/larcamon/baronproject/WIP/brainhack/axdki/data_sample/test_lucas/Wpara.nii.gz", X[:,:,:,4], affine)
#save_nifti("/nfs/khan/trainees/larcamon/baronproject/WIP/brainhack/axdki/data_sample/test_lucas/Wmean.nii.gz", X[:,:,:,5], affine)





# =========================
# DEFINE B-SHELLS
# =========================
b_unique = np.unique(bvals)
b_unique = np.sort(b_unique)

# tolérance (comme opt.bthresh MATLAB)
b_thresh = 50  

# =========================
# powder AVERAGE
# =========================
S_powder_list = []

for b in b_unique:
    inds = np.abs(bvals - b) < b_thresh
    # moyenne sur directions
    S_mean = np.mean(maskdata[..., inds], axis=-1)  # (nx,ny,nz)
    S_powder_list.append(S_mean)

S_powder = np.stack(S_powder_list, axis=0)
S_powder = np.clip(S_powder, 1e-6, None)
logS = np.log(S_powder)

nb = len(b_unique)
Nvox = nx * ny * nz

logS = logS.reshape(nb, -1)
#mask_flat = mask.reshape(-1) #mask is alreay flat

# =========================
# DESIGN MATRIX (Ap)
# =========================
b = b_unique[:, None]  # (nb,1)

Ap = np.concatenate([
    np.ones_like(b),   # S0
    -b,                # D
    (b**2) / 6         # W
], axis=1)             # (nb, 3)

# =========================
# FAST FULL VECTORIZED SOLVE
# =========================
# (Ap^T Ap)^-1 Ap^T
ATA = Ap.T @ Ap
ATA_inv = np.linalg.inv(ATA + 1e-6 * np.eye(3))
AT = Ap.T

# solution globale
X = ATA_inv @ (AT @ logS)   # (3, Nvox)

# appliquer masque
X[:, ~mask] = 0

# =========================
# RESHAPE OUTPUTS
# =========================
X = X.reshape(3, nx, ny, nz)

logS0 = X[0]
Dpowder = X[1]
Wpowder = X[2]

S0 = np.exp(logS0)

#save_nifti("/nfs/khan/trainees/larcamon/baronproject/WIP/brainhack/axdki/data_sample/test_lucas/S0_powder.nii.gz", S0.astype(np.float32), affine)
#save_nifti("/nfs/khan/trainees/larcamon/baronproject/WIP/brainhack/axdki/data_sample/test_lucas/Dpowder.nii.gz", Dpowder.astype(np.float32), affine)
#save_nifti("/nfs/khan/trainees/larcamon/baronproject/WIP/brainhack/axdki/data_sample/test_lucas/Wpowder.nii.gz", Wpowder.astype(np.float32), affine)


# =========================
# FINAL COMPUTATION 
# =========================

Dmean = 1/3*Dpara + 2/3*Dperp
Wperp = Wperp/(Dperp**2)
Wpara = Wpara/(Dpara**2)
Wmean = Wmean/(Dmean**2)
Wpowder = Wpowder/(Dpowder**2)

FA = np.sqrt( 3/2* ((Dpara-Dmean)**2+2*(Dperp-Dmean)**2) / (Dpara**2 + 2*Dperp**2) )

# need to setup direction to produce FA-RGB

save_nifti("/nfs/khan/trainees/larcamon/baronproject/WIP/brainhack/axdki/data_sample/test_lucas/Dmean_final.nii.gz", Dmean.astype(np.float32), affine)
save_nifti("/nfs/khan/trainees/larcamon/baronproject/WIP/brainhack/axdki/data_sample/test_lucas/Wperp_final.nii.gz", Wperp.astype(np.float32), affine)
save_nifti("/nfs/khan/trainees/larcamon/baronproject/WIP/brainhack/axdki/data_sample/test_lucas/Wpara_final.nii.gz", Wpara.astype(np.float32), affine)
save_nifti("/nfs/khan/trainees/larcamon/baronproject/WIP/brainhack/axdki/data_sample/test_lucas/Wmean_final.nii.gz", Wmean.astype(np.float32), affine)
save_nifti("/nfs/khan/trainees/larcamon/baronproject/WIP/brainhack/axdki/data_sample/test_lucas/Wpowder_final.nii.gz", Wpowder.astype(np.float32), affine)
save_nifti("/nfs/khan/trainees/larcamon/baronproject/WIP/brainhack/axdki/data_sample/test_lucas/FA_final.nii.gz", FA.astype(np.float32), affine)
