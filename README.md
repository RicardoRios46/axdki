# Hacking axisymmetric DKI into DIPY.  
**Western Univeristy Brainhack 2026 Project**

Diffusion Weighted MRI (dMRI) is a cool technology that has improved our understanding of brain microstructure and disease. In research, there are multiple approaches to model microstructure from dMRI data. For example, Diffusion Kurtosis Imaging (DKI) calculates quantitative metrics that potentially explain the brain's complex microstructural configuration. Here at Western, members of the CFMM have developed improvements for DKI making it more robust to noise while also reducing their acquisition time in the MRI scanner. However, these implementations were done on a closed platform (Matlab). This project aims to disseminate open science practices and research done here at Western, integrating these developments into the open source DIPY ecosystem (Python). We hope to give back powerful tools to the neuroscience community to tackle complex questions with dMRI. 

**Skills for project:** Familiarity with matlab, python, git, dMRI... But not really, everyone is welcome. If you are interested in any of these topics you can join. Skill level doesn't matter.

## Project Goals
Check the vibe coded plan [here](ThePlan.md)

### TL;DR
#### Day 1
1. Introduction. Hands on on `nii2kurt.m` and DIPY DKI fitting.
2. Undestand how `nii2kurt.m` operates. Identify the axysimmetric DKI fitting.
3. Translate the `nii2kurt.m` axisymmetric DKI fitting algorithm from MATLAB to Python.

[Day 1 Git Repository](https://github.com/RicardoRios46/axdki)

#### Day 2
4. Integrate implementation into DIPY following the existing class architecture.
5. Validate outputs against the MATLAB reference implementation.
6. Provide tests and a tutorial notebook for the DIPY community.

[Day 2 work packages](Day2.md)

[Day 2 Git Repository](https://github.com/RicardoRios46/dipy/tree/axdki)


### Code to Port
- **Source (MATLAB):** [`nii2kurt.m` in MatMRI](https://gitlab.com/cfmm/matlab/matmri/-/blob/master/dMRI/nii2kurt.m)
- **Target (Python):** Mirror the structure of [`dipy/reconst/dki.py`](https://github.com/dipy/dipy/blob/master/dipy/reconst/dki.py) and [`dipy/reconst/msdki.py`](https://github.com/dipy/dipy/blob/master/dipy/reconst/msdki.py)

## Relevant links:
- Poject repository
- [matMRI](https://gitlab.com/cfmm/matlab/matmri) (Matlab package where the method was originally implemented in function nii2kurt.m): 
- [Dipy](https://dipy.org/). Python ecosystem for models fitting and general proccessing of dMRI data. 
- Dipy [devolpers documentation](https://docs.dipy.org/stable/devel/index.html#development)

## The Axisymmetric Model (Quick Summary)

The **axisymmetric DKI** fitting approach:
- Reduces the number of free parameters by enforcing rotational symmetry about the principal diffusion axis
- Is significantly more robust to noise
- Enables clinically viable acquisition schemes (as few as 10 directions)

Standard DKI fits 22 free parameters (6 diffusion tensor + 15 kurtosis tensor + 1 S0). The axisymmetric model reduces this by assuming rotational symmetry around the principal eigenvector, leaving only **5 parameters**:

| Parameter | Symbol | Description |
|-----------|--------|-------------|
| Parallel diffusivity | D∥ | Along the fiber axis |
| Perpendicular diffusivity | D⊥ | Transverse to fiber |
| Parallel kurtosis | W∥ | Kurtosis along fiber |
| Perpendicular kurtosis | W⊥ | Kurtosis transverse |
| Symmetry axis | **n̂** | Principal eigenvector direction |

This reduced parameterization is what allows robust fitting from far fewer gradient directions.

### Output Metrics

The model produces the following maps (NIfTI images):
- `FA.nii.gz` — Fractional Anisotropy
- `MD.nii.gz` — Mean Diffusivity
- `AD.nii.gz` — Axial Diffusivity (D∥)
- `RD.nii.gz` — Radial Diffusivity (D⊥)
- `Wm.nii.gz` — Mean Kurtosis 
- `AK.nii.gz` — Axial Kurtosis (W∥)
- `RK.nii.gz` — Radial Kurtosis (W⊥)

## Background Reading 
1. [Hamilton et al. 2024](https://doi.org/10.1162/imag_a_00055) — The axisymmetric DKI paper
2. Hansel Axysimetric DKI paper 
3. DKI original paper
