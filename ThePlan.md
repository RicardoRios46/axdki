# Hackathon Project Plan
# Hacking Axisymmetric DKI into DIPY

---

## Day 1 — Saturday

### 1. Introduction & Theory

The goal of this block is to make sure everyone on the team has enough shared understanding to contribute meaningfully, regardless of their neuroimaging background.

#### What we will cover

**DTI and DKI — the basics**
- What is Diffusion Tensor Imaging (DTI) and what brain microstructural properties does it measure?
- How does Diffusion Kurtosis Imaging (DKI) extend DTI by capturing non-Gaussian water diffusion?
- What is the mathematical representation of the diffusion and kurtosis tensors?
- How do we fit these models to dMRI data? (linear vs. non-linear least squares, signal model)

**Axisymmetric DKI — the innovation**
- What does "axisymmetric" mean in this context? (rotational symmetry around the principal diffusion axis)
- How does the axisymmetric model reduce the parameter space from 22 to 5 free parameters?
- What are the advantages? (noise robustness, fewer required gradient directions, clinically viable acquisitions)
- What are the limitations? (assumes a single fiber population per voxel, less general than full DKI)
- Overview of the three components of the full `nii2kurt.m` implementation:
  1. Axisymmetric fitting ← **our focus for this hackathon**
  2. Spatial regularization
  3. OGSE support (multi-frequency acquisitions)

#### Hands-on: understanding both codebases with a real dataset

The goal here is not to write any new code — just to get everyone oriented by running existing implementations end-to-end.

1. **Download a DKI dataset** that we will use for development and validation throughout the hackathon
   - Lets look for an open dataset that we can use.
   - We could use on of the DIPY dataset available directly through DIPY. Check `get_fnames('cfin_multib')`

2. **Run the MATLAB reference implementation**
   - Run `nii2kurt.m` from [MatMRI](https://gitlab.com/cfmm/matlab/matmri/-/blob/master/dMRI/nii2kurt.m) on the dataset
   - Save the output metric maps (FA, MD, MK, AK, RK, W̄) — we will use these later for numerical validation

3. **Run the DIPY implementations**
   - Check DIPY [DKI tutorial](https://docs.dipy.org/stable/examples_built/reconstruction/reconst_dki.html)  and [MSDKI](https://docs.dipy.org/stable/examples_built/reconstruction/reconst_msdki.html)
   - Run standard `DiffusionKurtosisModel` from `dipy/reconst/dki.py`
   - Run `MeanDiffusionKurtosisModel` from `dipy/reconst/msdki.py` (the closest existing relative to what we are building)
   - Visualize and compare outputs — this builds intuition for what our implementation should produce

---

### 2. Translating `nii2kurt.m` to Python

The goal of this block is a working standalone Python translation of the axisymmetric fitting component of `nii2kurt.m`. We are not integrating into DIPY yet — that comes on Day 2. The priority is correctness first, DIPY conventions second.

#### Setup

> ⚠️ **Note on tooling:** For this standalone translation block we will use a simple `pixi` environment to iterate quickly. On Day 2 we move into the DIPY fork where `pyproject.toml` takes over (we are gong to use UV here).

#### Understanding the MATLAB source

Before writing any Python, the team should read and annotate `nii2kurt.m` together. The key questions to answer are:

- Where exactly does the axisymmetric fitting happen, separate from the regularization and OGSE logic?
- How does it parameterize the axisymmetric tensor? (eigenvalues + symmetry axis angles)
- What optimization routine does it call? (`lsqnonlin` or similar — we need a Python equivalent)
- What are the inputs and outputs of the fitting step?

This annotation exercise is also a good onboarding task for the team member new to neuroimaging — reading code alongside the theory section above is one of the fastest ways to build intuition.

#### Translation priorities

**What we will implement:**
- The axisymmetric signal forward model (given parameters → predicted signal)
- The fitting routine (given signal → estimated parameters)
- Output metric computation: D∥, D⊥, W∥, W⊥, FA, MD, MK, AK, RK, W̄

**What we will explicitly defer** (but design for):
- Spatial regularization
- OGSE / multi-frequency support
- GPU acceleration (see note below)

#### Design principles

**Modular from the start.** Even though we are only implementing axisymmetric fitting now, the code should be structured so that regularization and OGSE can be added as optional components later without requiring a rewrite. Concretely:

- The signal model, the fitting routine, and the metric computation should each be separate functions
- The fitting function should accept a `method` argument (e.g., `'ols'`, `'wls'`) as a hook for future extensions
- Avoid hardcoding assumptions that would break under OGSE (e.g., single b-value per shell)

**Use DIPY tools where they already exist.** The axisymmetric fitting algorithm has two steps: (1) estimate the diffusion tensor to find the principal axis of diffusivity, then (2) fit the axisymmetric kurtosis model constraining the symmetry axis to that direction. For step 1, DIPY already has a robust DTI fitting implementation — we should use it rather than reimplementing it.

**Code quality from day one:**
- All functions should have NumPy-style docstrings from the moment they are written (not retrofitted later)
- Code should pass `flake8` and `black` formatting checks
- Functions should have clear, single responsibilities

#### GPU fitting

`nii2kurt.m` uses MATLAB's GPU computing toolbox for accelerated fitting. In Python the equivalent options are:

- **CuPy** — drop-in NumPy replacement for NVIDIA GPUs; good if the optimization is vectorized over voxels
- **PyTorch** — better if the fitting is framed as an optimization problem (autodiff available)
- **SciPy on CPU** — simplest starting point; sufficient for a hackathon and can be swapped later

**Recommendation:** Start with `scipy.optimize.least_squares` on CPU. Structure the code so the fitting backend is interchangeable, and note GPU as a future enhancement in the docstring.

---

## Day 2 — Sunday

### 3. DIPY Integration

The goal of this block is to take yesterday's working Python translation and integrate it properly into the DIPY framework — classes, API, conventions and all.

#### Setup: DIPY fork

We are working from a fork of the official DIPY repository with a dedicated development branch. If you have not done this yet:

```bash
# Clone the fork (not dipy/dipy directly)
git clone git@github.com:RicardoRios46/dipy.git
cd dipy
git checkout axdki

# Install in editable mode — reads DIPY's pyproject.toml
uv pip install -e ".[dev]"

# Verify the install points to your local clone
python -c "import dipy; print(dipy.__file__)"
```

Always pull before starting work each session:

```bash
git pull origin axdki
```

#### Studying the DIPY architecture

Before writing integration code, study two existing implementations as blueprints:

| File | What to look for |
|------|-----------------|
| `dipy/reconst/dki.py` | Full DKI — `DiffusionKurtosisModel`, `DiffusionKurtosisFit`, metric methods |
| `dipy/reconst/msdki.py` | Mean DKI — simpler, closer in spirit to our reduced-parameter model |

Key questions to answer from reading these files:

- What base classes do they inherit from? (`ReconstModel`, `ReconstFit`)
- What is the expected interface? (`fit()`, `predict()`, metric properties)
- How do they handle gradient tables and b-values?
- How do they integrate with DIPY's masking and brain extraction utilities?
- Does DIPY have existing GPU-accelerated fitting methods we can hook into?

#### Translation into the DIPY framework

New files to create, following DIPY conventions:

```
dipy/reconst/axdki.py              ← AxSymDKIModel, AxSymDKIFit classes
dipy/reconst/tests/test_axdki.py  ← test suite
doc/examples/reconst/reconst_axdki.py  ← tutorial script
```

When translating from yesterday's standalone code into these classes:

- Map standalone fitting functions into `AxSymDKIModel.fit()`
- Map metric computations into properties on `AxSymDKIFit` (e.g., `.mk()`, `.ak()`, `.rk()`)
- Replace any custom tensor fitting with DIPY's `TensorModel` where applicable
- Check whether DIPY's `nlls_fit_tensor` or similar fitting utilities can be reused for our optimization step

---

### 4. Tests

Mirror the structure of `dipy/reconst/tests/test_dki.py` and `test_msdki.py`. At minimum, tests should cover:

- **Synthetic signal test:** generate a signal with known axisymmetric parameters, fit the model, verify recovered parameters are within tolerance
- **Metrics test:** verify FA, MD, MK, AK, RK, W̄ values on synthetic data against analytical ground truth
- **Numerical validation:** compare outputs against saved MATLAB `nii2kurt.m` results from Day 1 on the real dataset (within a reasonable numerical tolerance)
- **Gradient table handling:** verify the model raises clear errors on incompatible inputs (e.g., too few directions)

Run tests with:

```bash
pytest dipy/reconst/tests/test_axdki.py -v
```

---

### 5. Documentation

Mirror the structure of DIPY's existing DKI and MSDKI documentation pages.

#### Learn Sphinx (quick orientation)
- DIPY uses Sphinx with the `numpydoc` extension for API docs and `sphinx-gallery` for example scripts
- Example scripts in `doc/examples/reconst/` are auto-converted into documentation pages — writing the script *is* writing the docs page
- Study `doc/examples/reconst/reconst_dki.py` as the direct template

#### What to produce

- **`doc/examples/reconst/reconst_axdki.py`** — a gallery example script that:
  - Loads the CFIN dataset
  - Fits `AxSymDKIModel`
  - Visualizes output metric maps
  - Compares against standard DKI outputs
  - Includes explanatory prose between code blocks (Sphinx gallery style)

- **Docstrings in `axdki.py`** — NumPy-style, covering all public classes and methods, written during implementation (not after)

---

## End Goal: Pull Request to DIPY

Once the hackathon work is stable, the path to upstreaming is:

1. Open a PR from `YOUR_USER/dipy:feature/axsym-dki` → `dipy/dipy:master`
2. Follow DIPY's [contribution checklist](https://github.com/dipy/dipy/blob/master/CONTRIBUTING.rst)
3. All tests pass, docstrings are complete, example script runs end-to-end
4. Note explicitly in the PR description that spatial regularization and OGSE support are planned follow-up contributions

---

## Reference

| Resource | Link |
|----------|------|
| MATLAB source (`nii2kurt.m`) | [MatMRI on GitLab](https://gitlab.com/cfmm/matlab/matmri/-/blob/master/dMRI/nii2kurt.m) |
| Hamilton et al. 2024 | [PMC12224416](https://pmc.ncbi.nlm.nih.gov/articles/PMC12224416/) |
| DIPY DKI implementation | [`dipy/reconst/dki.py`](https://github.com/dipy/dipy/blob/master/dipy/reconst/dki.py) |
| DIPY MSDKI implementation | [`dipy/reconst/msdki.py`](https://github.com/dipy/dipy/blob/master/dipy/reconst/msdki.py) |
| DIPY DKI tutorial | [docs.dipy.org](https://docs.dipy.org/stable/examples_built/reconstruction/reconst_dki.html) |
| Jensen & Helpern 2010 (original DKI) | [doi:10.1002/nbm.1518](https://doi.org/10.1002/nbm.1518) |