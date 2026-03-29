# Day 2 Work Packages

## Prutvish
* Complete the dMRI intro from Day 1
* Run the DIPY DKI and MSDKI frameworks on the reference dataset and inspect the output metric maps

## Victor
* Build a pipeline to compare reference MATLAB outputs vs the Python implementation
* Axisymmetric DKI is only reliable in white matter — create a WM mask using FA thresholding (voxels with high FA are a good proxy for single-fiber white matter)
* Run the initial comparison and quantify the differences between implementations (mean, max absolute error per metric map)
* Run the comparison pipeline on the final implementation and all curated datasets
* Quantify the results (mean and max absolute error per metric) so we have concrete numbers ready for the presentation

## Alexandra-Anderson
* Curate a third (maybe even a 4th) dataset to broaden our testing coverage:
  * Find an open-access dataset that supports the DKI model (multi-shell, b ≥ 2000 s/mm², ≥ 30 directions)
  * Download and preprocess the data (DIPY or MRtrix are both fine)
  * Push the preprocessed dataset to the repository
  * Run Victor's comparison pipeline on this new dataset

## Prutvish + Ricardo
* Ricardo gives Prutvish a mini dMRI lesson

## Victor + Prutvish
* Start a documentation page for axisymmetric DKI in DIPY style
  * The MSDKI documentation page is the direct reference — mirror its structure
  * Consider using an LLM to draft the theory section, then carefully review for errors and hallucinations (Ricardo can help curate)
  * Include placeholder sections for the API reference and code examples — these will be filled in once the DIPY integration is complete

## Lucas
* Prepare metric map outputs for sub-01 using the provisional Python code, so Victor can plug them into his comparison pipeline
  * Investigate the NIfTI header issue — make sure affine and voxel dimensions are preserved correctly when saving outputs
  * Figure out whether we are missing a step in going from the fitted W parameters (W∥, W⊥) to the final K\_par and K\_perp metric maps — check against Hamilton et al. eq. and nii2kurt.m output
* Merge code from Day 1 into day 1 main branch.

## Djay
* Get familiar with DIPY's code conventions:
  * Read `msdki.py` end to end — it covers class structure, docstrings, design matrix, fitting methods, and how parameters are exposed
  * Also check DIPY's contributing guide for Python style requirements
* Establish the Git workflow for Day 2 — all changes go to [Ricardo's fork](git@github.com:RicardoRios46/dipy.git) on the `axdki` branch (discuss branch's name first), everyone pulls before starting work
* Lay out the file structure for the eventual PR:
  * Create `dipy/reconst/axdki.py`  (Lucas+Djay work here)
  * Create `dipy/reconst/tests/test\_axdki.py` (Ricardo work here)
  * Create `doc/examples/reconst/reconst\_axdki.py` (Victor-Prutvish here)
  * Check whether anything else is needed for a complete DIPY PR (e.g. `\_\_init\_\_` exports, changelog entry)
* Figure out the best dependency management approach for the DIPY fork:
   * DIPY uses pyproject.toml natively — Clude suggested uv pip install -e ".[dev]" as the clenaest option (its was influenced by me. We can discuss what is better)
   * Make sure every team member can install and run the final development version independently before the group QC session

## Lucas + Djay
* Port the Day 1 axisymmetric DKI code into the DIPY framework in `axdki.py`
* `msdki.py` is the primary reference — mirror its structure and reuse its conventions wherever possible:
  * Use the same `design\_matrix` and `fit` method signatures
  * Note that our design matrix is voxel-dependent (each voxel has a different symmetry axis n̂) — this needs special handling vs MSDKI's global matrix
  * Lucas's Day 1 implementation uses OLS — try to upgrade to WLS following MSDKI's approach, which is more robust and is the DIPY standard
  * Study how MSDKI exposes its fitted parameters as properties and mirror that pattern for D∥, D⊥, W∥, W⊥, FA, MD, MK, AK, RK

## Ricardo
* Bring the Mexican snacks
* Implement the forward signal model (needed for the pytest validation tests)
* Write tests in the DIPY pytest framework to validate the Lucas + Djay implementation — mirror `dipy/reconst/tests/test\_msdki.py`
* Prepare the presentation slides:
  * Day 1 achievements
  * Day 2 progress
  * Take photos throughout the day
* Ask the team for feedback on the presentation — we have a shot at winning the prize, so make sure the slides reflect everyone's contributions accurately
* Float and help unblock everyone

## Everyone
* Go to Djay's talk and give moral support
* Test the final implementation independently on the curated datasets
* Regroup for a collective QC review — flag any discrepancies before the presentation