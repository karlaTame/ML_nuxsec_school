# Tutorial guide

The portable setup cell locates the repository root (or clones/extracts it
in Colab); data paths then resolve relative to that root. Install with
`python -m pip install -e '.[notebook,dev]'`. The runner always uses its own Python
interpreter for fresh notebook kernels and stops on a cell error; a partial
executed notebook preserves the traceback.

## Local and Colab setup

See the [README](../README.md) for CPU-only installation, Colab buttons, and
private-repository ZIP upload. Both notebooks share the same standalone setup
cell. On Colab it installs the lightweight `colab` extra and adds `src/` to the
already-running kernel's import path. On a local installation it locates the
checkout without installing or downloading anything. CPU calculations are
explicit; no accelerator or machine-specific kernel is required.

The repository contains no generated outputs, models, or event samples.
Results are regenerated under ignored `outputs/` directories. The runner writes
executed copies rather than adding output to the two source notebooks.

## Portability validation

A shallow clone with no generated inputs was installed into a new Python 3.11
virtual environment with CPU-only PyTorch 2.15.0. Both notebooks completed their
smoke and default quick profiles; all 62 scientific/portability tests passed.

The Colab setup was also exercised locally in fresh Python 3.12 kernels using
IPython 7.34.0, ipykernel 6.17.1, and pandas 2.2.3, matching Google's published
kernel dependencies. The real setup cells cloned a local Git mirror or
extracted a repository ZIP, installed the package, and executed both smoke
tutorials. More Leptons also completed its quick profile on that stack, and all
62 tests passed. This validates the bootstrap and calculations; an authenticated
Google-hosted Colab session was not launched from this environment.

The README's Colab links open the public `main` branch. The ZIP-upload route
also supports private forks without placing GitHub credentials in notebook code.
No prior outputs or saved models are required by either route.

## Numerical profiles

| Setting | Tutorial 1 quick / full | More Leptons quick / full |
|---|---|---|
| ND training steps | 5,000 / 100,000 | 2,000 / 100,000 |
| FD FHC mu-minus events | 1,000 / 6,200 | 1,000 / 6,200 |
| Bootstrap draws | 128 / 25,000 | 32 / 4,096 |
| FD energy integration points | 128 / 512 | 128 / 512 |
| FD lepton grid points per axis | 64 / 256 | 64 / 256 |

Edit `PROFILE` in the first setup cell, or use the runner's `--profile` option.
The `smoke` profile checks execution only. For quantitative results, vary training
steps, integration grids, starts, and random seeds until the conclusions are
stable. A full-size bootstrap does not compensate for an undertrained network.

Tutorial 1 uses a single-channel shape-only likelihood. Its default five-output network is an alternative parametrization
of the fixed-mass problem; `N_SF=3` selects the minimal basis. It uses the
original ND-shaped unoscillated FD flux approximation and oscillation inputs.
The latest reproduction audit and the differences from the published first
paper are documented in [tutorial1-reproduction.md](tutorial1-reproduction.md).
Its inference now clamps negative learned cross sections, matches generation
and normalization domains, checks parameter-search boundaries, and scales plot
coordinates consistently with their units. The quick profile currently uses 5,000 steps. To repeat the 50,000-step
reproduction audit, set `ML4NU_TRAIN_STEPS=50000`; that audit is not a
pre-trained input required by the notebook.

Tutorial 2 uses independent neutrino/antineutrino five-output networks, the
current library's NuFit 6.1 inputs, separate unoscillated FD flux tables,
an ND integration ceiling of 20 GeV, and an FD window of 0.5–8 GeV. Its six
ND observations distinguish muon charges and combine electron charges; its
four FD observations combine both charges for each lepton flavor and beam mode.
The FHC mu-minus anchor fixes a single exposure for all channels. FD fitting
uses the extended Poisson likelihood with that recorded exposure, and Poisson
bootstrap counts. Network weights remain fixed during FD fitting.

The learned model is clamped to nonnegative cross sections at FD inference,
with a density floor of `1e-10` in the fitter. The notebook reports raw negative
grid fractions and counts of floored events. Check these alongside ND residuals
and FD total cross sections. Frequent corrections indicate an unreliable short
run. The notebook displays unfiltered bootstrap clouds and central quantiles;
it does not label a small bootstrap sample as a published confidence contour.

## Code map

| Calculation | Implementation |
|---|---|
| Flux tables and oscillation probabilities | `flux.py`, `nufast.py` |
| Kinematic transformations and factors | `kinematics.py` |
| PDF interpolation and synthetic structure functions | `structure_functions.py` |
| Known/learned cross sections | `cross_section.py` |
| Tutorial 1 network and variable projection | `model.py`, `varpro.py` |
| More Leptons setup, rates, exposure, loading | `pipeline/rates_common.py` |
| Joint ND training | `pipeline/train_near_detector.py` |
| FD accept/reject event generation | `pipeline/generate_fd_samples.py`, `sampling.py` |
| Batched FD fit and bootstrap | `pipeline/fd_optimize_torch.py` |
| SciPy reference for regression comparisons | `pipeline/optimize_far_detector.py` |
| Smearing and finite-statistics kernels | `pipeline/systematics.py` |
| Correlated fluxes and ND-to-FD propagation | `pipeline/flux_model/`, `pipeline/flux_nearfar.py` |
| Optional systematic-ensemble driver | `campaign/fwdfold_shot.py` |

`rates_common` sets the default torch dtype to float64 and disables autograd
on import. Trainers and fitters enable gradients in their optimization contexts.
When writing a new optimization loop, enable gradients explicitly.

## More Leptons artifacts

`outputs/more_leptons/<profile>/run.json` records settings, seeds, input SHA-256
hashes, package versions, oscillation parameters, likelihood, and exposure.
`bootstrap_summary.csv` records the displayed quantiles.

The ND checkpoint contains `model`, `model_bar`, and training `meta`. The fit
archive contains `minima_c` (truth cross section), `minima_c_` (learned cross
section), losses, gradient diagnostics, multi-start diagnostics, floor counts,
and `meta`. Each minima row is `[sin²(theta23), dm²31 (eV²), deltaCP (radians)]`.
Plotting wraps the periodic CP coordinate around the injected phase; it does
not alter saved fits. Only load trusted checkpoints: the legacy format uses
Python pickle metadata.

## Optional systematic ensembles

The last notebook cell is opt-in because the systematic calculation is much
larger than the nominal tutorial. The retained production driver jointly fits
flux nuisances and cross sections, carries the ND-constrained flux to the FD,
and samples the conditional FD residual. Smearing and finite ND statistics
are included for `total` runs. The model and data use correlated ND/FD flux
realizations from the bundled covariance.

For example, run one full draw from the repository root:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
NSTEP=100000 N_BOOTS=4096 PROGRESS=1 PROP=regress \
FLUXRESID=shot LIKELIHOOD=extended \
python -m ml4nu.campaign.fwdfold_shot 0 total ml
```

The arguments are **draw index**, **effects**, **fit**. `nom` has no experimental
systematics, `nd` adds finite ND statistics, `smear` adds energy smearing,
`flux` adds flux uncertainties, and `total` combines them and varies the
network initialization. `ml` fits the network and flux nuisances; `truth` uses
the known cross section while fitting flux nuisances. `fix` and `truthfix`
hold the nuisance correction fixed for comparisons. `LIKELIHOOD=shape` is
available for the shape-only comparison; `extended` is the main analysis.

Repeat with distinct draw indices for an ensemble, keeping `FLUXDATA_SHOT`
(default `424242`) and `FDSEED` (default `25`) fixed to compare analyses of
the same data. `NSTEP`, `N_BOOTS`, `TRAIN_SEED`, and `SMEAR_NGRID` control
the expensive stages. Lowering them is useful for debugging, not estimating
the final total uncertainty. The retained driver uses FD integration grids
of 512 energy points and 256 points per lepton axis.

The campaign writes generated inputs to `FD_samples_ffs_data*` and results to
`results_fwdfold/` (`results_fwdfold_shape/` for shape-only). Each archive
contains `minima`, `loss`, `gmax`, `found`, fitted `a_star`, flux draws, and
`meta`. Pool the minima across comparable systematic draws, using the same
likelihood and numerical settings; keep truth and learned ensembles separate.
This estimates the combined spread rather than the spread of draw medians.
The original campaign driver is retained for this extension; a full ensemble
is not part of the default tutorial execution check.
