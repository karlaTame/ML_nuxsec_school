# DUNE ND/FD flux covariance

A self-contained fractional flux covariance model for
`detector{ND,FD} × horn{FHC,RHC} × flavor{numu,numubar,nue,nuebar}`.
The package factory `load_default()` uses the joint ND/FD covariance; the
`NDFluxModel` constructor alone defaults to ND only for legacy callers.
It also carries tabulated ND mean fluxes.

Runtime dependency: **numpy only** (`scipy` not required). No ROOT / pyROOT.

## The model

Each channel has a mean flux tabulated in energy and a fractional
(`δφ/φ`) uncertainty tabulated in coarse energy bins, with correlations both
within and across channels.

- **Mean.** The tabulated flux is treated as point estimates of a smooth function
  and **linearly interpolated** (clamped past the tabulated range).
- **Covariance.** The coarse fractional covariance `C` is carried to arbitrary
  energies by the congruence `C_fine = M C Mᵀ`, where `M` is a linear
  interpolation operator (with **linear extrapolation** beyond the tabulated
  range). `M` is **block-diagonal by channel**, so energy is never interpolated
  across a channel boundary, and cross-channel correlations are preserved through
  the off-diagonal blocks of `C`. Because `C_fine` is a congruence of a PSD
  matrix it is PSD for any requested energies, and evaluating at a coarse bin
  centre reproduces that bin's uncertainty exactly.
- **Sampling.** A joint fractional shift `δ ~ N(0, C_fine)` is drawn (correctly
  correlated across channels and energies) and applied as `φ·(1+δ)`.

## Data files

Everything the model needs is read from these files; nothing is hardcoded.

- **`dune_flux_covariance.npz`** — the total (hadron-production + focusing)
  fractional flux covariance, `δφ/φ`. It is the full 136-bin matrix,
  `detector{ND,FD} × horn{FHC,RHC} × flavor{numu,numubar,nue,nuebar}`, with a
  per-bin layout (`detector`, `horn`, `flavor`, `E_low`, `E_high` in GeV). The
  factory uses both detector blocks. Source: `g4lbne v3r5p10 QGSP_BERT
  OfficialEngDesignSept2021`, on-axis.
- **`dune_nd_flux_means.npz`** — ND mean flux per channel vs energy (`energy`
  grid in GeV, `flux` array, `horn`/`flavor` labels, `units`). Source: `g4lbne
  v3r5p4 QGSP_BERT OptimizedEngineeredNov2017`, LBNE near detector.

## Usage

```python
import numpy as np
from ml4nu.pipeline.flux_model import load_default

m = load_default()
m.channels                      # ['ND_FHC_numu', ..., 'FD_RHC_nuebar']

# one energy grid shared by all channels
E = np.linspace(0.5, 20, 200)
means   = m.mean(E)             # {channel: array}
sigma   = m.stddev(E)           # {channel: fractional sigma}
C, layout = m.covariance(E)     # joint covariance + {channel: row/col slice}

# or a different energy grid per channel (subset allowed)
E = {"ND_FHC_numu": np.linspace(0.5, 20, 40),
     "FD_RHC_numubar": np.array([1., 3., 5., 8.])}
C, layout = m.covariance(E)

# draw flux realizations: {channel: array of shape (size, n_energies)}
flux = m.sample(np.linspace(1, 10, 50), size=1000, rng=np.random.default_rng(0))
```

`covariance(..., fractional=False)` and `stddev(..., fractional=False)` rescale to
absolute flux by the mean; `sample(..., fractional=True)` returns the shifts
`δφ/φ` instead of flux.

## Caveats

- **Means and covariance come from different beam productions** (`v3r5p4
  OptimizedEngineeredNov2017` vs `v3r5p10 OfficialEngDesignSept2021`). The mean
  shape is representative but is not the exact nominal the covariance was built
  around; swap in matching-production ND means when available. The model is
  otherwise insensitive to overall flux normalization.
- **The supplied means are ND means.** For compatibility, mean queries on FD
  channels fall back to the corresponding ND table. The tutorial and campaign
  use the separate GLoBES FD tables for nominal FD fluxes and this model for
  fractional uncertainties. Do not use `m.mean()` as a nominal FD prediction.
- **The covariance energy bin edges are not stored in the original matrices** and
  are reconstructed from external notes. They are consistent with the matrix
  dimensions but should be double-checked against a same-production flux
  histogram if one is available.
