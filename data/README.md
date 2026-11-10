# Tutorial inputs

These inputs are versioned and read without modification. No download is
required to run the nominal tutorials.

| Input | Use |
|---|---|
| `pdf/pdf_x1024_q1024.npy` | Original Tutorial 1 synthetic cross section |
| `pdf/pdf_x2048_q2048_z18_a40_q2max10000_linlog.h5` | More Leptons argon-40 neutrino synthetic cross section |
| `pdf/pdfb_x2048_q2048_z18_a40_q2max10000_linlog.h5` | More Leptons argon-40 antineutrino synthetic cross section |
| Four `histos_*_globes_flux.txt` tables | FHC/RHC unoscillated fluxes at ND and FD |

The grids were supplied with this research repository. The paper describes
the CT18NNLO-based toy cross section in Sec. III A of
[More Leptons](https://arxiv.org/abs/2610.30338); the library applies the
low-Q-squared suppression and W3 rescaling. LHAPDF is needed only to generate
new grids, not to interpolate these supplied grids. Duplicate formats and
older argon grids have been removed from this tutorial branch.

The GLoBES tables identify their beam production in their filenames:
`g4lbne_v3r5p4_QGSP_BERT_OptimizedEngineeredNov2017`, with `LBNEND` or
`LBNEFD` and neutrino or antineutrino horn mode. Their original distribution
is the [DUNE flux archive](https://glaucus.crc.nd.edu/DUNEFluxes/).
Read their headers for column definitions and units. The near/far ratio is
energy dependent; Tutorial 2 uses the FD tables for its FD prediction.

The optional flux-systematics input files live in
`src/ml4nu/pipeline/flux_model/`: the joint 136-bin fractional ND/FD
covariance `dune_flux_covariance.npz` and ND mean fluxes
`dune_nd_flux_means.npz`. See that directory's README for beam-production
differences and the reconstructed covariance-bin edges. These source-data
limitations are inherited from the supplied research repository.

Generated FD samples are intentionally absent. Tutorial 2 regenerates them,
including `exposure.npz`. Pre-existing samples in the original branch predate
the current FD analysis window and do not supply its recorded exposure.
