# First-tutorial reproduction audit

This audit compares the first tutorial with Fig. 6 of
[arXiv:2412.16303](https://arxiv.org/pdf/2412.16303), particularly Secs. III–V
and Appendix B. At the time of this audit the user's quick setting was 50,000 steps.
The ND training was rerun on the allocated A100 GPU, followed by fresh CPU notebook
executions and two FD-statistics comparisons using that same checkpoint.

## Why the previous comparison was misleading

| Item | Published first-paper calculation | Current tutorial |
|---|---|---|
| FD events | 6,200 | Quick: 1,000; full: 6,200 |
| Bootstrap draws | 25,000 | Quick: 128; full: 25,000 |
| Learned SF outputs | Three fixed-mass combinations | Five by default; three is configurable |
| ND grid | 256 energies × 128 × 128 lepton coordinates | 32 × 16 × 16 |
| Lepton-coordinate range | v1: 0.25–2.5; v2: 0–0.65 | v1: 0.25–3.25; v2: 0–0.65 |
| Training | 10,000 Adam steps on normalized ND density MSE | VarPro + Adam on marginal rates; audit used 50,000 steps; current quick uses 5,000 |
| Synthetic truth | First-paper quark-parton prescription | Shared later prescription: Q²/(Q²+1) damping and W3 rescaled by 0.1 |
| FD positivity | Clamp negative learned differential cross sections to zero | Previously absent; now applied before both FD integrals |

Increasing ND training steps cannot compensate for fewer FD events. In a
regular statistical regime, errors scale approximately as 1/sqrt(N); the
1,000-event quick example should therefore have wider intervals than Fig. 6.
The angle distribution also encounters the physical sin²(2 theta23)=1 boundary,
so its scaling need not follow that approximation exactly.

The original scan bounds, x=0.945–1 and dm²31=0.00244–0.002625 eV², were too
narrow for this quick example. Applied to the rerun's same data and bootstrap
weights, these bounds put **43/128 truth fits and 39/128 learned fits on a
numerical boundary**. These figures exclude x=1, which is a physical boundary.
Once a fit is clipped by its search domain, expanding only the plotting axes
cannot recover its missing tail.

There were also two consistency issues: sampling included intrinsic-electron
contributions that the fitted flux omitted, and unrestricted lepton samples
were normalized over a restricted lepton box. The revised notebook uses the
same muon-survival channel and lepton acceptance for generation and likelihood
normalization. It rejects nonpositive/nonfinite event likelihoods instead of
converting an invalid logarithm into a zero contribution.

## What the rerun establishes

The revised scan widens with reduced statistics and checks every bootstrap
optimum. It automatically expands numerical bounds when needed; x=1 remains
physical. Both reruns below have **zero optima at numerical scan edges**.
Plot limits cover both bootstrap clouds, the injection, and KDE-kernel tails.
Mass-splitting coordinates are actually multiplied by 1,000, matching the
10^-3 eV² axis label. Histograms retain estimates at x=1.

| FD events / bootstrap draws | Truth-cross-section std(dm²31) | Learned-cross-section std(dm²31) |
|---|---|---|
| 1,000 / 128 | 6.92e-5 eV² | 6.55e-5 eV² |
| 6,200 / 512 | 2.94e-5 eV² | 2.81e-5 eV² |

The truth mass-splitting spread decreases by **2.35×**, close to the rough
sqrt(6.2)=2.49 statistical expectation. These are measurements for one training
seed and FD draw, not published intervals. Bootstrap seed is 12; FD seed is 11.

The learned and truth fits still have a residual offset at 6,200 events:
median dm²31 is 0.002516 versus 0.002555 eV² in this draw. Exact Fig. 6
reproduction has **not** been established. The toy model, training objective,
resolution, and five-output parameterization differ from the paper. The
tutorial also retains its original upper-octant mapping. Selecting a network
using the known true FD answer would conceal these differences and is not
used here; model selection remains based on the ND training loss.

A resolution check of the same trained model illustrates the remaining
extrapolation error. These are relative L2 differences between normalized
true/model lepton densities; the FD model includes inference positivity:

| Evaluation grid (energy × v1 × v2) | ND relative L2 | FD relative L2 |
|---|---|---|
| 32 × 16 × 16 | 0.0023% | 4.04% |
| 64 × 32 × 32 | 0.349% | 4.08% |
| 128 × 64 × 64 | 0.335% | 4.06% |

A nearly exact fit on the training grid therefore does not establish equally
accurate FD predictions. The later truth prescription has not been changed
globally, which would also change the More Leptons calculation. Exact first-paper
reproduction needs a separate original truth/training configuration and a
convergence study. A 128- or 512-draw KDE is a diagnostic visualization,
particularly for its outer 99.7% contour.

The `inf` printed in the old training log's holdout column is an unused
test-dataset slot, not an infinite optimized loss. The separately tracked
`shape` diagnostic is finite.

## Artifacts and repeat commands

The audit checkpoint and measured comparisons were generated under
`outputs/tutorial1/paper_audit/`. These historical outputs have been removed;
none is required by a new clone. Normal quick results are under
`outputs/tutorial1/quick/`; the 6,200-event comparison is under
`outputs/tutorial1/quick_N6200_B512/`. Each includes a PNG/PDF final plot,
`bootstrap_fits.npz`, `fd_inputs.npz`, and `analysis.json`. Expensive ND weights
are saved as `nd_checkpoint.pt`, allowing FD experiments without retraining.

The current CPU/Colab quick default is 5,000 steps. To repeat this historical
50,000-step audit without retaining any old outputs, run:

```bash
ML4NU_TRAIN_STEPS=50000 python scripts/run_tutorials.py tutorial --profile quick
```

After a trained checkpoint exists, reuse it for plotting/statistics checks:

```bash
ML4NU_TRAIN_STEPS=50000 ML4NU_CHECKPOINT=outputs/tutorial1/quick/nd_checkpoint.pt \
python scripts/run_tutorials.py tutorial --profile quick

ML4NU_TRAIN_STEPS=50000 ML4NU_CHECKPOINT=outputs/tutorial1/quick/nd_checkpoint.pt \
ML4NU_FD_EVENTS=6200 ML4NU_N_BOOTS=512 \
python scripts/run_tutorials.py tutorial --profile quick
```

Checkpoints must match the structure-function count and training-step setting.
The first rerun command overwrites the profile's prior artifacts. Only load
trusted checkpoints. The notebook runner uses its own Python interpreter even
if saved notebook metadata names a Conda kernel unavailable on this machine.
