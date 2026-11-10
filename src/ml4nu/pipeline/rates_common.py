"""Shared setup, rates, exposure, and data loading for Tutorial 2.

The setup and inference conventions are explained in tutorial_more_leptons.ipynb
and docs/tutorials.md. Inputs resolve from the repository root. Importing this
module selects float64 and disables autograd; optimization code enables it in
its own context.
"""

from dataclasses import dataclass
import os

import numpy as np
import torch

from ml4nu.pdg import pid, masses
from ml4nu.flux import FluxNearDetector, FluxFarDetector
from ml4nu.structure_functions import PDFInterpolator, StructureFunctions
from ml4nu.kinematics import (
    KinematicMapping, KinematicsGrid, make_newk2, unmake_newk2,
    make_kinematic_factors_newk2, jacobian_newk2,
)
from ml4nu.cross_section import TruthCalc, ModelCalc
from ml4nu.utils import make_mlp, make_marginal_rate, grab, safe_make_dx
from ml4nu.far_detector_fit import make_I, make_dsigma_samp, safe_log
from ml4nu.nufast import Probability_Matter_LBL

# Match the notebook's global torch configuration (cell 0).
torch.set_default_dtype(torch.float64)
torch.set_grad_enabled(False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------------
# Defaults (notebook cells 2, 3, 8, 10, 11)
# ----------------------------------------------------------------------------
DATADIR = "./data/"
FLUX_FILE_FWD = DATADIR + "histos_g4lbne_v3r5p4_QGSP_BERT_OptimizedEngineeredNov2017_neutrino_LBNEND_globes_flux.txt"
FLUX_FILE_BWD = DATADIR + "histos_g4lbne_v3r5p4_QGSP_BERT_OptimizedEngineeredNov2017_antineutrino_LBNEND_globes_flux.txt"
# UNOSCILLATED FAR-detector fluxes (same g4lbne v3r5p4 production, LBNEFD).
# Until 2026-10-09 the FD flux was taken as the ND flux times P_osc, i.e. the
# near/far ratio was assumed energy-INDEPENDENT. It is not: FD/ND varies by
# ~20-70% across 0.5-8 GeV in the dominant numu channel (the ND views an
# extended decay pipe, the FD essentially a point source). Source:
# glaucus.crc.nd.edu/DUNEFluxes/OptimizedEngineeredNov2017/
FLUX_FILE_FD_FWD = DATADIR + "histos_g4lbne_v3r5p4_QGSP_BERT_OptimizedEngineeredNov2017_neutrino_LBNEFD_globes_flux.txt"
FLUX_FILE_FD_BWD = DATADIR + "histos_g4lbne_v3r5p4_QGSP_BERT_OptimizedEngineeredNov2017_antineutrino_LBNEFD_globes_flux.txt"
PDF_FN = "data/pdf/pdf_x2048_q2048_z18_a40_q2max10000_linlog.h5"
PDFB_FN = "data/pdf/pdfb_x2048_q2048_z18_a40_q2max10000_linlog.h5"

# # nufit 6.0, paper 1
# OSC_PARAMS = {
#     "s12": 0.554076,
#     "s23": 0.748999,
#     "s13": 0.148155,
#     "dCP": np.pi / 2,
#     "d21": 7.49e-5,
#     "d31": 2.534e-3,
# }

# # nufit 6.1, current
OSC_PARAMS = {
    "s12": 0.555698,
    "s23": 0.685565,
    "s13": 0.149967,
    "dCP": 3.61283,
    "d21": 7.537e-5,
    "d31": 2.521e-3,
}

# ND training grid (cell 8)
N_ENU_ND = 64   # 0.3 GeV spacing; doubled with ENUMAX 10 -> 20
N_V_ND = 16
V1_RANGE = (0.25, 3.25)
V2_RANGE = (0.0, 0.65)

# Upper end of the neutrino-energy integration AND the kinematic mask in
# TruthCalc/ModelCalc. Raising it to 20 would match the flux covariance's bin
# support (0-20 GeV), which is what makes all 68 ND flux nuisances observable
# instead of 60 -- see flux_nearfar.observable_subspace. It is NOT a free
# change, in two ways:
#   * ~20% of the ND rate lies above 10 GeV (measured, FHC numu, on the current
#     (v1,v2) box), so the learned cross sections, ND counts (Table II) and
#     everything downstream move. The current (v1,v2) box already captures most
#     of it: extending V1_RANGE to sqrt(20)=4.5 adds only ~3 points (19.6% ->
#     22.6%), so the histogram definition can stay as published.
#   * the PDF grids now reach Q^2 = 10^4 GeV^2 (Josh, 2026-10-08), so nothing
#     is clamped in range any more. For the record the old Q2max = 10 grids
#     were far less of a problem than the kinematic reach suggested: the rate
#     above Q^2 = 10 is only 0.05% at ENUMAX=10 and 1.5% at ENUMAX=20, because
#     the rate sits at low Q^2 and modest x (median x ~ 0.28), not out at the
#     2 M E boundary.
# The x grids are LINEAR in x in both the old and new files, so there are only
# ~21 points below x = 0.01 and the interpolation there is meaningless. It does
# not matter: x < 0.01 carries 0.002% of the rate. Do not read small-x
# structure out of these grids.
ENUMAX = 20

# Canonical order for the 8 fluxes / 8 marginals exchanged with the training
# script. mu/mubar use the muon kinematic grid; e/ebar use the electron grid.
CHANNELS = ["mu_fwd", "mubar_fwd", "e_fwd", "ebar_fwd",
            "mu_bwd", "mubar_bwd", "e_bwd", "ebar_bwd"]


# ----------------------------------------------------------------------------
# Model (notebook cell 16)
# ----------------------------------------------------------------------------
class SFModel(torch.nn.Module):
    def __init__(self, n_sf, H=64, depth=3, force_pos=False):
        super().__init__()
        self.n_sf = n_sf
        self.n_in = 2
        self.net = make_mlp(nin=self.n_in, nout=n_sf, H=H, depth=depth)

        self.pre_norm = 1
        self._log_norm = torch.nn.Parameter(torch.randn(n_sf))
        self.force_pos = force_pos

    def forward(self, X2, K, return_sf=False, return_dsigma=True):
        *shape, check_n_in = X2.shape
        assert self.n_in == check_n_in, (X2.shape, check_n_in)
        assert self.n_sf == K.shape[0]

        X2 = X2.reshape(-1, self.n_in)
        X2 = X2 * self.pre_norm
        F = self.net(X2) / torch.exp(self._log_norm)
        if self.force_pos:
            F = F ** 2

        F = F.permute(1, 0)  # SF index on axis 0
        F = F.reshape(self.n_sf, *shape)

        if return_dsigma:
            assert tuple(F.shape) == tuple(K.shape)
            dsigma = (K * F).sum(0)
        if return_sf and return_dsigma:
            return dsigma, F
        elif return_sf:
            return F
        elif return_dsigma:
            return dsigma


# ----------------------------------------------------------------------------
# Oscillation model (notebook cell 29)
# ----------------------------------------------------------------------------
s23_true = 0.685565
d31_true = 2.521e-3
dcp_true = 3.61283


def compute_osc_prob(enu, s23, d31, dcp):
    baseline = 1300
    s12 = 0.555698
    s13 = 0.149967
    d21 = 7.537e-5

    prob = Probability_Matter_LBL(s12 ** 2, s13 ** 2, s23 ** 2,
                                  dcp, d21, d31,
                                  baseline, enu)
    return prob


def make_phi_FD(phi_unosc_mu, phi_unosc_e, all_Enu, s23, d31, dcp=dcp_true, lepton="mu", anti=False):
    """Oscillate UNOSCILLATED FD fluxes into one FD flavor.

    The two inputs must be the unoscillated FAR-detector fluxes (setup
    .fd_unosc_*), NOT the ND fluxes: the near/far ratio is strongly
    energy-dependent (see FLUX_FILE_FD_FWD). Passing ND fluxes here was the
    pre-2026-10-09 behavior and silently assumed a constant near/far ratio.
    """

    if anti:
        p_osc = compute_osc_prob(grab(-all_Enu), s23=s23, d31=d31, dcp=dcp)
    else:
        p_osc = compute_osc_prob(grab(all_Enu), s23=s23, d31=d31, dcp=dcp)
    assert not np.isclose(phi_unosc_mu.sum() * safe_make_dx(all_Enu), 1)  # DON'T PASS PDF
    assert not np.isclose(phi_unosc_e.sum() * safe_make_dx(all_Enu), 1)  # DON'T PASS PDF
    if isinstance(phi_unosc_mu, torch.Tensor):
        p_osc = torch.as_tensor(p_osc, device=phi_unosc_mu.device)
    else:
        p_osc = np.array(grab(p_osc))

    if lepton == "mu":
        phi_FD = phi_unosc_mu * p_osc[1][1] + phi_unosc_e * p_osc[0][1]
    if lepton == "e":
        phi_FD = phi_unosc_mu * p_osc[1][0] + phi_unosc_e * p_osc[0][0]
    if lepton == "tau":
        phi_FD = phi_unosc_mu * p_osc[1][2] + phi_unosc_e * p_osc[0][2]
    nan_to_num = torch.nan_to_num if isinstance(phi_FD, torch.Tensor) else np.nan_to_num
    phi_FD = nan_to_num(phi_FD, 0)

    return phi_FD


def mix_sample_norm(I_l, I_lbar, phi_FD_l, phi_FD_lbar, denu):
    """Predicted total rate N^m_l(omega) for one charge-inclusive sample.

    Sum these over all four samples to build the grand-total normalizer
    the rate-aware likelihood needs (see make_mix_log_likelihood).
    """
    return (I_l * phi_FD_l).sum() * denu + (I_lbar * phi_FD_lbar).sum() * denu


def make_mix_log_likelihood(dsigma_samp_l, dsigma_samp_lbar, I_l, I_lbar, phi_FD_l, phi_FD_lbar, denu,
                            log_floor=None, norm_total=None, unnormalized=False):
    """Combined-charge ("confused") log-likelihood (notebook cell 64/85).

    log[P1 + P2] = log[R1 + R2] - log[Gamma1 + Gamma2].  NOTE: not minus-ll.

    ``log_floor``, if set, clamps sample densities at log_floor * norm before
    the log (mirror of fd_optimize_torch's LOG_FLOOR; see that module's
    docstring): a floored sample contributes the constant penalty
    log(log_floor) instead of being silently dropped at zero cost.
    """
    norm1 = (I_l * phi_FD_l).sum() * denu
    norm2 = (I_lbar * phi_FD_lbar).sum() * denu
    samp1 = (dsigma_samp_l @ phi_FD_l) * denu
    samp2 = (dsigma_samp_lbar @ phi_FD_lbar) * denu
    # norm_total: pass sum_{m,l} N^m_l (via mix_sample_norm) to use the
    # RATE-AWARE normalizer, mirroring fd_optimize_torch.LIKELIHOOD ==
    # "rate". Leaving it None keeps this sample's own total, the
    # shape-only form -- which discards the inter-sample rate ratios.
    own = norm1 + norm2
    if unnormalized:
        # EXTENDED form: no division at all; the predicted total enters the
        # likelihood separately as -alpha*N (see fd_optimize_torch.LIKELIHOOD
        # == "extended"). The floor still needs a scale, so it is taken
        # relative to this sample's own total.
        s = samp1 + samp2
        if log_floor is None:
            return safe_log(s)
        n = own.clamp_min(1e-300) if hasattr(own, "clamp_min") else max(float(own), 1e-300)
        return s.clamp_min(log_floor * n).log()
    norm = own if norm_total is None else norm_total
    if log_floor is None:
        return safe_log(samp1 + samp2) - safe_log(norm)
    n = norm.clamp_min(1e-300) if hasattr(norm, "clamp_min") else max(float(norm), 1e-300)
    s = (samp1 + samp2).clamp_min(log_floor * n)
    return s.log() - (n.log() if hasattr(n, "log") else float(np.log(n)))


# ----------------------------------------------------------------------------
# Setup container + builders
# ----------------------------------------------------------------------------
@dataclass
class Setup:
    osc_params: dict
    nd_flux_fwd: FluxNearDetector
    nd_flux_bwd: FluxNearDetector
    fd_flux_fwd: FluxFarDetector
    fd_flux_bwd: FluxFarDetector
    kin_mapping: KinematicMapping
    kin_grid_mu: KinematicsGrid
    kin_grid_e: KinematicsGrid
    truth: TruthCalc
    truth_bar: TruthCalc
    device: torch.device
    # UNOSCILLATED FD fluxes (LBNEFD tables). Defaulted so older pickled
    # setups still construct, but every FD rate must use these: the near/far
    # ratio is strongly energy-dependent (see FLUX_FILE_FD_FWD).
    fd_unosc_fwd: FluxNearDetector = None
    fd_unosc_bwd: FluxNearDetector = None


def build_setup(N_Enu_ND=N_ENU_ND, N_v_ND=N_V_ND, datadir=DATADIR,
                pdf_fn=PDF_FN, pdfb_fn=PDFB_FN, osc_params=None):
    """Build fluxes, kinematic grids, and truth calculators (cells 0-11)."""
    if osc_params is None:
        osc_params = dict(OSC_PARAMS)

    flux_file_fwd = datadir + os.path.basename(FLUX_FILE_FWD)
    flux_file_bwd = datadir + os.path.basename(FLUX_FILE_BWD)

    nd_flux_fwd = FluxNearDetector(flux_file_fwd)
    nd_flux_bwd = FluxNearDetector(flux_file_bwd)
    # FluxNearDetector is a generic GLoBES reader; these are the FD tables.
    fd_unosc_fwd = FluxNearDetector(datadir + os.path.basename(FLUX_FILE_FD_FWD))
    fd_unosc_bwd = FluxNearDetector(datadir + os.path.basename(FLUX_FILE_FD_BWD))
    fd_flux_fwd = FluxFarDetector(nd_flux=nd_flux_fwd, osc_params=osc_params,
                                  unosc_flux=fd_unosc_fwd)
    fd_flux_bwd = FluxFarDetector(nd_flux=nd_flux_bwd, osc_params=osc_params,
                                  unosc_flux=fd_unosc_bwd)

    v1 = torch.linspace(V1_RANGE[0], V1_RANGE[1], N_v_ND).to(device)
    v2 = torch.linspace(V2_RANGE[0], V2_RANGE[1], N_v_ND).to(device)
    # ENUMAX, not a literal: the integration ceiling and the calculators'
    # kinematic mask must never disagree (see the ENUMAX comment above).
    enu = torch.linspace(masses[pid.mu] + 1e-6, ENUMAX, N_Enu_ND).to(device)

    kin_mapping = KinematicMapping(unmake_func=unmake_newk2,
                                   make_kinematic_factors=make_kinematic_factors_newk2,
                                   jacobian=jacobian_newk2)
    kin_grid_mu = KinematicsGrid(v1, v2, enu, pid.mu, kin_mapping=kin_mapping)
    kin_grid_e = KinematicsGrid(v1, v2, enu, pid.e, kin_mapping=kin_mapping)

    pdf = PDFInterpolator(pdf_fn, clamp_q2=True, clamp_x=True)
    # Q^2 reach vs the PDF grid. At the published ENUMAX = 10 the reach is
    # already ~1.9x the grid max and clamp_q2 freezes the SFs above it -- that
    # is the accepted baseline, so no warning there. Doubling ENUMAX doubles
    # the ratio, which is a different kind of statement about the toy model.
    _q2_reach = 2 * 0.938 * ENUMAX                 # max Q^2 = 2 M E_nu
    _q2_grid = float(pdf.all_q2.max())
    if _q2_reach > 2.0 * _q2_grid:
        import warnings
        warnings.warn(
            f"ENUMAX={ENUMAX} reaches Q^2 = {_q2_reach:.1f} GeV^2 against a PDF "
            f"grid that stops at {_q2_grid:.1f} (ratio {_q2_reach/_q2_grid:.1f}x; "
            "the ENUMAX=10 baseline is 1.9x). clamp_q2 freezes the structure "
            "functions above the grid, so most of the added phase space would "
            "have NO Q^2 dependence. Regenerate the grids first with "
            "a larger LHAPDF grid (Q2max >~ 40; see data/README.md).",
            RuntimeWarning, stacklevel=2)
    sf = StructureFunctions(pdf)
    truth = TruthCalc(sf, enumax=ENUMAX, unique=False, anti=False, device=device)

    pdf_bar = PDFInterpolator(pdfb_fn, clamp_q2=True, clamp_x=True)
    sf_bar = StructureFunctions(pdf_bar)
    truth_bar = TruthCalc(sf_bar, enumax=ENUMAX, unique=False, anti=True, device=device)

    return Setup(osc_params, nd_flux_fwd, nd_flux_bwd, fd_flux_fwd, fd_flux_bwd,
                 kin_mapping, kin_grid_mu, kin_grid_e, truth, truth_bar, device,
                 fd_unosc_fwd, fd_unosc_bwd)


def _truth_dsigmas(setup):
    """Toy-model truth dsigma on the ND kinematic grid (cell 12, top)."""
    s = setup
    km = s.kin_mapping
    dsigma_mu = s.truth(s.kin_grid_mu.grid_enu, s.kin_grid_mu.grid1, s.kin_grid_mu.grid2,
                        flavor=pid.mu, kin_mapping=km)
    dsigma_mubar = s.truth_bar(s.kin_grid_mu.grid_enu, s.kin_grid_mu.grid1, s.kin_grid_mu.grid2,
                               flavor=pid.mubar, kin_mapping=km)
    dsigma_e = s.truth(s.kin_grid_e.grid_enu, s.kin_grid_e.grid1, s.kin_grid_e.grid2,
                       flavor=pid.e, kin_mapping=km)
    dsigma_ebar = s.truth_bar(s.kin_grid_e.grid_enu, s.kin_grid_e.grid1, s.kin_grid_e.grid2,
                              flavor=pid.ebar, kin_mapping=km)
    return dsigma_mu, dsigma_mubar, dsigma_e, dsigma_ebar


def build_training_inputs(setup):
    """"function 0": build the 8 ND fluxes and 8 marginals (cell 12).

    Returns (fluxes, marginals), each a list of 8 tensors in ``CHANNELS`` order.
    """
    s = setup
    dsigma_mu, dsigma_mubar, dsigma_e, dsigma_ebar = _truth_dsigmas(s)

    enu_mu = s.kin_grid_mu.all_enu
    enu_e = s.kin_grid_e.all_enu

    # 8 ND fluxes (forward then backward), moved to device.
    phi = {
        "mu_fwd": s.nd_flux_fwd(pid.numu, enu_mu),
        "mubar_fwd": s.nd_flux_fwd(pid.numubar, enu_mu),
        "e_fwd": s.nd_flux_fwd(pid.nue, enu_e),
        "ebar_fwd": s.nd_flux_fwd(pid.nuebar, enu_e),
        "mu_bwd": s.nd_flux_bwd(pid.numu, enu_mu),
        "mubar_bwd": s.nd_flux_bwd(pid.numubar, enu_mu),
        "e_bwd": s.nd_flux_bwd(pid.nue, enu_e),
        "ebar_bwd": s.nd_flux_bwd(pid.nuebar, enu_e),
    }
    phi = {k: torch.as_tensor(v, device=device) for k, v in phi.items()}

    dsig = {"mu": dsigma_mu, "mubar": dsigma_mubar, "e": dsigma_e, "ebar": dsigma_ebar}
    grid = {"mu": s.kin_grid_mu, "mubar": s.kin_grid_mu, "e": s.kin_grid_e, "ebar": s.kin_grid_e}

    # 8 marginals (target distributions for training).
    P = {}
    for ch in CHANNELS:
        base = ch.rsplit("_", 1)[0]  # mu / mubar / e / ebar
        kg = grid[base]
        P[ch] = make_marginal_rate(dsig[base], phi[ch], kg.denu, kg.dV2)

    fluxes = [phi[ch] for ch in CHANNELS]
    marginals = [P[ch] for ch in CHANNELS]
    return fluxes, marginals


# ----------------------------------------------------------------------------
# FD sample counts from rates (notebook cells 30-31)
# ----------------------------------------------------------------------------
def compute_exposure(setup, num_mu_fwd=6200, window=None):
    """Exposure ``dt`` converting arbitrary-unit FD rates to event COUNTS.

    Fixed by N(mu-, FHC) = ``num_mu_fwd`` evaluated with the TRUE cross
    section at the TRUE oscillation parameters -- i.e. the POT x mass of
    the simulated experiment. It is an experimental constant, NOT a fit
    parameter: that is what lets the extended FD likelihood
    (fd_optimize_torch.LIKELIHOOD == "extended") use the absolute event
    counts, whose normalization uncertainty is already carried by the flux
    covariance. Same single-anchor convention as compute_sample_counts;
    tests/test_fd_likelihood.py pins the two against each other.
    """
    s = setup
    dsigma_mu = _truth_dsigmas(s)[0]
    enu_mu = s.kin_grid_mu.all_enu
    phi_mu = s.fd_unosc_fwd(pid.numu, enu_mu)          # UNOSCILLATED FD flux
    phi_e = s.fd_unosc_fwd(pid.nue, s.kin_grid_e.all_enu)
    phi_FD_mu = make_phi_FD(phi_mu, phi_e, enu_mu, s23=s23_true, d31=d31_true,
                            dcp=dcp_true, lepton="mu")
    # Only the TDR window counts (see FD_EWINDOW): the anchor is quoted for
    # 0.5-8.0 GeV, so integrating over the full grid would make the inferred
    # exposure depend on ENUMAX, which is an ND-side choice.
    w = _window_weights(enu_mu, window)
    rate_mu = (grab(dsigma_mu) * grab(phi_FD_mu) * w).sum() * grab(s.kin_grid_mu.dV3)
    return num_mu_fwd / rate_mu


# Far-detector ANALYSIS window, in neutrino energy. This is the DUNE TDR's
# own window: TDR Vol. II Table 1.2 gives "the integrated rate of selected
# nu_mu CC-like events between 0.5 and 8.0 GeV assuming a 3.5-year (staged)
# exposure" as 6200 (and 2303 for nubar_mu in nubar mode). The pipeline cites
# that 6200 as its exposure anchor, so it has to count over the same window --
# otherwise the inferred POT x mass depends on an unrelated choice. It used to
# anchor over the WHOLE energy grid, which put only 78.5% of the counted rate
# inside [0.5, 8.0] at ENUMAX=10 (exposure 1.27x too large) and 44.2% at
# ENUMAX=20 (2.26x).
#
# Anchoring here also makes the exposure INVARIANT to ENUMAX, which is the
# property that matters: ENUMAX is an ND training choice (it sets how much of
# the flux covariance the ND can constrain) and must not move the FD
# normalization. tests/test_fd_window.py pins that invariance.
#
# The FD samples are drawn in this window and the FD likelihood integrates over
# it, so num_mu_fwd really is the number of analyzed mu- FHC events.
#
# Caveat for the paper: the TDR's 6200 are *selected* CC-like events, with
# efficiency and backgrounds folded in; the toy model has no selection, so the
# counts match in window but not in definition.
FD_EWINDOW = (0.5, 8.0)


def _window_weights(enu, window=None):
    """Per-grid-point weights for integrating over `window`.

    A boolean mask would snap the window to whole grid points, which makes
    the anchor wobble ~1% with N_ENU_ND -- the same size as the rate
    offsets we are trying to control. These are midpoint-rule weights:
    1 for interior points, the covered fraction for the two edge bins, 0
    outside. Exact for a uniform grid and smooth in the grid spacing.
    """
    lo, hi = FD_EWINDOW if window is None else window
    e = np.asarray(grab(enu), dtype=float)
    d = np.diff(e)
    if d.size == 0:
        return np.ones_like(e)
    step = float(np.median(d))
    left, right = e - 0.5 * step, e + 0.5 * step
    cov = np.clip(np.minimum(right, hi) - np.maximum(left, lo), 0.0, None)
    # clip to [0,1]: `step` is the MEDIAN spacing, so a linspace whose actual
    # diffs wobble at the 1e-17 level can hand back 1+eps for an interior bin
    return np.clip(cov / step, 0.0, 1.0)


EXPOSURE_FILE = "exposure.npz"


def load_exposure(sample_dir="FD_samples"):
    """The exposure that generated the FD samples in ``sample_dir``.

    ALWAYS prefer this over ``compute_exposure(setup)`` when an FD sample set
    is in play. The samples are drawn from whatever flux realization made
    them -- in the forward-fold campaign that is the fixed DATA draw b_data,
    not the nominal flux -- and are re-anchored to N(mu-,FHC) = num_mu_fwd
    under THAT flux. Recomputing the exposure from a nominal ``setup`` then
    disagrees with the samples (-2.4% for the 424242 data draw), which the
    shape-only likelihood hid but the extended one turns into a coherent
    absolute-rate offset in every channel of every flux-carrying shot.
    """
    path = os.path.join(sample_dir, EXPOSURE_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} is missing: it is written by generate_fd_samples and is "
            "the exposure that MADE these samples. Sample dirs produced "
            "before 2026-10-08 predate it -- regenerate them rather than "
            "falling back to compute_exposure(setup), which silently assumes "
            "the samples came from the nominal flux.")
    with np.load(path, allow_pickle=True) as z:
        return float(z["exposure"])


def compute_sample_counts(setup, num_mu_fwd=6200, window=None):
    """Far-detector event counts per channel from a SINGLE exposure anchor.

    One exposure ``dt`` is fixed by N(mu-, FHC/fwd) = ``num_mu_fwd``; all 8
    channels (both horn modes) then follow from truth sigma x oscillated FD flux
    under that single ``dt`` -- so the FHC/RHC split rides the physical relative
    flux, exactly like the ND (``compute_nd_sample_counts``). This is the
    toy-model-consistent normalization: NOT a per-mode hand-set anchor. (The old
    two-anchor form, num_mu_bwd=2303, additionally mis-pinned the wrong-sign mu-
    in RHC; both are superseded by this single-anchor treatment.)"""
    s = setup
    dsigma_mu, dsigma_mubar, dsigma_e, dsigma_ebar = _truth_dsigmas(s)
    enu_mu = s.kin_grid_mu.all_enu
    enu_e = s.kin_grid_e.all_enu
    dV3_mu = grab(s.kin_grid_mu.dV3)
    dV3_e = grab(s.kin_grid_e.dV3)
    # Every channel is counted in the SAME analysis window as the anchor, so
    # the reported counts are the events actually analyzed (see FD_EWINDOW).
    w_mu = _window_weights(enu_mu, window)
    w_e = _window_weights(enu_e, window)

    def nd(flux, flav, grid_enu):
        return flux(flav, grid_enu)

    counts = {}
    dt = None
    # UNOSCILLATED FD fluxes: the near/far ratio is energy-dependent, so the
    # FD counts must come from the FD tables, not from the ND ones.
    for mode, flux in (("fwd", s.fd_unosc_fwd), ("bwd", s.fd_unosc_bwd)):
        phi_mu = nd(flux, pid.numu, enu_mu)
        phi_e = nd(flux, pid.nue, enu_e)
        phi_mubar = nd(flux, pid.numubar, enu_mu)
        phi_ebar = nd(flux, pid.nuebar, enu_e)

        phi_FD_mu = make_phi_FD(phi_mu, phi_e, enu_mu, s23=s23_true, d31=d31_true,
                                dcp=dcp_true, lepton="mu")
        rate_mu = (grab(dsigma_mu) * grab(phi_FD_mu) * w_mu).sum() * dV3_mu
        if dt is None:                       # SINGLE anchor, fixed once on FHC mu-
            dt = num_mu_fwd / rate_mu

        phi_FD_mubar = make_phi_FD(phi_mubar, phi_ebar, enu_mu, s23=s23_true, d31=d31_true,
                                   dcp=dcp_true, lepton="mu", anti=True)
        rate_mubar = (grab(dsigma_mubar) * grab(phi_FD_mubar) * w_mu).sum() * dV3_mu

        phi_FD_e = make_phi_FD(phi_mu, phi_e, enu_e, s23=s23_true, d31=d31_true,
                               dcp=dcp_true, lepton="e")
        rate_e = (grab(dsigma_e) * grab(phi_FD_e) * w_e).sum() * dV3_e

        phi_FD_ebar = make_phi_FD(phi_mubar, phi_ebar, enu_e, s23=s23_true, d31=d31_true,
                                  dcp=dcp_true, lepton="e", anti=True)
        rate_ebar = (grab(dsigma_ebar) * grab(phi_FD_ebar) * w_e).sum() * dV3_e

        counts["mu_" + mode] = num_mu_fwd if mode == "fwd" else int(round(rate_mu * dt))
        counts["mubar_" + mode] = int(round(rate_mubar * dt))
        counts["e_" + mode] = int(round(rate_e * dt))
        counts["ebar_" + mode] = int(round(rate_ebar * dt))

    for mode in ("fwd", "bwd"):
        counts["etot_" + mode] = counts["e_" + mode] + counts["ebar_" + mode]
        counts["mutot_" + mode] = counts["mu_" + mode] + counts["mubar_" + mode]
    return counts


# Near-detector event counts, computed self-consistently the same way as the FD
# counts above but with the *un-oscillated* ND flux (each lepton channel fed by
# its own-flavor ND flux) and a single anchor: N(mu-, fwd) = ``num_mu_fwd``. One
# exposure ``dt`` is fixed by that anchor and applied to all 8 channels, so the
# per-channel counts (mu+/-, e+/-, fwd/bwd) follow from truth sigma x ND flux --
# exactly the FD recipe, minus oscillation. These are the correct per-channel
# multinomial totals for the finite-ND-statistics systematic (the old code used a
# single flat 3e7 for every channel, over-counting the rare channels' events).
def compute_nd_sample_counts(setup, num_mu_fwd=int(3e7)):
    """Per-channel ND event counts anchored on N(mu-, fwd)=``num_mu_fwd``.

    Returns a dict keyed by ``CHANNELS`` (plus etot_/mutot_ per mode), the total
    number of ND events per channel from the true toy cross section and the
    nominal (un-drawn) ND flux -- shot-independent, so the ND-stats effect stays
    orthogonal to the flux draw.
    """
    s = setup
    dsigma_mu, dsigma_mubar, dsigma_e, dsigma_ebar = _truth_dsigmas(s)
    dsig = {"mu": grab(dsigma_mu), "mubar": grab(dsigma_mubar),
            "e": grab(dsigma_e), "ebar": grab(dsigma_ebar)}
    enu = {"mu": s.kin_grid_mu.all_enu, "mubar": s.kin_grid_mu.all_enu,
           "e": s.kin_grid_e.all_enu, "ebar": s.kin_grid_e.all_enu}
    dV3 = {"mu": grab(s.kin_grid_mu.dV3), "mubar": grab(s.kin_grid_mu.dV3),
           "e": grab(s.kin_grid_e.dV3), "ebar": grab(s.kin_grid_e.dV3)}
    flav = {"mu": pid.numu, "mubar": pid.numubar, "e": pid.nue, "ebar": pid.nuebar}

    rate = {}
    for mode, flux in (("fwd", s.nd_flux_fwd), ("bwd", s.nd_flux_bwd)):
        for base in ("mu", "mubar", "e", "ebar"):
            phi = grab(flux(flav[base], enu[base]))          # un-oscillated ND flux
            rate[base + "_" + mode] = float((dsig[base] * phi).sum() * dV3[base])

    dt = num_mu_fwd / rate["mu_fwd"]                          # single exposure anchor
    counts = {ch: int(round(rate[ch] * dt)) for ch in CHANNELS}
    for mode in ("fwd", "bwd"):
        counts["etot_" + mode] = counts["e_" + mode] + counts["ebar_" + mode]
        counts["mutot_" + mode] = counts["mu_" + mode] + counts["mubar_" + mode]
    return counts


# ----------------------------------------------------------------------------
# Load FD samples + build I / dsigma_samp (notebook cells 34, 38-43)
# ----------------------------------------------------------------------------
SAMPLE_CHANNELS = ["mu_fwd", "mu_bwd", "e_fwd", "e_bwd",
                   "mubar_fwd", "mubar_bwd", "ebar_fwd", "ebar_bwd"]


@dataclass
class FDData:
    all_Enu_mu: torch.Tensor
    all_Enu_e: torch.Tensor
    dEnu_mu: torch.Tensor
    dEnu_e: torch.Tensor
    phi_ND: dict          # phi_ND[<chan>] numpy on FD grid, chan in CHANNELS
    phi_ND_model: dict    # ND fluxes used by the MODEL NLL path (== phi_ND
                          # unless a flux-uncertainty draw was applied)
    I: dict               # truth integrals I["mu"/"e"/"mubar"/"ebar"] (cpu)
    I_model: dict         # ML model integrals, same keys (cpu)
    dsigma_samp: dict     # truth dsigma_samp[<key>] (cpu), keys below
    dsigma_samp_model: dict   # ML model dsigma_samp[<key>] (cpu)
    n_samps: dict         # sample counts per channel incl etot/mutot


# keys used by the "confused" optimizer path (cell 43, subset)
DSAMP_KEYS = ["mutot_mu_fwd", "mutot_mubar_fwd", "etot_e_fwd", "etot_ebar_fwd",
              "mutot_mu_bwd", "mutot_mubar_bwd", "etot_e_bwd", "etot_ebar_bwd"]


def load_fd_samples(setup, model_calc, model_calc_bar,
                    N_Enu_FD=512, N_v_FD=256, sample_dir="FD_samples",
                    flux_shifts=None):
    """Load pre-saved FD samples and build the I / dsigma_samp arrays.

    Requires ``sample_dir`` to already contain the 8 base channel ``.npy`` files
    (see ``generate_fd_samples.py``). Raises FileNotFoundError otherwise.

    ``flux_shifts``, if given, is a dict ``channel -> fractional shift array``
    on the FD Enu grid (see systematics.draw_flux_shifts); it perturbs the ND
    fluxes used by the MODEL NLL path (``phi_ND_model``) only. The truth path
    and the FD samples always use the nominal fluxes.
    """
    s = setup
    km = s.kin_mapping

    samples = {}
    for ch in SAMPLE_CHANNELS:
        path = os.path.join(sample_dir, "samples_FD_%s.npy" % ch)
        if not os.path.exists(path):
            raise FileNotFoundError(
                "Missing FD sample file %r. Generate samples first with "
                "`python generate_fd_samples.py` (the optimizer requires "
                "pre-saved samples)." % path)
        samples[ch] = np.load(path)

    # combined-charge ("tot") samples by concatenation (cell 34)
    for mode in ("fwd", "bwd"):
        samples["etot_" + mode] = np.concatenate([samples["e_" + mode], samples["ebar_" + mode]])
        samples["mutot_" + mode] = np.concatenate([samples["mu_" + mode], samples["mubar_" + mode]])

    n_samps = {k: len(v) for k, v in samples.items()}

    # unpack the combined channels into (v1, v2) kinematic samples (cell 38)
    samp = {}
    for ch in ("mutot_fwd", "mutot_bwd", "etot_fwd", "etot_bwd"):
        enu_c, emu_c, cost_c = torch.tensor(samples[ch].T)
        samp[ch] = make_newk2(emu_c, cost_c)  # (v1, v2)

    # FD grids (cell 39)
    # FD integrals run over the ANALYSIS window, not the ND training grid:
    # the samples live there and the likelihood's normalizer must match them.
    all_Enu_mu = torch.linspace(FD_EWINDOW[0], FD_EWINDOW[1], N_Enu_FD)
    all_v1_mu = torch.linspace(s.kin_grid_mu.all_var1.min(), s.kin_grid_mu.all_var1.max(), N_v_FD)
    all_v2_mu = torch.linspace(s.kin_grid_mu.all_var2.min(), s.kin_grid_mu.all_var2.max(), N_v_FD)
    dEnu_mu = safe_make_dx(all_Enu_mu)

    all_Enu_e = torch.linspace(FD_EWINDOW[0], FD_EWINDOW[1], N_Enu_FD)
    all_v1_e = torch.linspace(s.kin_grid_e.all_var1.min(), s.kin_grid_e.all_var1.max(), N_v_FD)
    all_v2_e = torch.linspace(s.kin_grid_e.all_var2.min(), s.kin_grid_e.all_var2.max(), N_v_FD)
    dEnu_e = safe_make_dx(all_Enu_e)
    assert torch.allclose(all_Enu_e, all_Enu_mu)

    # UNOSCILLATED FD fluxes on the FD grid. The FD likelihood oscillates
    # these (fd_optimize_torch.batched_nll builds phi[ch_mu]*P + phi[ch_e]*P),
    # so they must be the FD tables; using the ND ones assumed a constant
    # near/far ratio. The name `phi_ND` is kept for the npz/record schema.
    phi_ND = {
        "mu_fwd": s.fd_unosc_fwd(pid.numu, all_Enu_mu),
        "mubar_fwd": s.fd_unosc_fwd(pid.numubar, all_Enu_mu),
        "e_fwd": s.fd_unosc_fwd(pid.nue, all_Enu_e),
        "ebar_fwd": s.fd_unosc_fwd(pid.nuebar, all_Enu_e),
        "mu_bwd": s.fd_unosc_bwd(pid.numu, all_Enu_mu),
        "mubar_bwd": s.fd_unosc_bwd(pid.numubar, all_Enu_mu),
        "e_bwd": s.fd_unosc_bwd(pid.nue, all_Enu_e),
        "ebar_bwd": s.fd_unosc_bwd(pid.nuebar, all_Enu_e),
    }

    if flux_shifts is None:
        phi_ND_model = phi_ND
    else:
        phi_ND_model = {ch: np.clip(np.asarray(phi) * (1 + flux_shifts[ch]), 0, None)
                        for ch, phi in phi_ND.items()}

    # integrals (cell 40). Moved to CPU for the CPU optimizer loop. The ML path
    # uses the *model* integrals (I_model), matching the truth path's use of the
    # truth integrals -- the notebook erroneously used truth I for both.
    I = {
        "mu": make_I(s.truth, all_Enu_mu, all_v1_mu, all_v2_mu, flavor=pid.mu, kin_mapping=km).cpu(),
        "e": make_I(s.truth, all_Enu_e, all_v1_e, all_v2_e, flavor=pid.e, kin_mapping=km).cpu(),
        "mubar": make_I(s.truth_bar, all_Enu_mu, all_v1_mu, all_v2_mu, flavor=pid.mubar, kin_mapping=km).cpu(),
        "ebar": make_I(s.truth_bar, all_Enu_e, all_v1_e, all_v2_e, flavor=pid.ebar, kin_mapping=km).cpu(),
    }
    I_model = {
        "mu": make_I(model_calc, all_Enu_mu, all_v1_mu, all_v2_mu, flavor=pid.mu, kin_mapping=km).cpu(),
        "e": make_I(model_calc, all_Enu_e, all_v1_e, all_v2_e, flavor=pid.e, kin_mapping=km).cpu(),
        "mubar": make_I(model_calc_bar, all_Enu_mu, all_v1_mu, all_v2_mu, flavor=pid.mubar, kin_mapping=km).cpu(),
        "ebar": make_I(model_calc_bar, all_Enu_e, all_v1_e, all_v2_e, flavor=pid.ebar, kin_mapping=km).cpu(),
    }

    # dsigma_samp on the combined channels (cell 43, subset used by the
    # "confused" optimizer). _ds(calc, mode-grid energies, channel, flavor)
    def _ds(calc, all_enu, ch, flavor):
        v1, v2 = samp[ch]
        return make_dsigma_samp(calc, all_enu, v1, v2, flavor=flavor, kin_mapping=km).cpu()

    dsigma_samp = {}
    dsigma_samp_model = {}
    for mode in ("fwd", "bwd"):
        # muon channels use the mu energy grid + mutot samples
        dsigma_samp["mutot_mu_" + mode] = _ds(s.truth, all_Enu_mu, "mutot_" + mode, pid.mu)
        dsigma_samp["mutot_mubar_" + mode] = _ds(s.truth_bar, all_Enu_mu, "mutot_" + mode, pid.mubar)
        dsigma_samp["etot_e_" + mode] = _ds(s.truth, all_Enu_e, "etot_" + mode, pid.e)
        dsigma_samp["etot_ebar_" + mode] = _ds(s.truth_bar, all_Enu_e, "etot_" + mode, pid.ebar)

        dsigma_samp_model["mutot_mu_" + mode] = _ds(model_calc, all_Enu_mu, "mutot_" + mode, pid.mu)
        dsigma_samp_model["mutot_mubar_" + mode] = _ds(model_calc_bar, all_Enu_mu, "mutot_" + mode, pid.mubar)
        dsigma_samp_model["etot_e_" + mode] = _ds(model_calc, all_Enu_e, "etot_" + mode, pid.e)
        dsigma_samp_model["etot_ebar_" + mode] = _ds(model_calc_bar, all_Enu_e, "etot_" + mode, pid.ebar)

    return FDData(all_Enu_mu, all_Enu_e, dEnu_mu, dEnu_e,
                  phi_ND, phi_ND_model, I, I_model,
                  dsigma_samp, dsigma_samp_model, n_samps)


def load_model(model_path, setup, force_ds_pos=False):
    """Rebuild (model, model_bar) from a saved checkpoint and wrap in ModelCalc.

    ``force_ds_pos=True`` clamps the model dsigma at 0 in the returned
    calculators (inference-side positivity; training is untouched -- the
    trainers build their own ModelCalc instances).
    """
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    n_sf = ckpt.get("meta", {}).get("n_sf", 5)
    model = SFModel(n_sf=n_sf, force_pos=False).to(setup.device)
    model_bar = SFModel(n_sf=n_sf, force_pos=False).to(setup.device)
    model.load_state_dict(ckpt["model"])
    model_bar.load_state_dict(ckpt["model_bar"])
    model_calc = ModelCalc(model, enumax=ENUMAX, unique=False, anti=False,
                           force_ds_pos=force_ds_pos, device=setup.device)
    model_calc_bar = ModelCalc(model_bar, enumax=ENUMAX, unique=False, anti=True,
                               force_ds_pos=force_ds_pos, device=setup.device)
    return model, model_bar, model_calc, model_calc_bar
