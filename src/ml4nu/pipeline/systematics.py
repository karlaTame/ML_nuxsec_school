"""Finite ND statistics, detector smearing, and correlated flux helpers.

The optional campaign.fwdfold_shot driver uses multinomial ND resampling,
radial-locus energy smearing at fixed angle for ND histograms, smeared FD
event densities, and a joint ND/FD flux covariance. Legacy var1-only kernels
and model-side flux-shift helpers remain for reference calculations.

Flux draws consume a grid-independent standard-normal vector. Matched seeds
therefore describe the same underlying nuisance draw on different grids.
"""

import numpy as np
import scipy.stats
import torch

from ml4nu.pipeline import rates_common as rc
from ml4nu.utils import grab

# rates_common channel <-> flux_model channel
FLUX_MODEL_CHANNEL = {
    "mu_fwd": "FHC_numu", "mubar_fwd": "FHC_numubar",
    "e_fwd": "FHC_nue", "ebar_fwd": "FHC_nuebar",
    "mu_bwd": "RHC_numu", "mubar_bwd": "RHC_numubar",
    "e_bwd": "RHC_nue", "ebar_bwd": "RHC_nuebar",
}

# ND exposure ANCHOR: N(mu-, fwd) = 3e7. The other 7 channels' event counts are
# derived self-consistently from truth sigma x un-oscillated ND flux via
# rc.compute_nd_sample_counts (NOT a flat count applied to every channel).
DEFAULT_N_ND_SAMPS = int(3e7)
# fractional E_lep resolution, all channels (sigma_E). The first paper uses 0.04;
# the draft syst notebook (cell 22) had 0.4, a factor-of-10 typo that had propagated
# here and into the campaign -- corrected 2026-09 (validated: 0.04 -> small FD-smear
# effect, 0.4 -> spurious octant-flipping shift). Anything run at 0.4 is superseded.
DEFAULT_SMEAR_WIDTH = 0.04


# ---------------------------------------------------------------------------
# Detector smearing (notebook cells 5 + 22, ported verbatim)
# ---------------------------------------------------------------------------

def make_1d_rel_smear_kernel(v, width, dv):
    """Relative-Gaussian smearing kernel; shape (v_smear, v_true).

    Columns are normalized so that int dv K(v|v') = 1.
    """
    assert len(v.shape) == 1
    v_true = v[None, :]
    v_smear = v[:, None]
    d = (v_true - v_smear) / v_true
    delta = np.exp(-d ** 2 / (2 * width ** 2))
    delta = delta / (delta.sum(0) * dv)[None, :]
    return delta

def make_smear_kernel(kg, width):
    """Kernel over var1 for smearing in E_lep = var1^2 (notebook make_smear1d).

    Built in energy space, transformed with the Jacobian J = 2*var1, then
    renormalized against dvar1 (the dv=1 in the first call is a placeholder;
    the final renormalization is what counts).
    """
    v1 = grab(kg.all_var1)
    smear = make_1d_rel_smear_kernel(v1 ** 2, width, 1.0)
    J = 2 * v1
    smear = smear * J
    smear = smear / (smear.sum(0) * float(grab(kg.dvar1)))[None, :]
    return torch.as_tensor(smear, dtype=torch.float64)


def make_smear_kernels(setup, width=DEFAULT_SMEAR_WIDTH):
    """Per-channel kernels in CHANNELS order keys; fwd/bwd share a grid."""
    kern = {
        "mu": make_smear_kernel(setup.kin_grid_mu, width).to(setup.device),
        "e": make_smear_kernel(setup.kin_grid_e, width).to(setup.device),
    }
    kern["mubar"], kern["ebar"] = kern["mu"], kern["e"]
    return {ch: kern[ch.rsplit("_", 1)[0]] for ch in rc.CHANNELS}


def apply_smearing(S, P, dvar1):
    """Smear the var1 axis of a (v1, v2) marginal (notebook cell 24/44).

    DEPRECATED for the FD study: this smears var1 only (holds v2 fixed), which is
    wrong -- an E_lep offset at fixed angle moves v1 AND v2 radially. Use
    make_locus_smear_operator / apply_locus_smear instead. Kept for the legacy
    single-axis path and back-compat.
    """
    return S.to(P.device) @ P * float(grab(dvar1))


def make_locus_smear_operator(kg, width, ngrid=64, nsig=5.0, device=None):
    """Sparse (N,N) operator: Gaussian E_lep=var1^2 smearing AT FIXED ANGLE.

    At fixed cos(theta) the ray v2/v1 = sqrt((1-cost)/2) is constant, and BOTH
    v1=sqrt(E_mu), v2=sqrt(E_mu(1-cost)/2) scale by s=sqrt(E_obs/E_true) -- so an
    E_mu smear is a purely radial rescaling in (v1, v2). Each true cell scatters
    its mass along its own ray with the relative-Gaussian E_mu weights
    (sigma = width*E_true, +-nsig, ngrid points) and a bilinear deposit onto the
    Cartesian (v1, v2) grid. Per-cell weights are normalized to unit sum (the mass
    Jacobian, also removing truncation/discretization bias). This is the binned
    analog of smeared_dsigma_samp and REPLACES the wrong var1-only kernel.

    Acts on a flattened (var1, var2) row-major marginal (N = dim1*dim2), matching
    make_marginal_rate's (v1, v2) output. Some mass scaled past the grid edge is
    dropped (physical acceptance loss); interior cells conserve to ~1e-6.
    """
    import scipy.sparse as sp
    v1 = grab(kg.all_var1).astype(np.float64)
    v2 = grab(kg.all_var2).astype(np.float64)
    n1, n2 = len(v1), len(v2); N = n1 * n2
    dv1 = float(grab(kg.dvar1)); dv2 = float(grab(kg.dvar2))
    V1, V2 = np.meshgrid(v1, v2, indexing="ij")            # (n1, n2) row-major
    emu_t = (V1 ** 2).ravel(); r1 = V1.ravel(); r2 = V2.ravel()   # (N,)
    t = np.linspace(0.0, 1.0, ngrid)
    lo = np.clip(emu_t * (1 - nsig * width), 1e-9, None)
    hi = emu_t * (1 + nsig * width)
    emu_o = lo[:, None] + (hi - lo)[:, None] * t[None, :]  # (N, g) true->obs E_mu grid
    demu = (hi - lo) / (ngrid - 1)
    sig = width * emu_t[:, None]                           # resolution vs TRUE energy
    w = np.exp(-0.5 * ((emu_o - emu_t[:, None]) / sig) ** 2) * demu[:, None]
    w /= w.sum(1, keepdims=True)                           # mass Jacobian / trunc-safe
    s = np.sqrt(emu_o / emu_t[:, None])                    # radial scale (N, g)
    f1 = (s * r1[:, None] - v1[0]) / dv1
    f2 = (s * r2[:, None] - v2[0]) / dv2
    i0 = np.floor(f1).astype(np.int64).ravel(); a = f1.ravel() - i0
    j0 = np.floor(f2).astype(np.int64).ravel(); b = f2.ravel() - j0
    src = np.repeat(np.arange(N), ngrid); wf = w.ravel()
    R = []; C = []; Vv = []
    for di, dj, wt in ((0, 0, (1 - a) * (1 - b)), (1, 0, a * (1 - b)),
                       (0, 1, (1 - a) * b), (1, 1, a * b)):
        ii = i0 + di; jj = j0 + dj
        ok = (ii >= 0) & (ii < n1) & (jj >= 0) & (jj < n2)
        R.append(ii[ok] * n2 + jj[ok]); C.append(src[ok]); Vv.append((wf * wt)[ok])
    # Dense (N,N): the ND grid is tiny (~16x16 -> 256), so dense costs <1 MB and
    # is fully autograd-friendly. scipy coo->array sums the duplicate bilinear
    # deposits; a torch sparse op is unnecessary and doesn't back-prop the dense
    # operand in this build.
    A = sp.coo_matrix((np.concatenate(Vv), (np.concatenate(R), np.concatenate(C))),
                      shape=(N, N)).toarray()
    T = torch.tensor(A, dtype=torch.float64)
    return T.to(device) if device is not None else T


def make_locus_smear_operators(setup, width=DEFAULT_SMEAR_WIDTH, ngrid=64, nsig=5.0):
    """Per-channel locus smear operators keyed by CHANNELS (fwd/bwd share a grid)."""
    op = {"mu": make_locus_smear_operator(setup.kin_grid_mu, width, ngrid, nsig, setup.device),
          "e": make_locus_smear_operator(setup.kin_grid_e, width, ngrid, nsig, setup.device)}
    op["mubar"], op["ebar"] = op["mu"], op["e"]
    return {ch: op[ch.rsplit("_", 1)[0]] for ch in rc.CHANNELS}


def apply_locus_smear(S, P):
    """Apply a locus operator to a (var1, var2) marginal. P may be 2-D (n1, n2)
    or batched (..., n1, n2); differentiable (dense matmul) for the ML path."""
    n1, n2 = P.shape[-2], P.shape[-1]
    flat = P.reshape(-1, n1 * n2)                          # (B, N)
    out = flat @ S.to(P.device).T                          # (B, N)  == S @ flat per row
    return out.reshape(P.shape)


# ---------------------------------------------------------------------------
# Finite ND statistics (notebook cells 5 + 24)
# ---------------------------------------------------------------------------

def noise_hist(P, n, rng):
    """Multinomial-resample a rate histogram, preserving its total norm."""
    P_np = grab(P)
    norm = P_np.sum()
    p = P_np.ravel() / norm
    counts = scipy.stats.multinomial(p=p, n=int(n)).rvs(random_state=rng)
    P_ = counts.reshape(P_np.shape).astype(np.float64)
    P_ = P_ / P_.sum() * norm
    if isinstance(P, torch.Tensor):
        return torch.as_tensor(P_, dtype=P.dtype, device=P.device)
    return P_


# ---------------------------------------------------------------------------
# Flux uncertainties (flux_model.NDFluxModel; replaces notebook cell 20)
# ---------------------------------------------------------------------------

def draw_flux_shifts(enu_by_channel, seed, model=None, detector="ND"):
    """One correlated fractional flux draw, evaluated per channel.

    enu_by_channel: {rc channel: 1-D energy array (GeV)}. Returns
    {rc channel: shift array delta(E)} so that phi' = phi * (1 + delta).

    The underlying nuisance vector is grid-independent, so calling this twice
    with the same seed on different grids (ND training grid, FD fit grid)
    yields the SAME physical draw -- this is how ND/FD draws stay matched.
    """
    from ml4nu.pipeline import flux_model
    if model is None:
        model = flux_model.load_default()
    # With the JOINT ND+FD model the channel labels carry a detector prefix.
    # The nuisance vector is drawn from `seed` alone, so asking for detector
    # "ND" and then "FD" with the SAME seed yields the two halves of ONE
    # correlated draw -- near and far now move together only as far as the
    # measured ~0.94 correlation allows, instead of identically.
    joint = len(getattr(model, "detectors", ("ND",))) > 1
    key = (lambda ch: f"{detector}_{FLUX_MODEL_CHANNEL[ch]}") if joint \
        else (lambda ch: FLUX_MODEL_CHANNEL[ch])
    enu = {key(ch): np.asarray(grab(E), dtype=np.float64)
           for ch, E in enu_by_channel.items()}
    rng = np.random.default_rng(seed)
    draw = model.sample(enu, size=1, rng=rng, fractional=True)
    return {ch: draw[key(ch)][0] for ch in enu_by_channel}


def perturb_flux(phi, delta):
    """phi * (1 + delta), clipped at zero, preserving type/device."""
    if isinstance(phi, torch.Tensor):
        d = torch.as_tensor(delta, dtype=phi.dtype, device=phi.device)
        return (phi * (1 + d)).clamp_min(0)
    return np.clip(np.asarray(phi) * (1 + delta), 0, None)


# ---------------------------------------------------------------------------
# Orchestration for the ND training inputs
# ---------------------------------------------------------------------------

def channel_dvar1(setup, ch):
    """dvar1 of the kinematics grid a channel's marginal lives on."""
    base = ch.rsplit("_", 1)[0]
    kg = setup.kin_grid_mu if base in ("mu", "mubar") else setup.kin_grid_e
    return kg.dvar1


def apply_nd_systematics(setup, fluxes, marginals, *, kernels=None,
                         shifts=None, n_samps=None, rng=None):
    """Transform nominal (fluxes, marginals) from rc.build_training_inputs.

    Returns (fluxes_model, marginals_data):
      * fluxes_model -- fluxes with the flux-uncertainty ``shifts`` applied
        (these feed the MODEL marginals in training);
      * marginals_data -- data targets, multinomial-resampled with ``n_samps``
        draws and then smeared with ``kernels`` (notebook order); built from
        NOMINAL fluxes.

    Any of kernels/shifts/n_samps left as None disables that effect; with all
    None the inputs pass through unchanged (baseline bit-compatibility).
    """
    phi = dict(zip(rc.CHANNELS, fluxes))
    P = dict(zip(rc.CHANNELS, marginals))

    if shifts is not None:
        phi = {ch: perturb_flux(phi[ch], shifts[ch]) for ch in rc.CHANNELS}

    if n_samps is not None:
        rng = np.random.default_rng(rng)
        # n_samps may be a per-channel dict {ch: N} (self-consistent ND counts
        # from rc.compute_nd_sample_counts) or a single scalar (legacy flat count).
        n_of = n_samps.__getitem__ if isinstance(n_samps, dict) else (lambda ch: n_samps)
        P = {ch: noise_hist(P[ch], n_of(ch), rng) for ch in rc.CHANNELS}

    if kernels is not None:
        # kernels are full-locus operators (make_locus_smear_operators): smear
        # E_lep at fixed angle (radial in v1,v2), not the wrong var1-only kernel.
        P = {ch: apply_locus_smear(kernels[ch], P[ch]) for ch in rc.CHANNELS}

    return [phi[ch] for ch in rc.CHANNELS], [P[ch] for ch in rc.CHANNELS]


# ---------------------------------------------------------------------------
# Far-detector detector smearing (unbinned, pointwise E_lep resolution)
# ---------------------------------------------------------------------------
# DATA: per-event random E_mu offset (one reconstruction draw per shot).
# MODEL: smeared model density at each observed event,
#   p_smeared(v') = int dEmu_true Smear(Emu_obs|Emu_true) p(v1,v2)|_(Emu_true, cos_obs)
# following the fixed-angle locus (v1,v2)=make_newk2(Emu_true, cos_obs) -- both v1
# and v2 move radially, which a v1-only histogram smear cannot capture. Replaces the
# per-event dsigma_samp only (the norm integral I is smearing-invariant). Independent
# of the oscillation params, so it is a one-time precompute; the FD fit loop is
# unchanged. Ported from the single-channel event-level offset study.

def load_tot_events(sample_dir):
    """{tot_ch: (emu, cost)} for mutot/etot x fwd/bwd, concatenated as in load_fd_samples."""
    import os
    ev = {}
    for mode in ("fwd", "bwd"):
        for tot, bases in (("mutot", ("mu", "mubar")), ("etot", ("e", "ebar"))):
            a = np.concatenate([np.load(os.path.join(sample_dir, f"samples_FD_{b}_{mode}.npy"))
                                for b in bases], axis=0)
            ev[f"{tot}_{mode}"] = (a[:, 1].astype(np.float64), a[:, 2].astype(np.float64))  # emu, cost
    return ev


def smeared_dsigma_samp(calc, all_enu, emu_obs, cost, flavor, km, width, ngrid, dev, chunk=None, nsig=5.0):
    """(n_events, n_enu) smeared model density at observed events (CPU).

    make_dsigma_samp loops Python over n_enu, so cost is launch-bound in the number
    of event-chunks; pick the chunk from a fixed points budget (chunk*ngrid) so memory
    stays bounded while the launch count drops ~1/chunk. chunk=None auto-sizes it.
    """
    from ml4nu.kinematics import make_newk2
    from ml4nu import far_detector_fit as _fdf
    _fdf.tqdm = lambda x, **k: x  # silence the per-Enu progress bar
    make_dsigma_samp = _fdf.make_dsigma_samp
    if chunk is None:
        chunk = max(16, 262144 // ngrid)   # ~0.25M event*grid points per make_dsigma_samp call
    emu_obs = torch.as_tensor(emu_obs, dtype=torch.float64, device=dev)
    cost = torch.as_tensor(cost, dtype=torch.float64, device=dev)
    n = emu_obs.shape[0]; ne = len(all_enu)
    out = torch.empty(n, ne, dtype=torch.float64)
    t = torch.linspace(0.0, 1.0, ngrid, device=dev, dtype=torch.float64)
    norm_c = 1.0 / np.sqrt(2 * np.pi)
    for st in range(0, n, chunk):
        e = emu_obs[st:st + chunk]; c = cost[st:st + chunk]; cc = e.shape[0]
        lo = (e * (1.0 - nsig * width)).clamp_min(1e-3); hi = e * (1.0 + nsig * width)
        emu_g = lo[:, None] + (hi - lo)[:, None] * t[None, :]          # (cc, ngrid) true E_mu grid
        demu = (hi - lo) / (ngrid - 1)
        sig = width * emu_g                                           # relative E_lep resolution
        w = norm_c / sig * torch.exp(-0.5 * ((e[:, None] - emu_g) / sig) ** 2) * demu[:, None]
        v1g, v2g = make_newk2(emu_g.reshape(-1), c.repeat_interleave(ngrid))  # fixed-angle locus
        ds = make_dsigma_samp(calc, all_enu, v1g, v2g, flavor=flavor, kin_mapping=km)
        ds = ds.reshape(cc, ngrid, ne)
        out[st:st + cc] = (w[:, :, None] * ds).sum(1).cpu()
    return out


def build_smeared_dsigma(sample_dir, calc, calc_bar, all_Enu_mu, all_Enu_e, km, width,
                         ngrid, offset_seed, dev, chunk=None):
    """Full 8-entry smeared dsigma_samp dict (per-event E_mu offset + model smear).
    calc/calc_bar are truth (s.truth/s.truth_bar) or model (mc/mcb) calculators."""
    from ml4nu.pdg import pid
    ev = load_tot_events(sample_dir)
    rng = np.random.default_rng(offset_seed)
    out = {}
    for mode in ("fwd", "bwd"):
        for tot, enu, hyps in (("mutot", all_Enu_mu, [("mu", pid.mu, calc), ("mubar", pid.mubar, calc_bar)]),
                               ("etot", all_Enu_e, [("e", pid.e, calc), ("ebar", pid.ebar, calc_bar)])):
            emu, cost = ev[f"{tot}_{mode}"]
            emu_obs = np.clip(emu * (1.0 + rng.normal(0.0, width, size=emu.shape)), 1e-3, None)  # DATA offset
            for name, flav, cc in hyps:
                out[f"{tot}_{name}_{mode}"] = smeared_dsigma_samp(cc, enu, emu_obs, cost, flav, km,
                                                                 width, ngrid, dev, chunk)
    return out
