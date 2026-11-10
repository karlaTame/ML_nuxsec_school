"""Shot-structured forward-fold: one flux draw per shot into the DATA, model
absorbs it via a_i nuisances (priors OFF). Mirrors the campaign shot convention
(flux draw = default_rng(SeedSequence(shot)) -> same draws as systflux/total).

args: shot(int)  effects(flux|total)  fit(ml|truth|gen)
  effects flux : flux draw + smear (ND finite-stats OFF)   == systflux analog
  effects total: flux draw + ND finite-stats + smear       == total analog
  fit ml   : a_i gradient-trained with the network (VarPro solves c)
  fit truth: a_i linear least-squares with the TRUE xsec  (new truth_syst)
  fit gen  : only (re)generate the drawn-flux FD samples, then exit
env: NSTEP(100000) LR_A(1e-3) FDSEED(25=best-combined-bias no-syst seed)
"""
import sys, os, numpy as np, torch
from dataclasses import replace
from ml4nu.pipeline import rates_common as rc, systematics as syst, fd_optimize_torch as ft, flux_model
from ml4nu.pipeline import generate_fd_samples as gfd
from ml4nu.pdg import pid
from ml4nu.cross_section import ModelCalc
from ml4nu.flux import FluxFarDetector
from ml4nu.utils import precompute_bootstrap_tensor
from ml4nu.pipeline.train_near_detector import _sf_basis, _fold_c, LOSS_NORM

SHOT = int(sys.argv[1]); EFFECTS = sys.argv[2]; FIT = sys.argv[3]
NSTEP = int(os.environ.get("NSTEP", "100000")); LR_A = float(os.environ.get("LR_A", "1e-3"))
# PROGRESS=1 prints a periodic training-progress line (step/NSTEP, loss, best);
# PROGRESS_EVERY sets the cadence. Off by default -> no effect on numerics.
_PROG = os.environ.get("PROGRESS", "0") == "1"; _PROG_EVERY = int(os.environ.get("PROGRESS_EVERY", "2000"))
LAMBDA_A = float(os.environ.get("LAMBDA_A", "0"))   # flux prior strength on (b_model+a); 0 = off
# Flux systematic draw: an offset of the model's FD flux, taken from the
# systematics model's own covariance -- the standard "throw universes"
# procedure. It lives in the ND->FD residual eps ~ N(0, C_fd|nd), because that
# is the ONLY subspace where an offset survives the ND constraint: the forward
# fold fits all 68 ND-determined directions, so an offset there is absorbed
# exactly (with LAMBDA_A=0 the model draw b_model is a pure reparametrization
# of the starting point and provably cannot move the converged fit -- measured
# shot-to-shot FD flux spread 0.0000% over 100 b_model draws). Throwing the
# offset in the eps directions instead keeps the systematic where the physics
# is and needs no new FD samples.
#   shot (default) -- one universe per shot, fixed across its bootstraps. The
#                     shot-to-shot spread IS the flux systematic, and pooling
#                     N_syst x N_boot is the paper's own prescription.
#   boot           -- one universe per bootstrap replicate: marginalizes inside
#                     a single run (cheaper, but conflates systematic and
#                     statistical within one bootstrap ensemble, and the scipy
#                     reference path cannot consume it).
#   0 / off        -- no residual; the FD flux is the bare conditional mean and
#                     the flux systematic is understated by construction.
# Output directory. Non-default likelihoods get their OWN directory as well
# as a filename suffix. The suffix alone is a trap: reproduce_paper_plots'
# FF_RE does not accept it and raises on any file it cannot parse, so one
# shape run written beside the extended results aborts the figure notebook.
# Separate directories also make it impossible to pool two likelihoods by
# accident. Resolved from the env here because ft.LIKELIHOOD is not set
# until the FD stage, while the network is written before it.
LIKELIHOOD_ENV = os.environ.get("LIKELIHOOD", "extended")
OUTDIR = ("results_fwdfold" if LIKELIHOOD_ENV == "extended"
          else f"results_fwdfold_{LIKELIHOOD_ENV}")

FLUXRESID = os.environ.get("FLUXRESID", "shot").lower()
if FLUXRESID in ("1", "true", "yes"):
    FLUXRESID = "shot"
if FLUXRESID in ("0", "false", "no", "none"):
    FLUXRESID = "off"
if FLUXRESID not in ("shot", "boot", "off"):
    raise ValueError(f"FLUXRESID must be shot|boot|off, got {FLUXRESID!r}")
FDSEED = int(os.environ.get("FDSEED", "25"))
SMEAR_W = float(os.environ.get("SMEAR_W", str(syst.DEFAULT_SMEAR_WIDTH)))   # 0.04 (paper); 0.4 was a typo
SMEAR_NGRID = int(os.environ.get("SMEAR_NGRID", "64"))   # converged (==256==1024); ~3 min uncontended
# ND network init seed. `total` marginalizes init like the old campaign
# (VARY=both: train seed = shot); other effects hold init fixed at 17. An explicit
# TRAIN_SEED env overrides (for the isolated init-variation pack).
_canonical_ts = SHOT if EFFECTS == 'total' else 17
TRAIN_SEED = int(os.environ["TRAIN_SEED"]) if "TRAIN_SEED" in os.environ else _canonical_ts
# output tag (also the saved-network name); non-canonical init gets a _ts suffix
_tag = f"sh{SHOT}_{EFFECTS}_{FIT}" + ("_regress" if os.environ.get("PROP", "mode") == "regress" else "") + (f"_ts{TRAIN_SEED}" if TRAIN_SEED != _canonical_ts else "")
# REUSE_NET: for ml/fix, load the saved ND network + a* and skip training (valid when
# only the FD data changed, e.g. an FD-count renorm -- the ND side is untouched).
REUSE_NET = os.environ.get("REUSE_NET", "0") == "1"
# effect flags: flux (ND flux draw), ndstats (finite ND stats), fdsmear = detector
# smearing (ND-marginal + FD together, gated as one physical effect). 'nd' isolates
# ND finite-stats; 'nom' has no effects (the no-syst reference).
use_flux = EFFECTS in ('flux', 'fluxsmear', 'total')
use_ndstats = EFFECTS in ('nd', 'total')
use_fdsmear = EFFECTS in ('smear', 'fdsmear', 'fluxsmear', 'total')
NB = int(os.environ.get("N_BOOTS", "4096")); dev = rc.device   # match the results_61 campaign
s = rc.build_setup(); km = s.kin_mapping
fluxes, marg = rc.build_training_inputs(s)
phi_nom = dict(zip(rc.CHANNELS, [f.clone() for f in fluxes]))
DT = phi_nom['mu_fwd'].dtype
# Detector smearing (ND-marginal + FD) is one physical effect, gated together:
# on only for smear/fluxsmear/total, off (raw marginals, no kernel/measure) for
# flux/nom -- so the flux-only and nd-only decompositions are smear-free.
# ND-marginal smear: full-locus (E_mu at fixed angle -> radial rescaling of both
# v1 and v2), the binned analog of the FD smeared_dsigma_samp, NOT the old
# var1-only kernel (which wrongly held v2 fixed). Mass-conserving sparse operator.
kern = syst.make_locus_smear_operators(s, SMEAR_W, ngrid=SMEAR_NGRID, nsig=5.0) if use_fdsmear else None
def nd_smear(ch, P):
    return syst.apply_locus_smear(kern[ch], P) if use_fdsmear else P

# JOINT ND+FD flux covariance (2026-10-09): ONE nuisance vector a of length
# NMODE shifts both detectors, but through their real ~0.94 correlation rather
# than identically. Before this, ND modes were reused verbatim for the FD flux,
# which asserted a perfectly correlated near/far flux and removed the ~3%
# residual FD uncertainty that no ND measurement can constrain away.
fm = flux_model.load_default(); L = fm._L; NMODE = L.shape[1]
_JOINT = len(getattr(fm, "detectors", ("ND",))) > 1
def _fmkey(ch, det):
    base = syst.FLUX_MODEL_CHANNEL[ch]
    return f"{det}_{base}" if _JOINT else base
def basis_on(enu_by_channel, det="ND"):
    """Fractional flux modes f_j(E) for `det`, as (n_E, NMODE) per channel."""
    enu_fm = {_fmkey(ch, det): np.asarray(rc.grab(E)) for ch, E in enu_by_channel.items()}
    M, layout = fm._build_M(enu_fm); ML = M @ L
    return {ch: ML[layout[_fmkey(ch, det)]] for ch in enu_by_channel}
enu_nd = {ch: (s.kin_grid_mu if ch.split('_')[0] in ('mu', 'mubar') else s.kin_grid_e).all_enu for ch in rc.CHANNELS}

# PROP selects how the fitted flux nuisances reach the FAR detector.
#   mode    (the PRE-2026-10-08 campaign behavior): coefficients of the
#           136 joint eigenmodes; the FD flux reuses those modes' FD rows.
#           The ND response has rank 68, so 68 directions are ND-blind and
#           whatever lands there is decided by the regularizer.
#   regress (DEFAULT, the proper near/far extrapolation): a unit-normal
#           u of length 68 over the ND flux covariance, and the FD flux is
#           predicted by the conditional-Gaussian regression B = C_fn C_nd^-1.
#           Full rank, prior exactly N(0,I), and the irreducible residual
#           C_fd|nd becomes an explicit term instead of being asserted zero.
# Only the MODEL side changes: the data draw b_data, the ND data marginals and
# the FD samples are identical in both modes, so a shot run both ways isolates
# the propagation.
PROP = os.environ.get("PROP", "regress")
if PROP not in ("mode", "regress"):
    raise ValueError(f"PROP must be 'mode' or 'regress', got {PROP!r}")
_nf = None
if PROP == "regress":
    from ml4nu.pipeline.flux_nearfar import NearFarFluxModel
    _nf = NearFarFluxModel(fm)
    _base = {ch: syst.FLUX_MODEL_CHANNEL[ch] for ch in rc.CHANNELS}
    NMODE = _nf.n_nd

def model_basis(enu_by_channel, det="ND"):
    """Basis the MODEL's flux nuisances act through, per PROP.

    PROP=mode: the joint eigenmodes (same as basis_on). PROP=regress: the
    unit-normal u-basis at the ND, and the near/far REGRESSION response of
    that same u at the FD. ``basis_on`` itself is left alone -- the data
    draw b_data is a joint 136-mode realization in both modes.
    """
    if _nf is None:
        return basis_on(enu_by_channel, det=det)
    by_base = {_base[ch]: np.asarray(rc.grab(E)) for ch, E in enu_by_channel.items()}
    bas = _nf.nd_basis(by_base) if det == "ND" else _nf.fd_basis(by_base)
    return {ch: bas[_base[ch]] for ch in enu_by_channel}

Fnd = {ch: torch.as_tensor(v, device=dev, dtype=DT)
       for ch, v in model_basis(enu_nd, det="ND").items()}
# The DATA's ND flux shift is always a joint 136-mode realization, so it needs
# the joint basis regardless of how the MODEL is parametrized.
Fnd_data = (Fnd if _nf is None else
            {ch: torch.as_tensor(v, device=dev, dtype=DT)
             for ch, v in basis_on(enu_nd, det="ND").items()})

# Flux draws (real-data-like): the DATA carries ONE fixed realization b_data
# (shared across all shots, like the single FD-stats seed); the MODEL carries a
# per-shot random draw b_model and fits [1+a*f]*Phitilde, Phitilde=phi_nom*(1+b_model*f).
# Per-shot variation is thus on the MODEL side; a absorbs it (a ~ b_data - b_model),
# and the residual over b_model draws is the flux systematic.
flux_seed, noise_seed, smear_seed, resid_seed = (
    int(x) for x in np.random.SeedSequence(SHOT).generate_state(4))
if PROP == "regress":
    # FLUX-MODEL INIT: a per-shot draw of the flux the MODEL assumes, from the
    # same covariance. This is the campaign's long-standing proxy for
    # "sensitivity to the assumed flux", and it is kept deliberately distinct
    # from the NN-weight init, which is the proxy for "sensitivity to the
    # function parameterization". Both are initializations mechanically; they
    # probe different ingredients of the model.
    #
    # How much it BITES depends on the fit, and this is the part worth being
    # precise about:
    #   fix / truthfix (a frozen at 0) -- b_model is NOT an initialization. It
    #       is a hard, unfitted flux error the model is stuck with, so it
    #       propagates at full strength and its shot-to-shot variation is a
    #       genuine flux systematic. These are the configs behind the "flux
    #       sensitivity is large in multi-channel analyses" figure, and that
    #       result is untouched by anything in the near/far rework.
    #   ml / truth (a fitted) -- the forward fold absorbs the ND-determined
    #       part by construction, so what survives is an optimizer-path effect
    #       (the loss is non-convex and the baseline enters as a PRODUCT,
    #       phi(1+Fb)(1+Fa), not phi(1+F(b+a)), so it is not an exact
    #       reparametrization either). The irreducible piece there is eps, not
    #       b_model -- see FLUXRESID.
    # FLUXMODEL_DRAW=0 pins b_model to nominal (isolates eps).
    if os.environ.get("FLUXMODEL_DRAW", "1") != "0" and use_flux:
        _b136 = np.random.default_rng(flux_seed).standard_normal(fm._L.shape[1])
        b_model = _nf.u_from_bin_shift(_nf.nd_block_of(_b136, fm._L))
    else:
        b_model = np.zeros(NMODE)
else:
    b_model = np.random.default_rng(flux_seed).standard_normal(NMODE) if use_flux else np.zeros(NMODE)
bm_t = torch.as_tensor(b_model, device=dev, dtype=DT)
# Fixed DATA flux realization, drawn like a b_model but from a RESERVED shot index
# far outside the ensemble range -- never run as a real shot, so b_data's flux seed
# is guaranteed never reused as a per-shot b_model seed.
# ONE fixed true flux realization, as in a real experiment: nature drew once.
# Varying it per shot would be a coverage study, not an analysis procedure --
# nothing an experiment could do with its own data. The flux uncertainty is
# instead propagated INSIDE the fit, by marginalizing the irreducible ND->FD
# residual eps ~ N(0, C_fd|nd) over the bootstrap ensemble (FLUXRESID below).
# That is what a real analysis does with ND-constrained flux nuisances, and it
# needs only this one dataset.
DATA_SHOT = os.environ.get("FLUXDATA_SHOT", "424242")
# Only effects that INCLUDE the flux systematic (flux/fluxsmear/total) carry a data
# flux realization. Single-systematic packs that don't (nom/smear/nd) use NOMINAL
# flux (b_data=0) so each isolates exactly one effect -- and so their learned xsec
# isn't biased by the data draw's flux normalization (a single-detector fit can't
# separate flux-norm from sigma-norm; a non-zero b_data launders into learned sigma).
if not use_flux:
    DATA_SHOT = 'none'
_NDATA = fm._L.shape[1]          # b_data is always a joint-mode vector
if DATA_SHOT in ("none", "nominal"):
    b_data = np.zeros(_NDATA)
else:
    _bd_seed = int(np.random.SeedSequence(int(DATA_SHOT)).generate_state(3)[0])
    # The DATA always carries a joint 136-mode draw -- it is a real (ND, FD)
    # flux realization, drawn from the full covariance, and it must be
    # identical in both PROP modes so the FD samples are shared. Under
    # PROP=regress the model's 68 nuisances cannot represent its FD part
    # exactly; the shortfall IS the irreducible residual C_fd|nd.
    b_data = np.random.default_rng(_bd_seed).standard_normal(_NDATA)
bd_t = torch.as_tensor(b_data, device=dev, dtype=DT)
phitilde = {ch: phi_nom[ch] * (1 + Fnd[ch] @ bm_t) for ch in rc.CHANNELS}   # model baseline flux
# Flux nuisances a exist only to reconcile a flux perturbation (model draw b_model
# or fixed data draw b_data). With neither -- e.g. the flux-free nom baseline --
# there are NO flux nuisances: freeze a=0 rather than fit it (a spurious fit lets a
# absorb the network/xsec residual and bias the oscillation params).
has_flux_nuis = bool(use_flux or np.linalg.norm(b_data) > 1e-12)

dsig_mu, dsig_mubar, dsig_e, dsig_ebar = rc._truth_dsigmas(s)
dsig = {'mu': dsig_mu, 'mubar': dsig_mubar, 'e': dsig_e, 'ebar': dsig_ebar}
grid = {'mu': s.kin_grid_mu, 'mubar': s.kin_grid_mu, 'e': s.kin_grid_e, 'ebar': s.kin_grid_e}

# ---- DATA marginals: drawn flux, true xsec, [ND stats], ND smear ----
# Finite ND statistics: per-channel event counts anchored on N(mu-,fwd)=3e7 and
# derived self-consistently from truth sigma x un-oscillated ND flux (like FD),
# so each channel's multinomial noise ~1/sqrt(N_ch) reflects its true rarity
# (rare e+-/e- channels are ~50-200x below mu-). Old code used a flat 3e7 for all.
nd_counts = rc.compute_nd_sample_counts(s) if use_ndstats else None
rng = np.random.default_rng(noise_seed)
Pd = {}
for ch in rc.CHANNELS:
    base = ch.rsplit('_', 1)[0]; kg = grid[base]
    P0 = rc.make_marginal_rate(dsig[base], phi_nom[ch] * (1 + Fnd_data[ch] @ bd_t), kg.denu, kg.dV2)
    if nd_counts is not None:
        P0 = syst.noise_hist(P0, nd_counts[ch], rng)
    Pd[ch] = nd_smear(ch, P0)

# ---- FD samples: FIXED data flux b_data + FD-stats seed FDSEED. ONE set shared by
#      all shots (data is fixed), keyed by the data realization, not the shot. ----
SD = f"FD_samples_ffs_data{DATA_SHOT}"
if not os.path.exists(os.path.join(SD, "samples_FD_mu_fwd.npy")):
    fine = np.linspace(0.02, rc.ENUMAX, 4000)
    # Two halves of ONE correlated draw: the ND flux moves with the ND block,
    # the FD flux with the FD block. b_data is a single nuisance vector of
    # length NMODE spanning both.
    dfine_nd = {ch: (basis_on({ch: fine}, det="ND")[ch] @ b_data) for ch in rc.CHANNELS}
    dfine_fd = {ch: (basis_on({ch: fine}, det="FD")[ch] @ b_data) for ch in rc.CHANNELS}

    class Perturbed:
        def __init__(self, base, mode, shifts):
            self._base = base; self._shifts = shifts
            self._d = {pid.numu: f"mu_{mode}", pid.numubar: f"mubar_{mode}", pid.nue: f"e_{mode}", pid.nuebar: f"ebar_{mode}"}
        def __getattr__(self, k): return getattr(object.__getattribute__(self, '_base'), k)
        def __call__(self, flavor, energy):
            base = self._base(flavor, energy); ch = self._d.get(flavor)
            if ch is None: return base
            E = energy.detach().cpu().numpy() if isinstance(energy, torch.Tensor) else np.asarray(energy)
            return base * (1 + np.interp(E, fine, self._shifts[ch]))
    ndf = Perturbed(s.nd_flux_fwd, 'fwd', dfine_nd)
    ndb = Perturbed(s.nd_flux_bwd, 'bwd', dfine_nd)
    # The FD samples are drawn from the perturbed UNOSCILLATED FD flux.
    fdf = Perturbed(s.fd_unosc_fwd, 'fwd', dfine_fd)
    fdb = Perturbed(s.fd_unosc_bwd, 'bwd', dfine_fd)
    s_data = replace(s, nd_flux_fwd=ndf, nd_flux_bwd=ndb,
                     fd_unosc_fwd=fdf, fd_unosc_bwd=fdb,
                     fd_flux_fwd=FluxFarDetector(nd_flux=ndf, unosc_flux=fdf,
                                                 osc_params=dict(rc.OSC_PARAMS)),
                     fd_flux_bwd=FluxFarDetector(nd_flux=ndb, unosc_flux=fdb,
                                                 osc_params=dict(rc.OSC_PARAMS)))
    gfd.generate_fd_samples(s_data, seed=FDSEED, sample_dir=SD)
if FIT == 'gen':
    print(f"FFGEN shot={SHOT} -> {SD} done", flush=True); sys.exit(0)

# REUSE_NET is valid only if the saved net was trained on the SAME ND data, i.e.
# the same flux draw (b_data + b_model). A change that alters the training data --
# e.g. the flux-free nom (b_data 424242->0) -- makes the net + its fitted a* stale,
# so it must be RETRAINED, not reused (else a* perturbs the model flux against
# now-nominal data and biases the fit).
_pt = f"{OUTDIR}/ndmodel_ff_{_tag}.pt"
_ck = None; _reuse = False
if REUSE_NET and FIT in ('ml', 'fix') and os.path.exists(_pt):
    _ck = torch.load(_pt, weights_only=False)
    _reuse = np.allclose(_ck['b_data'], b_data) and np.allclose(_ck['b_model'], b_model)
    if not _reuse:
        print(f"[REUSE_NET] {_tag}.pt trained on different flux (b_data/b_model) -> RETRAIN", flush=True)

# ================= a_i inference =================
if FIT == 'truthfix':
    # laundered TRUTH control: true xsec, flux nuisances FROZEN at a=0
    # (the truth analog of fit=fix) — the model keeps its drawn flux
    # Phitilde with no freedom to reconcile it to the data.
    a_star = np.zeros(NMODE)
    mc = ModelCalc(rc.SFModel(n_sf=5).to(dev), enumax=rc.ENUMAX, unique=False, anti=False, force_ds_pos=True, device=dev)
    mcb = ModelCalc(rc.SFModel(n_sf=5).to(dev), enumax=rc.ENUMAX, unique=False, anti=True, force_ds_pos=True, device=dev)
elif FIT == 'truth':
    # Truth floor uses the TRUE flux nuisance directly -- NOT an ND re-fit. The old
    # path solved 68 free flux modes from the ND marginals with only a 1e-9 ridge; with
    # ND shot noise (total) that is ill-posed and overfits (total_truth |b+a*|=41.7, 5x
    # the sqrt(68) prior scale; dcp shot-to-shot 17x its components). The learned side
    # escapes via the implicit regularization of finite-step training; the truth side
    # has no such thing. Since the data's true flux IS b_data, set b_model+a = b_data
    # exactly (a = b_data - b_model): the truth FD fit uses the true flux x xsec model,
    # FD-only, no ND problem to mis-define. The residual FD flux uncertainty is then
    # purely eps (FLUXRESID, one ND->FD correlation universe per shot) -- the genuine
    # irreducible truth-side flux systematic. (b_data=0 for nd/smear/nom -> a=0.)
    if has_flux_nuis and _nf is not None:
        # b_data is the 136-mode joint draw; its model-space value is the ND-block
        # projection into the 68 u-modes -- the SAME map that builds b_model. Then
        # b_model+a = u_true = the true ND flux in u-space (not an ND re-fit).
        u_true = np.asarray(_nf.u_from_bin_shift(_nf.nd_block_of(b_data, fm._L)))
        a_star = u_true - b_model
    else:
        a_star = np.zeros(NMODE)
    mc = ModelCalc(rc.SFModel(n_sf=5).to(dev), enumax=rc.ENUMAX, unique=False, anti=False, force_ds_pos=True, device=dev)
    mcb = ModelCalc(rc.SFModel(n_sf=5).to(dev), enumax=rc.ENUMAX, unique=False, anti=True, force_ds_pos=True, device=dev)
elif _reuse:
    # reuse the trained ND network + a* (verified same-flux training; only FD differs)
    a_star = _ck['a_star']
    best_sd = (_ck['model'], _ck['model_bar'])
    mf = rc.SFModel(n_sf=5).to(dev); mf.load_state_dict(best_sd[0])
    mfb = rc.SFModel(n_sf=5).to(dev); mfb.load_state_dict(best_sd[1])
    mc = ModelCalc(mf, enumax=rc.ENUMAX, unique=False, anti=False, force_ds_pos=True, device=dev)
    mcb = ModelCalc(mfb, enumax=rc.ENUMAX, unique=False, anti=True, force_ds_pos=True, device=dev)
    print(f"[REUSE_NET] loaded ndmodel_ff_{_tag}.pt, skipping ND training", flush=True)
else:  # ml (a_i gradient-trained) or fix (a_i frozen at 0 = laundered control)
    FF = (FIT == 'ml') and has_flux_nuis     # no flux nuisances -> don't fit a
    torch.manual_seed(TRAIN_SEED)
    model = rc.SFModel(n_sf=5).to(dev); mbar = rc.SFModel(n_sf=5).to(dev)
    for m in (model, mbar):
        with torch.no_grad(): m._log_norm.zero_()
        m._log_norm.requires_grad_(False)
    a = torch.zeros(NMODE, device=dev, dtype=DT, requires_grad=FF)
    net_params = list(model.net.parameters()) + list(mbar.net.parameters())
    opt = (torch.optim.Adam([{'params': net_params}, {'params': [a], 'lr': LR_A}], lr=1e-3)
           if FF else torch.optim.Adam(net_params, lr=1e-3))
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2000, min_lr=1e-5, threshold=1e-2)
    flav = {'mu': pid.mu, 'e': pid.e, 'mubar': pid.mubar, 'ebar': pid.ebar}
    nets = {'mu': model, 'e': model, 'mubar': mbar, 'ebar': mbar}
    gr = {'mu': s.kin_grid_mu, 'e': s.kin_grid_e, 'mubar': s.kin_grid_mu, 'ebar': s.kin_grid_e}
    calcs = {bb: ModelCalc(nets[bb], enumax=rc.ENUMAX, unique=False, anti=(bb in ('mubar', 'ebar')), device=dev) for bb in gr}
    with torch.no_grad():
        for bb, c in calcs.items():
            kg = gr[bb]; c(kg.grid_enu, kg.grid1, kg.grid2, flavor=flav[bb], kin_mapping=km)
    dEnu = {'mu': s.kin_grid_mu.denu.item(), 'e': s.kin_grid_e.denu.item()}
    dv1 = {'mu': s.kin_grid_mu.dvar1, 'e': s.kin_grid_e.dvar1}
    def chrow(bb, mode, G):
        ch = f"{bb}_{mode}"; g = 'mu' if bb in ('mu', 'mubar') else 'e'
        M = G[bb] @ (phitilde[ch] * (1 + Fnd[ch] @ a)) * dEnu[g]
        if use_fdsmear:
            M = syst.apply_locus_smear(kern[ch], M)   # radial locus smear (mass-conserving)
        return M.flatten(1)
    BLK = [('mu', 'fwd'), ('mubar', 'fwd'), ('mu', 'bwd'), ('mubar', 'bwd'), ('etot', 'fwd'), ('etot', 'bwd')]
    def build():
        G = {bb: _sf_basis(nets[bb], calcs[bb]) for bb in calcs}
        blocks = []; tg = []
        for base, mode in BLK:
            if base == 'etot':
                e = chrow('e', mode, G); eb = chrow('ebar', mode, G)
                blocks.append(torch.cat([e.T, eb.T], 1)); tg.append((Pd[f'e_{mode}'] + Pd[f'ebar_{mode}']).flatten())
            else:
                r = chrow(base, mode, G); blk = torch.zeros(r.shape[1], 10, device=dev, dtype=r.dtype)
                col = 0 if base == 'mu' else 5; blk[:, col:col + 5] = r.T
                blocks.append(blk); tg.append(Pd[f'{base}_{mode}'].flatten())
        return blocks, tg
    eye = torch.eye(10, device=dev)
    best = np.inf; best_sd = (_fold_c(model, torch.ones(5)), _fold_c(mbar, torch.ones(5))); best_a = a.detach().clone()
    params = net_params + ([a] if FF else [])
    with torch.enable_grad():
        for step in range(NSTEP):
            opt.zero_grad(); blocks, tg = build()
            with torch.no_grad():
                H = sum(blocks[i].T @ blocks[i] for i in range(6)); rhs = sum(blocks[i].T @ tg[i] for i in range(6))
                c = torch.linalg.solve(H + 1e-12 * H.diagonal().mean() * eye, rhs)
            loss = sum(((blocks[i] @ c - tg[i]) ** 2).sum() for i in range(6)) * LOSS_NORM
            # flux prior: the total model flux nuisance (b_model+a) ~ N(0,I) (the f_i are
            # covariance eigenmodes). Breaks the sigma<->flux degeneracy that otherwise lets
            # a absorb cross-section shape (learned sigma biased ~12% high). LAMBDA_A=0 -> off.
            if LAMBDA_A > 0:
                loss = loss + LAMBDA_A * ((bm_t + a) ** 2).sum()
            if loss.item() < best:
                best = loss.item(); best_sd = (_fold_c(model, c[:5]), _fold_c(mbar, c[5:])); best_a = a.detach().clone()
            loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sch.step(loss.item())
            if _PROG and (step % _PROG_EVERY == 0 or step == NSTEP - 1):
                print(f"[train sh{SHOT} {EFFECTS} {FIT} ts{TRAIN_SEED}] step {step+1}/{NSTEP} "
                      f"loss {loss.item():.4e} best {best:.4e} lr {opt.param_groups[0]['lr']:.2e}", flush=True)
    a_star = rc.grab(best_a)
    mf = rc.SFModel(n_sf=5).to(dev); mf.load_state_dict(best_sd[0])
    mfb = rc.SFModel(n_sf=5).to(dev); mfb.load_state_dict(best_sd[1])
    mc = ModelCalc(mf, enumax=rc.ENUMAX, unique=False, anti=False, force_ds_pos=True, device=dev)
    mcb = ModelCalc(mfb, enumax=rc.ENUMAX, unique=False, anti=True, force_ds_pos=True, device=dev)

# ---- persist the trained ND network NOW, before the FD analysis ----
# Saving here rather than at the end means a crash, a kill or an interrupted FD
# fit never costs the training run: re-run with REUSE_NET=1 and the FD side
# redoes itself in seconds. Campaign-neutral (same file, same contents), but it
# is what makes iterating on the FD analysis cheap.
os.makedirs(OUTDIR, exist_ok=True)
_meta = dict(shot=SHOT, eff=EFFECTS, fit=FIT, fdseed=FDSEED, data_shot=DATA_SHOT,
             train_seed=TRAIN_SEED, use_flux=use_flux, use_ndstats=use_ndstats, prop=PROP,
             use_fdsmear=use_fdsmear, smear_w=SMEAR_W, smear_ngrid=SMEAR_NGRID,
             n_boots=NB, starts=4, likelihood=os.environ.get("LIKELIHOOD", "extended"),
             flux_seed=flux_seed, noise_seed=noise_seed,
             smear_seed=smear_seed, resid_seed=resid_seed, fluxresid=FLUXRESID,
             nmode=NMODE, params="s23sq,d31,dcp",
             nd_counts=(nd_counts if use_ndstats else None))
if FIT in ('ml', 'fix'):
    torch.save({'model': best_sd[0], 'model_bar': best_sd[1], 'n_sf': 5,
                'a_star': a_star, 'b_model': b_model, 'b_data': b_data, 'meta': _meta},
               f"{OUTDIR}/ndmodel_ff_{_tag}.pt")
    print(f"saved {OUTDIR}/ndmodel_ff_{_tag}.pt (pre-FD)", flush=True)

# Diagnostic: learned-vs-true sigma ratio on the ND grid (skip FD fit, exit).
if os.environ.get("CHECK_MARGINAL") == "1":
    _cm = {'mu': mc, 'e': mc, 'mubar': mcb, 'ebar': mcb}
    _fl = {'mu': pid.mu, 'e': pid.e, 'mubar': pid.mubar, 'ebar': pid.ebar}
    for _b in ('mu', 'mubar', 'e', 'ebar'):
        _kg = grid[_b]
        _dm = np.nan_to_num(rc.grab(_cm[_b](_kg.grid_enu, _kg.grid1, _kg.grid2, flavor=_fl[_b], kin_mapping=km)))
        _dt = np.nan_to_num(rc.grab(dsig[_b]))
        print(f"CHECKMARG eff={EFFECTS} ch={_b}: sigma_model/sigma_true = {_dm.sum()/_dt.sum():.4f}", flush=True)
    if a_star.shape == b_data.shape:
        _al = np.dot(a_star, b_data - b_model) / (np.linalg.norm(a_star) * np.linalg.norm(b_data - b_model) + 1e-12)
    else:
        _al = float("nan")      # PROP=regress: a* (68 u-coords) and b_data (136 modes) differ
    print(f"CHECKMARG eff={EFFECTS} |a*|={np.linalg.norm(a_star):.3f} align={_al:.3f}", flush=True)
    sys.exit(0)

# ================= FD oscillation fit with frozen a* =================
# must match load_fd_samples' FD grid (the analysis window), or the flux
# shifts land on different energies than the likelihood integrates over
allE = np.linspace(rc.FD_EWINDOW[0], rc.FD_EWINDOW[1], 512)
Ffd = model_basis({ch: allE for ch in rc.CHANNELS}, det="FD")   # FAR response
if PROP == "regress":
    # The model's FD flux is the near/far REGRESSION of its total fitted ND
    # deviation: one additive shift M_fd B chol(C_nd) (b_model + a*), not a
    # product of an independent FD draw with a correction. Leaving the
    # ND-blind coordinates of (b_model + a*) at their prior mean -- which
    # zero gradient does by itself -- makes this exactly the conditional
    # mean E[delta_fd | what the ND measured].
    sfd = {ch: (Ffd[ch] @ (b_model + a_star)) for ch in rc.CHANNELS}
    # ... and the conditional mean is not the whole story: no ND measurement
    # determines eps ~ N(0, C_fd|nd), the irreducible near/far residual
    # (median 0.94% per bin, up to 2.86%). A real analysis propagates it by
    # carrying the flux nuisances into the FD fit; here we MARGINALIZE it,
    # drawing one universe per bootstrap replicate. That is the same pooling
    # the paper already prescribes (N_syst x N_boot), it needs only this one
    # dataset, and the resulting CI widening IS the flux systematic -- no
    # shot-to-shot ensemble required. FLUXRESID=0 switches it off (then the
    # FD flux is the bare conditional mean and the flux systematic is
    # understated by construction).
    if FLUXRESID != "off" and has_flux_nuis:
        _Rb = _nf.fd_residual_basis({_base[ch]: allE for ch in rc.CHANNELS})
        Rfd = {ch: _Rb[_base[ch]] for ch in rc.CHANNELS}     # keys are BASE names
        _k = next(iter(Rfd.values())).shape[1]
        _rng = np.random.default_rng(resid_seed)
        if FLUXRESID == "shot":
            # ONE universe for this shot: a fixed offset of the model's FD
            # flux, drawn from C_fd|nd. Stays 1-D, so the scipy reference path
            # still works and the FD fit is unchanged in cost.
            _eps = _rng.standard_normal(_k)
            sfd = {ch: sfd[ch] + Rfd[ch] @ _eps for ch in rc.CHANNELS}
            _rms = np.sqrt(np.mean([(Rfd[ch] @ _eps) ** 2 for ch in rc.CHANNELS]))
        else:
            # one universe per bootstrap; (NB, n_E) broadcasts for free
            _eps = _rng.standard_normal((NB, _k))
            sfd = {ch: sfd[ch][None, :] + _eps @ Rfd[ch].T for ch in rc.CHANNELS}
            _rms = np.sqrt(np.mean([(_eps @ Rfd[ch].T) ** 2 for ch in rc.CHANNELS]))
        print(f"[fluxresid={FLUXRESID}] eps offset: {_k} modes, "
              f"rms {100*_rms:.3f}% per bin", flush=True)
else:
    # FD model flux = Phitilde_fd * (1 + a*f) = phi_nom * (1+b_model*f) * (1+a*f)
    sfd = {ch: ((1 + Ffd[ch] @ b_model) * (1 + Ffd[ch] @ a_star) - 1) for ch in rc.CHANNELS}
fd = rc.load_fd_samples(s, mc, mcb, sample_dir=SD, flux_shifts=sfd)
# FD detector smearing: replace the per-event dsigma_samp with the E_lep-smeared
# version at offset events (I is smearing-invariant, left as-is). One-time precompute.
if use_fdsmear:
    import time as _t; _t0 = _t.time()
    calc, calc_bar = (s.truth, s.truth_bar) if FIT in ('truth', 'truthfix') else (mc, mcb)
    smeared = syst.build_smeared_dsigma(SD, calc, calc_bar, fd.all_Enu_mu, fd.all_Enu_e, km,
                                        SMEAR_W, SMEAR_NGRID, smear_seed, dev)
    if FIT in ('truth', 'truthfix'):
        fd.dsigma_samp = smeared
    else:
        fd.dsigma_samp_model = smeared
    print(f"[FD smear ngrid={SMEAR_NGRID} w={SMEAR_W}] precompute {_t.time()-_t0:.1f}s", flush=True)
np.random.seed(0)
# Rate-aware FD likelihood (2026-10-08): the grand-total normalizer keeps
# the inter-sample rate ratios, which carry real dcp information. Set
# LIKELIHOOD=shape to reproduce the published shape-only results.
ft.LIKELIHOOD = LIKELIHOOD_ENV
# "extended" uses the ABSOLUTE event counts, so it needs the exposure that
# converts arbitrary-unit FD rates into counts (the POT x mass of the
# simulated experiment). Its normalization uncertainty is already carried
# by the flux covariance, so it is NOT profiled here.
# NOT compute_exposure(s): the FD samples in SD were drawn from the DATA flux
# realization b_data and re-anchored to 6200 under THAT flux, so the nominal-flux
# exposure disagrees with them by -2.4% -- a coherent absolute-rate offset in every
# channel of every flux-carrying shot. load_exposure reads the value recorded when
# the samples were generated, so the two can never drift apart again.
if ft.LIKELIHOOD == "extended":
    ft.EXPOSURE = rc.load_exposure(SD)
# poisson=True is REQUIRED with "rate": fixed replicate sizes freeze the
# observed ratios and make the rate constraint fluctuation-free.
bs = {c: torch.tensor(precompute_bootstrap_tensor(
        fd.n_samps[c], NB, poisson=(ft.LIKELIHOOD in ("rate", "extended"))))
      for c in ("mutot_fwd", "etot_fwd", "mutot_bwd", "etot_bwd")}
ft.LOG_FLOOR = 1e-10
if FIT in ('truth', 'truthfix'):
    dsamp, I = fd.dsigma_samp, fd.I
else:
    dsamp, I = fd.dsigma_samp_model, fd.I_model
mn, loss, gmax, found = ft.fit_all_boots(fd, bs, dsamp, I, dev, NB, phi=fd.phi_ND_model, seed=0, starts=4)
mn = rc.grab(mn); m = np.median(mn, axis=0)
# persist the full per-bootstrap minima + certificates + nuisances + metadata
# (campaign-parity: minima = [s23^2, d31, dcp] per boot; a* = fitted flux nuisances)
_meta["likelihood"] = ft.LIKELIHOOD        # authoritative once the fit has run
_lsuf = "" if ft.LIKELIHOOD == "extended" else f"_{ft.LIKELIHOOD}"  # keep extended canonical; shape/rate get a suffix so they don't clobber it
np.savez(f"{OUTDIR}/minima_ff_{_tag}{_lsuf}.npz",
         minima=mn, loss=rc.grab(loss), gmax=rc.grab(gmax), found=rc.grab(found),
         a_star=a_star, b_model=b_model, b_data=b_data, meta=_meta)
# the ND network was already written before the FD fit; rewrite it only to pick
# up the now-final meta (same weights, same a*).
if FIT in ('ml', 'fix'):
    torch.save({'model': best_sd[0], 'model_bar': best_sd[1], 'n_sf': 5,
                'a_star': a_star, 'b_model': b_model, 'b_data': b_data, 'meta': _meta},
               f"{OUTDIR}/ndmodel_ff_{_tag}.pt")
print(f"FFSHOT shot={SHOT} eff={EFFECTS} fit={FIT} fdseed={FDSEED} "
      f"[flux={int(use_flux)} nd={int(use_ndstats)} fdsmear={int(use_fdsmear)} ts={TRAIN_SEED}]: "
      f"d31 {m[1]:.5f} std {mn[:,1].std():.5f} | s23^2 {m[0]:.4f} dcp {m[2]:.3f} | "
      f"prop={PROP} |a*|={np.linalg.norm(a_star):.3f} |b_model|={np.linalg.norm(b_model):.3f} "
      f"|b_data|={np.linalg.norm(b_data):.3f} |a*-(b_data-b_model)|="
      # only comparable when model and data share coordinates (PROP=mode)
      f"{np.linalg.norm(a_star-(b_data-b_model)) if a_star.shape == b_data.shape else float('nan'):.3f}", flush=True)
