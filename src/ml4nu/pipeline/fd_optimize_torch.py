"""GPU-native, batched far-detector oscillation fit.

The scipy path (optimize_far_detector.py) runs 5000 independent L-BFGS-B fits,
each a CPU-driven loop of tiny tensor ops -- GPU-hostile (host<->device syncs).

Here the whole ensemble is ONE differentiable forward pass: parameters are a
``(N_boots, 3)`` tensor ``[s23, d31, dcp]`` and the negative log-likelihood is
computed for all bootstraps simultaneously as batched matmuls, then minimized
with a torch optimizer via autograd. Everything stays on the GPU.

The batched likelihood is validated to match the scalar scipy ``compute_ll``.

Key enabler: ``Probability_Matter_LBL`` allocates its output as
``zeros((3,3)+E.shape)`` from pure elementwise ops, so passing energies shaped
``(B, N_E)`` and params shaped ``(B, 1)`` yields ``(3, 3, B, N_E)`` for free.
"""

import os

import numpy as np
import torch

from ml4nu.pipeline import rates_common as rc
from ml4nu.nufast import Probability_Matter_LBL

# fixed (non-fitted) oscillation params, SOURCED from rates_common.OSC_PARAMS so
# they cannot drift from the sample generation / scipy reference. (They did: this was
# hard-coded at NuFit 6.0 while the data moved to 6.1 -> P(numu->nue) ~2.4% off.)
_S12, _S13, _D21 = rc.OSC_PARAMS["s12"], rc.OSC_PARAMS["s13"], rc.OSC_PARAMS["d21"]
_BASELINE = 1300


def osc_prob_batched(E, s23, d31, dcp, device, dtype=torch.float64):
    """Batched matter oscillation probs.

    E: (N_E,) energies (negative => antineutrino). s23,d31,dcp: (B,) params.
    Returns (3, 3, B, N_E): probs[final, initial, boot, energy].
    """
    B = s23.shape[0]
    Eb = E.reshape(1, -1).expand(B, -1)                      # (B, N_E), view
    return Probability_Matter_LBL(
        _S12 ** 2, _S13 ** 2, (s23 ** 2).reshape(B, 1),
        dcp.reshape(B, 1), _D21, d31.reshape(B, 1),
        _BASELINE, Eb, device=device, dtype=dtype)           # (3,3,B,N_E)


def _safe_log(x):
    return torch.nan_to_num(x.log())


# Density floor for the per-sample likelihood (None = off, the notebook
# behavior). The notebook's _safe_log maps a NEGATIVE model density at an
# observed event to log -> nan -> 0: the sample is silently DROPPED,
# contributing zero penalty -- the model is rewarded for predicting negative
# probability density exactly where data exists, which carves the cliffs the
# certificates flag (and lets large-flux-draw shots reach absurd NLLs).
# With LOG_FLOOR = eps, each sample density is floored at eps * norm
# (scale-free), so a floored sample contributes the constant penalty log(eps)
# with zero gradient: negative predictions are expensive, the surface has no
# reward-cliff, and well-behaved fits (densities >> eps * norm) are unchanged.
LOG_FLOOR = None


# Which likelihood the FD fit maximizes.
#
#   "shape" (the pre-2026-10-08 behavior, == the published Eq.):
#       ln L = sum_{m,l} [ sum_i ln q^m_l(x_i) - n^m_l ln N^m_l ]
#     Every sample is divided by ITS OWN predicted total, so neither the
#     absolute rate nor the RATIOS between the four samples inform the fit.
#     That discards real delta_CP sensitivity: the e-like/mu-like rate
#     ratio is a primary appearance observable.
#
#   "rate" (correct; default since 2026-10-08):
#     Add a Poisson term per sample with one unknown overall normalization
#     alpha (mu^m_l = alpha * N^m_l), then profile alpha analytically:
#       alpha_hat = n_tot / sum_{m,l} N^m_l
#       ln L = sum_{m,l} sum_i ln q^m_l(x_i) - n_tot ln( sum_{m,l} N^m_l )
#     i.e. REPLACE each sample's own normalizer by the GRAND TOTAL. One
#     free scale (the genuinely unknown exposure/attenuation) is removed;
#     the three independent inter-sample ratios now constrain omega.
#
# NOTE on bootstrapping: with "rate", the per-sample counts must be allowed
# to fluctuate (Poisson) across bootstrap replicates. precompute_bootstrap_
# tensor draws exactly n^m_l events every replicate, which freezes the
# observed ratios and would make the rate constraint look infinitely sharp.
# See fit_all_boots(poisson_counts=...).
#   "extended" (correct; DEFAULT since 2026-10-09):
#     The full unbinned extended likelihood, with no free normalization:
#       L = prod_{m,l} exp(-N^m_l(w)) prod_i q^m_l(x_i; w)
#     equivalently  prod_{m,l} Pois(n^m_l | N^m_l(w)) * prod_i q/N^m_l.
#     Writing q = alpha r and N = alpha R for the arbitrary-unit rates the
#     code carries, and dropping the w-independent n_tot*ln(alpha):
#       ln L = sum_{m,l} [ sum_i ln r^m_l(x_i; w) - alpha R^m_l(w) ]
#     so the ABSOLUTE counts constrain omega, not just the ratios. alpha is
#     the exposure (see EXPOSURE below); it is an experimental constant, and
#     its normalization uncertainty is already carried by the flux
#     covariance, so profiling it here would double count.
LIKELIHOOD = "extended"

# Exposure alpha converting the arbitrary-unit rates in this module into
# event counts. REQUIRED for LIKELIHOOD == "extended"; set it from
# rates_common.load_exposure(sample_dir), i.e. the exposure RECORDED WHEN THE
# SAMPLES WERE GENERATED. compute_exposure(setup) is only equivalent when the
# samples came from the nominal flux; for a drawn data flux it is off by the
# flux normalization shift (-2.4% for the campaign's 424242 draw), which lands
# as a coherent absolute-rate offset in every channel. Unused by
# "rate"/"shape", which are invariant to it.
EXPOSURE = None


def _mix_parts(dsl, dslb, Il, Ilb, phil, philb, denu):
    """(samp, norm) for one charge-inclusive sample.

    samp: (B, N_samp) per-event rate density. norm: (B,) predicted total
    rate N^m_l(omega) for this sample.
    """
    norm = (Il * phil).sum(-1) * denu + (Ilb * philb).sum(-1) * denu     # (B,)
    samp = (phil @ dsl.T) * denu + (philb @ dslb.T) * denu               # (B, N_samp)
    return samp, norm


def _ll_unnormalized(samp, norm):
    """log(samp), with the optional density floor taken relative to `norm`.

    The extended likelihood does not divide by a normalizer, but the floor
    still needs a scale; `norm` supplies it so the floor stays scale-free.
    """
    if LOG_FLOOR is None:
        return _safe_log(samp)
    return samp.clamp_min(LOG_FLOOR * norm.clamp_min(1e-300)[:, None]).log()


def _ll_from(samp, norm):
    """log(samp) - log(norm), with the optional density floor."""
    if LOG_FLOOR is None:
        return _safe_log(samp) - _safe_log(norm)[:, None]
    n = norm.clamp_min(1e-300)
    s = samp.clamp_min(LOG_FLOOR * n[:, None])
    return s.log() - n.log()[:, None]


def _mix_ll(dsl, dslb, Il, Ilb, phil, philb, denu):
    """Per-sample-normalized log-likelihood (the "shape" form).

    Kept for the scipy cross-check, which mirrors rates_common.
    make_mix_log_likelihood; batched_nll routes through _mix_parts so it
    can share one global normalizer across samples.
    """
    return _ll_from(*_mix_parts(dsl, dslb, Il, Ilb, phil, philb, denu))


def batched_nll(params, T):
    """Negative log-likelihood for all bootstraps at once.

    params: (B, 3) = [s23, d31, dcp]. T: dict of device tensors (see build_tensors).
    Returns (B,) per-bootstrap NLL.
    """
    s23, d31, dcp = params[:, 0], params[:, 1], params[:, 2]
    E = T["E"]
    p_nu = osc_prob_batched(E, s23, d31, dcp, T["device"])     # (3,3,B,N_E)
    p_an = osc_prob_batched(-E, s23, d31, dcp, T["device"])
    phi = T["phi_ND"]

    def fFD(ch_mu, ch_e, p, lepton):
        if lepton == "mu":
            v = phi[ch_mu] * p[1][1] + phi[ch_e] * p[0][1]
        else:
            v = phi[ch_mu] * p[1][0] + phi[ch_e] * p[0][0]
        return torch.nan_to_num(v)   # matches make_phi_FD (nan->0 before matmul)

    f_mu_fwd = fFD("mu_fwd", "e_fwd", p_nu, "mu")
    f_e_fwd = fFD("mu_fwd", "e_fwd", p_nu, "e")
    f_mubar_fwd = fFD("mubar_fwd", "ebar_fwd", p_an, "mu")
    f_ebar_fwd = fFD("mubar_fwd", "ebar_fwd", p_an, "e")
    f_mu_bwd = fFD("mu_bwd", "e_bwd", p_nu, "mu")
    f_e_bwd = fFD("mu_bwd", "e_bwd", p_nu, "e")
    f_mubar_bwd = fFD("mubar_bwd", "ebar_bwd", p_an, "mu")
    f_ebar_bwd = fFD("mubar_bwd", "ebar_bwd", p_an, "e")

    ds, I, bs, dmu, de = T["dsamp"], T["I"], T["bs"], T["dEnu_mu"], T["dEnu_e"]
    p_mutot_fwd = _mix_parts(ds["mutot_mu_fwd"], ds["mutot_mubar_fwd"], I["mu"], I["mubar"], f_mu_fwd, f_mubar_fwd, dmu)
    p_mutot_bwd = _mix_parts(ds["mutot_mu_bwd"], ds["mutot_mubar_bwd"], I["mu"], I["mubar"], f_mu_bwd, f_mubar_bwd, dmu)
    p_etot_fwd = _mix_parts(ds["etot_e_fwd"], ds["etot_ebar_fwd"], I["e"], I["ebar"], f_e_fwd, f_ebar_fwd, de)
    p_etot_bwd = _mix_parts(ds["etot_e_bwd"], ds["etot_ebar_bwd"], I["e"], I["ebar"], f_e_bwd, f_ebar_bwd, de)

    rate_penalty = 0.0
    if LIKELIHOOD == "extended":
        # no normalizer at all; the predicted total enters as -alpha*R
        if EXPOSURE is None:
            raise ValueError(
                "LIKELIHOOD == 'extended' needs EXPOSURE (events per unit "
                "rate); set it from rates_common.load_exposure(sample_dir), NOT\n"
                "from compute_exposure(setup) unless the samples really came\n"
                "from the nominal flux")
        ll_mutot_fwd = _ll_unnormalized(*p_mutot_fwd)
        ll_mutot_bwd = _ll_unnormalized(*p_mutot_bwd)
        ll_etot_fwd = _ll_unnormalized(*p_etot_fwd)
        ll_etot_bwd = _ll_unnormalized(*p_etot_bwd)
        rate_penalty = EXPOSURE * (p_mutot_fwd[1] + p_mutot_bwd[1]
                                   + p_etot_fwd[1] + p_etot_bwd[1])
    elif LIKELIHOOD == "rate":
        # one normalizer for all samples => the inter-sample ratios inform
        # omega (alpha profiled out analytically; see LIKELIHOOD above)
        n_all = (p_mutot_fwd[1] + p_mutot_bwd[1]
                 + p_etot_fwd[1] + p_etot_bwd[1])
        ll_mutot_fwd = _ll_from(p_mutot_fwd[0], n_all)
        ll_mutot_bwd = _ll_from(p_mutot_bwd[0], n_all)
        ll_etot_fwd = _ll_from(p_etot_fwd[0], n_all)
        ll_etot_bwd = _ll_from(p_etot_bwd[0], n_all)
    elif LIKELIHOOD == "shape":
        ll_mutot_fwd = _ll_from(*p_mutot_fwd)
        ll_mutot_bwd = _ll_from(*p_mutot_bwd)
        ll_etot_fwd = _ll_from(*p_etot_fwd)
        ll_etot_bwd = _ll_from(*p_etot_bwd)
    else:
        raise ValueError("LIKELIHOOD must be 'extended', 'rate' or 'shape', "
                         f"got {LIKELIHOOD!r}")

    # weighted sums: len(bs)*(ll * bs).sum over samples, per bootstrap
    ll = (bs["mutot_fwd"].shape[1] * (ll_mutot_fwd * bs["mutot_fwd"]).sum(-1)
          + bs["etot_fwd"].shape[1] * (ll_etot_fwd * bs["etot_fwd"]).sum(-1)
          + bs["mutot_bwd"].shape[1] * (ll_mutot_bwd * bs["mutot_bwd"]).sum(-1)
          + bs["etot_bwd"].shape[1] * (ll_etot_bwd * bs["etot_bwd"]).sum(-1))
    return -(ll - rate_penalty)   # (B,)


def count_floored(fd, bs, dsamp, I, phi, minima, dev, floor):
    """Per-bootstrap count of ACTIVE events at/below the density floor.

    Evaluates the per-event densities at each bootstrap's fitted minimum
    (minima rows are [sin^2 t23, d31, dcp]) and counts events with
    samp <= floor * norm among events with nonzero bootstrap weight.
    This is the "how often does safe_log/the floor still fire" statistic.
    """
    B = len(minima)
    T = build_tensors(fd, bs, dev, dsamp, I, phi=phi)
    m = torch.as_tensor(minima, device=dev)
    params = torch.stack([m[:, 0].sqrt(), m[:, 1], m[:, 2]], -1)
    s23, d31, dcp = params[:, 0], params[:, 1], params[:, 2]
    with torch.no_grad():
        p_nu = osc_prob_batched(T["E"], s23, d31, dcp, dev)
        p_an = osc_prob_batched(-T["E"], s23, d31, dcp, dev)
        phi_T = T["phi_ND"]

        def fFD(ch_mu, ch_e, p, lepton):
            if lepton == "mu":
                v = phi_T[ch_mu] * p[1][1] + phi_T[ch_e] * p[0][1]
            else:
                v = phi_T[ch_mu] * p[1][0] + phi_T[ch_e] * p[0][0]
            return torch.nan_to_num(v)

        pairs = [
            ("mutot_fwd", fFD("mu_fwd", "e_fwd", p_nu, "mu"), fFD("mubar_fwd", "ebar_fwd", p_an, "mu"), "mutot_mu_fwd", "mutot_mubar_fwd", "mu", "mubar", "dEnu_mu"),
            ("mutot_bwd", fFD("mu_bwd", "e_bwd", p_nu, "mu"), fFD("mubar_bwd", "ebar_bwd", p_an, "mu"), "mutot_mu_bwd", "mutot_mubar_bwd", "mu", "mubar", "dEnu_mu"),
            ("etot_fwd", fFD("mu_fwd", "e_fwd", p_nu, "e"), fFD("mubar_fwd", "ebar_fwd", p_an, "e"), "etot_e_fwd", "etot_ebar_fwd", "e", "ebar", "dEnu_e"),
            ("etot_bwd", fFD("mu_bwd", "e_bwd", p_nu, "e"), fFD("mubar_bwd", "ebar_bwd", p_an, "e"), "etot_e_bwd", "etot_ebar_bwd", "e", "ebar", "dEnu_e"),
        ]
        n_flr = torch.zeros(B, device=dev, dtype=torch.long)
        thr = floor if floor is not None else 0.0
        for name, phil, philb, k1, k2, ka, kb, dk in pairs:
            denu = T[dk]
            norm = ((T["I"][ka] * phil).sum(-1) + (T["I"][kb] * philb).sum(-1)) * denu
            samp = ((phil @ T["dsamp"][k1].T) + (philb @ T["dsamp"][k2].T)) * denu
            active = T["bs"][name] > 0
            n_flr += ((samp <= thr * norm.clamp_min(1e-300)[:, None]) & active).sum(-1)
    return rc.grab(n_flr)


def build_tensors(fd, bs, device, dsamp, I, phi=None, dtype=torch.float64):
    """Move an FDData + bootstrap-weight dict onto ``device`` for batched eval.

    ``dsamp``/``I`` select the cross-section model: fd.dsigma_samp/fd.I (truth) or
    fd.dsigma_samp_model/fd.I_model (ML). ``phi`` selects the ND flux dict
    (fd.phi_ND for truth, fd.phi_ND_model for the ML path when a
    flux-uncertainty draw is in play; default fd.phi_ND).
    bs[ch]: (B, N_samp_ch) float weights.
    """
    if phi is None:
        phi = fd.phi_ND
    def t(x):
        return torch.as_tensor(rc.grab(x) if isinstance(x, torch.Tensor) else x,
                               device=device, dtype=dtype)
    return {
        "device": device,
        "E": t(fd.all_Enu_mu),                       # == all_Enu_e (asserted)
        "dEnu_mu": t(fd.dEnu_mu), "dEnu_e": t(fd.dEnu_e),
        "phi_ND": {k: t(v) for k, v in phi.items()},
        "I": {k: t(v) for k, v in I.items()},
        "dsamp": {k: t(v) for k, v in dsamp.items()},
        "bs": {k: t(v) for k, v in bs.items()},
    }


# ---------------------------------------------------------------------------
# Batched fit
# ---------------------------------------------------------------------------
# Unconstrained reparametrization: s23 = sigmoid(u) in (0,1); d31 = exp(w) > 0
# (log-space is natural for a mass-squared); dcp = z (free, reported mod 2pi).
def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _unconstrained_nll(x, T):
    """NLL as a function of the unconstrained (B, 3) coords [u, w, z]."""
    params = torch.stack([torch.sigmoid(x[:, 0]), torch.exp(x[:, 1]), x[:, 2]], -1)
    return batched_nll(params, T)


def _masked_sum(f):
    """Sum of per-boot losses with non-finite entries masked out.

    Summing raw per-boot NLLs lets a single trajectory with a non-finite loss
    poison the batch: the scalar sum goes nan, autograd then returns nan
    gradients for EVERY row, the shared LBFGS line search dies for the whole
    chunk, and every certificate in the chunk reads inf (observed as
    exactly-chunk-sized blocks of inf gmax at max_batch=8192). Masked boots
    contribute zero gradient to themselves and nothing to anyone else; they
    are flagged separately via their own non-finite loss value.
    """
    return torch.where(torch.isfinite(f), f, torch.zeros_like(f)).sum()


# Newton polish trust region. Steps are clamped per unconstrained coordinate,
# and candidates must keep d31 = exp(w) in a physically sane window and u out
# of sigmoid saturation. Without these an early ill-conditioned Newton step can
# jump to d31 ~ 1e11 where the rapidly-oscillating likelihood aliases to a
# finite (sometimes locally lower) NLL but the Hessian overflows -> nan, and
# the bootstrap freezes there.
_POLISH_STEP_MAX = 1.0
_POLISH_W_MIN, _POLISH_W_MAX = float(np.log(1e-5)), 0.0   # d31 in (1e-5, 1) eV^2
_POLISH_U_MAX = 40.0                                       # sigmoid saturates


def _newton_polish(x, T, steps=25, gtol=1e-6):
    """Per-bootstrap damped Newton polish in unconstrained coords.

    LBFGS drives the whole batch with one quasi-Newton history and one shared
    line-search step on the SUMMED loss, so individual bootstraps can stop
    slightly off their own stationary point (and a single pathological
    bootstrap can stall the shared line search for everyone). The objective is
    separable, so exact per-bootstrap 3x3 Hessians come from three
    double-backward passes, and each bootstrap gets its own Levenberg damping
    and its own convergence test.

    Returns (x, nll, gmax): polished coords (B, 3), per-boot NLL (B,), and the
    per-boot max |grad| certificate (B,) evaluated at the returned x
    (inf if the gradient/Hessian is non-finite there).
    """
    x = x.detach().clone()
    B = x.shape[0]
    lam = torch.full((B,), 1e-3, device=x.device, dtype=x.dtype)
    eye = torch.eye(3, device=x.device, dtype=x.dtype)
    with torch.no_grad():
        nll = _unconstrained_nll(x, T)

    def grad_hess(xx):
        xg = xx.detach().requires_grad_(True)
        f = _unconstrained_nll(xg, T)
        fin = torch.isfinite(f).detach()
        (g,) = torch.autograd.grad(_masked_sum(f), xg, create_graph=True)
        # separable objective: d(sum_i g[i, k])/dx has per-boot Hessian rows
        H = torch.stack([torch.autograd.grad(
                torch.where(torch.isfinite(g[:, k]), g[:, k],
                            torch.zeros_like(g[:, k])).sum(), xg,
                retain_graph=(k < 2))[0]
                         for k in range(3)], 1)              # (B, 3, 3)
        return g.detach(), H.detach(), fin

    def certificate(g, fin):
        gmax = g.abs().amax(-1)
        gmax = torch.where(fin, gmax, torch.full_like(gmax, float("inf")))
        return torch.nan_to_num(gmax, nan=float("inf"), posinf=float("inf"))

    g, H, fin = grad_hess(x)
    for _ in range(steps):
        if (certificate(g, fin) <= gtol).all():
            break
        # zero out non-finite rows so solve stays finite; those boots take no
        # step and stay flagged by the certificate
        ok = (fin & torch.isfinite(g).all(-1)
              & torch.isfinite(H).reshape(B, -1).all(-1))
        g_s = torch.where(ok[:, None], g, torch.zeros_like(g))
        H_s = torch.where(ok[:, None, None], H, eye.expand(B, -1, -1))
        step = torch.linalg.solve(H_s + lam[:, None, None] * eye,
                                  -g_s[..., None]).squeeze(-1)
        step = step.clamp(-_POLISH_STEP_MAX, _POLISH_STEP_MAX)
        with torch.no_grad():
            cand = x + step
            in_dom = ((cand[:, 1] >= _POLISH_W_MIN) & (cand[:, 1] <= _POLISH_W_MAX)
                      & (cand[:, 0].abs() <= _POLISH_U_MAX))
            f_new = _unconstrained_nll(cand, T)
            accept = ok & in_dom & torch.isfinite(f_new) & (f_new < nll)
        x = torch.where(accept[:, None], cand, x)
        nll = torch.where(accept, f_new, nll)
        lam = torch.where(accept, lam * 0.3, lam * 10.0).clamp(1e-8, 1e8)
        g, H, fin = grad_hess(x)
    return x, nll, certificate(g, fin)


def _fit_octant(T, u0, w0, z0, max_iter, lr, polish_steps=25, gtol=1e-6):
    dev = T["device"]
    u = torch.tensor(u0, device=dev, dtype=torch.float64, requires_grad=True)
    w = torch.tensor(w0, device=dev, dtype=torch.float64, requires_grad=True)
    z = torch.tensor(z0, device=dev, dtype=torch.float64, requires_grad=True)

    # LBFGS shares one line search across the summed loss: a single bootstrap
    # in pathological territory can zero the shared step, freeze the sum, and
    # trip tolerance_change for the WHOLE batch while other bootstraps are
    # still mid-descent (seen on the rugged ML surface: the fit exited ~7x
    # earlier than the truth fit). So on early exit, restart with a fresh
    # optimizer (new curvature history + line search) and keep going until the
    # iteration budget is spent or the loss truly stops improving.
    #
    # Per-boot best tracking: the shared step length can also DRAG individual
    # bootstraps to worse points while the batch sum improves -- on the kinky
    # ML surface, at wide batches, whole boots ended on the "zero flux"
    # plateau (extreme params -> masked fluxes -> NLL = -0.0, zero gradient,
    # no way back). LBFGS is therefore treated as a proposal generator: each
    # bootstrap keeps the best finite point it has visited, checked at every
    # restart boundary, and the polish starts from those.
    with torch.no_grad():
        best_x = torch.stack([u, w, z], -1).detach().clone()
        best_f = _unconstrained_nll(best_x, T)

    iters_left = max_iter
    prev_loss = float("inf")
    crashes = 0
    # Env-gated heartbeat (FD_FIT_HEARTBEAT=1): one line per LBFGS segment
    # (<=250 iters) with iters used, loss, and ETA to the iteration budget.
    # Print-only; default off so cluster logs are unchanged.
    _hb = os.environ.get("FD_FIT_HEARTBEAT") == "1"
    if _hb:
        import time as _tt
        _t0 = _tt.time()
    while iters_left > 0:
        # segment the budget so best-tracking checkpoints at least every 250
        # iterations (bounds how much progress a dragged bootstrap can lose)
        seg = min(iters_left, 250)
        opt = torch.optim.LBFGS([u, w, z], lr=lr, max_iter=seg,
                                line_search_fn="strong_wolfe",
                                tolerance_grad=1e-10, tolerance_change=1e-14)

        def closure():
            opt.zero_grad()
            params = torch.stack([torch.sigmoid(u), torch.exp(w), z], -1)
            loss = _masked_sum(batched_nll(params, T))
            loss.backward()
            return loss

        try:
            loss = float(opt.step(closure).detach())
            crashes = 0
        except ZeroDivisionError:
            # torch's strong-Wolfe cubic interpolation divides by zero when
            # its bracket degenerates (x1 == x2) on a flat/masked stretch.
            # Restore the per-boot best params and retry with a fresh line
            # search; give up (polish from best) if it keeps recurring.
            crashes += 1
            with torch.no_grad():
                u.copy_(best_x[:, 0])
                w.copy_(best_x[:, 1])
                z.copy_(best_x[:, 2])
            iters_left -= seg
            if crashes >= 3:
                break
            continue
        iters_left -= max(opt.state[u].get("n_iter", seg), 1)
        with torch.no_grad():
            x_now = torch.stack([u, w, z], -1).detach()
            f_now = _unconstrained_nll(x_now, T)
            better = torch.isfinite(f_now) & (f_now < best_f)
            best_x = torch.where(better[:, None], x_now, best_x)
            best_f = torch.where(better, f_now, best_f)
        if _hb:
            _used = max_iter - iters_left
            _el = _tt.time() - _t0
            _eta = _el / max(_used, 1) * iters_left
            print(f"  [fd fit] iters {_used}/{max_iter} loss {loss:.6e} "
                  f"eta<= {_eta/60:.0f}m (budget; early exit on convergence)",
                  flush=True)
        if not np.isfinite(loss) or loss >= prev_loss - 1e-10:
            break
        prev_loss = loss
    x = best_x
    if polish_steps > 0:
        x, nll, gmax = _newton_polish(x, T, steps=polish_steps, gtol=gtol)
    else:
        with torch.no_grad():
            nll = _unconstrained_nll(x, T)
        gmax = torch.full_like(nll, float("nan"))
    with torch.no_grad():
        params = torch.stack([torch.sigmoid(x[:, 0]), torch.exp(x[:, 1]),
                              x[:, 2]], -1)
    return params.detach(), nll.detach(), gmax.detach()


def _chunk_phi(phi, lo, hi, K, B):
    """Slice a per-bootstrap flux to a chunk and widen it by the start count.

    `phi` entries are either (n_E,) -- one flux for every bootstrap, the
    usual case -- or (B, n_E), one universe per bootstrap, which is how
    the irreducible ND->FD flux residual is marginalized (see
    fwdfold_shot's FLUXRESID). The 2-D case has to be chunked and
    repeat_interleave'd exactly like the bootstrap tensors, or it
    broadcasts against the wrong axis of the n*K trajectory batch.
    """
    if phi is None:
        return None
    out = {}
    for ch, v in phi.items():
        a = v if torch.is_tensor(v) else np.asarray(v)
        if a.ndim == 1:
            out[ch] = v
        elif a.ndim == 2:
            if a.shape[0] != B:
                raise ValueError(
                    f"per-bootstrap flux '{ch}' has leading dim {a.shape[0]}, "
                    f"expected B={B}; it must carry one universe per bootstrap")
            out[ch] = (a[lo:hi].repeat_interleave(K, dim=0) if torch.is_tensor(a)
                       else np.repeat(a[lo:hi], K, axis=0))
        else:
            raise ValueError(f"flux '{ch}' must be 1-D or 2-D, got {a.shape}")
    return out


def fit_all_boots(fd, bs, dsamp, I, dev, B, *, phi=None, seed=0, starts=4,
                  max_iter=2000, lr=1.0, polish_steps=25, gtol=1e-6,
                  max_batch=8192):
    """Multistart batched fit: B bootstraps x `starts` random inits, both
    octant mirrors per start, keeping the per-bootstrap best across all of it.

    Single-start fits fail two ways at scale: ~0.5-2% of trajectories settle
    in an unphysical aliasing basin (d31 >> 1e-2 with a genuine local
    minimum), and on the kinky ML surface basin selection is a lottery.
    Best-of-K suppresses both geometrically. K multiplies the batch width, not
    the wall clock, until the GPU saturates (~1.4x time for 8x width on an
    A100); `max_batch` chunks the work so memory stays bounded at any B*K.

    max_iter is shared by each chunk's batch (LBFGS stops on the summed loss),
    so it must cover the SLOWEST trajectory: 200 left ~1e-3 NLL on the table
    at B=64; 2000 matched scipy L-BFGS-B to <1e-5 per bootstrap.

    Returns minima (B, 3) = [sin^2(theta23), Delta m^2_31, delta_CP], NLL (B,),
    the per-bootstrap max |grad| certificate gmax (B,) from the Newton polish
    (see _newton_polish; nan if polish_steps=0), and n_found (B,) = number of
    starts that reached (within 1e-6 of) the per-bootstrap best NLL.
    """
    K = max(1, starts)
    # deterministic inits for all B*K trajectories, independent of chunking
    np.random.seed(seed)
    s23_0 = np.random.uniform(0, 1, (B, K))
    d31_0 = np.random.uniform(0.0022, 0.0028, (B, K))
    dcp_0 = np.random.uniform(0, 2 * np.pi, (B, K))

    chunk = max(1, max_batch // K)
    out_min, out_nll, out_gmax, out_found = [], [], [], []
    for lo in range(0, B, chunk):
        hi = min(B, lo + chunk)
        n = hi - lo
        bs_c = {ch: v[lo:hi].repeat_interleave(K, dim=0) for ch, v in bs.items()}
        T = build_tensors(fd, bs_c, dev, dsamp, I,
                          phi=_chunk_phi(phi, lo, hi, K, B))

        s23 = s23_0[lo:hi].reshape(n * K)
        w0 = np.log(d31_0[lo:hi].reshape(n * K))
        dcp = dcp_0[lo:hi].reshape(n * K)
        with torch.enable_grad():
            pA, nA, gA = _fit_octant(T, _logit(s23), w0, dcp, max_iter,
                                     lr, polish_steps, gtol)
            pB, nB, gB = _fit_octant(T, _logit(1 - s23), w0, dcp, max_iter,
                                     lr, polish_steps, gtol)

        win_B = (nB < nA)
        params = torch.where(win_B[:, None], pB, pA).reshape(n, K, 3)
        nll = torch.where(win_B, nB, nA).reshape(n, K)
        gmax = torch.where(win_B, gB, gA).reshape(n, K)

        best = nll.argmin(dim=1)                                   # (n,)
        p = torch.gather(params, 1, best[:, None, None].expand(-1, 1, 3)).squeeze(1)
        nl = torch.gather(nll, 1, best[:, None]).squeeze(1)
        gm = torch.gather(gmax, 1, best[:, None]).squeeze(1)
        found = (nll <= nl[:, None] + 1e-6).sum(dim=1)

        out_min.append(torch.stack([p[:, 0] ** 2, p[:, 1],
                                    torch.remainder(p[:, 2], 2 * np.pi)], -1))
        out_nll.append(nl)
        out_gmax.append(gm)
        out_found.append(found)
        if B > chunk:
            print(f"  chunk {lo}-{hi}/{B} done")

    minima = torch.cat(out_min)
    nll = torch.cat(out_nll)
    gmax = torch.cat(out_gmax)
    n_found = torch.cat(out_found)
    if polish_steps > 0:
        n_cert = int((gmax <= gtol).sum())
        print(f"converged per max|grad|<={gtol:g}: {n_cert}/{B}"
              + ("" if n_cert == B else f"  (worst max|grad|={gmax.max():.2e})"))
    print(f"starts reaching best basin: median {int(n_found.median())}/{K}")
    return rc.grab(minima), rc.grab(nll), rc.grab(gmax), rc.grab(n_found)


def run_fd_fit(fd, dev, *, model_path="", seed=0, N_boots=4096, N_Enu_FD=512,
               N_v_FD=256, tag="", out_dir="results", max_iter=2000, lr=1.0,
               polish_steps=25, gtol=1e-6, starts=4, max_batch=8192,
               truth=True, truth_syst=False, model_fit=True, log_floor=None,
               extra_arrays=None, extra_meta=None, fname_prefix="minima_torch"):
    """Bootstrap + truth/model multistart fits + npz save for a built FDData.

    Three possible fits per bootstrap ensemble:
      * truth      (minima_c):  truth xsec + NOMINAL flux -- the
        perfect-knowledge control; shot-independent, so shared via a
        reference run (default off in shots).
      * truth_syst (minima_ct): truth xsec + the DRAWN flux
        (fd.phi_ND_model) -- perfect xsec, imperfect flux knowledge: the
        pure flux-systematic floor. Shot-DEPENDENT. Needs flux_shifts to be
        meaningful (equals the truth fit otherwise).
      * model      (minima_c_): ML xsec + the drawn flux -- the analysis.
        ``model_fit=False`` skips it (e.g. adding truth_syst to an
        already-fitted ensemble without recomputing the ML fit).

    ``extra_arrays``/``extra_meta`` let callers attach provenance.
    Returns the npz path.
    """
    from ml4nu.utils import precompute_bootstrap_tensor
    import os
    import time
    global LOG_FLOOR
    LOG_FLOOR = log_floor
    if log_floor is not None:
        print(f"density floor: samp >= {log_floor:g} * norm (penalty {np.log(log_floor):.1f}/sample)")
    t0 = time.time()
    np.random.seed(seed)
    # poisson=True is REQUIRED for any rate-sensitive likelihood (see the
    # LIKELIHOOD docstring): fixed replicate sizes freeze the observed
    # totals, so the rate constraint carries no statistical fluctuation
    # and the CIs come out spuriously tight. Only "shape" is immune.
    bs = {ch: torch.tensor(precompute_bootstrap_tensor(fd.n_samps[ch], N_boots,
                                                       poisson=(LIKELIHOOD != "shape")))
          for ch in ("mutot_fwd", "etot_fwd", "mutot_bwd", "etot_bwd")}

    fit_kw = dict(seed=seed, starts=starts, max_iter=max_iter, lr=lr,
                  polish_steps=polish_steps, gtol=gtol, max_batch=max_batch)
    arrays = {}
    if truth:
        minima_c, loss_c, gmax_c, found_c = fit_all_boots(
            fd, bs, fd.dsigma_samp, fd.I, dev, N_boots, phi=fd.phi_ND, **fit_kw)
        arrays.update(minima_c=minima_c, loss_c=loss_c,
                      gmax_c=gmax_c, found_c=found_c)
        print(f"truth fit: {time.time() - t0:.1f}s ({N_boots} boots x {starts} starts)")
    else:
        print("truth fit: skipped (shot-independent; see a truth reference run)")

    if truth_syst:
        ts = time.time()
        minima_ct, loss_ct, gmax_ct, found_ct = fit_all_boots(
            fd, bs, fd.dsigma_samp, fd.I, dev, N_boots,
            phi=fd.phi_ND_model, **fit_kw)
        arrays.update(minima_ct=minima_ct, loss_ct=loss_ct,
                      gmax_ct=gmax_ct, found_ct=found_ct)
        print(f"truth-syst fit (truth xsec, drawn flux): {time.time() - ts:.1f}s")

    if model_fit:
        t1 = time.time()
        minima_c_, loss_c_, gmax_c_, found_c_ = fit_all_boots(
            fd, bs, fd.dsigma_samp_model, fd.I_model, dev, N_boots,
            phi=fd.phi_ND_model, **fit_kw)
        print(f"model fit: {time.time() - t1:.1f}s ({N_boots} boots x {starts} starts)")

        # how often the density floor still fires at the fitted ML minima
        nflr_c_ = count_floored(fd, bs, fd.dsigma_samp_model, fd.I_model,
                                fd.phi_ND_model, minima_c_, dev, log_floor)
        print(f"floored events at ML minima: boots with any {int((nflr_c_ > 0).sum())}/{N_boots}, "
              f"median {int(np.median(nflr_c_))}, max {int(nflr_c_.max())}")
        arrays.update(minima_c_=minima_c_, loss_c_=loss_c_, gmax_c_=gmax_c_,
                      found_c_=found_c_, nflr_c_=nflr_c_)

    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(model_path))[0]
    suffix = ("_" + tag) if tag else ""
    fname = "%s_%s_seed%d_nboots%d_E%d_v%d%s.npz" % (
        fname_prefix, stem, seed, N_boots, N_Enu_FD, N_v_FD, suffix)
    path = os.path.join(out_dir, fname)
    meta = dict(model_path=model_path, seed=seed, N_boots=N_boots,
                N_Enu_FD=N_Enu_FD, N_v_FD=N_v_FD, tag=tag,
                starts=starts, max_batch=max_batch, truth_fit=truth,
                truth_syst_fit=truth_syst, model_fit=model_fit,
                log_floor=log_floor, optimizer="torch-lbfgs")
    meta.update(extra_meta or {})
    np.savez(path, **arrays, **(extra_arrays or {}), meta=meta)
    print("saved", path)
    return path


def optimize_far_detector_torch(model_path, *, seed=0, N_boots=4096, N_Enu_FD=512,
                                N_v_FD=256, tag="", out_dir="results",
                                sample_dir="FD_samples", max_iter=2000, lr=1.0,
                                polish_steps=25, gtol=1e-6, starts=4,
                                max_batch=8192, log_floor=1e-10,
                                force_ds_pos=True, device=None):
    import time
    t0 = time.time()
    setup = rc.build_setup()
    dev = torch.device(device) if device else rc.device
    _, _, mc, mcb = rc.load_model(model_path, setup, force_ds_pos=force_ds_pos)
    fd = rc.load_fd_samples(setup, mc, mcb, N_Enu_FD=N_Enu_FD, N_v_FD=N_v_FD,
                            sample_dir=sample_dir)
    print("n_samps:", fd.n_samps, "| device:", dev, "| starts:", starts)
    print(f"FD build: {time.time() - t0:.1f}s")
    return run_fd_fit(fd, dev, model_path=model_path, seed=seed, N_boots=N_boots,
                      N_Enu_FD=N_Enu_FD, N_v_FD=N_v_FD, tag=tag, out_dir=out_dir,
                      max_iter=max_iter, lr=lr, polish_steps=polish_steps,
                      gtol=gtol, starts=starts, max_batch=max_batch,
                      log_floor=log_floor if log_floor and log_floor > 0 else None)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="GPU-batched far-detector oscillation fit.")
    ap.add_argument("model_path")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boots", type=int, default=4096)
    ap.add_argument("--n-enu-fd", type=int, default=512)
    ap.add_argument("--n-v-fd", type=int, default=256)
    ap.add_argument("--max-iter", type=int, default=2000,
                    help="LBFGS iterations, shared by the whole batch; must cover the slowest bootstrap")
    ap.add_argument("--lr", type=float, default=1.0)
    ap.add_argument("--polish-steps", type=int, default=25,
                    help="per-bootstrap damped-Newton polish steps after LBFGS (0 to disable)")
    ap.add_argument("--gtol", type=float, default=1e-6,
                    help="per-bootstrap max|grad| convergence certificate threshold")
    ap.add_argument("--starts", type=int, default=4,
                    help="random multistart inits per bootstrap (each also fit from its "
                         "octant mirror); raise for ML-critical runs, the ML surface is heavily multimodal")
    ap.add_argument("--log-floor", type=float, default=1e-10,
                    help="relative density floor for the ML likelihood (0 = off)")
    ap.add_argument("--force-ds-pos", action=argparse.BooleanOptionalAction, default=True,
                    help="clamp model dsigma at 0 for inference (training never clamped)")
    ap.add_argument("--max-batch", type=int, default=8192,
                    help="max concurrent trajectories (boots x starts) per GPU chunk")
    ap.add_argument("--device", default=None, help="e.g. cuda / cpu (default: auto)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--sample-dir", default="FD_samples")
    args = ap.parse_args()
    optimize_far_detector_torch(args.model_path, seed=args.seed, N_boots=args.n_boots,
                                N_Enu_FD=args.n_enu_fd, N_v_FD=args.n_v_fd,
                                max_iter=args.max_iter, lr=args.lr, device=args.device,
                                polish_steps=args.polish_steps, gtol=args.gtol,
                                starts=args.starts, max_batch=args.max_batch,
                                log_floor=args.log_floor, force_ds_pos=args.force_ds_pos,
                                tag=args.tag, out_dir=args.out_dir, sample_dir=args.sample_dir)


if __name__ == "__main__":
    main()
