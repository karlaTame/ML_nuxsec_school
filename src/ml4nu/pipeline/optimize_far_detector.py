"""Far-detector bootstrap optimizer ("function 2").

Faithful, plotting-free port of notebook cells 54, 84-86. Loads a trained ND
model and pre-saved FD samples, then runs the bootstrap maximum-likelihood
oscillation fit, saving the per-bootstrap minima ``minima_c`` (truth cross
section) and ``minima_c_`` (ML cross section).

Usage:
    python optimize_far_detector.py models/ndmodel_nsteps3000_lr0.001_seed17.pt \
        --seed 0 --n-boots 5000 --n-enu-fd 512 --n-v-fd 256 --tag base

Each minima row is [sin^2(theta23), Delta m^2_31, delta_CP].

NOTE: This fixes a bug in the notebook (cell 86), where the ML path
(``minima_c_``) used the *truth* integrals with the *model* dsigma_samp. Here
the ML path consistently uses the *model* integrals (fd.I_model) with the model
dsigma_samp, while the truth path uses the truth integrals (fd.I).
"""

import argparse
import multiprocessing as mp
import os

import numpy as np
import scipy.optimize
import torch
from tqdm import tqdm

from ml4nu.pipeline import rates_common as rc
from ml4nu.utils import precompute_bootstrap_tensor, safe_make_dx, grab

# L-BFGS-B convergence tolerance for the per-bootstrap MLE. 1e-6 is plenty for a
# 3-parameter fit over physical ranges; the notebook's 1e-12 just burned
# function evaluations. Tune here.
OPT_TOL = 1e-6


def compute_ll_for_params_confused(s23, d31, dcp_mod, fd, dsamp, I,
                                   bs_mutot_fwd, bs_etot_fwd, bs_mutot_bwd, bs_etot_bwd,
                                   phi=None, log_floor=None,
                                   likelihood="extended", exposure=None):
    """Combined-charge log-likelihood at one (s23, d31, dcp) point (cell 85).

    ``dsamp`` is the dsigma_samp dict and ``I`` the matching integrals dict --
    truth (fd.dsigma_samp / fd.I) or model (fd.dsigma_samp_model / fd.I_model).
    ``bs_*`` are single bootstrap rows. ``fd`` is the FDData container.
    """
    dcp = np.mod(dcp_mod, 2 * np.pi)
    # phi selects the ND flux dict: fd.phi_ND (truth) or fd.phi_ND_model
    # (ML path under a flux-uncertainty draw); default truth.
    phi = fd.phi_ND if phi is None else phi
    # This reference path evaluates ONE bootstrap at a time, so it cannot
    # consume the per-bootstrap flux universes used to marginalize the
    # irreducible ND->FD residual (fwdfold_shot's FLUXRESID). Fail loudly
    # rather than broadcast one universe across every replicate: run the
    # torch/scipy cross-check with FLUXRESID=0.
    for _ch, _v in phi.items():
        if np.ndim(_v) > 1:
            raise NotImplementedError(
                f"flux '{_ch}' is per-bootstrap (shape {np.shape(_v)}); the "
                "scipy reference path is single-bootstrap. Cross-check with "
                "FLUXRESID=0, or use fd_optimize_torch, which chunks it.")

    # Oscillation probabilities depend only on (s23, d31, dcp) and the sign of E,
    # so compute them ONCE for nu (+E) and once for nubar (-E) and reuse across
    # all 8 channels -- fwd/bwd and mu/e share the same p_osc; only the ND flux
    # differs. (all_Enu_mu == all_Enu_e, asserted in load_fd_samples.) This
    # replaces 8 make_phi_FD calls (each recomputing an osc prob) with 2.
    E = grab(fd.all_Enu_mu)
    p_nu = np.asarray(grab(rc.compute_osc_prob(E, s23=s23, d31=d31, dcp=dcp)))
    p_anti = np.asarray(grab(rc.compute_osc_prob(-E, s23=s23, d31=d31, dcp=dcp)))

    def fFD(ch_mu, ch_e, p, lepton):
        pm, pe = grab(phi[ch_mu]), grab(phi[ch_e])
        if lepton == "mu":
            v = pm * p[1][1] + pe * p[0][1]
        else:  # "e"
            v = pm * p[1][0] + pe * p[0][0]
        return torch.as_tensor(np.nan_to_num(v, nan=0.0))

    f_FD_mu_fwd = fFD("mu_fwd", "e_fwd", p_nu, "mu")
    f_FD_e_fwd = fFD("mu_fwd", "e_fwd", p_nu, "e")
    f_FD_mubar_fwd = fFD("mubar_fwd", "ebar_fwd", p_anti, "mu")
    f_FD_ebar_fwd = fFD("mubar_fwd", "ebar_fwd", p_anti, "e")

    f_FD_mu_bwd = fFD("mu_bwd", "e_bwd", p_nu, "mu")
    f_FD_e_bwd = fFD("mu_bwd", "e_bwd", p_nu, "e")
    f_FD_mubar_bwd = fFD("mubar_bwd", "ebar_bwd", p_anti, "mu")
    f_FD_ebar_bwd = fFD("mubar_bwd", "ebar_bwd", p_anti, "e")

    dEnu_mu = safe_make_dx(fd.all_Enu_mu)
    dEnu_e = safe_make_dx(fd.all_Enu_e)

    # Mirrors fd_optimize_torch.LIKELIHOOD; keep the two in sync or the
    # torch/scipy cross-check compares different estimators.
    #   "extended" (default): no normalizer; the predicted totals enter as
    #                         -exposure * sum_g N_g, so absolute counts count.
    #   "rate"              : one grand-total normalizer (profiled alpha).
    #   "shape"             : each sample by its own total (published form).
    if likelihood not in ("extended", "rate", "shape"):
        raise ValueError(f"likelihood must be extended/rate/shape, got {likelihood!r}")
    norms = (
        rc.mix_sample_norm(I["mu"], I["mubar"], f_FD_mu_fwd, f_FD_mubar_fwd, dEnu_mu),
        rc.mix_sample_norm(I["mu"], I["mubar"], f_FD_mu_bwd, f_FD_mubar_bwd, dEnu_mu),
        rc.mix_sample_norm(I["e"], I["ebar"], f_FD_e_fwd, f_FD_ebar_fwd, dEnu_e),
        rc.mix_sample_norm(I["e"], I["ebar"], f_FD_e_bwd, f_FD_ebar_bwd, dEnu_e))
    norm_total = sum(norms) if likelihood == "rate" else None
    unnormalized = (likelihood == "extended")
    if unnormalized and exposure is None:
        raise ValueError("likelihood='extended' needs `exposure` (events per "
                         "unit rate); use rates_common.load_exposure(sample_dir)")

    ll_mutot_fwd = rc.make_mix_log_likelihood(dsamp["mutot_mu_fwd"], dsamp["mutot_mubar_fwd"],
                                              I["mu"], I["mubar"], f_FD_mu_fwd, f_FD_mubar_fwd, dEnu_mu,
                                              log_floor=log_floor, norm_total=norm_total,
                                             unnormalized=unnormalized)
    ll_mutot_bwd = rc.make_mix_log_likelihood(dsamp["mutot_mu_bwd"], dsamp["mutot_mubar_bwd"],
                                              I["mu"], I["mubar"], f_FD_mu_bwd, f_FD_mubar_bwd, dEnu_mu,
                                              log_floor=log_floor, norm_total=norm_total,
                                             unnormalized=unnormalized)
    ll_etot_fwd = rc.make_mix_log_likelihood(dsamp["etot_e_fwd"], dsamp["etot_ebar_fwd"],
                                             I["e"], I["ebar"], f_FD_e_fwd, f_FD_ebar_fwd, dEnu_e,
                                             log_floor=log_floor, norm_total=norm_total,
                                             unnormalized=unnormalized)
    ll_etot_bwd = rc.make_mix_log_likelihood(dsamp["etot_e_bwd"], dsamp["etot_ebar_bwd"],
                                             I["e"], I["ebar"], f_FD_e_bwd, f_FD_ebar_bwd, dEnu_e,
                                             log_floor=log_floor, norm_total=norm_total,
                                             unnormalized=unnormalized)

    ll_tot = (
        len(bs_mutot_fwd) * (ll_mutot_fwd * bs_mutot_fwd).sum().item()
        + len(bs_etot_fwd) * (ll_etot_fwd * bs_etot_fwd).sum().item()
        + len(bs_mutot_bwd) * (ll_mutot_bwd * bs_mutot_bwd).sum().item()
        + len(bs_etot_bwd) * (ll_etot_bwd * bs_etot_bwd).sum().item()
    )
    if unnormalized:          # extended: subtract alpha * sum_g N_g
        ll_tot -= float(exposure) * float(sum(float(n) for n in norms))
    return ll_tot


def make_mle_for_one_boot(fd, dsamp, I, bs_mutot_fwd, bs_etot_fwd, bs_mutot_bwd, bs_etot_bwd,
                          init=None, phi=None, likelihood="extended", exposure=None):
    """One bootstrap MLE with octant degeneracy handling (cell 85).

    ``init`` = (s23_init, d31_init_x1000, dcp_init); if None, drawn from the
    global numpy RNG (legacy behavior). Passing it explicitly makes the fit
    deterministic and safe to run in parallel workers.
    """
    if init is None:
        s23_init = np.random.uniform(0, 1)
        d31_init = np.random.uniform(0.0022, 0.0028) * 1000
        dcp_init = np.random.uniform(0, 2 * np.pi)
    else:
        s23_init, d31_init, dcp_init = init

    def neg_ll(x):
        return -compute_ll_for_params_confused(x[0], x[1] / 1000, x[2], fd, dsamp, I,
                                               bs_mutot_fwd, bs_etot_fwd, bs_mutot_bwd, bs_etot_bwd,
                                               phi=phi, likelihood=likelihood,
                                               exposure=exposure)

    opts = {'maxiter': 10000, 'gtol': OPT_TOL}
    res = scipy.optimize.minimize(neg_ll, x0=(s23_init, d31_init, dcp_init),
                                  bounds=[(0, 1), (1e-6, None), (None, None)],
                                  method='L-BFGS-B', tol=OPT_TOL, options=opts)
    res_2 = scipy.optimize.minimize(neg_ll, x0=(1 - s23_init, d31_init, dcp_init),
                                    bounds=[(0, 1), (1e-6, None), (None, None)],
                                    method='L-BFGS-B', tol=OPT_TOL, options=opts)

    if res.fun > res_2.fun:
        return [res_2.x[0] ** 2, res_2.x[1] / 1000, np.mod(res_2.x[2], 2 * np.pi)], res_2.fun
    return [res.x[0] ** 2, res.x[1] / 1000, np.mod(res.x[2], 2 * np.pi)], res.fun


# Module-level worker state, populated in the parent before the worker pool is
# forked so children inherit the (large, read-only) FD arrays copy-on-write.
_WORKER = {}


def _worker_init():
    # Each worker is single-threaded: with N workers on an N-core box, per-op
    # multithreading would oversubscribe. Parallelism comes from the pool.
    torch.set_num_threads(1)


def _run_boot(task):
    """Run one bootstrap (truth + model fits). Returns (i, m, l, m_, l_)."""
    i, init_t, init_m = task
    fd, bs = _WORKER["fd"], _WORKER["bs"]
    rows = (bs["mutot_fwd"][i], bs["etot_fwd"][i], bs["mutot_bwd"][i], bs["etot_bwd"][i])
    lk, xp = _WORKER.get("likelihood", "extended"), _WORKER.get("exposure")
    m, l = make_mle_for_one_boot(fd, fd.dsigma_samp, fd.I, *rows, init=init_t,
                                 phi=fd.phi_ND, likelihood=lk, exposure=xp)
    m_, l_ = make_mle_for_one_boot(fd, fd.dsigma_samp_model, fd.I_model, *rows, init=init_m,
                                   phi=fd.phi_ND_model, likelihood=lk, exposure=xp)
    return i, m, l, m_, l_


def optimize_far_detector(model_path, *, seed=0, N_boots=5000, N_Enu_FD=512, N_v_FD=256,
                          tag="", out_dir="results", sample_dir="FD_samples",
                          loop_threads=4, jobs=1, force_ds_pos=True,
                          likelihood="extended"):
    # With jobs>1 we fork workers, which is unsafe once a CUDA context exists, so
    # build on CPU. (The loop is CPU-only anyway; this only affects the one-time
    # FD build.) jobs==1 keeps the GPU for the faster build.
    if jobs != 1:
        rc.device = torch.device("cpu")
    setup = rc.build_setup()
    model, model_bar, model_calc, model_calc_bar = rc.load_model(
        model_path, setup, force_ds_pos=force_ds_pos)

    fd = rc.load_fd_samples(setup, model_calc, model_calc_bar,
                            N_Enu_FD=N_Enu_FD, N_v_FD=N_v_FD, sample_dir=sample_dir)
    print("n_samps:", fd.n_samps)

    # bootstrap tensors (cell 54). CPU for the CPU optimizer loop.
    np.random.seed(seed)
    bs = {}
    for ch in ("mutot_fwd", "etot_fwd", "mutot_bwd", "etot_bwd"):
        # poisson=True is REQUIRED with the rate-aware likelihood: the
        # default freezes each sample size, so the inter-sample ratios do
        # not fluctuate and the rate constraint looks infinitely sharp.
        bs[ch] = torch.tensor(precompute_bootstrap_tensor(fd.n_samps[ch], N_boots,
                                                          poisson=True))

    # Deterministic per-bootstrap optimizer inits, drawn in the same order as the
    # serial loop (truth fit then model fit per bootstrap) so parallel and serial
    # runs give identical results for a given seed.
    np.random.seed(seed)
    tasks = []
    for i in range(N_boots):
        it = (np.random.uniform(0, 1), np.random.uniform(0.0022, 0.0028) * 1000, np.random.uniform(0, 2 * np.pi))
        im = (np.random.uniform(0, 1), np.random.uniform(0.0022, 0.0028) * 1000, np.random.uniform(0, 2 * np.pi))
        tasks.append((i, it, im))

    # Match the torch path (fd_optimize_torch.LIKELIHOOD): the extended form
    # keeps the absolute counts, so it needs the exposure that GENERATED these
    # samples -- read from sample_dir, never recomputed from `setup` (which
    # assumes the nominal flux; see rates_common.load_exposure).
    _WORKER["fd"], _WORKER["bs"] = fd, bs
    _WORKER["likelihood"] = likelihood
    _WORKER["exposure"] = (rc.load_exposure(sample_dir)
                           if likelihood == "extended" else None)

    # The per-bootstrap likelihood is many tiny per-sample torch ops (a log and an
    # (N_samp, N_Enu) x (N_Enu,) matmul). On a high-core box torch's default thread
    # count fans each across all cores and the dispatch overhead swamps the work
    # (~10x slower). Cap threads; parallelism across bootstraps comes from the pool.
    prev_threads = torch.get_num_threads()
    if jobs == 1:
        torch.set_num_threads(max(1, loop_threads))
        results = [_run_boot(t) for t in tqdm(tasks)]
        torch.set_num_threads(prev_threads)
    else:
        print("running %d bootstraps across %d workers" % (N_boots, jobs))
        ctx = mp.get_context("fork")
        with ctx.Pool(jobs, initializer=_worker_init) as pool:
            results = list(tqdm(pool.imap_unordered(_run_boot, tasks, chunksize=8),
                                total=N_boots))
        results.sort(key=lambda r: r[0])

    minima_c = np.stack([r[1] for r in results])
    loss_c = [r[2] for r in results]
    minima_c_ = np.stack([r[3] for r in results])
    loss_c_ = [r[4] for r in results]

    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(model_path))[0]
    suffix = ("_" + tag) if tag else ""
    fname = "minima_%s_seed%d_nboots%d_E%d_v%d%s.npz" % (stem, seed, N_boots, N_Enu_FD, N_v_FD, suffix)
    path = os.path.join(out_dir, fname)
    np.savez(path,
             minima_c=minima_c, minima_c_=minima_c_,
             loss_c=np.array(loss_c), loss_c_=np.array(loss_c_),
             meta=dict(model_path=model_path, seed=seed, N_boots=N_boots,
                       N_Enu_FD=N_Enu_FD, N_v_FD=N_v_FD, tag=tag))
    print("saved", path)
    return path


def main():
    ap = argparse.ArgumentParser(description="Far-detector bootstrap oscillation fit.")
    ap.add_argument("model_path", type=str, help="Trained ND model .pt file")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boots", type=int, default=5000)
    ap.add_argument("--n-enu-fd", type=int, default=512)
    ap.add_argument("--n-v-fd", type=int, default=256)
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--out-dir", type=str, default="results")
    ap.add_argument("--sample-dir", type=str, default="FD_samples")
    ap.add_argument("--threads", type=int, default=4,
                    help="torch threads for the serial (jobs=1) bootstrap loop; low is faster on many-core boxes")
    ap.add_argument("--jobs", type=int, default=1,
                    help="parallel worker processes over bootstraps (each single-threaded). "
                         ">1 builds the FD grid on CPU so forking is safe. Use os.cpu_count() on a big machine.")
    args = ap.parse_args()

    print("device:", rc.device)
    optimize_far_detector(args.model_path, seed=args.seed, N_boots=args.n_boots,
                          N_Enu_FD=args.n_enu_fd, N_v_FD=args.n_v_fd, tag=args.tag,
                          out_dir=args.out_dir, sample_dir=args.sample_dir,
                          loop_threads=args.threads, jobs=args.jobs)


if __name__ == "__main__":
    main()
