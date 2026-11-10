"""FD likelihood: shape-only vs rate-aware.

Pins the 2026-10-08 change. The published likelihood divides each of the
four samples by ITS OWN predicted total, so the relative rates between
samples carry no information -- yet the e-like/mu-like ratio is a primary
delta_CP observable. The rate-aware form profiles a single unknown
normalization alpha, which replaces the per-sample normalizers with one
grand total and restores the three inter-sample ratios.
"""
import numpy as np
import pytest
import torch

from ml4nu.pipeline import fd_optimize_torch as ft

torch.set_default_dtype(torch.float64)


def _fake_T(B=3, NE=6, NS=5, seed=0):
    """Minimal tensor bundle shaped like build_tensors' output."""
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.rand(*s, generator=g) + 0.5
    chans = ("mu_fwd", "e_fwd", "mubar_fwd", "ebar_fwd",
             "mu_bwd", "e_bwd", "mubar_bwd", "ebar_bwd")
    bs = {}
    for c in ("mutot_fwd", "etot_fwd", "mutot_bwd", "etot_bwd"):
        counts = torch.zeros(B, NS)
        for b in range(B):                      # fixed total per replicate
            idx = torch.randint(0, NS, (NS,), generator=g)
            counts[b] = torch.bincount(idx, minlength=NS).double()
        bs[c] = counts / NS
    return dict(
        E=torch.linspace(0.5, 5.0, NE), device=torch.device("cpu"),
        phi_ND={c: r(NE) for c in chans},
        dsamp={k: r(NS, NE) for k in
               ("mutot_mu_fwd", "mutot_mubar_fwd", "mutot_mu_bwd",
                "mutot_mubar_bwd", "etot_e_fwd", "etot_ebar_fwd",
                "etot_e_bwd", "etot_ebar_bwd")},
        I={k: r(NE) for k in ("mu", "mubar", "e", "ebar")},
        bs=bs, dEnu_mu=0.3, dEnu_e=0.3)


def _params(B=3):
    return torch.tensor([[0.7, 2.5e-3, 1.0]] * B, dtype=torch.float64)


def _scale_e_channels(T, factor):
    """A genuine 'e-like cross section is `factor` times larger' change:
    scale the per-event densities AND the matching rate integrals."""
    T2 = {**T, "dsamp": dict(T["dsamp"]), "I": dict(T["I"])}
    for k in ("etot_e_fwd", "etot_ebar_fwd", "etot_e_bwd", "etot_ebar_bwd"):
        T2["dsamp"][k] = T["dsamp"][k] * factor
    for k in ("e", "ebar"):
        T2["I"][k] = T["I"][k] * factor
    return T2


def test_shape_mode_is_blind_to_inter_sample_ratios():
    T, p = _fake_T(), _params()
    ft.LIKELIHOOD = "shape"
    try:
        base = ft.batched_nll(p, T)
        scaled = ft.batched_nll(p, _scale_e_channels(T, 2.0))
    finally:
        ft.LIKELIHOOD = "rate"
    assert torch.allclose(base, scaled), \
        "shape-only must be invariant to the e/mu rate ratio (that is the flaw)"


def test_rate_mode_sees_inter_sample_ratios():
    T, p = _fake_T(), _params()
    ft.LIKELIHOOD = "rate"
    base = ft.batched_nll(p, T)
    scaled = ft.batched_nll(p, _scale_e_channels(T, 2.0))
    assert not torch.allclose(base, scaled), \
        "rate-aware likelihood must respond to the e/mu rate ratio"


def test_rate_mode_invariant_to_a_common_scale():
    """alpha is profiled out, so scaling ALL samples together changes nothing."""
    T, p = _fake_T(), _params()
    ft.LIKELIHOOD = "rate"
    T2 = {**T, "dsamp": {k: v * 3.0 for k, v in T["dsamp"].items()},
          "I": {k: v * 3.0 for k, v in T["I"].items()}}
    assert torch.allclose(ft.batched_nll(p, T), ft.batched_nll(p, T2)), \
        "a common normalization must profile out of the rate-aware form"


def test_modes_differ_by_the_expected_log_ratio_term():
    """ll_rate - ll_shape = sum_g n_g ln N_g - n_tot ln(sum_g N_g)."""
    T, p = _fake_T(), _params()
    ft.LIKELIHOOD = "shape"
    try:
        nll_shape = ft.batched_nll(p, T)
    finally:
        ft.LIKELIHOOD = "rate"
    nll_rate = ft.batched_nll(p, T)

    # rebuild the four predicted totals the same way batched_nll does
    s23, d31, dcp = p[:, 0], p[:, 1], p[:, 2]
    pn = ft.osc_prob_batched(T["E"], s23, d31, dcp, T["device"])
    pa = ft.osc_prob_batched(-T["E"], s23, d31, dcp, T["device"])
    phi = T["phi_ND"]
    f = lambda cm, ce, P, lep: torch.nan_to_num(
        phi[cm] * P[1][1 if lep == "mu" else 0] + phi[ce] * P[0][1 if lep == "mu" else 0])
    groups = [
        ("mutot_fwd", f("mu_fwd", "e_fwd", pn, "mu"), f("mubar_fwd", "ebar_fwd", pa, "mu"), "mu", "mubar", T["dEnu_mu"]),
        ("mutot_bwd", f("mu_bwd", "e_bwd", pn, "mu"), f("mubar_bwd", "ebar_bwd", pa, "mu"), "mu", "mubar", T["dEnu_mu"]),
        ("etot_fwd", f("mu_fwd", "e_fwd", pn, "e"), f("mubar_fwd", "ebar_fwd", pa, "e"), "e", "ebar", T["dEnu_e"]),
        ("etot_bwd", f("mu_bwd", "e_bwd", pn, "e"), f("mubar_bwd", "ebar_bwd", pa, "e"), "e", "ebar", T["dEnu_e"]),
    ]
    N = {g: (T["I"][a] * pl).sum(-1) * d + (T["I"][b] * plb).sum(-1) * d
         for g, pl, plb, a, b, d in groups}
    n = {g: T["bs"][g].shape[1] * T["bs"][g].sum(-1) for g in N}
    n_tot = sum(n.values())
    expect = sum(n[g] * N[g].log() for g in N) - n_tot * sum(N.values()).log()
    assert torch.allclose(-(nll_rate - nll_shape), expect, rtol=1e-10)


def test_bootstrap_counts_fixed_vs_poisson():
    """Default replicates freeze the sample size; poisson=True lets it vary."""
    from ml4nu.utils import precompute_bootstrap_tensor as pbt
    n, nb = 200, 400
    np.random.seed(0)
    fixed = pbt(n, nb) * n
    np.random.seed(0)
    pois = pbt(n, nb, poisson=True) * n
    tot_fixed, tot_pois = fixed.sum(1), pois.sum(1)
    assert np.allclose(tot_fixed, n), "default must draw exactly n events"
    assert tot_pois.std() > 0.5 * np.sqrt(n), \
        "poisson=True must give ~sqrt(n) spread in the replicate total"
    assert abs(tot_pois.mean() - n) < 5 * np.sqrt(n / nb) + 1


def test_extended_requires_exposure():
    T, p = _fake_T(), _params()
    ft.LIKELIHOOD, ft.EXPOSURE = "extended", None
    try:
        with pytest.raises(ValueError, match="EXPOSURE"):
            ft.batched_nll(p, T)
    finally:
        ft.LIKELIHOOD = "extended"


def test_extended_sees_the_absolute_rate():
    """Unlike 'rate', the extended form is NOT invariant to a common scale:
    the total event count is information the oscillation fit may use."""
    T, p = _fake_T(), _params()
    T2 = {**T, "dsamp": {k: v * 3.0 for k, v in T["dsamp"].items()},
          "I": {k: v * 3.0 for k, v in T["I"].items()}}
    ft.EXPOSURE = 1.0
    ft.LIKELIHOOD = "rate"
    assert torch.allclose(ft.batched_nll(p, T), ft.batched_nll(p, T2)), \
        "'rate' profiles the normalization out"
    ft.LIKELIHOOD = "extended"
    assert not torch.allclose(ft.batched_nll(p, T), ft.batched_nll(p, T2)), \
        "'extended' must respond to the absolute normalization"


def test_extended_matches_the_screenshot_formula():
    """ln L = sum_g [ sum_i ln r_i - alpha R_g ], up to w-independent terms.

    Checked as a DIFFERENCE against the shape form, which removes the
    per-event log-density bookkeeping:
        ln L_ext - ln L_shape = sum_g [ n_g ln N_g ] - alpha sum_g N_g
    """
    T, p = _fake_T(), _params()
    alpha = 0.37
    ft.LIKELIHOOD = "shape"
    nll_shape = ft.batched_nll(p, T)
    ft.LIKELIHOOD, ft.EXPOSURE = "extended", alpha
    nll_ext = ft.batched_nll(p, T)

    s23, d31, dcp = p[:, 0], p[:, 1], p[:, 2]
    pn = ft.osc_prob_batched(T["E"], s23, d31, dcp, T["device"])
    pa = ft.osc_prob_batched(-T["E"], s23, d31, dcp, T["device"])
    phi = T["phi_ND"]
    f = lambda cm, ce, P, lep: torch.nan_to_num(
        phi[cm] * P[1][1 if lep == "mu" else 0] + phi[ce] * P[0][1 if lep == "mu" else 0])
    groups = [
        ("mutot_fwd", f("mu_fwd", "e_fwd", pn, "mu"), f("mubar_fwd", "ebar_fwd", pa, "mu"), "mu", "mubar", T["dEnu_mu"]),
        ("mutot_bwd", f("mu_bwd", "e_bwd", pn, "mu"), f("mubar_bwd", "ebar_bwd", pa, "mu"), "mu", "mubar", T["dEnu_mu"]),
        ("etot_fwd", f("mu_fwd", "e_fwd", pn, "e"), f("mubar_fwd", "ebar_fwd", pa, "e"), "e", "ebar", T["dEnu_e"]),
        ("etot_bwd", f("mu_bwd", "e_bwd", pn, "e"), f("mubar_bwd", "ebar_bwd", pa, "e"), "e", "ebar", T["dEnu_e"]),
    ]
    N = {g: (T["I"][a] * pl).sum(-1) * d + (T["I"][b] * plb).sum(-1) * d
         for g, pl, plb, a, b, d in groups}
    n = {g: T["bs"][g].shape[1] * T["bs"][g].sum(-1) for g in N}
    expect = sum(n[g] * N[g].log() for g in N) - alpha * sum(N.values())
    assert torch.allclose(-(nll_ext - nll_shape), expect, rtol=1e-10)


def test_exposure_matches_the_generation_anchor():
    """compute_exposure must reproduce the dt that compute_sample_counts
    used to generate the samples, or the extended likelihood would be
    normalized against a different experiment than the data came from."""
    pytest_importorskip = pytest.importorskip
    rc = pytest_importorskip("ml4nu.pipeline.rates_common")
    import os
    if not os.path.exists("data"):
        pytest.skip("flux data not present")
    s = rc.build_setup()
    counts = rc.compute_sample_counts(s)
    dt = rc.compute_exposure(s)
    # dt is defined by N(mu-,FHC) = 6200, so the anchor must come back exactly
    assert counts["mu_fwd"] == 6200
    assert float(dt) > 0
    # and a second call must agree (no hidden state)
    assert float(rc.compute_exposure(s)) == float(dt)


def test_scipy_and_torch_agree_in_every_mode():
    """The scipy reference and the torch path must implement the SAME
    estimator in all three modes -- that cross-check is the repo's main
    correctness guarantee, and it silently broke while only one side had
    the rate/extended forms."""
    import os
    if not os.path.exists("data"):
        pytest.skip("flux data not present")
    from ml4nu.pipeline import rates_common as rc
    from ml4nu.pipeline import optimize_far_detector as ofd

    T = _fake_T(B=1, NE=6, NS=5, seed=3)
    alpha = 0.37
    # Build the scipy-side inputs from the SAME fake tensors.
    I = {k: v for k, v in T["I"].items()}
    dsamp = {k: v for k, v in T["dsamp"].items()}
    norms = None

    for mode in ("shape", "rate", "extended"):
        # torch side
        ft.LIKELIHOOD, ft.EXPOSURE = mode, alpha
        nll_torch = float(ft.batched_nll(_params(1), T)[0])
        # scipy side: replicate via make_mix_log_likelihood on the same arrays
        phil = {}
        s23, d31, dcp = (_params(1)[:, i] for i in range(3))
        pn = ft.osc_prob_batched(T["E"], s23, d31, dcp, T["device"])
        pa = ft.osc_prob_batched(-T["E"], s23, d31, dcp, T["device"])
        phi = T["phi_ND"]
        f = lambda cm, ce, P, lep: torch.nan_to_num(
            phi[cm] * P[1][1 if lep == "mu" else 0]
            + phi[ce] * P[0][1 if lep == "mu" else 0])[0]
        g = [("mutot_mu_fwd", "mutot_mubar_fwd", "mu", "mubar",
              f("mu_fwd", "e_fwd", pn, "mu"), f("mubar_fwd", "ebar_fwd", pa, "mu"),
              T["dEnu_mu"], "mutot_fwd"),
             ("mutot_mu_bwd", "mutot_mubar_bwd", "mu", "mubar",
              f("mu_bwd", "e_bwd", pn, "mu"), f("mubar_bwd", "ebar_bwd", pa, "mu"),
              T["dEnu_mu"], "mutot_bwd"),
             ("etot_e_fwd", "etot_ebar_fwd", "e", "ebar",
              f("mu_fwd", "e_fwd", pn, "e"), f("mubar_fwd", "ebar_fwd", pa, "e"),
              T["dEnu_e"], "etot_fwd"),
             ("etot_e_bwd", "etot_ebar_bwd", "e", "ebar",
              f("mu_bwd", "e_bwd", pn, "e"), f("mubar_bwd", "ebar_bwd", pa, "e"),
              T["dEnu_e"], "etot_bwd")]
        norms = [rc.mix_sample_norm(I[a], I[b], pl, plb, d)
                 for _, _, a, b, pl, plb, d, _ in g]
        ntot = sum(norms) if mode == "rate" else None
        ll = 0.0
        for (k1, k2, a, b, pl, plb, d, bsk), _n in zip(g, norms):
            lls = rc.make_mix_log_likelihood(dsamp[k1], dsamp[k2], I[a], I[b],
                                             pl, plb, d, norm_total=ntot,
                                             unnormalized=(mode == "extended"))
            w = T["bs"][bsk][0]
            ll += len(w) * float((lls * w).sum())
        if mode == "extended":
            ll -= alpha * float(sum(float(x) for x in norms))
        assert abs(-ll - nll_torch) < 1e-8 * max(1.0, abs(nll_torch)), \
            f"scipy and torch disagree in mode {mode}"


def test_exposure_travels_with_the_samples_not_the_setup():
    """The decisive regression for the 2026-10-08 bug.

    The forward-fold campaign draws its FD samples from a PERTURBED data
    flux and re-anchors them to N(mu-,FHC) = 6200 under that flux, while
    the fit used to take its exposure from the NOMINAL setup. The two
    differ by the flux normalization shift, which the shape-only
    likelihood could not see but the extended one turns into a coherent
    absolute-rate offset in every channel. So: a perturbed flux must give
    a measurably different exposure, and load_exposure must hand back the
    recorded one rather than recomputing it.
    """
    rc = pytest.importorskip("ml4nu.pipeline.rates_common")
    import os
    import numpy as np
    from dataclasses import replace
    if not os.path.exists("data"):
        pytest.skip("flux data not present")
    s = rc.build_setup()
    dt_nom = rc.compute_exposure(s)

    class Scaled:                      # +5% flat on the unoscillated FD flux
        def __init__(self, base):
            self._base = base
        def __getattr__(self, k):
            return getattr(object.__getattribute__(self, "_base"), k)
        def __call__(self, flavor, energy):
            return self._base(flavor, energy) * 1.05

    s_pert = replace(s, fd_unosc_fwd=Scaled(s.fd_unosc_fwd))
    dt_pert = rc.compute_exposure(s_pert)
    # a 5% brighter flux needs 5% less exposure for the same 6200 events
    assert abs(float(dt_nom / dt_pert) - 1.05) < 1e-6, (dt_nom, dt_pert)

    # load_exposure must READ, not recompute: hand it a dir holding the
    # perturbed value and it must not come back with the nominal one.
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        np.savez(os.path.join(d, rc.EXPOSURE_FILE),
                 exposure=float(dt_pert), num_mu_fwd=6200)
        assert rc.load_exposure(d) == float(dt_pert)
        assert rc.load_exposure(d) != float(dt_nom)


def test_load_exposure_refuses_a_sample_dir_without_one():
    """A pre-2026-10-08 sample dir must fail loudly: silently falling back
    to the nominal exposure is exactly the bug."""
    rc = pytest.importorskip("ml4nu.pipeline.rates_common")
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(FileNotFoundError, match="regenerate"):
            rc.load_exposure(d)


# --------------------------------------------------------------------------
# Per-bootstrap flux universes (marginalizing the ND->FD residual)
# --------------------------------------------------------------------------

def test_batched_nll_accepts_one_flux_universe_per_bootstrap():
    """A (B, n_E) flux must give each bootstrap its own universe.

    This is how the irreducible ND->FD flux residual is marginalized
    (fwdfold_shot's FLUXRESID). It works only because make_I is
    flux-independent and _mix_parts contracts the energy axis last, so
    pin it: row b of the batched result must equal a single-flux
    evaluation with universe b.
    """
    T = _fake_T(B=4, NE=6, NS=5)
    ft.LIKELIHOOD, ft.EXPOSURE = "extended", 10.0
    g = torch.Generator().manual_seed(11)
    B = 4
    phi2 = {c: v[None, :] * (1 + 0.1 * torch.rand(B, v.numel(), generator=g))
            for c, v in T["phi_ND"].items()}
    T_multi = {**T, "phi_ND": phi2}
    nll_multi = ft.batched_nll(_params(B), T_multi)
    for b in range(B):
        T_b = {**T, "phi_ND": {c: v[b] for c, v in phi2.items()},
               "bs": {k: v[b:b + 1] for k, v in T["bs"].items()}}
        nll_b = ft.batched_nll(_params(1), T_b)
        assert torch.allclose(nll_multi[b], nll_b[0], rtol=1e-12), b


def test_chunk_phi_slices_and_interleaves_like_the_bootstraps():
    """The 2026-10-08 bug: fit_all_boots widens the bootstrap tensors to
    n*K trajectories but passed `phi` through untouched, so a
    per-bootstrap flux broadcast against the wrong axis -- and wherever
    B happened to equal n*K it would have silently paired universe i with
    trajectory i, mixing universes across replicates and starts."""
    B, NE, K = 6, 4, 3
    one = {"mu_fwd": torch.arange(NE).double()}
    many = {"mu_fwd": torch.arange(B * NE).double().reshape(B, NE)}

    # 1-D fluxes are shared by every trajectory and must pass through
    assert ft._chunk_phi(one, 2, 5, K, B)["mu_fwd"] is one["mu_fwd"]
    assert ft._chunk_phi(None, 0, B, K, B) is None

    out = ft._chunk_phi(many, 2, 5, K, B)["mu_fwd"]
    assert out.shape == (3 * K, NE)                      # n=3 chunk, K starts
    for i, b in enumerate(range(2, 5)):                  # each universe K times
        for k in range(K):
            assert torch.equal(out[i * K + k], many["mu_fwd"][b])

    with pytest.raises(ValueError, match="leading dim"):
        ft._chunk_phi({"mu_fwd": torch.zeros(B + 1, NE)}, 0, 2, K, B)
    with pytest.raises(ValueError, match="1-D or 2-D"):
        ft._chunk_phi({"mu_fwd": torch.zeros(B, NE, 2)}, 0, 2, K, B)


def test_shot_universes_stay_one_dimensional():
    """FLUXRESID=shot (the default) draws ONE flux universe per shot, so the
    FD flux stays (n_E,) and every path -- including the single-bootstrap
    scipy reference -- can consume it. Only FLUXRESID=boot goes 2-D."""
    import numpy as np
    rc = pytest.importorskip("ml4nu.pipeline.rates_common")
    nf = pytest.importorskip("ml4nu.pipeline.flux_nearfar")
    m = nf.load_default()
    E = np.linspace(*rc.FD_EWINDOW, 64)
    R = m.fd_residual_basis({"FHC_numu": E})["FHC_numu"]
    eps_shot = np.random.default_rng(0).standard_normal(R.shape[1])
    assert (R @ eps_shot).shape == (64,)
    eps_boot = np.random.default_rng(0).standard_normal((8, R.shape[1]))
    assert (eps_boot @ R.T).shape == (8, 64)
    # and the offset is the size C_fd|nd predicts, not something else
    rms = np.sqrt(np.mean((R @ eps_shot) ** 2))
    sig = np.sqrt(np.mean(np.diag(m.C_cond)))
    assert 0.3 * sig < rms < 3 * sig, (rms, sig)


def test_scipy_path_refuses_per_bootstrap_fluxes():
    """It evaluates one bootstrap at a time, so it cannot consume flux
    universes; it must say so instead of broadcasting one across all."""
    ofd = pytest.importorskip("ml4nu.pipeline.optimize_far_detector")
    import numpy as np

    class _FD:
        phi_ND = {"mu_fwd": np.ones((4, 6))}
        all_Enu_mu = np.linspace(0.5, 5.0, 6)

    z = np.zeros(3)
    with pytest.raises(NotImplementedError, match="FLUXRESID=0"):
        ofd.compute_ll_for_params_confused(
            0.7, 2.5e-3, 1.0, _FD(), None, None, z, z, z, z,
            phi=_FD.phi_ND)
