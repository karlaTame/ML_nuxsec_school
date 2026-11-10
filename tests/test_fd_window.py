"""The FD analysis window, and the exposure anchored on it.

DUNE TDR Vol. II Table 1.2 quotes 6200 as "the integrated rate of selected
nu_mu CC-like events between 0.5 and 8.0 GeV". The pipeline cites that
number as its exposure anchor, so it has to count over the same window.
It used to anchor over the whole energy grid, which left only 78.5% of
the counted rate inside [0.5, 8.0] at ENUMAX=10 (exposure 1.27x too
large) and 44.2% at ENUMAX=20 (2.26x) -- i.e. the inferred POT x mass
depended on ENUMAX, which is an ND-side training choice.
"""
import os

import numpy as np
import pytest

rc = pytest.importorskip("ml4nu.pipeline.rates_common")


def _needs_data():
    if not os.path.exists("data"):
        pytest.skip("flux data not present")


def test_window_is_the_tdr_window():
    assert rc.FD_EWINDOW == (0.5, 8.0)


def test_window_weights_cover_edge_bins_fractionally():
    """A boolean mask snaps the window to grid points and makes the anchor
    wobble ~1% with N_ENU_ND -- the same size as the rate offsets this
    whole exercise is about. Midpoint weights remove that."""
    e = np.linspace(0.0, 10.0, 101)                 # step 0.1
    w = rc._window_weights(e, (0.5, 8.0))
    assert w.shape == e.shape
    assert np.all((w >= 0) & (w <= 1))
    interior = (e > 0.55) & (e < 7.95)
    assert np.allclose(w[interior], 1.0)
    assert np.all(w[e < 0.4] == 0.0)
    assert np.all(w[e > 8.1] == 0.0)
    # total coverage reproduces the window width
    assert abs(w.sum() * 0.1 - 7.5) < 1e-9

    # a window landing mid-bin must still integrate exactly
    w2 = rc._window_weights(e, (0.53, 7.97))
    assert abs(w2.sum() * 0.1 - (7.97 - 0.53)) < 1e-9


def test_exposure_is_invariant_to_enumax():
    """THE property worth having. ENUMAX sets how much of the flux
    covariance the ND can constrain; it must not move the FD
    normalization. Before the anchor was windowed, going 10 -> 20 moved
    the exposure by -43%."""
    _needs_data()
    enumax0, nenu0 = rc.ENUMAX, rc.N_ENU_ND
    try:
        vals = {}
        for emax, nenu in ((10, 32), (20, 64), (20, 128)):
            rc.ENUMAX, rc.N_ENU_ND = emax, nenu
            s = rc.build_setup(N_Enu_ND=nenu)
            vals[(emax, nenu)] = (float(rc.compute_exposure(s)),
                                  rc.compute_sample_counts(s))
    finally:
        rc.ENUMAX, rc.N_ENU_ND = enumax0, nenu0
    dts = np.array([v[0] for v in vals.values()])
    assert np.ptp(dts) / dts.mean() < 0.01, vals          # was 0.43
    # and the analyzed counts come along
    for ch in ("e_fwd", "mu_bwd", "mubar_bwd"):
        c = np.array([v[1][ch] for v in vals.values()], dtype=float)
        assert np.ptp(c) / c.mean() < 0.02, (ch, c)


def test_anchor_channel_is_exactly_num_mu_fwd():
    _needs_data()
    s = rc.build_setup()
    assert rc.compute_sample_counts(s, num_mu_fwd=6200)["mu_fwd"] == 6200
    assert rc.compute_sample_counts(s, num_mu_fwd=1234)["mu_fwd"] == 1234


@pytest.fixture(scope="module")
def generated_samples(tmp_path_factory):
    """Regenerate current-window data instead of relying on stale tracked samples."""
    from ml4nu.pipeline.generate_fd_samples import generate_fd_samples

    setup = rc.build_setup(N_Enu_ND=32, N_v_ND=8)
    sample_dir = tmp_path_factory.mktemp("fd_samples")
    generate_fd_samples(setup, seed=19, sample_dir=str(sample_dir), num_mu_fwd=200)
    return setup, sample_dir


def test_fd_integration_grid_spans_the_window(generated_samples):
    """The likelihood's normalizer must integrate the same window the
    samples were drawn from, or the extended form's absolute rates are
    normalized against a different experiment."""
    _needs_data()
    setup, sample_dir = generated_samples
    import torch
    from ml4nu.cross_section import ModelCalc
    model = rc.SFModel(n_sf=5)
    mc = ModelCalc(model, enumax=rc.ENUMAX, unique=False, anti=False, device=setup.device)
    mcb = ModelCalc(rc.SFModel(n_sf=5), enumax=rc.ENUMAX, unique=False, anti=True, device=setup.device)
    fd = rc.load_fd_samples(setup, mc, mcb, N_Enu_FD=64, N_v_FD=32,
                            sample_dir=str(sample_dir))
    for grid in (fd.all_Enu_mu, fd.all_Enu_e):
        g = np.asarray(rc.grab(grid))
        assert abs(g.min() - rc.FD_EWINDOW[0]) < 1e-9
        assert abs(g.max() - rc.FD_EWINDOW[1]) < 1e-9


def test_drawn_samples_lie_inside_the_window(generated_samples):
    """So num_mu_fwd really is the number of events analyzed. The flux's
    rvs() samples its whole support (to 125 GeV), so this is the Sampler's
    enu_window doing the work, not the enumax mask."""
    _, sample_dir = generated_samples
    lo, hi = rc.FD_EWINDOW
    for ch in ("mu_fwd", "e_fwd", "mubar_bwd", "ebar_bwd"):
        a = np.load(sample_dir / f"samples_FD_{ch}.npy")
        assert a[:, 0].min() >= lo - 1e-9, ch
        assert a[:, 0].max() <= hi + 1e-9, ch
    assert len(np.load(sample_dir / "samples_FD_mu_fwd.npy")) == 200
    assert rc.load_exposure(str(sample_dir)) > 0
