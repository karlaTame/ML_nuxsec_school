"""Near/far flux treatment (2026-10-09).

Until this date the FD flux was the ND flux times P_osc, and the ND flux
covariance modes were reused verbatim for the FD flux. Both assumed a
perfectly correlated, energy-independent near/far relationship. The
released tables say otherwise, and these tests pin the corrections.
"""
import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(not os.path.exists("data"),
                                reason="flux data not present")

B = ("data/histos_g4lbne_v3r5p4_QGSP_BERT_OptimizedEngineeredNov2017_")
COL = {"nue": 1, "numu": 2, "nuebar": 4, "numubar": 5}


def test_fd_tables_present_and_aligned():
    for mode in ("neutrino", "antineutrino"):
        nd = np.loadtxt(B + f"{mode}_LBNEND_globes_flux.txt")
        fd = np.loadtxt(B + f"{mode}_LBNEFD_globes_flux.txt")
        assert nd.shape == fd.shape
        assert np.allclose(nd[:, 0], fd[:, 0]), "ND/FD energy grids must match"


def test_near_far_ratio_is_NOT_energy_independent():
    """The assumption that was wrong: FD/ND is not a constant alpha."""
    nd = np.loadtxt(B + "neutrino_LBNEND_globes_flux.txt")
    fd = np.loadtxt(B + "neutrino_LBNEFD_globes_flux.txt")
    E, n, f = nd[:, 0], nd[:, COL["numu"]], fd[:, COL["numu"]]
    band = (E > 0.5) & (E < 8) & (n > 0) & (f > 0)
    r = f[band] / n[band]
    assert r.max() / r.min() > 1.3, \
        "near/far ratio looks flat; check the FD table is really the FD one"


def test_setup_uses_the_fd_tables_for_fd_rates():
    from ml4nu.pipeline import rates_common as rc
    s = rc.build_setup()
    assert s.fd_unosc_fwd is not None and s.fd_unosc_bwd is not None
    from ml4nu.pdg import pid
    E = s.kin_grid_mu.all_enu
    nd_phi = np.asarray(rc.grab(s.nd_flux_fwd(pid.numu, E)))
    fd_phi = np.asarray(rc.grab(s.fd_unosc_fwd(pid.numu, E)))
    assert fd_phi.max() < 1e-3 * nd_phi.max(), \
        "FD flux should be ~1e-7 of ND; Setup may still hold the ND table"


def test_joint_flux_covariance_is_not_perfectly_correlated():
    """The FD block exists and matched ND/FD bins correlate at ~0.94, not 1:
    a perfect ND measurement still leaves residual FD flux uncertainty."""
    from ml4nu.pipeline import flux_model
    m = flux_model.load_default()                 # joint by default now
    assert set(m.detectors) == {"ND", "FD"}
    assert len(m.channels) == 16
    assert m._L.shape[0] == 136

    d = np.load(flux_model.COVARIANCE_FILE, allow_pickle=True)
    det = d["detector"].astype(str)
    C = np.asarray(d["cov"], float)
    sd = np.sqrt(np.diag(C))
    R = C / np.outer(sd, sd)
    lab = np.array([f"{h}_{f}_{lo:.4f}" for h, f, lo in
                    zip(d["horn"].astype(str), d["flavor"].astype(str), d["E_low"])])
    fd_idx = {l: i for i, l in zip(np.where(det == "FD")[0], lab[det == "FD"])}
    pairs = [(i, fd_idx[l]) for i, l in zip(np.where(det == "ND")[0], lab[det == "ND"])
             if l in fd_idx]
    assert pairs, "no matched ND/FD bins found"
    rho = np.median([R[i, j] for i, j in pairs])
    assert 0.8 < rho < 0.999, f"matched ND/FD correlation {rho:.3f} unexpected"


def test_nd_only_model_still_available():
    """The historical (perfectly correlated) behavior stays reproducible."""
    from ml4nu.pipeline import flux_model
    m = flux_model.load_default(detectors=("ND",))
    assert m._L.shape[0] == 68 and len(m.channels) == 8


def test_nd_and_fd_draws_share_one_nuisance_vector():
    """Same seed, different detector -> two halves of ONE correlated draw:
    correlated but NOT identical."""
    from ml4nu.pipeline import systematics as syst
    import torch
    E = torch.linspace(0.5, 8.0, 40)
    chans = {ch: E for ch in ("mu_fwd", "e_fwd")}
    nd = syst.draw_flux_shifts(chans, 1234, detector="ND")
    fd = syst.draw_flux_shifts(chans, 1234, detector="FD")
    a, b = nd["mu_fwd"], fd["mu_fwd"]
    assert not np.allclose(a, b), "ND and FD shifts must not be identical"
    assert np.corrcoef(a, b)[0, 1] > 0.5, "ND and FD shifts should be correlated"
