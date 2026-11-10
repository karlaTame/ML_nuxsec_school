"""Near/far flux extrapolation as a conditional Gaussian.

These pin the replacement for the rank-deficient 136-mode nuisance basis:
the ND fit basis must be full rank with a unit-normal prior, and the FD
prediction must come from the real near/far regression rather than from
either degenerate alternative (reuse the ND modes = assert perfect
correlation; use the FD block alone = assert none).
"""
import numpy as np
import pytest

pytest.importorskip("ml4nu.pipeline.flux_nearfar")
from ml4nu.pipeline import flux_nearfar as nf, flux_model as fmod


@pytest.fixture(scope="module")
def m():
    return nf.load_default()


def test_blocks_are_square_and_full_rank(m):
    """The whole point: no degeneracy to regularize away."""
    assert m.n_nd == m.n_fd == 68
    assert np.linalg.matrix_rank(m.C_nd) == 68
    # C_fd|nd is a genuine 68-dim residual, not a near-null remainder
    assert np.linalg.matrix_rank(m.C_cond, tol=1e-10 * np.abs(m.C_cond).max()) == 68
    # and the ND fit basis, stacked over all 8 ND channels, spans all 68
    # directions -- every nuisance is ND-constrained. Contrast the joint
    # eigenbasis, whose ND response is 68 x 136 of rank 68, leaving 68
    # ND-blind directions for the regularizer to decide.
    E = np.linspace(0.02, 20.0, 800)        # the FULL support of the cov bins
    F = m.nd_basis({b: E for b in m._bases})
    assert F["FHC_numu"].shape == (800, 68)
    stacked = np.concatenate([F[b] for b in m._bases], 0)
    assert np.linalg.matrix_rank(stacked) == 68
    # each single channel only reaches its OWN bins, which is why the
    # rank statement has to be made on the stack
    assert np.linalg.matrix_rank(F["FHC_numu"]) < 68


def test_nd_basis_reproduces_the_nd_flux_covariance(m):
    """F @ F.T must equal the fractional ND covariance on the same grid,
    i.e. u ~ N(0,I) is exactly the right prior for these coordinates."""
    fm = fmod.load_default(detectors=("ND", "FD"))
    E = np.linspace(0.6, 8.0, 40)
    for base in ("FHC_numu", "FHC_nue", "RHC_numubar"):
        F = m.nd_basis({base: E})[base]
        C = fm.covariance({f"ND_{base}": E}, fractional=True)[0]
        assert np.allclose(F @ F.T, C, atol=1e-13), base


def test_fd_basis_is_the_regression_of_the_nd_nuisances(m):
    """fd_basis(u) must equal M_fd @ B @ chol @ u, evaluated independently."""
    E = np.linspace(0.6, 8.0, 24)
    base = "FHC_numu"
    G = m.fd_basis({base: E})[base]
    rng = np.random.default_rng(3)
    u = rng.standard_normal(m.n_nd)
    # independent route: u -> ND bin shift -> regression -> interpolate
    d_nd = m.chol_nd @ u
    d_fd_bins = m.B @ d_nd
    M = m._M({base: E}, "FD", m._fd_cols)[base]
    assert np.allclose(G @ u, M @ d_fd_bins, atol=1e-12)


def test_conditioning_shrinks_the_fd_uncertainty_everywhere(m):
    """C_fd|nd must be PSD and strictly tighter than C_fd in every bin --
    the ND is informative about the FD flux, which is exactly what using
    the FD-only block would throw away."""
    s = m.conditional_summary()
    assert np.all(np.linalg.eigvalsh(m.C_cond) > -1e-12)
    assert np.all(s["shrinkage"] < 1.0)
    # measured for the DUNE matrix: ~89% of the FD flux is ND-determined
    assert 0.05 < np.median(s["shrinkage"]) < 0.25
    assert np.median(s["sigma_fd"]) > 5 * np.median(s["sigma_cond"])


def test_regression_beats_both_degenerate_alternatives(m):
    """On joint draws from the real covariance, predicting the FD flux by
    regression must beat (a) reusing the ND shift verbatim -- what the
    136-mode basis did -- and must agree with C_fd|nd."""
    fm = fmod.load_default(detectors=("ND", "FD"))
    L = fm._L
    rng = np.random.default_rng(0)
    err_reg, err_reuse = [], []
    for _ in range(300):
        b = rng.standard_normal(L.shape[1])
        d_nd, d_fd = m.nd_block_of(b, L), m.fd_block_of(b, L)
        err_reg.append(np.abs(d_fd - m.B @ d_nd))
        err_reuse.append(np.abs(d_fd - d_nd))
    reg, reuse = np.median(err_reg), np.median(err_reuse)
    assert reg < 0.5 * reuse, (reg, reuse)
    # and the residual must match the conditional covariance it predicts:
    # E|x| = sqrt(2/pi) sigma for a centered Gaussian
    pred = np.sqrt(2 / np.pi) * np.median(np.sqrt(np.diag(m.C_cond)))
    assert abs(reg / pred - 1) < 0.35, (reg, pred)


def test_matched_correlation_is_not_the_regression_coefficient(m):
    """A trap worth pinning: matched bins correlate at ~0.94, but the
    diagonal of B is ~0.37 -- a shift in one ND bin is redistributed over
    many correlated FD bins. 'Scale the ND shift by rho' is wrong."""
    s = m.conditional_summary()
    assert 0.9 < np.median(s["matched_rho"]) < 0.99
    assert np.median(s["B_diag"]) < 0.6


def test_u_from_bin_shift_roundtrips(m):
    rng = np.random.default_rng(7)
    u = rng.standard_normal(m.n_nd)
    assert np.allclose(m.u_from_bin_shift(m.chol_nd @ u), u, atol=1e-10)


def test_residual_basis_has_the_conditional_covariance(m):
    E = np.linspace(0.6, 8.0, 30)
    base = "FHC_numu"
    R = m.fd_residual_basis({base: E})[base]
    M = m._M({base: E}, "FD", m._fd_cols)[base]
    assert np.allclose(R @ R.T, M @ m.C_cond @ M.T, atol=1e-13)


def test_refuses_an_nd_only_model():
    nd_only = fmod.load_default(detectors=("ND",))
    with pytest.raises(ValueError, match="JOINT"):
        nf.NearFarFluxModel(nd_only)


def test_enumax_leaves_eight_directions_blind(m):
    """The covariance's flux bins run to 20 GeV but the pipeline trains on
    ENUMAX = 10, so 8 of the 68 nuisance directions are invisible to the
    ND fit for lack of coverage -- and they still move the FD prediction.
    Conditioning on what the ND actually observes is what accounts for
    them; forgetting to understates the irreducible FD flux residual."""
    E_full = np.linspace(0.02, 20.0, 800)
    E_pipe = np.linspace(0.02, 10.0, 400)

    def stack(E):
        b = m.nd_basis({k: E for k in m._bases})
        return np.concatenate([b[k] for k in m._bases], 0)

    Q_full, Qp_full = m.observable_subspace(stack(E_full))
    Q_pipe, Qp_pipe = m.observable_subspace(stack(E_pipe))
    assert Q_full.shape[0] == 68 and Qp_full.shape[0] == 0
    assert Q_pipe.shape[0] == 60 and Qp_pipe.shape[0] == 8

    # full coverage reproduces the plain conditional; ENUMAX=10 does not
    assert np.allclose(m.residual_covariance(stack(E_full)), m.C_cond, atol=1e-12)
    s_best = np.sqrt(np.diag(m.C_cond))
    s_pipe = np.sqrt(np.diag(m.residual_covariance(stack(E_pipe))))
    assert np.all(s_pipe >= s_best - 1e-12)
    assert 1.0 < 100 * np.median(s_pipe) < 1.2          # ~1.07%
    assert 3.5 < 100 * s_pipe.max() < 4.5               # ~4.00%, vs 2.86% best case
