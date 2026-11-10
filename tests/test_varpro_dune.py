"""Near-duplicate consistency: ml4nu.{model,varpro} vs the production DUNE
pipeline (rates_common.SFModel, train_near_detector's VarPro assembly).

The production files stay untouched while the paper is in production; these
tests are the tripwire that keeps the shared library core and the pipeline
from silently diverging. Also pins the Phase-D identity: the identity-
covariance chi^2 equals the DUNE notebook's marginal MSE, and the marginal
basis is exact (the model is linear in c).
"""

import os
import sys

import numpy as np
import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from ml4nu.pipeline import rates_common as rc                     # noqa: E402  (sets float64,
from ml4nu.pipeline import train_near_detector as tnd             # noqa: E402   disables grad)

torch.set_grad_enabled(True)                  # undo rates_common side effect

from ml4nu.pdg import pid                     # noqa: E402
from ml4nu.kinematics import KinematicsGrid, KinematicMapping, \
    unmake_newk2, make_kinematic_factors_newk2, jacobian_newk2   # noqa: E402
from ml4nu.cross_section import ModelCalc     # noqa: E402
from ml4nu.model import SFModel, make_sf_model, fold_c            # noqa: E402
from ml4nu import varpro                      # noqa: E402
from ml4nu.utils import make_marginal_rate    # noqa: E402


def test_sfmodel_matches_rates_common():
    """ml4nu SFModel(c) == rates_common SFModel(_log_norm) at c=exp(-ln)."""
    n_sf = 5
    torch.manual_seed(3)
    m_rc = rc.SFModel(n_sf=n_sf, force_pos=False)
    m_new = SFModel(n_sf=n_sf, force_pos=False)
    m_new.net.load_state_dict(m_rc.net.state_dict())
    with torch.no_grad():
        m_new.c.copy_(torch.exp(-m_rc._log_norm.detach()))

    torch.manual_seed(4)
    X2 = torch.rand(7, 11, 2)
    K = torch.rand(n_sf, 7, 11)
    with torch.no_grad():
        out_rc = m_rc(X2, K)
        out_new = m_new(X2, K)
    denom = out_rc.abs().clamp(min=1e-300)
    assert ((out_rc - out_new).abs() / denom).max() < 1e-14


def test_fold_c_matches_train_near_detector():
    torch.manual_seed(5)
    model = make_sf_model(n_sf=3, seed=11)
    c = torch.tensor([2.5, -0.75, 1e-3])
    sd_new = fold_c(model, c)
    sd_tnd = tnd._fold_c(model, c)
    assert set(sd_new) == set(sd_tnd)
    for k in sd_new:
        if k == 'c':
            # ours resets c to ones; tnd leaves the buffer untouched (it
            # operates on rates_common models that have no c). Both fold the
            # same values into net[-1]; model.c was ones here so they agree.
            assert torch.equal(sd_new[k], torch.ones(3))
        assert torch.equal(sd_new[k], sd_tnd[k]), k


def test_identity_covariance_joint_solve_matches_tnd_assembly():
    """varpro_solve_joint(cov_inv=None) == train_near_detector's
    concatenated least-squares solve (H = A.T A + 1e-12 ridge)."""
    torch.manual_seed(6)
    n_sf = 5

    class _DS:
        def __init__(self, n_bins):
            B = torch.randn(n_bins, n_sf) * torch.logspace(
                -3, 0, n_sf)  # spread column scales like real bases
            self.B = B
            self.b = torch.randn(n_bins).abs()
            self.fit_mask = torch.ones(n_bins, dtype=torch.bool)
            self.cov_inv = None
            self.Cinv_b = self.b
            self.const = (self.b @ self.b).item()
            self.name = f'ds{n_bins}'

    ds1, ds2 = _DS(40), _DS(25)
    ridge_rel = 1e-12
    c_joint, chi_joint, per = varpro.varpro_solve_joint(
        [ds1.B, ds2.B], [ds1, ds2], ridge=ridge_rel)

    # train_near_detector assembly: concatenate rows, one ridge-solved system
    A = torch.cat([ds1.B, ds2.B])
    bvec = torch.cat([ds1.b, ds2.b])
    H = A.T @ A
    ridge = ridge_rel * H.diagonal().mean()
    c_tnd = torch.linalg.solve(
        H + ridge * torch.eye(n_sf), A.T @ bvec)

    assert torch.allclose(c_joint, c_tnd, rtol=1e-10, atol=0)
    loss_tnd = ((A @ c_tnd - bvec) ** 2).sum().item()
    assert abs(chi_joint - loss_tnd) <= 1e-11 * abs(loss_tnd)
    # per-dataset split is additive
    assert abs(sum(per) - chi_joint) <= 1e-12 * abs(chi_joint)


@pytest.fixture(scope="module")
def dune_grid_setup():
    """Small DUNE-style newk2 grid + ModelCalc, no PDF files needed."""
    kin_mapping = KinematicMapping(
        unmake_func=unmake_newk2,
        make_kinematic_factors=make_kinematic_factors_newk2,
        jacobian=jacobian_newk2)
    v1 = torch.linspace(0.25, 3.25, 8)
    v2 = torch.linspace(0.0, 0.65, 6)
    enu = torch.linspace(0.011, 10.0, 12)
    kg = KinematicsGrid(v1, v2, enu, pid.mu, kin_mapping=kin_mapping)
    model = make_sf_model(n_sf=3, seed=17)
    mc = ModelCalc(model, enumax=10)
    phi = torch.rand(12).abs() + 0.1
    return model, mc, kg, phi, kin_mapping


def test_marginal_dataset_chi2_equals_inline_mse(dune_grid_setup):
    """chi2_of on a MarginalVarProDataset == ((marginal - P)**2).sum(),
    the DUNE tutorial's training loss, at arbitrary weights and c."""
    model, mc, kg, phi, km = dune_grid_setup
    torch.manual_seed(8)
    P = torch.rand(kg.dim1, kg.dim2) * 1e-38
    ds = varpro.make_marginal_dataset('dune_nd', mc, kg, phi, P,
                                      kin_mapping=km)
    with torch.no_grad():
        model.c.copy_(torch.tensor([2.0, -0.5, 1.3]))
        B = ds.basis(model)
        chi2 = varpro.chi2_of(B, model.c, ds).item()
        marg = make_marginal_rate(mc.dsigma(), phi, kg.denu, kg.dV2)
        mse = ((marg.flatten() - P.flatten()) ** 2).sum().item()
    assert abs(chi2 - mse) <= 1e-12 * max(abs(mse), 1e-300)


def test_marginal_basis_linearity(dune_grid_setup):
    """A @ c must equal the marginal of the full model forward (dsigma with
    c inside) — the model is linear in c, so the basis is exact."""
    model, mc, kg, phi, km = dune_grid_setup
    ds = varpro.make_marginal_dataset('dune_nd', mc, kg, phi,
                                      torch.zeros(kg.dim1, kg.dim2),
                                      kin_mapping=km)
    with torch.no_grad():
        model.c.copy_(torch.tensor([0.7, 3.1, -1.2]))
        lhs = ds.basis(model) @ model.c
        rhs = make_marginal_rate(mc.dsigma(), phi, kg.denu, kg.dV2).flatten()
    denom = rhs.abs().clamp(min=1e-300)
    assert ((lhs - rhs).abs() / denom).max() < 1e-12


def test_varpro_optimum_beats_perturbations(dune_grid_setup):
    """Sanity: the solved c is the argmin of the quadratic chi^2."""
    model, mc, kg, phi, km = dune_grid_setup
    torch.manual_seed(9)
    P = torch.rand(kg.dim1, kg.dim2) * 1e-38
    ds = varpro.make_marginal_dataset('dune_nd', mc, kg, phi, P,
                                      kin_mapping=km)
    with torch.no_grad():
        B = ds.basis(model)
        c_star, chi2 = varpro.varpro_solve(B, ds)
        for _ in range(10):
            c_pert = c_star * (1 + 0.01 * torch.randn(3))
            assert varpro.chi2_of(B, c_pert, ds).item() >= chi2


def test_stop_at_chi2_bookkeeps_crossing(dune_grid_setup):
    """The discrepancy-stop crossing step must enter history and best
    tracking, and final_model must reproduce the recorded chi2 exactly
    (the 2026-08-23 bug: the crossing step was dropped, leaving best_loss
    systematically above the stopping threshold)."""
    model, mc, kg, phi, km = dune_grid_setup
    torch.manual_seed(12)
    P = torch.rand(kg.dim1, kg.dim2) * 1e-38
    ds = varpro.make_marginal_dataset('nd', mc, kg, phi, P, kin_mapping=km)
    result = varpro.fit_varpro(model, [ds], n_steps=50, log_every=10**9,
                               progress=False, stop_at_chi2=1e30)
    assert result.history['stopped_at'] == 0
    assert len(result.history['loss']) == 1
    assert result.best_loss == result.history['loss'][0]
    assert result.best_loss <= 1e30 * ds.fit_mask.sum().item()
    model.load_state_dict(result.final_model)
    with torch.no_grad():
        chi2 = varpro.chi2_of(ds.basis(model), model.c, ds).item()
    assert abs(chi2 - result.best_loss) <= 1e-9 * abs(result.best_loss)


def test_unique_sf_combination_identity():
    """Sum K5.F == sum K3u.unique_sf_combination(F) for arbitrary fields —
    the exact single-lepton 5->3 reduction (rung-0 identity, now in the
    library for SF-map comparisons)."""
    from ml4nu.model import unique_sf_combination
    from ml4nu.kinematics import make_kinematic_factors
    from ml4nu.pdg import masses as _m, pid as _p
    torch.manual_seed(21)
    ml2 = _m[_p.mu] ** 2
    x = torch.rand(50) * 0.9 + 0.05
    q2 = torch.rand(50) * 8 + 0.5
    y = torch.rand(50) * 0.8 + 0.1
    K5 = make_kinematic_factors(q2, x, y, unique=False, ml2=ml2, enumax=1e9)
    K3 = make_kinematic_factors(q2, x, y, unique=True, ml2=ml2, enumax=1e9)
    F = torch.randn(5, 50)
    Fu = unique_sf_combination(F, x, q2, ml2=ml2)
    lhs = sum(K5[j] * F[j] for j in range(5))
    rhs = sum(K3[i] * Fu[i] for i in range(3))
    denom = lhs.abs().clamp(min=1e-300)
    mask = lhs.abs() > 1e-12
    assert (((lhs - rhs).abs() / denom)[mask] < 1e-10).all()
