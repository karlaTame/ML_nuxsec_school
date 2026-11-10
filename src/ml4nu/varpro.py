"""Covariance-weighted VarPro training core.

The model prediction is linear in per-SF coefficients c, so each step solves
c analytically from the normal equations of a chi^2 with an arbitrary data
covariance, and Adam trains only the network shape with the gradient flowing
through the basis. The production DUNE trainer's unweighted least squares
(train_near_detector.train_near_detector_varpro) is the identity-covariance
special case; tests/test_varpro_dune.py pins the two implementations
against each other.

Datasets are duck-typed (see MarginalVarProDataset):

    name      str
    b         (N_fit,)  data values on fitted bins
    fit_mask  (N_targets,) bool
    cov_inv   (N_fit, N_fit) inverse covariance, or None for identity
    Cinv_b    (N_fit,)  cov_inv @ b   (== b when cov_inv is None)
    const     float     b @ Cinv_b    (chi^2 at m = 0)
    basis(model) -> (N_targets, n_sf), gradient flows through the net

Joint training on several independent datasets is the block sum of their
normal equations (block-diagonal covariance => the joint chi^2 is exactly
additive), with one shared c.
"""

import copy
from dataclasses import dataclass, field

import numpy as np
import torch
from tqdm import tqdm

from .utils import make_marginal_rate


# ----------------------------------------------------------------------------
# Basis and solve (notebook cell 16)
# ----------------------------------------------------------------------------

def minerva_basis(model, X2_batch, kun_batch, phi, denu):
    """Basis matrix B (N_bins, n_sf) for flux-averaged bin-averaged densities.

    B[i, j] = (1/Phi) * bin-AVERAGED density of (K_j * net_j) over the
              cell centres of bin i, flux-integrated over E,
              with Phi = integral of phi(E) dE on the same grid.
    The model prediction vector is then  m = B @ c.

    Two load-bearing conventions (2026-08-11 fixes, do not "simplify"):
    * the sub-cell MEAN with no bin-area multiply — MINERvA publishes the
      DENSITY d2sigma/dPz/dPt, and multiplying by the bin area contaminated
      the learned SFs with the binning scheme;
    * the explicit 1/Phi computed from the SAME flux array on the SAME grid —
      without it VarPro's c silently absorbs 1/Phi_train, mis-normalizing
      every other dataset by Phi_other/Phi_train (and the flux files' per-bin
      vs per-GeV unit conventions on top). Computing Phi this way cancels any
      flux-file normalization convention exactly, which is also what makes a
      shared c meaningful across datasets in a joint fit.
    """
    G = model.forward_basis(X2_batch, kun_batch)  # (n_sf,N_bins,Ns,Ns,Nenu)
    avg1 = G.mean(dim=2)                 # (n_sf,N_bins,Ns,Nenu)
    avg2 = avg1.mean(dim=2)              # (n_sf,N_bins,Nenu)  bin-avg density
    Phi = phi.sum() * denu               # integrated flux on this grid
    B = (avg2 * phi[None, None, :]).sum(dim=2) * denu / Phi  # (n_sf,N_bins)
    return B.T                                            # (N_bins,n_sf)


def varpro_solve_joint(Bs, datasets, ridge=0.0):
    """Solve the joint normal equations over one or more datasets.

    H = sum_d B_d[fit].T @ Cinv_d @ B_d[fit],  rhs = sum_d B_d[fit].T @ Cinv_b_d
    Returns (c_star, chi2_total, chi2_per_dataset). cov_inv=None is treated
    as the identity. With a single dataset this reproduces the notebook's
    varpro_solve bit-exactly (ridge defaults to 0.0, notebook-exact;
    train_near_detector uses a 1e-12 relative ridge if ever needed).
    """
    H = None
    rhs = None
    B_fits = []
    for B, ds in zip(Bs, datasets):
        B_fit = B[ds.fit_mask]                       # (N_fit, n_sf)
        B_fits.append(B_fit)
        if ds.cov_inv is None:
            H_d = B_fit.T @ B_fit
        else:
            H_d = B_fit.T @ ds.cov_inv @ B_fit       # (n_sf, n_sf)
        rhs_d = B_fit.T @ ds.Cinv_b                  # (n_sf,)
        H = H_d if H is None else H + H_d
        rhs = rhs_d if rhs is None else rhs + rhs_d
    if ridge:
        n_sf = H.shape[0]
        H = H + ridge * H.diagonal().mean() * torch.eye(
            n_sf, device=H.device, dtype=H.dtype)
    c_star = torch.linalg.solve(H, rhs)              # (n_sf,)
    # Optimal chi^2 per dataset (identical to direct computation — check)
    chi2_per = []
    for B_fit, ds in zip(B_fits, datasets):
        m_fit = B_fit @ c_star
        if ds.cov_inv is None:
            chi2_per.append(
                (m_fit @ m_fit - 2 * ds.Cinv_b @ m_fit + ds.const).item())
        else:
            chi2_per.append(
                (m_fit @ ds.cov_inv @ m_fit
                 - 2 * ds.Cinv_b @ m_fit + ds.const).item())
    return c_star, sum(chi2_per), chi2_per


def varpro_solve(B, ds, ridge=0.0):
    """Single-dataset wrapper: returns (c_star, chi2)."""
    c_star, chi2, _ = varpro_solve_joint([B], [ds], ridge=ridge)
    return c_star, chi2


def chi2_of(B, c, ds):
    """Differentiable chi^2 in the training-loop form (cell 18).

    m = B[fit] @ c;  chi^2 = m @ (Cinv @ m) - 2 Cinv_b @ m + const.
    c is used as-is (a buffer — treated as constant), so the gradient flows
    only through B. Identity covariance: chi^2 = m@m - 2 b@m + b@b, which is
    exactly ((m - b)**2).sum() — the DUNE marginal MSE.
    """
    m = B[ds.fit_mask] @ c
    if ds.cov_inv is None:
        Cinv_m = m
    else:
        Cinv_m = ds.cov_inv @ m
    return m @ Cinv_m - 2 * ds.Cinv_b @ m + ds.const


def _chi2_eval(B, c, ds):
    """no-grad chi^2 in the cell-18 test-evaluation form (chained matmuls)."""
    m = B[ds.fit_mask] @ c
    if ds.cov_inv is None:
        return (m @ m - 2 * ds.Cinv_b @ m + ds.const).item()
    return (m @ ds.cov_inv @ m - 2 * ds.Cinv_b @ m + ds.const).item()


def predict(model, ds):
    """Model prediction B @ c on ALL targets of ds (no fit mask applied)."""
    with torch.no_grad():
        return ds.basis(model) @ model.c


def resolve_c(model, datasets, ridge=0.0):
    """Re-solve c for the model's current weights and copy it into model.c.

    Needed after loading a training snapshot: the loop stores c from one
    step before opt.step(), so the stored c is stale for those weights
    (cells 20/21). Returns (chi2_total, chi2_per_dataset).
    """
    if not isinstance(datasets, (list, tuple)):
        datasets = [datasets]
    with torch.no_grad():
        Bs = [ds.basis(model) for ds in datasets]
        c_star, chi2, chi2_per = varpro_solve_joint(Bs, datasets)
        model.c.copy_(c_star)
    return chi2, chi2_per


def scale_report(model, ds):
    """Print the cell-17 scale diagnostics: c ~ data/B is a units effect."""
    with torch.no_grad():
        B = ds.basis(model)                                   # (N_bins, n_sf)
        X2_flat = ds.X2.reshape(-1, model.n_in)
        net_raw = model._net_out(X2_flat)                     # (pts, n_sf)
        K = ds.kun
        b = ds.b

        print("=== Scale diagnostics ===\n")
        print("Data b (cross section per fitted bin):")
        print(f"  mean={b.mean():.3e}  min={b.min():.3e}  max={b.max():.3e}\n")
        print("Net outputs net_j(x,Q²)  [raw NN, no c, no K]:")
        for j in range(model.n_sf):
            v = net_raw[:, j]
            print(f"  j={j}: mean|net|={v.abs().mean():.3e}  "
                  f"min={v.min():.3e}  max={v.max():.3e}")
        print("\nKinematic factors K_j(x,Q²,E):")
        for j in range(model.n_sf):
            v = K[j]
            print(f"  j={j}: mean|K|={v.abs().mean():.3e}  "
                  f"min={v.abs().min():.3e}  max={v.abs().max():.3e}")
        print(f"\nFlux phi(E):   mean={ds.phi.mean():.3e}   max={ds.phi.max():.3e}")
        print(f"dE (energy step): {ds.denu:.3e}")
        print("\nBasis B[i,j]:")
        for j in range(model.n_sf):
            v = B[:, j]
            print(f"  j={j}: mean|B|={v.abs().mean():.3e}  "
                  f"min={v.abs().min():.3e}  max={v.abs().max():.3e}")
        print(f"\nc (VarPro): {[f'{v:.3e}' for v in model.c.tolist()]}")
        pred_fit = (B @ model.c)[ds.fit_mask]
        print("\nB @ c (model predictions, fitted bins):")
        print(f"  mean={pred_fit.mean():.3e}  min={pred_fit.min():.3e}  "
              f"max={pred_fit.max():.3e}")
        print(f"\nRatio (B@c) / data:  mean={(pred_fit/b).mean():.3f}  "
              f"(should be ~1 if normalisation is correct)")
        print("\n--- Bottom line ---")
        print(f"  data scale:  {b.mean():.3e}")
        print(f"  B scale:     {B.abs().mean():.3e}")
        print(f"  → c ~ data/B = {(b.mean()/B.abs().mean()):.3e}  "
              f"(matches c above?)")


# ----------------------------------------------------------------------------
# Trainer (notebook cell 18, minus the frozen-out stage machinery)
# ----------------------------------------------------------------------------

@dataclass
class FitResult:
    best_loss_model: dict
    best_loss: float
    best_loss_step: int
    best_test_model: dict
    best_test: float
    best_test_step: int
    final_model: dict
    history: dict = field(default_factory=dict)


def _cpu_snapshot(model):
    return copy.deepcopy(
        {k: v.clone().detach().cpu() for k, v in model.state_dict().items()})


def fit_varpro(model, train, test=(), *, n_steps=100_000, lr=1e-3,
               clip_norm=1.0, factor=0.5, patience=2000, min_lr=1e-5,
               threshold=1e-2, extra_metrics=None, best_metric=None,
               penalty=None, loss_norm=None, ridge=0.0, stop_at_chi2=None,
               log_every=5000, vp_check_every=5000, progress=True):
    """VarPro + Adam + ReduceLROnPlateau training loop.

    train / test : lists of dataset objects (see module docstring). The loss
    is the sum of train-dataset chi^2's at the jointly VarPro-optimal c
    (gradient through the bases only); test datasets and ``extra_metrics``
    (dict name -> callable(model) -> float) are evaluated each step under
    no_grad with the train-solved c. ``best_test`` tracks the sum over test
    datasets, or the ``best_metric`` extra metric when there are no test
    datasets. Runs under ``torch.enable_grad()`` so it works even when grad
    is globally disabled (rates_common import side effect).

    penalty : optional callable(model) -> differentiable scalar, ADDED to
    the chi^2 loss each step (e.g. a density-positivity term
    ``lam * relu(-dsigma)**2``). It shapes the network only through the
    gradient — the VarPro c solve stays the pure-chi^2 optimum, so model
    linearity in c is untouched, and a one-sided penalty vanishes exactly
    once satisfied. The VP self-check and ``history['loss']`` refer to the
    chi^2 part alone (comparable across runs); the penalty value is logged
    as ``history['penalty']`` and best-loss tracking uses the total
    objective.

    loss_norm : None | float | 'auto'. Multiplies the chi^2 in the optimized
    objective (the production trainer's LOSS_NORM=1e11, train_near_detector
    cell-21 semantics). Physically-scaled losses like the DUNE marginal MSE
    are O(1e-11): their ~1e-9 gradients sit below Adam's eps=1e-8, which
    suppresses and distorts updates and keeps clip_norm from ever engaging —
    the 2026-08 bisection measured a 5-13x FD-fidelity penalty for training
    unscaled. 'auto' measures the initial VarPro-solved chi^2 and uses
    1/chi2_init, so the objective starts at exactly 1 with no hand-picked
    constant (the chosen value is printed and stored as
    ``history['loss_norm_used']``). Dimensionless covariance-weighted chi^2s
    (MINERvA) are already O(1)-scaled and need None. ``history['loss']``
    always records the PHYSICAL (unscaled) chi^2; ``best_loss`` and the
    scheduler follow the optimized objective (the plateau scheduler uses
    relative thresholds, so its behavior is scale-invariant).

    ridge : relative Tikhonov term in the per-step VarPro c solve
    (H += ridge * mean(diag H) * I; train_near_detector uses 1e-12 as a
    numerical guard). Larger values bias training against large-cancellation
    c solutions; evaluation-time resolve_c stays unridged.

    stop_at_chi2 : float | None. Discrepancy-principle early stop: halt as
    soon as EVERY train dataset individually satisfies
    chi2_d <= stop_at_chi2 * n_fit_d (checked at the current weights,
    before the optimizer step, so ``final_model`` IS the stopping state
    with its VarPro-solved c). Meaningful only for statistically-scaled
    chi^2 targets (real-data covariances, e.g. MINERvA); an exact-toy MSE
    has no noise floor. Rationale: fitting real data past chi2/N ~ 1 is
    fitting noise — the 2026-08-22 study measured half the sigma(E)
    squiggliness of a 100k-step joint MINERvA fit appearing between
    chi2/N = 1 and 0.16. The stopping step lands in
    ``history['stopped_at']``; the crossing step is fully bookkept
    (histories and best-model tracking — its snapshot is the exact
    crossing state, free of the usual one-step-late snapshot drift), and
    the stopping state is also ``final_model``.

    Hyperparameter defaults are the MINERvA notebook's (clip_norm=1.0
    prevents the collapse-at-LR-reset failure; threshold=1e-2 makes the
    plateau scheduler respond to genuine stalls, not noise-level progress).
    Multi-resolution subgrid stages were retired (subgrid_schedule=[1] is
    frozen); staged training = sequential fit_varpro calls with warm start.
    """
    train = list(train)
    test = list(test)
    extra_metrics = dict(extra_metrics or {})
    if best_metric is not None:
        assert best_metric in extra_metrics, (best_metric, list(extra_metrics))
    assert next(model.parameters()).dtype == torch.float64, \
        "float64 required (pipeline convention)"

    opt = torch.optim.Adam(model.net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=factor, patience=patience, min_lr=min_lr,
        threshold=threshold)

    best_loss, best_loss_step = np.inf, 0
    best_loss_model = _cpu_snapshot(model)
    best_test, best_test_step = np.inf, 0
    best_test_model = _cpu_snapshot(model)

    history = {'loss': [], 'test': [], 'grad_norm': [], 'lr': []}
    if penalty is not None:
        history['penalty'] = []
    for ds in train:
        history[f'chi2/{ds.name}'] = []
    for ds in test:
        history[f'chi2/{ds.name}'] = []
    for name in extra_metrics:
        history[name] = []

    # Initial VarPro step: set c optimally for the current (e.g. random) net.
    with torch.no_grad():
        Bs0 = [ds.basis(model) for ds in train]
        c0, chi0, _ = varpro_solve_joint(Bs0, train, ridge=ridge)
        model.c.copy_(c0)
        n_fit = sum(int(ds.fit_mask.sum().item()) for ds in train)
        print(f'VarPro init: c = {[f"{v:.3e}" for v in c0.tolist()]}')
        print(f'  initial chi2 = {chi0:.4e}  (chi2/Ndof = {chi0/n_fit:.2f})')
    print(f'  LR set to {lr:.2e}')

    if loss_norm == 'auto':
        assert chi0 != 0, 'auto loss_norm undefined: initial chi2 is 0'
        loss_norm = 1.0 / abs(chi0)
        print(f'  loss_norm (auto) = {loss_norm:.3e}  (objective starts at 1)')
    if loss_norm is not None:
        history['loss_norm_used'] = float(loss_norm)

    steps = tqdm(range(n_steps)) if progress else range(n_steps)
    vp_ok = True
    with torch.enable_grad():
        for nth_step in steps:
            opt.zero_grad()

            # -- VarPro: solve for optimal c given current network shape ----
            Bs = [ds.basis(model) for ds in train]
            with torch.no_grad():
                c_star, chi2_vp, chi2_vp_per = varpro_solve_joint(
                    [B.detach() for B in Bs], train, ridge=ridge)
                model.c.copy_(c_star)

            # -- Loss: chi^2 at VarPro-optimal c (gradient flows via B) -----
            per_ds = [chi2_of(B, model.c, ds) for B, ds in zip(Bs, train)]
            if stop_at_chi2 is not None and all(
                    t.item() <= stop_at_chi2 * ds.fit_mask.sum().item()
                    for t, ds in zip(per_ds, train)):
                # Bookkeep the crossing step fully (no optimizer step is
                # taken, so the current weights + just-solved c ARE the
                # stopping state), then halt. Without this the crossing
                # step never enters history/best tracking and best_loss
                # sits systematically above the threshold.
                history['stopped_at'] = nth_step
                stop_val = sum(t.item() for t in per_ds)
                with torch.no_grad():
                    test_per = [_chi2_eval(ds.basis(model), model.c, ds)
                                for ds in test]
                    test_loss = sum(test_per) if test_per else np.inf
                    metric_vals = {name: fn(model)
                                   for name, fn in extra_metrics.items()}
                history['loss'].append(stop_val)
                if penalty is not None:
                    history['penalty'].append(float('nan'))
                history['test'].append(test_loss)
                history['grad_norm'].append(float('nan'))
                history['lr'].append(opt.param_groups[0]['lr'])
                for ds, t in zip(train, per_ds):
                    history[f'chi2/{ds.name}'].append(t.item())
                for ds, t in zip(test, test_per):
                    history[f'chi2/{ds.name}'].append(t)
                for name, v in metric_vals.items():
                    history[name].append(v)
                if stop_val < best_loss:
                    best_loss, best_loss_step = stop_val, nth_step
                    best_loss_model = _cpu_snapshot(model)
                track = (metric_vals[best_metric]
                         if (not test and best_metric is not None)
                         else test_loss)
                if track < best_test:
                    best_test, best_test_step = track, nth_step
                    best_test_model = _cpu_snapshot(model)
                print(f'stop_at_chi2: all train datasets at chi2/N <= '
                      f'{stop_at_chi2:g} at step {nth_step} '
                      f'({", ".join(f"{ds.name} {t.item()/ds.fit_mask.sum().item():.3f}" for t, ds in zip(per_ds, train))})')
                break
            chi2_loss = per_ds[0]
            for t in per_ds[1:]:
                chi2_loss = chi2_loss + t
            loss = chi2_loss if loss_norm is None else chi2_loss * loss_norm
            if penalty is not None:
                pen = penalty(model)
                loss = loss + pen
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                  max_norm=clip_norm)
            opt.step()

            # -- Test datasets + extra metrics with the train-solved c ------
            with torch.no_grad():
                test_per = [_chi2_eval(ds.basis(model), model.c, ds)
                            for ds in test]
                test_loss = sum(test_per) if test_per else np.inf
                metric_vals = {name: fn(model)
                               for name, fn in extra_metrics.items()}

            loss_val = loss.item()   # optimized objective (scaled chi^2 + pen)
            chi2_val = (loss_val if (penalty is None and loss_norm is None)
                        else chi2_loss.item())   # physical chi^2
            history['loss'].append(chi2_val)
            if penalty is not None:
                history['penalty'].append(pen.item())
            history['test'].append(test_loss)
            history['grad_norm'].append(norm.item())
            history['lr'].append(opt.param_groups[0]['lr'])
            for ds, t in zip(train, per_ds):
                history[f'chi2/{ds.name}'].append(t.item())
            for ds, t in zip(test, test_per):
                history[f'chi2/{ds.name}'].append(t)
            for name, v in metric_vals.items():
                history[name].append(v)

            scheduler.step(loss_val)

            if loss_val < best_loss:
                best_loss = loss_val
                best_loss_step = nth_step
                best_loss_model = _cpu_snapshot(model)

            track = (metric_vals[best_metric]
                     if (not test and best_metric is not None) else test_loss)
            if track < best_test:
                best_test = track
                best_test_step = nth_step
                best_test_model = _cpu_snapshot(model)

            if nth_step % vp_check_every == 0:
                # Correctness check: VarPro chi^2 must match direct loss
                # (reported, not asserted — matches the notebook's behavior)
                vp_ok = abs(chi2_val - chi2_vp) < max(1e-3 * abs(chi2_vp), 1e-6)
                if not vp_ok:
                    print(f'WARNING step {nth_step}: VP self-check failed: '
                          f'chi2={chi2_val!r} vs varpro chi2={chi2_vp!r}')
            if nth_step % log_every == 0:
                print(nth_step,
                      f'{loss_val:.6g}', f'{test_loss:.6g}',
                      f'lr={opt.param_groups[0]["lr"]:.2e}',
                      f'c=[{" ".join(f"{v:.2e}" for v in model.c.tolist())}]',
                      f'VP_ok={vp_ok}')

    final_model = _cpu_snapshot(model)
    print(f'Best loss @ {best_loss_step} = {best_loss:.4g}')
    n_fit_train = sum(int(ds.fit_mask.sum().item()) for ds in train)
    print(f'  chi2/Ndof = {best_loss / n_fit_train:.3f}  (Ndof={n_fit_train})')
    if test:
        n_fit_test = sum(int(ds.fit_mask.sum().item()) for ds in test)
        print(f'Best test @ {best_test_step} = {best_test:.4g}')
        print(f'  chi2/Ndof = {best_test / n_fit_test:.3f}  (Ndof={n_fit_test})')
    elif best_metric is not None:
        print(f'Best {best_metric} @ {best_test_step} = {best_test:.4g}')

    return FitResult(
        best_loss_model=best_loss_model, best_loss=best_loss,
        best_loss_step=best_loss_step,
        best_test_model=best_test_model, best_test=best_test,
        best_test_step=best_test_step,
        final_model=final_model, history=history)


# ----------------------------------------------------------------------------
# Identity-covariance dataset over an ND marginal (DUNE tutorial backport)
# ----------------------------------------------------------------------------

@dataclass
class MarginalVarProDataset:
    """Unweighted least squares on a flattened ND marginal rate.

    chi^2 = m@m - 2 b@m + b@b == ((m - b)**2).sum(), the DUNE notebook MSE.
    """
    name: str
    X2: torch.Tensor          # cached ModelCalc kinematic inputs
    kun: torch.Tensor         # cached kinematic factors (n_sf leading axis)
    phi: torch.Tensor         # ND flux on the energy grid
    denu: torch.Tensor        # energy step
    dV2: torch.Tensor         # (v1, v2) cell area (make_marginal_rate arg)
    b: torch.Tensor           # (N_bins,) flattened target marginal
    fit_mask: torch.Tensor    # (N_bins,) bool, all True
    Cinv_b: torch.Tensor      # == b
    const: float              # == (b @ b).item()
    cov_inv = None

    def basis(self, model):
        G = model.forward_basis(self.X2, self.kun)   # (n_sf, N1, N2, Nenu)
        A = torch.stack([
            make_marginal_rate(G[j], self.phi, self.denu, self.dV2)
            for j in range(G.shape[0])])              # (n_sf, N1, N2)
        return A.reshape(A.shape[0], -1).T            # (N1*N2, n_sf)


def make_marginal_dataset(name, model_calc, kin_grid, phi, P, *,
                          kin_mapping, flavor=None):
    """Build a MarginalVarProDataset from a KinematicsGrid + target marginal P.

    Calls model_calc once on the grid to cache X2/kun (the generic ModelCalc
    hook both basis builders use). flavor defaults to pid.mu.
    """
    from .pdg import pid
    if flavor is None:
        flavor = pid.mu
    with torch.no_grad():
        model_calc(kin_grid.grid_enu, kin_grid.grid1, kin_grid.grid2,
                   flavor=flavor, kin_mapping=kin_mapping)
        X2 = model_calc.X2.clone()
        kun = model_calc.kun.clone()
    dev = X2.device
    phi = torch.as_tensor(phi).to(dev)
    b = torch.as_tensor(P).to(dev).flatten()
    return MarginalVarProDataset(
        name=name, X2=X2, kun=kun, phi=phi,
        denu=torch.as_tensor(kin_grid.denu).to(dev),
        dV2=torch.as_tensor(kin_grid.dV2).to(dev),
        b=b, fit_mask=torch.ones_like(b, dtype=torch.bool),
        Cinv_b=b, const=(b @ b).item())
