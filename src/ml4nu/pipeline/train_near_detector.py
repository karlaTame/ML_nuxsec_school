"""Near-detector model training ("function 1").

Faithful, plotting-free port of the training loop in
the original research notebook; used by ``tutorial_more_leptons.ipynb``.

Usage:
    python -m ml4nu.pipeline.train_near_detector --n-steps 3000 --lr 1e-3 --seed 17 --tag base

Produces ``models/ndmodel_nsteps{N}_lr{lr}_seed{seed}[_{tag}].pt`` containing the
best-loss state dicts for ``model`` and ``model_bar`` plus a ``meta`` dict.
"""

import argparse
import copy
import os

import numpy as np
import torch
from tqdm import tqdm

from ml4nu.pipeline import rates_common as rc
from ml4nu.pdg import pid
from ml4nu.cross_section import ModelCalc
from ml4nu.utils import make_marginal_rate, grab

LOSS_NORM = 1e11   # notebook cell 21
CLIP_NORM = np.inf  # notebook cell 8


def train_near_detector(setup, fluxes, marginals, *, N_steps=100000, lr=1e-3, seed=17,
                        n_sf=5, tag="", out_dir="models", smear=None):
    """Train model/model_bar against the 8 fluxes and 8 marginals.

    ``fluxes`` and ``marginals`` are lists of 8 tensors in ``rc.CHANNELS`` order
    (as returned by ``rc.build_training_inputs``). ``smear``, if given, is a
    dict ``channel -> (N_v1, N_v1) kernel`` (see systematics.make_smear_kernels)
    applied to the MODEL marginals each step; the data-side ``marginals`` must
    then be passed in already smeared with the same kernels so the two match.
    """
    s = setup
    km = s.kin_mapping
    phi = dict(zip(rc.CHANNELS, fluxes))
    P = dict(zip(rc.CHANNELS, marginals))

    dEnu_mu = s.kin_grid_mu.denu.item()
    dV2_mu = s.kin_grid_mu.dV2.item()
    dEnu_e = s.kin_grid_e.denu.item()
    dV2_e = s.kin_grid_e.dV2.item()

    # --- build model (cell 16) ---
    torch.manual_seed(seed)
    model = rc.SFModel(n_sf=n_sf, force_pos=False).to(s.device)
    model_bar = rc.SFModel(n_sf=n_sf, force_pos=False).to(s.device)
    opt = torch.optim.Adam(list(model.parameters()) + list(model_bar.parameters()), lr=lr)

    model_calc = ModelCalc(model, enumax=rc.ENUMAX, unique=False, anti=False, device=s.device)
    model_calc_bar = ModelCalc(model_bar, enumax=rc.ENUMAX, unique=False, anti=True, device=s.device)

    # truth dsigma used only for the (logged) test metric
    dsigma_mu, dsigma_mubar, dsigma_e, dsigma_ebar = rc._truth_dsigmas(s)

    # --- training loop (cell 22) ---
    best_loss = np.inf
    best_loss_step = 0
    best_loss_model = copy.deepcopy({k: v.clone().detach().cpu() for k, v in model.state_dict().items()})
    best_loss_model_bar = copy.deepcopy({k: v.clone().detach().cpu() for k, v in model_bar.state_dict().items()})

    with torch.enable_grad():
        for nth_step in tqdm(range(N_steps)):
            opt.zero_grad()

            if nth_step > 0 and nth_step % 10000 == 0:
                opt.param_groups[0]['lr'] *= (2 / 3)

            dsigma_mu_ = model_calc(s.kin_grid_mu.grid_enu, s.kin_grid_mu.grid1, s.kin_grid_mu.grid2, flavor=pid.mu, kin_mapping=km)
            dsigma_mubar_ = model_calc_bar(s.kin_grid_mu.grid_enu, s.kin_grid_mu.grid1, s.kin_grid_mu.grid2, flavor=pid.mubar, kin_mapping=km)
            dsigma_e_ = model_calc(s.kin_grid_e.grid_enu, s.kin_grid_e.grid1, s.kin_grid_e.grid2, flavor=pid.e, kin_mapping=km)
            dsigma_ebar_ = model_calc_bar(s.kin_grid_e.grid_enu, s.kin_grid_e.grid1, s.kin_grid_e.grid2, flavor=pid.ebar, kin_mapping=km)

            md_mu_fwd = make_marginal_rate(dsigma_mu_, phi["mu_fwd"], dEnu_mu, dV2_mu)
            md_mubar_fwd = make_marginal_rate(dsigma_mubar_, phi["mubar_fwd"], dEnu_mu, dV2_mu)
            md_e_fwd = make_marginal_rate(dsigma_e_, phi["e_fwd"], dEnu_e, dV2_e)
            md_ebar_fwd = make_marginal_rate(dsigma_ebar_, phi["ebar_fwd"], dEnu_e, dV2_e)

            md_mu_bwd = make_marginal_rate(dsigma_mu_, phi["mu_bwd"], dEnu_mu, dV2_mu)
            md_mubar_bwd = make_marginal_rate(dsigma_mubar_, phi["mubar_bwd"], dEnu_mu, dV2_mu)
            md_e_bwd = make_marginal_rate(dsigma_e_, phi["e_bwd"], dEnu_e, dV2_e)
            md_ebar_bwd = make_marginal_rate(dsigma_ebar_, phi["ebar_bwd"], dEnu_e, dV2_e)

            if smear is not None:
                # detector smearing on the model marginals (S @ md * dvar1);
                # the data targets P were smeared identically before training
                dv1_mu = s.kin_grid_mu.dvar1
                dv1_e = s.kin_grid_e.dvar1
                md_mu_fwd = smear["mu_fwd"] @ md_mu_fwd * dv1_mu
                md_mubar_fwd = smear["mubar_fwd"] @ md_mubar_fwd * dv1_mu
                md_e_fwd = smear["e_fwd"] @ md_e_fwd * dv1_e
                md_ebar_fwd = smear["ebar_fwd"] @ md_ebar_fwd * dv1_e
                md_mu_bwd = smear["mu_bwd"] @ md_mu_bwd * dv1_mu
                md_mubar_bwd = smear["mubar_bwd"] @ md_mubar_bwd * dv1_mu
                md_e_bwd = smear["e_bwd"] @ md_e_bwd * dv1_e
                md_ebar_bwd = smear["ebar_bwd"] @ md_ebar_bwd * dv1_e

            # charge ID for muons
            loss_mu_fwd = ((P["mu_fwd"] - md_mu_fwd) ** 2).sum()
            loss_mubar_fwd = ((P["mubar_fwd"] - md_mubar_fwd) ** 2).sum()
            loss_mu_bwd = ((P["mu_bwd"] - md_mu_bwd) ** 2).sum()
            loss_mubar_bwd = ((P["mubar_bwd"] - md_mubar_bwd) ** 2).sum()

            # no charge ID for electrons (e + ebar combined)
            loss_etot_fwd = (((P["e_fwd"] + P["ebar_fwd"]) - (md_e_fwd + md_ebar_fwd)) ** 2).sum()
            loss_etot_bwd = (((P["e_bwd"] + P["ebar_bwd"]) - (md_e_bwd + md_ebar_bwd)) ** 2).sum()

            loss = (loss_mu_fwd + loss_mubar_fwd + loss_etot_fwd
                    + loss_mu_bwd + loss_mubar_bwd + loss_etot_bwd)
            loss *= LOSS_NORM
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CLIP_NORM)
            opt.step()

            with torch.no_grad():
                test_loss = grab(((dsigma_mu_ - dsigma_mu) ** 2).sum().item())
                test_loss += grab(((dsigma_mubar_ - dsigma_mubar) ** 2).sum().item())
                test_loss += grab(((dsigma_e_ - dsigma_e) ** 2).sum().item())
                test_loss += grab(((dsigma_ebar_ - dsigma_ebar) ** 2).sum().item())

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_loss_step = nth_step
                best_loss_model = copy.deepcopy({k: v.clone().detach().cpu() for k, v in model.state_dict().items()})
                best_loss_model_bar = copy.deepcopy({k: v.clone().detach().cpu() for k, v in model_bar.state_dict().items()})

            if nth_step % 500 == 0:
                print(nth_step, f"{loss.item():.6g}", f"{test_loss:.6g}")

    print("Best loss @", best_loss_step, "=", best_loss)

    # --- save ---
    os.makedirs(out_dir, exist_ok=True)
    suffix = ("_" + tag) if tag else ""
    fname = "ndmodel_nsteps%d_lr%g_seed%d%s.pt" % (N_steps, lr, seed, suffix)
    path = os.path.join(out_dir, fname)
    torch.save({
        "model": best_loss_model,
        "model_bar": best_loss_model_bar,
        "meta": {"N_steps": N_steps, "lr": lr, "seed": seed, "tag": tag,
                 "n_sf": n_sf, "best_loss": best_loss, "best_loss_step": best_loss_step},
    }, path)
    print("saved", path)
    return path


# ----------------------------------------------------------------------------
# VarPro + learning-rate-scheduled variant used by Tutorial 2
# ----------------------------------------------------------------------------
# The baseline SFModel learns a per-SF normalization (_log_norm) by gradient.
# Here the model prediction is linear in a per-SF coefficient vector c, so the
# optimal c is solved analytically (variable projection) at every step from
# the (2*n_sf x 2*n_sf) normal equations of the joint marginal-matching loss;
# Adam trains only the network shape, with ReduceLROnPlateau and the minerva
# gradient clip (norm<=1) that prevents the collapse-at-LR-reset failure.
# Saved checkpoints fold c into the last linear layer, so they load and run
# through rc.load_model / the FD pipeline exactly like baseline models.

def _sf_basis(model, calc):
    """Per-SF dsigma basis G with G.sum(0) == calc.dsigma(); (n_sf, *grid)."""
    X2 = calc.X2.reshape(-1, model.n_in) * model.pre_norm
    F = model.net(X2) / torch.exp(model._log_norm)
    F = F.permute(1, 0).reshape(model.n_sf, *calc.X2.shape[:-1])
    return calc.kun * F


def _fold_c(model, c):
    """CPU state dict with c folded into the last linear layer's rows."""
    m = copy.deepcopy(model).cpu()
    with torch.no_grad():
        m.net[-1].weight.mul_(c.cpu()[:, None])
        m.net[-1].bias.mul_(c.cpu())
    return {k: v.clone().detach() for k, v in m.state_dict().items()}


def train_near_detector_varpro(setup, fluxes, marginals, *, N_steps=100000,
                               lr=1e-3, seed=17, n_sf=5, tag="",
                               out_dir="models", smear=None, clip_norm=1.0,
                               patience=2000, min_lr=1e-5, threshold=1e-2):
    """VarPro + ReduceLROnPlateau variant of train_near_detector.

    Same inputs/outputs/loss as the baseline; only the optimization differs.
    The saved .pt is a plain SFModel checkpoint (c folded into net[-1]).
    """
    s = setup
    km = s.kin_mapping
    phi = dict(zip(rc.CHANNELS, fluxes))
    P = dict(zip(rc.CHANNELS, marginals))

    dEnu_mu = s.kin_grid_mu.denu.item()
    dEnu_e = s.kin_grid_e.denu.item()
    dv1 = {"mu": s.kin_grid_mu.dvar1, "e": s.kin_grid_e.dvar1}

    torch.manual_seed(seed)
    model = rc.SFModel(n_sf=n_sf, force_pos=False).to(s.device)
    model_bar = rc.SFModel(n_sf=n_sf, force_pos=False).to(s.device)
    assert not model.force_pos, "VarPro requires the model linear in its SF outputs"
    for m in (model, model_bar):
        # normalization now lives in c (solved analytically); freeze at 1
        with torch.no_grad():
            m._log_norm.zero_()
        m._log_norm.requires_grad_(False)

    opt = torch.optim.Adam(list(model.net.parameters())
                           + list(model_bar.net.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=patience, min_lr=min_lr,
        threshold=threshold)

    # X2/kun depend only on grid+flavor: call each ModelCalc once to cache them
    calcs = {
        "mu": ModelCalc(model, enumax=rc.ENUMAX, unique=False, anti=False, device=s.device),
        "e": ModelCalc(model, enumax=rc.ENUMAX, unique=False, anti=False, device=s.device),
        "mubar": ModelCalc(model_bar, enumax=rc.ENUMAX, unique=False, anti=True, device=s.device),
        "ebar": ModelCalc(model_bar, enumax=rc.ENUMAX, unique=False, anti=True, device=s.device),
    }
    grids = {"mu": s.kin_grid_mu, "e": s.kin_grid_e,
             "mubar": s.kin_grid_mu, "ebar": s.kin_grid_e}
    flav = {"mu": pid.mu, "e": pid.e, "mubar": pid.mubar, "ebar": pid.ebar}
    nets = {"mu": model, "e": model, "mubar": model_bar, "ebar": model_bar}
    with torch.no_grad():
        for b, calc in calcs.items():
            kg = grids[b]
            calc(kg.grid_enu, kg.grid1, kg.grid2, flavor=flav[b], kin_mapping=km)
            # basis fidelity: sum of per-SF slices must equal the plain dsigma
            assert torch.allclose(_sf_basis(nets[b], calc).sum(0), calc.dsigma(),
                                  atol=0, rtol=1e-12)

    dsigma_truth = dict(zip(("mu", "mubar", "e", "ebar"), rc._truth_dsigmas(s)))
    n2 = 2 * n_sf
    dtype = calcs["mu"].X2.dtype

    def channel_rows(b, G):
        """Flattened per-SF marginal basis for every channel of base b."""
        dEnu = dEnu_mu if b in ("mu", "mubar") else dEnu_e
        out = {}
        for mode in ("fwd", "bwd"):
            ch = f"{b}_{mode}"
            M = G @ phi[ch] * dEnu                       # (n_sf, N_v1, N_v2)
            if smear is not None:
                M = torch.einsum("ab,jbc->jac", smear[ch],
                                 M) * dv1["mu" if b in ("mu", "mubar") else "e"]
            out[ch] = M.flatten(1)                        # (n_sf, N_bins)
        return out

    best_loss, best_loss_step = np.inf, 0
    best_sd = (_fold_c(model, torch.ones(n_sf)),
               _fold_c(model_bar, torch.ones(n_sf)))

    with torch.enable_grad():
        for nth_step in tqdm(range(N_steps)):
            opt.zero_grad()
            G = {b: _sf_basis(nets[b], calcs[b]) for b in calcs}
            rows = {}
            for b in calcs:
                rows.update(channel_rows(b, G[b]))

            # joint least-squares system in c = (c_model, c_bar)
            blocks, targets = [], []
            for base, mode in (("mu", "fwd"), ("mubar", "fwd"),
                               ("mu", "bwd"), ("mubar", "bwd")):
                ch = f"{base}_{mode}"
                blk = torch.zeros(rows[ch].shape[1], n2, device=s.device, dtype=dtype)
                col = 0 if base == "mu" else n_sf
                blk[:, col:col + n_sf] = rows[ch].T
                blocks.append(blk)
                targets.append(P[ch].flatten())
            for mode in ("fwd", "bwd"):
                e, eb = rows[f"e_{mode}"], rows[f"ebar_{mode}"]
                blocks.append(torch.cat([e.T, eb.T], dim=1))
                targets.append((P[f"e_{mode}"] + P[f"ebar_{mode}"]).flatten())
            A = torch.cat(blocks)                        # (N_obs, 2*n_sf)
            bvec = torch.cat(targets)

            with torch.no_grad():
                H = A.T @ A
                ridge = 1e-12 * H.diagonal().mean()
                c = torch.linalg.solve(
                    H + ridge * torch.eye(n2, device=s.device, dtype=dtype),
                    A.T @ bvec)

            # chi^2 at the VarPro-optimal c; gradient flows through A only
            loss = ((A @ c - bvec) ** 2).sum() * LOSS_NORM

            # snapshot BEFORE opt.step(): c was solved for the CURRENT weights,
            # and folding it into next step's weights loses the correspondence
            # (the minerva notebook re-solves c post hoc for the same reason)
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_loss_step = nth_step
                best_sd = (_fold_c(model, c[:n_sf]), _fold_c(model_bar, c[n_sf:]))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(model_bar.parameters()),
                max_norm=clip_norm)
            opt.step()
            scheduler.step(loss.item())

            with torch.no_grad():
                test_loss = sum(
                    (((G[b] * c[(0 if b in ("mu", "e") else n_sf)
                                + torch.arange(n_sf, device=s.device),
                                None, None, None]).sum(0)
                      - dsigma_truth[b]) ** 2).sum().item()
                    for b in ("mu", "mubar", "e", "ebar"))

            if nth_step % 500 == 0:
                print(nth_step, f"{loss.item():.6g}", f"{test_loss:.6g}",
                      f"lr={opt.param_groups[0]['lr']:.2e}")

    print("Best loss @", best_loss_step, "=", best_loss)

    os.makedirs(out_dir, exist_ok=True)
    suffix = ("_" + tag) if tag else ""
    fname = "ndmodel_nsteps%d_lr%g_seed%d%s.pt" % (N_steps, lr, seed, suffix)
    path = os.path.join(out_dir, fname)
    torch.save({
        "model": best_sd[0],
        "model_bar": best_sd[1],
        "meta": {"N_steps": N_steps, "lr": lr, "seed": seed, "tag": tag,
                 "n_sf": n_sf, "best_loss": best_loss,
                 "best_loss_step": best_loss_step, "trainer": "varpro"},
    }, path)
    print("saved", path)
    return path


def main():
    ap = argparse.ArgumentParser(description="Train the near-detector cross-section model.")
    ap.add_argument("--n-steps", type=int, default=100000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--n-sf", type=int, default=5)
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--out-dir", type=str, default="models")
    ap.add_argument("--varpro", action=argparse.BooleanOptionalAction, default=True,
                    help="VarPro-normalized, ReduceLROnPlateau-scheduled trainer "
                         "(analytic output coefficients); "
                         "checkpoint format unchanged; --no-varpro restores baseline")
    args = ap.parse_args()

    print("device:", rc.device)
    setup = rc.build_setup()
    fluxes, marginals = rc.build_training_inputs(setup)
    trainer = train_near_detector_varpro if args.varpro else train_near_detector
    trainer(setup, fluxes, marginals, N_steps=args.n_steps, lr=args.lr,
            seed=args.seed, n_sf=args.n_sf, tag=args.tag, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
