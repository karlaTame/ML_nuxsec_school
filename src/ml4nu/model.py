"""Structure-function model with VarPro linear coefficients.

Used by tutorial.ipynb. Per-SF linear norms live in a registered buffer
``c`` that VarPro solves analytically each step.

Relationship to the production pipeline's ``rates_common.SFModel`` (which
stays untouched while the paper is in production): that class keeps a
trainable ``_log_norm`` parameter instead of ``c``; the two agree exactly
when ``c == exp(-_log_norm)`` (see tests/test_varpro_dune.py, which pins
this equivalence). A later unification would land here as a
``from_rates_state_dict`` classmethod.

SF count is configuration, not architecture: ``n_sf=3`` pairs with
``unique=True`` kinematic factors (single-lepton data in the
muon-disappearance tutorial); ``n_sf=5`` pairs with ``unique=False``
(+ ``anti`` for the antiparticle model) as required when fitting multiple
lepton species (the DUNE 8-channel pipeline). The ``unique``/``anti``
choice lives in ModelCalc / the kinematic-factor functions, never here.
"""

import copy

import torch

from .utils import make_mlp


class SFModel(torch.nn.Module):
    def __init__(self, n_sf, H=64, depth=3, force_pos=False, pre_norm=1):
        super().__init__()
        self.n_sf = n_sf
        self.n_in = 2
        self.net = make_mlp(nin=self.n_in, nout=n_sf, H=H, depth=depth)
        self.pre_norm = pre_norm
        self.force_pos = force_pos
        # Linear norm coefficients per structure function (one per SF).
        # NOT updated by the optimizer — solved analytically by VarPro at
        # each step.  Stored as a buffer so state_dict saves/loads them.
        self.register_buffer('c', torch.ones(n_sf))

    def _net_out(self, X2):
        """Shared MLP forward (no c). Returns (total_pts, n_sf)."""
        X2 = X2.reshape(-1, self.n_in) * self.pre_norm
        F = self.net(X2)
        if self.force_pos:
            F = F ** 2
        return F

    def forward(self, X2, K, return_sf=False, return_dsigma=True):
        *shape, check_n_in = X2.shape
        assert self.n_in == check_n_in, (X2.shape, check_n_in)
        F = self._net_out(X2)                     # (total_pts, n_sf)
        F = F * self.c.unsqueeze(0)               # multiply by linear norms
        F = F.permute(1, 0)                        # (n_sf, total_pts)
        F = F.reshape(self.n_sf, *shape)           # (n_sf, *shape)
        if return_dsigma:
            assert tuple(F.shape) == tuple(K.shape)
            dsigma = (K * F).sum(0)
        if return_sf and return_dsigma:
            return dsigma, F
        elif return_sf:
            return F
        elif return_dsigma:
            return dsigma

    def forward_basis(self, X2, K):
        """Per-SF dsigma WITHOUT c multiplier.

        Returns G of shape (n_sf, *X2.shape[:-1]).
        Each slice G[j] = K[j] * net_j(X2) is the contribution of the
        j-th structure function before the linear norm is applied.
        """
        *shape, _ = X2.shape
        F = self._net_out(X2)                     # (total_pts, n_sf), no c
        F = F.permute(1, 0).reshape(self.n_sf, *shape)
        return K * F                               # (n_sf, *shape)


def make_sf_model(n_sf=3, H=64, depth=3, seed=17, pre_norm=1,
                  force_pos=False, device=None, dtype=torch.float64):
    """Seed torch then construct an SFModel, reproducing the notebook init.

    ``torch.manual_seed(seed)`` immediately followed by construction under a
    ``dtype`` default reproduces the notebook's cell-10 initialization
    bit-exactly (Linear-layer init consumes the RNG in dtype-dependent
    order, so the default dtype is set for the construction and restored
    after — no global state leaks out of this call).
    """
    old_dtype = torch.get_default_dtype()
    torch.manual_seed(seed)
    try:
        torch.set_default_dtype(dtype)
        model = SFModel(n_sf=n_sf, H=H, depth=depth,
                        force_pos=force_pos, pre_norm=pre_norm)
    finally:
        torch.set_default_dtype(old_dtype)
    if device is not None:
        model = model.to(device)
    return model


def unique_sf_combination(F, x, q2, ml2=None, M=None):
    """Map 5-SF fields to the single-lepton unique 3-SF frame [W2_, Wy, Wc].

    F : (n_sf, *shape) SF fields (include the c normalization first if you
    want physical combinations); x, q2 : broadcastable to *shape. For
    n_sf == 3 the input is already the unique frame and passes through.
    The reduction is exact for a fixed lepton mass (pinned to 1.6e-15
    against TruthCalc in the 2026-08 rung-0 check); coefficients follow
    StructureFunctions.calc_unique. ml2 defaults to the muon.
    """
    from .pdg import pid, masses, constants
    if ml2 is None:
        ml2 = masses[pid.mu] ** 2
    if M is None:
        M = constants.M_A
    if F.shape[0] == 3:
        return F
    W1, W2_, W3, W4, W5 = F
    W2 = M ** 2 * W2_
    sq2 = torch.where(q2 == 0, torch.ones_like(q2), q2)
    sx = torch.where(x == 0, torch.ones_like(x), x)
    Wy = W1 - (x / 2) * W2 - W3 / 2 + (ml2 / sq2) * W4
    Wc = W3 - 2 * (ml2 / sq2) * W5 - sq2 / (2 * M ** 2 * sx) * W2
    return torch.stack([W2_, Wy, Wc])


def fold_c(model, c=None):
    """CPU state dict with c folded into net[-1] rows and c reset to ones.

    Mirror of ``train_near_detector._fold_c`` (pinned equivalent by
    tests/test_varpro_dune.py): scales the last linear layer's rows by c so
    the checkpoint runs as a plain net with unit norms. ``c`` defaults to
    the model's own buffer.
    """
    if c is None:
        c = model.c
    m = copy.deepcopy(model).cpu()
    with torch.no_grad():
        m.net[-1].weight.mul_(c.cpu()[:, None])
        m.net[-1].bias.mul_(c.cpu())
        m.c.fill_(1.0)
    return {k: v.clone().detach() for k, v in m.state_dict().items()}
