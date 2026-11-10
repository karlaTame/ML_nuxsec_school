import torch
import numpy as np
import scipy


def get_default_device():
    """Return CUDA if available, else CPU. Called at construction time, not import time."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------
# Utilities
# -------------------------

def grab(x):
    """
    Safe conversion helper:
    - torch.Tensor -> CPU numpy
    - numpy / python -> unchanged
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


# -------------------------
# Histogram / bin utilities
# -------------------------

def make_symm_bin_edges(x):
    x = grab(x)
    return np.concatenate([
        [x[0] - (x[1] - x[0]) / 2],
        (x[1:] + x[:-1]) / 2,
        [x[-1] + (x[-1] - x[-2]) / 2],
    ])


# -------------------------
# Bootstrap utilities
# (CPU by construction; physics/statistics unchanged)
# -------------------------

def precompute_bootstrap_tensor(nsamps, nboots, nsub=None, weights=None,
                                poisson=False, rng=None):
    """Bootstrap weight matrix, (nboots, nsamps), normalized by ``nsub``.

    Downstream the per-replicate event sum is ``nsub * (ll * bs).sum(-1)``,
    i.e. ``sum_i counts_i * ll_i``, which stays correct whether or not the
    replicate total equals ``nsub``.

    poisson=False (default): every replicate draws exactly ``nsub`` events,
    so the sample SIZE is frozen. That is right for a shape-only
    likelihood, where counts carry no information.

    poisson=True: the replicate total is first drawn as
    ``Poisson(nsub)``. REQUIRED when the likelihood uses rate information
    (fd_optimize_torch.LIKELIHOOD == "rate"): otherwise the observed
    inter-sample ratios are identical in every replicate, the rate
    constraint carries no statistical fluctuation, and the resulting
    confidence intervals are spuriously tight.
    """
    if weights is None:
        weights = np.ones(nsamps)
    if nsub is None:
        nsub = nsamps
    draw = rng if rng is not None else np.random

    weights = np.asarray(weights, dtype=np.float64)
    weights /= weights.sum()

    samp_ids = np.arange(nsamps)
    bs_tensor = np.zeros((nboots, nsamps))
    if poisson:
        totals = draw.poisson(nsub, size=nboots)
        for bs_id, n_b in enumerate(totals):
            if n_b == 0:
                continue
            ids, counts = np.unique(
                draw.choice(samp_ids, p=weights, size=int(n_b), replace=True),
                return_counts=True)
            bs_tensor[bs_id, ids] = counts
    else:
        bs_samp_ids = draw.choice(
            samp_ids, p=weights, size=(nboots, nsub), replace=True
        )
        for bs_id in range(nboots):
            ids, counts = np.unique(bs_samp_ids[bs_id], return_counts=True)
            bs_tensor[bs_id, ids] = counts

    return bs_tensor / nsub


def fast_boostrap_average(data):
    data = grab(data)
    nbs, *shape = data.shape
    data = data.reshape(nbs, -1)

    bs_mu = np.median(data, axis=0)
    bs_corr = np.corrcoef(data, rowvar=False)

    bdry = (0.5 - 0.341344746) * 100
    bs_dl = bs_mu - np.percentile(data, bdry, axis=0)
    bs_dr = np.percentile(data, 100 - bdry, axis=0) - bs_mu

    spreads = np.max(
        np.stack([bs_dl, bs_dr], axis=0),
        axis=0,
    )

    bs_cov = bs_corr * np.einsum("a,b->ab", spreads, spreads)

    return (
        bs_mu.reshape(shape),
        bs_cov.reshape(shape + shape),
    )


def fast_boostrap_average_decorr(data):
    data = grab(data)
    nbs, *shape = data.shape
    data = data.reshape(nbs, -1)

    bs_mu = np.median(data, axis=0)

    bdry = (0.5 - 0.341344746) * 100
    bs_dl = bs_mu - np.percentile(data, bdry, axis=0)
    bs_dr = np.percentile(data, 100 - bdry, axis=0) - bs_mu

    spreads = np.max(
        np.stack([bs_dl, bs_dr], axis=0),
        axis=0,
    )

    return bs_mu.reshape(shape), spreads.reshape(shape)


# -------------------------
# Resampling
# -------------------------

def resample_hist(P, n):
    P = grab(P)
    p = P.ravel() / P.sum()
    dist = scipy.stats.multinomial(p=p, n=n)
    return dist.rvs().reshape(P.shape)


# -------------------------
# Grid spacing utilities
# -------------------------

def safe_make_dx(x):
    assert len(x.shape) == 1
    dx = x[1:] - x[:-1]

    if isinstance(dx, torch.Tensor):
        dx0 = dx[0]
        assert torch.allclose(dx, dx0, rtol=1e-5, atol=1e-5)
        return dx0
    else:
        dx0 = dx[0]
        assert np.allclose(dx, dx0)
        return dx0


def safe_make_dx_2(x):
    return safe_make_dx(x)


# -------------------------
# Neural network helper
# -------------------------

def make_mlp(nin, nout, H, depth):
    layers = [
        torch.nn.Linear(nin, H),
        torch.nn.LeakyReLU(),
    ]
    for _ in range(depth):
        layers += [
            torch.nn.Linear(H, H),
            torch.nn.LeakyReLU(),
        ]
    layers.append(torch.nn.Linear(H, nout))
    return torch.nn.Sequential(*layers)


# -------------------------
# Physics-preserving marginals
# -------------------------

def make_marginal(dsigma, phi, dEnu, dV2):
    """
    GPU-safe, torch-only.
    Physics unchanged.
    """
    device = dsigma.device
    phi  = torch.as_tensor(phi,  device=device)
    dEnu = torch.as_tensor(dEnu, device=device)
    dV2  = torch.as_tensor(dV2,  device=device)

    Y = dsigma @ phi * dEnu
    Y = Y / (Y.sum() * dV2)
    return Y


def make_marginal_rate(dsigma, phi, dEnu, dV2):
    """
    GPU-safe, torch-only.
    Physics unchanged.
    """
    device = dsigma.device
    phi  = torch.as_tensor(phi,  device=device)
    dEnu = torch.as_tensor(dEnu, device=device)

    return dsigma @ phi * dEnu
