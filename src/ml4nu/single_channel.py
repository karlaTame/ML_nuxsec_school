"""Numerically checked FD inference helpers for the first tutorial."""

import warnings

import numpy as np
import scipy.stats
import torch

from .kinematics import make_newk2
from .utils import grab


def in_lepton_region(energy, cosine, v1_range, v2_range):
    """Use the same lepton acceptance in event generation and normalization."""
    v1, v2 = make_newk2(energy, cosine)
    return ((v1 >= v1_range[0]) & (v1 <= v1_range[1])
            & (v2 >= v2_range[0]) & (v2 <= v2_range[1]))


def shape_log_likelihood(dsigma, integral, flux, denu):
    """Normalized log density; invalid densities must never become log(1)."""
    flux = torch.as_tensor(flux, dtype=dsigma.dtype, device=dsigma.device)
    integral = torch.as_tensor(integral, dtype=dsigma.dtype, device=dsigma.device)
    norm = integral @ flux * denu
    rate = dsigma @ flux * denu
    if not torch.isfinite(norm).all() or (norm <= 0).any():
        raise ValueError("FD normalization is nonpositive or nonfinite.")
    if not torch.isfinite(rate).all() or (rate <= 0).any():
        raise ValueError("FD event density is nonpositive or nonfinite. "
                         "Check training and inference positivity; do not drop these events.")
    return rate.log() - norm.log()


def likelihood_grid(dsigma, integral, enu, phi_mu, grid_s23, grid_d31,
                    oscillation_probability, *, phi_e=None, chunk_size=256):
    """Vectorized scan; s23 is sin(theta23), not sin-squared or sin(2 theta23)."""
    if grid_s23.shape != grid_d31.shape:
        raise ValueError("Oscillation parameter grids must have identical shapes.")
    enu = torch.as_tensor(enu, device=dsigma.device, dtype=dsigma.dtype)
    phi_mu = torch.as_tensor(phi_mu, device=dsigma.device, dtype=dsigma.dtype)
    phi_e = torch.zeros_like(phi_mu) if phi_e is None else torch.as_tensor(
        phi_e, device=dsigma.device, dtype=dsigma.dtype)
    denu = enu[1] - enu[0]
    flat_s23, flat_d31 = grid_s23.flatten(), grid_d31.flatten()
    values = []
    for start in range(0, flat_s23.numel(), chunk_size):
        stop = start + chunk_size
        # NuFast allocates its output from E.shape, so explicitly broadcast E
        # to the parameter batch instead of passing a (1, N_energy) view.
        energies = enu[None, :].expand(len(flat_s23[start:stop]), -1)
        probability = oscillation_probability(
            energies, s23=flat_s23[start:stop, None],
            d31=flat_d31[start:stop, None])
        probability = torch.as_tensor(probability, device=dsigma.device)
        flux = phi_mu * probability[1, 1] + phi_e * probability[0, 1]
        values.append(shape_log_likelihood(dsigma, integral, flux.T, denu))
    return torch.cat(values, dim=1).reshape(dsigma.shape[0], *grid_s23.shape)


def bootstrap_minima(log_likelihood, weights, grid_x, grid_y, batch_size=64):
    """Find each bootstrap's optimum without allocating boots x grid x grid."""
    if log_likelihood.shape[1:] != grid_x.shape or grid_x.shape != grid_y.shape:
        raise ValueError("Likelihood and parameter grid shapes disagree.")
    likelihood = log_likelihood.flatten(1)
    weights = torch.as_tensor(weights, device=likelihood.device, dtype=likelihood.dtype)
    if weights.shape[1] != likelihood.shape[0]:
        raise ValueError("Bootstrap weights must have one column per observed event.")
    if not torch.isfinite(likelihood).all():
        raise ValueError("Cannot bootstrap nonfinite event log likelihoods.")
    indices = []
    for start in range(0, len(weights), batch_size):
        totals = weights[start:start + batch_size] @ likelihood
        indices.append(totals.argmax(1).cpu().numpy())
    indices = np.concatenate(indices)
    return np.asarray(grab(grid_x)).ravel()[indices], np.asarray(grab(grid_y)).ravel()[indices]


def boundary_counts(x, y, all_x, all_y):
    """The x=1 boundary is physical; the other scan edges are numerical."""
    return dict(x_lower=int(np.isclose(x, float(all_x[0]), atol=1e-12, rtol=0).sum()),
                x_physical_upper=int(np.isclose(x, float(all_x[-1]), atol=1e-12, rtol=0).sum()),
                y_lower=int(np.isclose(y, float(all_y[0]), atol=1e-12, rtol=0).sum()),
                y_upper=int(np.isclose(y, float(all_y[-1]), atol=1e-12, rtol=0).sum()))


def bootstrap_plot_limits(samples, injected, physical_x=(0.0, 1.0), padding=0.18):
    """Display every bootstrap estimate and the injection, with visible margins."""
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2 or samples.shape[1] != 2 or not np.isfinite(samples).all():
        raise ValueError("Expected finite bootstrap pairs with shape (N, 2).")
    lower = np.minimum(samples.min(0), injected)
    upper = np.maximum(samples.max(0), injected)
    span = np.maximum(upper - lower, [0.01, 1e-5])
    # Outer KDE contours extend beyond the most extreme observed bootstrap.
    # Include Gaussian-kernel tails as well as the empirical cloud; otherwise
    # an apparently generous scatter-plot limit can still cut the 3-sigma line.
    kernel_tail = 5 * 1.5 * len(samples) ** (-1 / 6) * samples.std(0)
    margin = np.maximum(padding * span, kernel_tail)
    lower -= margin
    upper += margin
    lower[0] = max(physical_x[0], lower[0])
    upper[0] = min(physical_x[1], upper[0])
    return (float(lower[0]), float(upper[0])), (float(lower[1]), float(upper[1]))


def bootstrap_kde(x, y, all_x, all_y, smudge=1.5):
    """Smooth bootstrap pairs, or report a degenerate smoke-run cloud honestly."""
    samples = np.stack([grab(x), grab(y)])
    try:
        kde = scipy.stats.gaussian_kde(
            samples, bw_method=lambda k: k.n ** (-1 / (k.d + 4)) * smudge)
    except (ValueError, np.linalg.LinAlgError):
        warnings.warn("Bootstrap cloud is degenerate: showing points and histograms "
                      "instead of inventing smooth confidence contours.", stacklevel=2)
        return None
    xx, yy = np.meshgrid(grab(all_x), grab(all_y), indexing="ij")
    density = kde(np.stack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    cell = float(all_x[1] - all_x[0]) * float(all_y[1] - all_y[0])
    return density / (density.sum() * cell)
