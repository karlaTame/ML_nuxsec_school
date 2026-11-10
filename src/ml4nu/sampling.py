import torch
import scipy
from tqdm.auto import tqdm
from .pdg import pid, lepton_flavor
from .nufast import compute_osc_prob
from .utils import get_default_device
import numpy as np

# Resolved once at import time; GPU if available, else CPU.
device = get_default_device()


def make_cdf(x, pdf):
    assert x.shape == pdf.shape
    assert len(x.shape) == 1
    cdf = torch.cumsum(torch.tensor(pdf, device=device), dim=0)
    return cdf / cdf[-1]


def sample_cdf(x, cdf, n):
    x = torch.tensor(x, device=device) if not torch.is_tensor(x) else x.to(device)
    cdf = torch.tensor(cdf, device=device) if not torch.is_tensor(cdf) else cdf.to(device)
    cdf = cdf / cdf[-1]  # normalize

    r = torch.rand(n, device=device)  # random samples in [0,1)
    idx = torch.searchsorted(cdf, r)  # find index in CDF

    # Clamp indices to valid range
    idx = torch.clamp(idx, 0, len(x) - 1)
    return x[idx]


class Sampler:
    def __init__(self, flux, compute_dsigma, osc_params=None, enu_window=None):
        self.flux = flux
        self.compute_dsigma = compute_dsigma
        self.osc_params = osc_params
        # Restrict the proposal to an (lo, hi) neutrino-energy window. The
        # flux's own rvs() samples its whole support (to 125 GeV for the
        # DUNE tables), so without this the accepted sample is bounded only
        # by the calculator's enumax mask -- which is an ND-side choice.
        self.enu_window = enu_window

    def proposal_density(self, enu, emu, cost, *, flavor=pid.mu):
        phi = 1
        if self.osc_params is not None:
            phi = compute_osc_prob(enu, self.osc_params)  # keep physics unchanged
        q = 1 / enu / 2
        p = self.compute_dsigma(enu, emu, cost, flavor=lepton_flavor(flavor))
        p = p.to(device) if torch.is_tensor(p) else torch.tensor(p, device=device)
        phi = torch.tensor(phi, device=device) if not torch.is_tensor(phi) else phi.to(device)
        out = p * phi / q
        if self.enu_window is not None:
            lo, hi = self.enu_window
            e = enu if torch.is_tensor(enu) else torch.tensor(enu, device=out.device)
            out = torch.where((e >= lo) & (e <= hi), out, torch.zeros_like(out))
        return out

    def propose(self, nsamples, *, flavor):
        enu = torch.tensor(self.flux.rvs(flavor, nsamples), device=device)
        emu = torch.rand(nsamples, device=device) * enu
        cost = 2 * torch.rand(nsamples, device=device) - 1

        pq = self.proposal_density(enu, emu, cost, flavor=flavor)
        samples = torch.stack([enu, emu, cost], dim=1)
        return samples, pq


def fmax(f, ranges):
    def _f_scalar(x):
        val = f(x)
        if isinstance(val, torch.Tensor):
            val = val.detach().cpu().item()
        return float(val)
    return -scipy.optimize.differential_evolution(lambda x: -_f_scalar(x), ranges).fun


def accept_reject(func, nsamples, max_val, nbatch):
    samples = torch.zeros((nsamples, 3), device=device)
    i0 = 0
    nit = 0
    ar = []

    max_val = torch.tensor(max_val, device=device)

    with tqdm(total=nsamples) as pbar:
        while i0 < nsamples:
            nit += 1
            x, fsamples = func(nbatch)  # x, fsamples are device tensors
            r = torch.rand(nbatch, device=device)
            accepted = r < fsamples / max_val
            naccepted = accepted.sum().item()
            ar.append(accepted.float().mean().item())
            i1 = min(i0 + naccepted, nsamples)
            samples[i0:i1] = x[accepted][:i1-i0]
            i0 = i1
            pbar.update(naccepted)

    print('AR', np.mean(ar), '+/-', np.std(ar))
    print('n_iterations', nit)
    return samples.cpu().numpy()  # final output as NumPy for legacy code
