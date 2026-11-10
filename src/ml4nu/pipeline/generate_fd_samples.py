"""Far-detector accept/reject sampling helper.

Faithful, plotting-free port of notebook cells 30-33. Populates ``FD_samples/``
with the 8 base-channel ``.npy`` files that ``optimize_far_detector.py`` requires.

Usage:
    python generate_fd_samples.py --seed 11

Sample counts are derived from the true FD event rates (normalized to
``num_mu_fwd`` FHC mu-, single-anchor; all other channels/modes from rel. flux).
"""

import argparse
import os

import numpy as np
import torch

from ml4nu.pipeline import rates_common as rc
from ml4nu.pdg import pid, masses
from ml4nu.sampling import Sampler, fmax, accept_reject

NBATCH = 16384  # notebook cell 32


def generate_fd_samples(setup, *, seed=11, sample_dir="FD_samples",
                        num_mu_fwd=6200):
    s = setup
    counts = rc.compute_sample_counts(s, num_mu_fwd=num_mu_fwd)   # single-anchor
    print("sample counts:", counts)
    # The exposure that MADE these samples. It must travel with them: the
    # extended FD likelihood compares ABSOLUTE counts, so an exposure taken
    # from a different flux realization than the one the samples were drawn
    # from puts a coherent rate offset into every channel (a 424242 data draw
    # vs the nominal flux is -2.4%, ~3 sigma of the combined rate constraint).
    # Recomputing it later from a nominal `setup` is exactly that bug, so
    # record it here and read it back via rates_common.load_exposure.
    exposure = float(rc.compute_exposure(s, num_mu_fwd=num_mu_fwd))
    print("exposure:", exposure)

    np.random.seed(seed)
    torch.manual_seed(seed)

    os.makedirs(sample_dir, exist_ok=True)

    # Sample inside the ANALYSIS window, so the drawn events are exactly the
    # events analyzed and num_mu_fwd is literally the mu- FHC sample size.
    # The envelope search must use the same bounds: it used to be a hardcoded
    # [m_l, 10] box, which silently stopped covering the density once ENUMAX
    # went past 10 and would have biased accept/reject.
    _lo, _hi = rc.FD_EWINDOW
    mu_range = [(_lo, _hi), (masses[pid.mu], _hi), (-1, 1)]
    e_range = [(_lo, _hi), (masses[pid.e], _hi), (-1, 1)]

    # (mode, fd_flux) -> (sampler for nu, sampler for nubar)
    for mode, fd_flux in (("fwd", s.fd_flux_fwd), ("bwd", s.fd_flux_bwd)):
        sampler = Sampler(fd_flux, s.truth, enu_window=rc.FD_EWINDOW)
        sampler_bar = Sampler(fd_flux, s.truth_bar, enu_window=rc.FD_EWINDOW)

        max_mu = fmax(f=lambda x: sampler.proposal_density(x[0], x[1], x[2], flavor=pid.numu), ranges=mu_range)
        max_e = fmax(f=lambda x: sampler.proposal_density(x[0], x[1], x[2], flavor=pid.nue), ranges=e_range)
        max_mubar = fmax(f=lambda x: sampler_bar.proposal_density(x[0], x[1], x[2], flavor=pid.numubar), ranges=mu_range)
        max_ebar = fmax(f=lambda x: sampler_bar.proposal_density(x[0], x[1], x[2], flavor=pid.nuebar), ranges=e_range)

        jobs = [
            ("mu_" + mode, sampler, pid.numu, max_mu),
            ("e_" + mode, sampler, pid.nue, max_e),
            ("mubar_" + mode, sampler_bar, pid.numubar, max_mubar),
            ("ebar_" + mode, sampler_bar, pid.nuebar, max_ebar),
        ]
        for ch, smp, flav, max_val in jobs:
            n = counts[ch]
            print("sampling", ch, "n=", n, "max_val=", max_val)
            arr = accept_reject(lambda nn, smp=smp, flav=flav: smp.propose(nn, flavor=flav),
                                n, max_val, NBATCH)
            path = os.path.join(sample_dir, "samples_FD_%s.npy" % ch)
            np.save(path, arr)
            print("saved", path)

    meta_path = os.path.join(sample_dir, rc.EXPOSURE_FILE)
    np.savez(meta_path, exposure=exposure, num_mu_fwd=num_mu_fwd,
             channels=np.array(list(counts), dtype=object),
             counts=np.array([counts[c] for c in counts], dtype=float))
    print("saved", meta_path)


def main():
    ap = argparse.ArgumentParser(description="Generate far-detector accept/reject samples.")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--sample-dir", type=str, default="FD_samples")
    ap.add_argument("--num-mu-fwd", type=int, default=6200)
    args = ap.parse_args()

    print("device:", rc.device)
    setup = rc.build_setup()
    generate_fd_samples(setup, seed=args.seed, sample_dir=args.sample_dir,
                        num_mu_fwd=args.num_mu_fwd)


if __name__ == "__main__":
    main()
