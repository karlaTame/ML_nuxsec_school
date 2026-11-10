"""
DUNE near-detector flux model.

An ND flux model with a mean flux and a fractional flux covariance for the eight
channels  horn{FHC,RHC} x flavor{numu,numubar,nue,nuebar}.  It can be queried and
sampled at arbitrary neutrino energies: means are linearly interpolated point
estimates of the tabulated flux, and the coarse-binned fractional covariance is
carried to those energies by the congruence  C_fine = M @ C @ M.T, where M is a
per-channel (block-diagonal) linear interpolation operator with linear
extrapolation beyond the tabulated range.  Because C_fine is a congruence
transform of a PSD matrix it is PSD for any requested energies, and the
per-channel block structure guarantees no interpolation ever mixes energies
across channels.

All numbers (bin edges, covariance, means) are read from the .npz data files;
nothing is hardcoded here.  Depends only on numpy.
"""

import numpy as np

__all__ = ["NDFluxModel"]


class NDFluxModel:
    def __init__(self, covariance_file, means_file, detectors=("ND",)):
        """Fractional flux uncertainty model.

        detectors: which detector blocks of the 136-bin covariance to keep.
          ("ND",)        -- ND sub-block only (the pre-2026-10-09 default).
                            Channels are labelled "FHC_numu" etc.
          ("ND", "FD")   -- the FULL matrix, so one nuisance vector shifts the
                            near and far fluxes together THROUGH THEIR ACTUAL
                            CORRELATIONS rather than identically. Channels are
                            labelled "ND_FHC_numu" / "FD_FHC_numu".

        Using only the ND block and reusing those modes for the FD flux (what
        the pipeline did until 2026-10-09) asserts a perfectly correlated
        near/far flux. The released matrix says matched ND/FD bins correlate
        at ~0.94, not 1, so that assumption removes a genuine ~3% residual FD
        flux uncertainty that no ND measurement can constrain away.
        """
        cd = np.load(covariance_file, allow_pickle=True)
        det = cd["detector"].astype(str)
        detectors = tuple(detectors)
        keep = np.isin(det, detectors)
        if not keep.any():
            raise ValueError(f"no covariance bins for detectors={detectors}; "
                             f"file has {sorted(set(det.tolist()))}")
        C = np.asarray(cd["cov"], float)[np.ix_(keep, keep)]
        horn = cd["horn"].astype(str)[keep]
        flav = cd["flavor"].astype(str)[keep]
        elo = np.asarray(cd["E_low"], float)[keep]
        ehi = np.asarray(cd["E_high"], float)[keep]
        self.detectors = detectors
        if len(detectors) == 1:                       # back-compatible labels
            label = np.array([f"{h}_{f}" for h, f in zip(horn, flav)])
        else:
            label = np.array([f"{d}_{h}_{f}"
                              for d, h, f in zip(det[keep], horn, flav)])

        self.channels = list(dict.fromkeys(label.tolist())) # unique, order preserved (plain str)
        self._cov = C
        self._slice = {}                                    # channel -> columns in C
        self._nodes = {}                                    # channel -> interp nodes (bin centers)
        self._range = {}                                    # channel -> (Emin_edge, Emax_edge)
        for ch in self.channels:
            idx = np.where(label == ch)[0]
            self._slice[ch] = slice(idx[0], idx[-1] + 1)
            self._nodes[ch] = 0.5 * (elo[idx] + ehi[idx])
            self._range[ch] = (float(elo[idx].min()), float(ehi[idx].max()))

        # symmetric PSD square root for sampling: C = V diag(w) V^T,  L = V sqrt(w)
        w, V = np.linalg.eigh(0.5 * (C + C.T))
        # Pin each eigenvector's sign (largest-|component| positive) so a FIXED
        # draw (e.g. the b_data=424242 data flux) is REPRODUCIBLE across LAPACK/
        # numpy builds. eigh leaves column signs arbitrary and environment-
        # dependent; unpinned, the same seed gives a ~1% different flux
        # realization on a different machine. (b_data=0 is immune, so datanone
        # matched while data424242 drifted.) Convention-only: C = L L^T is
        # unchanged, so sampling statistics are identical.
        _piv = np.argmax(np.abs(V), axis=0)
        _sgn = np.sign(V[_piv, np.arange(V.shape[1])])
        _sgn[_sgn == 0] = 1.0
        V = V * _sgn
        self._L = V * np.sqrt(np.clip(w, 0.0, None))

        # --- means (tabulated flux vs energy, per channel) ---
        md = np.load(means_file, allow_pickle=True)
        self._mE = np.asarray(md["energy"], float)
        mflux = np.asarray(md["flux"], float)
        mlab = [f"{h}_{f}" for h, f in zip(md["horn"].astype(str), md["flavor"].astype(str))]
        self._mean = {lab: mflux[i] for i, lab in enumerate(mlab)}
        self.mean_units = str(md["units"]) if "units" in md.files else ""

    # ------------------------------------------------------------------ #
    #  energy handling
    # ------------------------------------------------------------------ #
    def _resolve(self, Enu):
        """Enu may be one array (shared by all channels) or {channel: array}."""
        if isinstance(Enu, dict):
            unknown = set(Enu) - set(self.channels)
            if unknown:
                raise KeyError(f"unknown channel(s): {sorted(unknown)}")
            return {c: np.atleast_1d(np.asarray(Enu[c], float))
                    for c in self.channels if c in Enu}
        arr = np.atleast_1d(np.asarray(Enu, float))
        return {c: arr for c in self.channels}

    @staticmethod
    def _weights(E, nodes):
        """Linear-interpolation weights with linear extrapolation past the ends.

        Row i has (at most) two nonzero entries summing to 1.  For E inside the
        node range this is ordinary linear interpolation; outside, the nearest
        segment is extended (weights leave [0,1], which is correct linear
        extrapolation and remains PSD-safe under the congruence).
        """
        n = np.asarray(nodes, float)
        m = n.size
        W = np.zeros((E.size, m))
        if m == 1:
            W[:, 0] = 1.0
            return W
        k = np.clip(np.searchsorted(n, E, side="right") - 1, 0, m - 2)
        t = (E - n[k]) / (n[k + 1] - n[k])
        rows = np.arange(E.size)
        W[rows, k] = 1.0 - t
        W[rows, k + 1] = t
        return W

    def _build_M(self, enu):
        """Block-diagonal interpolation operator (sum(len) x n_nodes) and row layout."""
        total = sum(v.size for v in enu.values())
        M = np.zeros((total, self._cov.shape[0]))
        layout, start = {}, 0
        for ch, E in enu.items():
            M[start:start + E.size, self._slice[ch]] = self._weights(E, self._nodes[ch])
            layout[ch] = slice(start, start + E.size)
            start += E.size
        return M, layout

    def _mean_at(self, ch, E):
        # The means file is ND-only; a detector-prefixed channel falls back to
        # the matching ND spectrum. Means are not used by the nuisance-basis
        # path (_build_M @ _L), only by mean()/sample(fractional=False).
        key = ch if ch in self._mean else ch.split("_", 1)[1]
        return np.interp(E, self._mE, self._mean[key])      # linear, clamped past ends

    # ------------------------------------------------------------------ #
    #  public API
    # ------------------------------------------------------------------ #
    def energy_range(self, channel):
        """(Emin, Emax) edges over which the covariance is tabulated (beyond -> extrapolated)."""
        return self._range[channel]

    def mean(self, Enu):
        """Mean flux at the requested energies. Returns {channel: array}."""
        return {ch: self._mean_at(ch, E) for ch, E in self._resolve(Enu).items()}

    def stddev(self, Enu, fractional=True):
        """Per-bin standard deviation. fractional=True -> delta_phi/phi; else absolute flux."""
        enu = self._resolve(Enu)
        M, layout = self._build_M(enu)
        var = np.einsum("ij,jk,ik->i", M, self._cov, M)     # diag(M C M^T)
        out = {}
        for ch, E in enu.items():
            s = np.sqrt(np.clip(var[layout[ch]], 0.0, None))
            out[ch] = s if fractional else s * self._mean_at(ch, E)
        return out

    def covariance(self, Enu, fractional=True):
        """Joint covariance at the requested energies.

        Returns (C, layout): C is the (N x N) covariance over all requested
        (channel, energy) points concatenated in channel order; layout maps each
        channel to its slice of rows/cols.  fractional=False rescales to absolute
        flux covariance by the mean flux at each point.
        """
        enu = self._resolve(Enu)
        M, layout = self._build_M(enu)
        C = M @ self._cov @ M.T
        C = 0.5 * (C + C.T)
        if not fractional:
            mu = np.concatenate([self._mean_at(ch, E) for ch, E in enu.items()])
            C = C * np.outer(mu, mu)
        return C, layout

    def sample(self, Enu, size=1, rng=None, fractional=False, clip_negative=True):
        """Draw flux realizations at the requested energies.

        Samples a joint fractional shift delta ~ N(0, M C M^T) (correctly
        correlated across channels and energies) using the coarse-matrix square
        root, then returns, per channel, an array of shape (size, n_energies).
        fractional=True returns the shifts delta_phi/phi; otherwise absolute flux
        phi*(1+delta) (clipped at zero by default).
        """
        enu = self._resolve(Enu)
        M, layout = self._build_M(enu)
        rng = np.random.default_rng() if rng is None else rng
        xi = rng.standard_normal((self._L.shape[1], size))
        dfrac = M @ (self._L @ xi)                           # (N, size)
        out = {}
        for ch, E in enu.items():
            d = dfrac[layout[ch]].T                          # (size, n_E)
            if fractional:
                out[ch] = d
            else:
                f = self._mean_at(ch, E) * (1.0 + d)
                out[ch] = np.clip(f, 0.0, None) if clip_negative else f
        return out
