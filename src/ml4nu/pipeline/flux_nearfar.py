"""Near/far flux extrapolation as a conditional Gaussian.

The forward-fold fit needs two things from the flux model: a basis of ND
flux perturbations to fit against ND data, and a rule for turning the
fitted ND perturbation into an FD prediction. The joint 136-mode
eigenbasis (``flux_model.load_default()._L``) supplies both at once, but
badly: its response to ND data has rank 68, not 136, so half the
parameters are ND-blind and whatever the solve puts in that subspace is
set by the regularizer rather than by the physics (``fwdfold_shot``'s
truth path leans on a 1e-9 relative ridge for exactly this reason).

The right object is the conditional distribution of the FD flux given the
ND flux. Writing the joint covariance of the fractional flux deviations
as

    C = [[C_nd , C_nf],
         [C_fn , C_fd]]

the FD deviation splits into a part the ND determines and a part it
cannot:

    delta_fd = B delta_nd + eps,   B    = C_fn C_nd^-1
                                   eps ~ N(0, C_fd|nd)
    C_fd|nd = C_fd - B C_nf

Both pieces are full rank 68 for the DUNE matrix, so the degeneracy is
gone -- not damped, gone. The nuisance parameter becomes a unit-normal
``u`` of length 68 via the Cholesky factor of ``C_nd``, giving a prior of
exactly N(0,I) (so a prior penalty has an unambiguous scale), and the FD
response of that same ``u`` is ``M_fd B chol(C_nd)``.

For the DUNE ND+FD matrix the split is (see ``conditional_summary``):

    FD fractional flux uncertainty, unconditional : 6.1% - 14.3% (med 8.4%)
                          residual given the ND   : 0.23% - 2.86% (med 0.94%)

i.e. the ND pins down ~89% of the FD flux and ~1% is irreducible. Reusing
ND modes for the FD flux asserted that residual was zero; using the
FD-only block would assert the ND says nothing and overstate it ~9x.

Note that the matched-bin ND/FD *correlation* (median 0.94) is not the
regression coefficient (median 0.37 on the diagonal of ``B``): a shift in
one ND bin is redistributed over many correlated FD bins, so only the
full matrix is right.
"""
import numpy as np


class NearFarFluxModel:
    """Conditional near/far flux extrapolation built from a joint model.

    Takes a joint ND+FD ``flux_model.NDFluxModel`` (i.e. one loaded with
    ``detectors=("ND", "FD")``) and exposes the ND fit basis, the FD
    response of the same nuisances, and the irreducible FD residual.
    """

    def __init__(self, flux_model, jitter=0.0):
        chans = list(flux_model.channels)
        nd = [c for c in chans if c.startswith("ND_")]
        fd = [c for c in chans if c.startswith("FD_")]
        if not nd or not fd:
            raise ValueError(
                "NearFarFluxModel needs a JOINT model: load it with "
                f"detectors=('ND','FD'). Got channels {chans}")
        self._fm = flux_model
        # base names ("FHC_numu", ...) shared by the two detectors; the
        # conditional only needs the blocks, but a base present at one
        # detector and not the other would silently drop out of B.
        self._bases = [c[3:] for c in nd]
        missing = set(self._bases) - {c[3:] for c in fd}
        if missing:
            raise ValueError(f"channels present at the ND but not the FD: {missing}")

        sl = flux_model._slice
        self._nd_cols = np.concatenate([np.arange(sl[c].start, sl[c].stop) for c in nd])
        self._fd_cols = np.concatenate([np.arange(sl[c].start, sl[c].stop) for c in fd])
        # column index WITHIN the ND (FD) block, per channel
        self._nd_local = {c: np.searchsorted(self._nd_cols,
                                             np.arange(sl[c].start, sl[c].stop)) for c in nd}

        C = np.asarray(flux_model._cov, dtype=float)
        C_nd = C[np.ix_(self._nd_cols, self._nd_cols)]
        C_fd = C[np.ix_(self._fd_cols, self._fd_cols)]
        C_fn = C[np.ix_(self._fd_cols, self._nd_cols)]
        if jitter:
            C_nd = C_nd + jitter * np.trace(C_nd) / C_nd.shape[0] * np.eye(C_nd.shape[0])

        self.C_nd, self.C_fd, self.C_fn = C_nd, C_fd, C_fn
        # B solves B C_nd = C_fn, i.e. the least-squares regression of the FD
        # deviation on the ND one. Solved, never inverted explicitly.
        self.B = np.linalg.solve(C_nd, C_fn.T).T
        C_cond = C_fd - self.B @ C_fn.T
        self.C_cond = 0.5 * (C_cond + C_cond.T)              # kill asymmetric roundoff
        self.chol_nd = np.linalg.cholesky(C_nd)
        self.n_nd = C_nd.shape[0]
        self.n_fd = C_fd.shape[0]

    # ---- bases -------------------------------------------------------
    def _M(self, enu_by_base, det, cols):
        """Rows of the bin->energy interpolation for `det`, on `cols`."""
        enu = {f"{det}_{b}": np.asarray(E) for b, E in enu_by_base.items()}
        M, layout = self._fm._build_M(enu)
        M = M[:, cols]
        return {b: M[layout[f"{det}_{b}"]] for b in enu_by_base}

    def nd_basis(self, enu_by_base):
        """ND fractional-shift response of a unit-normal u, per channel.

        ``(n_E, n_nd)``; ``M_nd @ chol(C_nd)``, so ``F @ u`` with
        ``u ~ N(0,I)`` reproduces the ND flux covariance exactly. Full
        rank, unlike the ND response of the joint eigenmodes.
        """
        return {b: M @ self.chol_nd for b, M in
                self._M(enu_by_base, "ND", self._nd_cols).items()}

    def fd_basis(self, enu_by_base):
        """FD response of the SAME u: ``M_fd @ B @ chol(C_nd)``.

        This is the whole point -- the FD flux is predicted from the
        fitted ND nuisances through the real near/far regression, not by
        reusing the ND modes (which asserts perfect correlation) and not
        from an independent FD draw (which asserts none).
        """
        return {b: M @ self.B @ self.chol_nd for b, M in
                self._M(enu_by_base, "FD", self._fd_cols).items()}

    def fd_residual_basis(self, enu_by_base, rcond=1e-12):
        """FD response of the irreducible residual eps ~ N(0, C_fd|nd).

        ``(n_E, k)`` for a unit-normal coefficient vector. Use this to
        draw the part of the FD flux no ND measurement can constrain.
        """
        w, V = np.linalg.eigh(self.C_cond)
        keep = w > rcond * w.max()
        R = V[:, keep] * np.sqrt(w[keep])
        return {b: M @ R for b, M in
                self._M(enu_by_base, "FD", self._fd_cols).items()}

    # ---- what the ND fit can actually see ----------------------------
    def observable_subspace(self, nd_basis, rcond=1e-10):
        """Split u into the part the ND fit determines and the part it cannot.

        ``nd_basis`` is the stacked ND basis on the ACTUAL training grid
        (``np.concatenate(list(nd_basis(...).values()))``). Returns
        ``(Q, Qperp)``, orthonormal rows spanning the identifiable and
        blind subspaces of u.

        This is not a formality. The covariance's flux bins run to 20 GeV
        while the pipeline trains on ENUMAX = 10, so **8 of the 68
        directions are invisible to the ND fit** purely for lack of
        energy coverage -- and they still move the FD prediction, because
        B couples high-E ND bins to FD bins at all energies. Because u is
        unit-normal and Q is orthonormal, the identifiable and blind
        coordinates are independent N(0,I), so leaving the blind ones at
        0 (which zero gradient does by itself, and which the prior also
        prescribes) makes ``B chol u`` exactly the conditional mean. What
        must NOT be forgotten is their contribution to the residual --
        see ``residual_covariance``.
        """
        F = np.asarray(nd_basis, dtype=float)
        sv = np.linalg.svd(F, compute_uv=False)
        k = int((sv > rcond * sv[0]).sum())
        Vt = np.linalg.svd(F, full_matrices=True)[2]
        return Vt[:k], Vt[k:]

    def residual_covariance(self, nd_basis=None, rcond=1e-10):
        """Irreducible FD flux covariance given what the ND fit observes.

        With ``nd_basis=None`` this is ``C_fd|nd``, the best case in which
        every ND bin is measured. Passing the stacked ND basis on the real
        training grid adds the FD response of the ND-blind directions,
        which for the pipeline's ENUMAX = 10 grid takes the per-bin
        residual from (median 0.94%, max 2.86%) to (median 1.07%, max
        4.00%).
        """
        if nd_basis is None:
            return self.C_cond
        _, Qperp = self.observable_subspace(nd_basis, rcond)
        if Qperp.size == 0:
            return self.C_cond
        X = self.B @ self.chol_nd @ Qperp.T
        return self.C_cond + X @ X.T

    # ---- diagnostics -------------------------------------------------
    def u_from_bin_shift(self, delta_nd):
        """The u reproducing a given ND bin-space fractional shift."""
        return np.linalg.solve(self.chol_nd, np.asarray(delta_nd, dtype=float))

    def nd_block_of(self, joint_vec, L):
        """ND bin-space shift of a joint 136-mode coefficient vector."""
        return (np.asarray(L) @ np.asarray(joint_vec))[self._nd_cols]

    def fd_block_of(self, joint_vec, L):
        """FD bin-space shift of a joint 136-mode coefficient vector."""
        return (np.asarray(L) @ np.asarray(joint_vec))[self._fd_cols]

    def conditional_summary(self):
        """Per-bin FD uncertainty before/after conditioning on the ND."""
        s_fd = np.sqrt(np.diag(self.C_fd))
        s_cond = np.sqrt(np.clip(np.diag(self.C_cond), 0.0, None))
        rho = np.diag(self.C_fn) / (np.sqrt(np.diag(self.C_nd)) * s_fd)
        return dict(sigma_fd=s_fd, sigma_cond=s_cond, shrinkage=s_cond / s_fd,
                    matched_rho=rho, B_diag=np.diag(self.B))


def load_default(**kw):
    """NearFarFluxModel over the packaged joint ND+FD covariance."""
    from ml4nu.pipeline import flux_model
    return NearFarFluxModel(flux_model.load_default(detectors=("ND", "FD")), **kw)
