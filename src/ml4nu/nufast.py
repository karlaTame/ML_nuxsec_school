# Probability_Matter_LBL obtained from: https://github.com/PeterDenton/NuFast/blob/main/py/NuFast.py
import torch
from dataclasses import dataclass, asdict
# import numpy as np

# Some constants
eVsqkm_to_GeV_over4 = 1e-9 / 1.97327e-7 * 1e3 / 4
YerhoE2a = 1.52588e-4


# Probability_Matter_LBL calculates all nine oscillation probabilities including
# the matter effect in an optimized, fast, and efficient way. The precision can
# be controlled with N_Newton. For many applications N_Newton=0 may be enough,
# but many years of DUNE or HK-LBL may require N_Newton=1. This code may be
# suitable for atmospheric neutrinos. The code is standalone with the Parameters
# module for several constants.
#
# Inputs:
#   mixing angles (usual parameterization)
#   phase (usual parameterization) make Dmsq31 positive/negative for the NO/IO
#   Delta msq's (eV^2)
#   L (km)
#   E (GeV) positive for neutrinos, negative for antineutrinos
#   rho (g/cc)
#   Ye: electron fraction, typically around 0.5
#   N_Newton: number of Newton's method iterations to do. should be zero, one, two (or higher)
# Outputs:
#   probs_returned is all nine oscillation probabilities: e.g. probs_returned[1][0] is mu->e


def Probability_Matter_LBL(
    s12sq, s13sq, s23sq, delta, Dmsq21, Dmsq31, L, E,
    rho=3.0, Ye=0.5, N_Newton=0, device="cpu", dtype=torch.float64
):
    # ------------------------------------------------------------------
    # Ensure torch tensors
    # ------------------------------------------------------------------
    E = torch.as_tensor(E, device=device, dtype=dtype)
    s12sq = torch.as_tensor(s12sq, device=device, dtype=dtype)
    s13sq = torch.as_tensor(s13sq, device=device, dtype=dtype)
    s23sq = torch.as_tensor(s23sq, device=device, dtype=dtype)
    delta = torch.as_tensor(delta, device=device, dtype=dtype)
    Dmsq21 = torch.as_tensor(Dmsq21, device=device, dtype=dtype)
    Dmsq31 = torch.as_tensor(Dmsq31, device=device, dtype=dtype)

    # ------------------------------------------------------------------
    # Mixing quantities
    # ------------------------------------------------------------------
    c13sq = 1.0 - s13sq

    Ue2sq = c13sq * s12sq
    Ue3sq = s13sq

    Um3sq = c13sq * s23sq
    Ut2sq = s13sq * s12sq * s23sq
    Um2sq = (1.0 - s12sq) * (1.0 - s23sq)

    Jrr = torch.sqrt(Um2sq * Ut2sq)
    sind = torch.sin(delta)
    cosd = torch.cos(delta)

    Um2sq = Um2sq + Ut2sq - 2.0 * Jrr * cosd
    Jmatter = 8.0 * Jrr * c13sq * sind

    Amatter = Ye * rho * E * YerhoE2a
    Dmsqee = Dmsq31 - s12sq * Dmsq21

    # ------------------------------------------------------------------
    # Polynomial coefficients
    # ------------------------------------------------------------------
    A = Dmsq21 + Dmsq31
    See = A - Dmsq21 * Ue2sq - Dmsq31 * Ue3sq
    Tmm = Dmsq21 * Dmsq31
    Tee = Tmm * (1.0 - Ue3sq - Ue2sq)
    C = Amatter * Tee
    A = A + Amatter

    # ------------------------------------------------------------------
    # Lambda_3 (DMP)
    # ------------------------------------------------------------------
    xmat = Amatter / Dmsqee
    tmp = 1.0 - xmat
    lambda3 = (
        Dmsq31
        + 0.5 * Dmsqee
        * (xmat - 1.0 + torch.sqrt(tmp * tmp + 4.0 * s13sq * xmat))
    )

    # Newton refinement
    B = Tmm + Amatter * See
    for _ in range(N_Newton):
        lambda3 = (
            lambda3 * lambda3 * (2.0 * lambda3 - A) + C
        ) / (
            lambda3 * (3.0 * lambda3 - 2.0 * A) + B
        )

    # ------------------------------------------------------------------
    # Eigenvalue splittings
    # ------------------------------------------------------------------
    tmp = A - lambda3
    Dlambda21 = torch.sqrt(tmp * tmp - 4.0 * C / lambda3)

    lambda2 = 0.5 * (A - lambda3 + Dlambda21)
    Dlambda32 = lambda3 - lambda2
    Dlambda31 = Dlambda32 + Dlambda21

    # ------------------------------------------------------------------
    # Mixing in matter
    # ------------------------------------------------------------------
    PiDlambdaInv = 1.0 / (Dlambda31 * Dlambda32 * Dlambda21)
    Xp3 = PiDlambdaInv * Dlambda21
    Xp2 = -PiDlambdaInv * Dlambda31

    Ue3sq = (lambda3 * (lambda3 - See) + Tee) * Xp3
    Ue2sq = (lambda2 * (lambda2 - See) + Tee) * Xp2

    Smm = A - Dmsq21 * Um2sq - Dmsq31 * Um3sq
    Tmm = Tmm * (1.0 - Um3sq - Um2sq) + Amatter * (See + Smm - A)

    Um3sq = (lambda3 * (lambda3 - Smm) + Tmm) * Xp3
    Um2sq = (lambda2 * (lambda2 - Smm) + Tmm) * Xp2

    Jmatter = (
        Jmatter
        * Dmsq21
        * Dmsq31
        * (Dmsq31 - Dmsq21)
        * PiDlambdaInv
    )

    Ue1sq = 1.0 - Ue3sq - Ue2sq
    Um1sq = 1.0 - Um3sq - Um2sq

    Ut3sq = 1.0 - Um3sq - Ue3sq
    Ut2sq = 1.0 - Um2sq - Ue2sq
    Ut1sq = 1.0 - Um1sq - Ue1sq

    # ------------------------------------------------------------------
    # Oscillation phases
    # ------------------------------------------------------------------
    Lover4E = eVsqkm_to_GeV_over4 * L / E

    D21 = Dlambda21 * Lover4E
    D32 = Dlambda32 * Lover4E

    sinD21 = torch.sin(D21)
    sinD31 = torch.sin(D32 + D21)
    sinD32 = torch.sin(D32)

    triple_sin = sinD21 * sinD31 * sinD32

    sinsqD21_2 = 2.0 * sinD21 * sinD21
    sinsqD31_2 = 2.0 * sinD31 * sinD31
    sinsqD32_2 = 2.0 * sinD32 * sinD32

    # ------------------------------------------------------------------
    # Probabilities
    # ------------------------------------------------------------------
    Pme_CPC = (
        (Ut3sq - Um2sq * Ue1sq - Um1sq * Ue2sq) * sinsqD21_2
        + (Ut2sq - Um3sq * Ue1sq - Um1sq * Ue3sq) * sinsqD31_2
        + (Ut1sq - Um3sq * Ue2sq - Um2sq * Ue3sq) * sinsqD32_2
    )
    Pme_CPV = -Jmatter * triple_sin

    Pmm = 1.0 - 2.0 * (
        Um2sq * Um1sq * sinsqD21_2
        + Um3sq * Um1sq * sinsqD31_2
        + Um3sq * Um2sq * sinsqD32_2
    )

    Pee = 1.0 - 2.0 * (
        Ue2sq * Ue1sq * sinsqD21_2
        + Ue3sq * Ue1sq * sinsqD31_2
        + Ue3sq * Ue2sq * sinsqD32_2
    )

    # ------------------------------------------------------------------
    # Stack output (3×3×...)
    # ------------------------------------------------------------------
    probs = torch.zeros((3, 3) + E.shape, device=device, dtype=dtype)

    probs[0, 0] = Pee
    probs[0, 1] = Pme_CPC - Pme_CPV
    probs[0, 2] = 1.0 - probs[0, 0] - probs[0, 1]

    probs[1, 0] = Pme_CPC + Pme_CPV
    probs[1, 1] = Pmm
    probs[1, 2] = 1.0 - probs[1, 0] - probs[1, 1]

    probs[2, 0] = 1.0 - probs[0, 0] - probs[1, 0]
    probs[2, 1] = 1.0 - probs[0, 1] - probs[1, 1]
    probs[2, 2] = 1.0 - probs[0, 2] - probs[1, 2]

    return probs




@dataclass
class OscillationParams:
    # NOTE: these built-in defaults are STALE (NuFit 6.0-era). The pipeline passes
    # explicit params (rates_common.OSC_PARAMS, NuFit 6.1); anything relying on
    # these defaults is a silent-mismatch bug, so __post_init__ warns loudly.
    s12: float = 0.554076
    s23: float = 0.748999
    s13: float = 0.148155
    dCP: float = 3.08923
    d21: float = 7.49e-5
    d31: float = 2.534e-3

    def __post_init__(self):
        flds = OscillationParams.__dataclass_fields__
        if all(getattr(self, k) == flds[k].default for k in flds):
            import warnings
            warnings.warn(
                "nufast.OscillationParams: falling back to built-in DEFAULT oscillation "
                "parameters (stale NuFit-6.0 values). Pass explicit osc_params "
                "(e.g. rates_common.OSC_PARAMS, NuFit 6.1) to avoid a data/fit mismatch.",
                stacklevel=2)

    def to_dict(self):
        return asdict(self)


def compute_osc_prob(enu: torch.Tensor, *, osc_params: OscillationParams = None,
                     baseline: float = 1300, device='cpu', **kwargs):
    if osc_params is None:
        osc_params = asdict(OscillationParams())
    osc_params = {**osc_params, **kwargs}
    return Probability_Matter_LBL(osc_params["s12"]**2, osc_params["s13"]**2,
                                  osc_params["s23"]**2, osc_params["dCP"],
                                  osc_params["d21"], osc_params["d31"], baseline,
                                  enu, device=device)
