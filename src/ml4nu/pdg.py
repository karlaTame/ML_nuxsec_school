from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class constants:
    Vud2: float = 1
    gf2: float = 1
    M_A: float = 1
    #M_A: float = 0.938
    #M_A: float = 35.3


@dataclass(frozen=True)
class pid:
    e: int = 11
    nue: int = 12
    mu: int = 13
    numu: int = 14
    tau: int = 15
    nutau: int = 16
    ebar: int = -11
    nuebar: int = -12
    mubar: int = -13
    numubar: int = -14
    taubar: int = -15
    nutaubar: int = -16


def lepton_flavor(flavor: int):
    if abs(flavor) % 2 == 1:
        return flavor
    else:
        return np.sign(flavor) * (abs(flavor) - 1)


masses = {
    pid.e: 0.0005109989461,
    pid.mu: 0.10565837,
    pid.tau: 1.77686,
}
