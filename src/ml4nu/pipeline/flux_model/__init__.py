"""DUNE ND flux uncertainty model (see README.md in this directory)."""
import os

from .flux_model import NDFluxModel

_HERE = os.path.dirname(os.path.abspath(__file__))
COVARIANCE_FILE = os.path.join(_HERE, "dune_flux_covariance.npz")
MEANS_FILE = os.path.join(_HERE, "dune_nd_flux_means.npz")


def load_default(detectors=("ND", "FD")):
    """Flux model from the data files shipped here.

    Defaults to the JOINT ND+FD covariance (2026-10-09): one nuisance vector
    shifts both detectors through their real correlations. Pass
    detectors=("ND",) for the historical ND-only behavior, which implicitly
    treats the near/far flux as perfectly correlated.
    """
    return NDFluxModel(COVARIANCE_FILE, MEANS_FILE, detectors=detectors)
