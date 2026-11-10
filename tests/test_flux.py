import numpy as np

from ml4nu.flux import FluxNearDetector, FluxFarDetector
from ml4nu.pdg import pid


def test_flux_near_detector(global_datadir):
    flux_file = (global_datadir / "histos_g4lbne_v3r5p4_QGSP_BERT_OptimizedEngineeredNov2017_neutrino_LBNEND_globes_flux.txt")

    # Create a FluxNearDetector object
    nd_flux = FluxNearDetector(flux_file)

    # Test the initialization
    assert isinstance(nd_flux, FluxNearDetector)
    assert nd_flux._flux is not None
    assert nd_flux._energies is not None
    assert nd_flux.force_emax == np.inf
    assert nd_flux.force_emin == 0


def test_flux_far_detector(global_datadir):
    flux_file = (global_datadir / "histos_g4lbne_v3r5p4_QGSP_BERT_OptimizedEngineeredNov2017_neutrino_LBNEND_globes_flux.txt")

    # Create a FluxNearDetector object
    nd_flux = FluxNearDetector(flux_file)

    # Create a FluxFarDetector object
    fd_flux = FluxFarDetector(nd_flux=nd_flux)

    # Test the initialization
    assert isinstance(fd_flux, FluxFarDetector)
    assert fd_flux._nd_flux is not None
    assert fd_flux._osc_params["s12"] == 0.554076
    assert fd_flux._osc_params["s23"] == 0.748999
    assert fd_flux._osc_params["s13"] == 0.148155
    assert fd_flux._osc_params["dCP"] == 3.08923
    assert fd_flux._osc_params["d21"] == 7.49e-5
    assert fd_flux._osc_params["d31"] == 2.534e-3

    assert fd_flux._baseline == 1300
