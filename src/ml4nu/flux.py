import pandas as pd
from . import nufast
from .sampling import make_cdf, sample_cdf
from .pdg import pid
import scipy
import numpy as np
import torch


MAPPING = {
    pid.nue: "phi(nue)", pid.numu: "phi(numu)", pid.nutau: "phi(nutau)",
    -pid.nue: "phi(nuebar)", -pid.numu: "phi(numubar)", -pid.nutau: "phi(nutaubar)"
}


class FluxNearDetector:
    def __init__(self, filename: str, force_emax: float = np.inf,
                 force_emin: float = 0, cdf_res: int = 16384):
        self._flux = pd.read_csv(
            filename, sep=' ', index_col=False,
            names=["enu", "phi(nue)", "phi(numu)",
                   "phi(nutau)", "phi(nuebar)",
                   "phi(numubar)", "phi(nutaubar)"]
        )
        self._energies = self._flux["enu"].to_numpy()
        self.force_emax = force_emax
        self.force_emin = force_emin

        self._integral = {}
        self._cdf_res = cdf_res

    def __call__(self, flavor: int, energy):
        """
        CPU-only ND flux evaluation.
        Accepts numpy or torch input.
        """

        if isinstance(energy, torch.Tensor):
            energy = energy.detach().cpu().numpy()

        col = MAPPING[flavor]
        flux = self._flux[col].to_numpy()

        phi = np.interp(energy, xp=self._energies, fp=flux)
        phi = np.where(energy < self.emin, 0.0, phi)
        phi = np.where(energy > self.emax, 0.0, phi)

        return phi

    def integral(self, flavor: int):
        if flavor not in self._integral:  # cache
            self._integral[flavor] = scipy.integrate.quad(
                lambda E: self(flavor, E),
                self.emin, self.emax,
                limit=500,
                epsabs=1e-12
            )[0]
        return self._integral[flavor]

    @property
    def emin(self):
        return np.maximum(self._energies[0], self.force_emin)

    @property
    def emax(self):
        return np.minimum(self._energies[-1], self.force_emax)

    def pdf(self, flavor, enu):
        return self(flavor, enu) / self.integral(flavor)

    def rvs(self, flavor, npts):
        enu = np.linspace(self.emin, self.emax, self._cdf_res)
        pdf = self.pdf(flavor, enu)
        cdf = make_cdf(enu, pdf)
        samps = sample_cdf(enu, cdf, npts)
        return samps


class FluxFarDetector:
    def __init__(self, *, nd_flux: FluxNearDetector,
                 osc_params: nufast.OscillationParams = None,
                 osc_flavors: list = [pid.nue, pid.numu, pid.nutau],
                 baseline: float = 1300,
                 unosc_flux: FluxNearDetector = None,
                 cdf_res: int = 16384, **kwargs):
        # unosc_flux: the UNOSCILLATED FAR-detector flux table. If omitted the
        # ND flux is used, which assumes an energy-independent near/far ratio
        # -- false at the 20-70% level across 0.5-8 GeV (see
        # rates_common.FLUX_FILE_FD_FWD). Prefer passing the LBNEFD tables.
        self._baseline = baseline
        self._nd_flux = unosc_flux if unosc_flux is not None else nd_flux
        self._flux = self._nd_flux._flux
        self._energies = self._flux["enu"].to_numpy()
        self.force_emax = nd_flux.force_emax
        self.force_emin = nd_flux.force_emin

        if osc_params is None:
            osc_params = nufast.OscillationParams().to_dict()
        osc_params = {**osc_params, **kwargs}
        self._osc_params = osc_params

        self.osc_flavors = osc_flavors
        self._integral = {}
        self._cdf_res = cdf_res

    # Update oscillation parameters and reset the integral cache for now
    # TODO: Is it worthwhile to store osc_params in a cache?
    def update_osc_params(self, *, osc_params: nufast.OscillationParams = None,
                          **kwargs):
        if osc_params is None:
            osc_params = nufast.OscillationParams()
        osc_params = {**osc_params, **kwargs}
        self._osc_params = osc_params
        self._integral = {}

    def __call__(self, flavor: int, energy):
        """
        CUDA-safe wrapper:
        - Accepts torch.Tensor (CPU or CUDA) or numpy array
        - Forces NUFAST + ND flux evaluation on CPU
        - Returns torch.Tensor iff input was torch.Tensor
        """

        if isinstance(energy, torch.Tensor):
            energy_cpu = energy.detach().cpu().numpy()
            return_torch = True
            device = energy.device
            dtype = energy.dtype
        else:
            energy_cpu = energy
            return_torch = False

        sign = np.sign(flavor)

        # Oscillation probabilities (CPU / numpy). NuFast convention: E > 0
        # for neutrinos, E < 0 for antineutrinos -- the sign flips the matter
        # potential and delta_CP. (Previously +E was used for all flavors,
        # so antineutrino samples carried neutrino-like appearance
        # probabilities; the fit path uses -E, biasing truth fits.)
        probs = nufast.compute_osc_prob(
            sign * energy_cpu,
            osc_params=self._osc_params,
            baseline=self._baseline
        )

        if isinstance(probs, torch.Tensor):
            probs = probs.detach().cpu().numpy()

        electron_flux = 0.0
        muon_flux = 0.0
        tau_flux = 0.0

        # ND fluxes (FORCED numpy)
        if pid.nue in self.osc_flavors:
            electron_flux = self._nd_flux(sign * pid.nue, energy_cpu)
        if pid.numu in self.osc_flavors:
            muon_flux = self._nd_flux(sign * pid.numu, energy_cpu)
        if pid.nutau in self.osc_flavors:
            tau_flux = self._nd_flux(sign * pid.nutau, energy_cpu)

        # Flavor projection
        if abs(flavor) == pid.nue:
            out = (
                electron_flux * probs[0, 0]
              + muon_flux     * probs[1, 0]
              + tau_flux      * probs[2, 0]
            )
        elif abs(flavor) == pid.numu:
            out = (
                electron_flux * probs[0, 1]
              + muon_flux     * probs[1, 1]
              + tau_flux      * probs[2, 1]
            )
        elif abs(flavor) == pid.nutau:
            out = (
                electron_flux * probs[0, 2]
              + muon_flux     * probs[1, 2]
              + tau_flux      * probs[2, 2]
            )

        if return_torch:
            return torch.as_tensor(out, device=device, dtype=dtype)

        return out

    def integral(self, flavor: int):
        # CPU-only by definition (scipy.integrate.quad)
        if flavor not in self._integral:
            self._integral[flavor] = scipy.integrate.quad(
                lambda E: self(flavor, E),
                self.emin, self.emax,
                limit=500,
                epsabs=1e-12
            )[0]
        return self._integral[flavor]

    @property
    def emin(self):
        return self._nd_flux.emin

    @property
    def emax(self):
        return self._nd_flux.emax

    def pdf(self, flavor, enu):
        return self(flavor, enu) / self.integral(flavor)

    def rvs(self, flavor, npts):
        enu = np.linspace(self.emin, self.emax, self._cdf_res)
        pdf = self.pdf(flavor, enu)
        cdf = make_cdf(enu, pdf)
        samps = sample_cdf(enu, cdf, npts)
        return samps
