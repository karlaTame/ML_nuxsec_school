import numpy as np
import torch
import h5py
from scipy.interpolate import RegularGridInterpolator
from .pdg import constants, pid, masses


def load_h5_as_dict(fout, h5f=None):
    if h5f is None:
        with h5py.File(fout, "r") as h5f:
            return load_h5_as_dict(fout, h5f=h5f)
    d = {}
    for k, v in h5f.items():
        if isinstance(v, h5py.Dataset):
            d[k] = v[...]
        elif isinstance(v, h5py.Group):
            d[k] = load_h5_as_dict(fout, h5f=v)
    return d


class PDFInterpolator:

    def __init__(self, grid_fn, clamp_q2=False, clamp_x=False, log=False):
        self.log = log
        self.grid_fn = grid_fn

        self.grid_x, self.grid_q2, self.grid_xf, self.all_x, self.all_q2 = \
            PDFInterpolator._load_pdf_grid(grid_fn, log)

        xf2, xf3 = self.grid_xf

        def _make_interp(f):
            return RegularGridInterpolator(
                (self.all_x, self.all_q2),
                f,
                bounds_error=False,
                fill_value=np.nan,
                method="linear",
            )

        self.interp_xf2 = _make_interp(xf2)
        self.interp_xf3 = _make_interp(xf3)

        self.clamp_q2 = clamp_q2
        self.max_q2 = self.all_q2.max()

        self.clamp_x = clamp_x
        self.min_x = self.all_x.min()

    @staticmethod
    def _load_pdf_grid(grid_fn, log=False):
        prefix = "log" if log else ""
        if grid_fn.endswith(".npy"):
            d = np.load(grid_fn, allow_pickle=True).item()
        elif grid_fn.endswith(".h5"):
            d = load_h5_as_dict(grid_fn)

        grid_x = d[prefix + "x"]
        grid_q2 = d[prefix + "q2"]
        grid_xf = d[prefix + "xf"]

        all_x = np.ascontiguousarray(grid_x[:, 0])
        all_q2 = np.ascontiguousarray(grid_q2[0, :])

        assert np.all(all_x[:, None] == grid_x)
        assert np.all(all_q2[None, :] == grid_q2)

        return grid_x, grid_q2, grid_xf, all_x, all_q2

    def _fix_vars_cpu(self, x_np, q2_np):
        if self.clamp_q2:
            q2_np = np.where(q2_np > self.max_q2, self.max_q2, q2_np)
        if self.clamp_x:
            x_np = np.where(x_np < self.min_x, self.min_x, x_np)
        if self.log:
            x_np = np.log(x_np)
            q2_np = np.log(q2_np)
        return x_np, q2_np

    def _interp(self, interp_fn, x, q2):
        device = x.device
        x_np = x.detach().cpu().numpy()
        q2_np = q2.detach().cpu().numpy()

        x_np, q2_np = self._fix_vars_cpu(x_np, q2_np)
        out_np = interp_fn((x_np, q2_np))

        return torch.as_tensor(out_np, device=device)

    def xf2(self, x, q2):
        return self._interp(self.interp_xf2, x, q2)

    def xf3(self, x, q2):
        return self._interp(self.interp_xf3, x, q2)

    def calc_all(self, x, q2):
        return self.xf2(x, q2), self.xf3(x, q2)


class StructureFunctions:
    """
    Fully torch/GPU-compatible structure functions.
    Physics unchanged.
    """

    def __init__(self, pdf, device="cpu"):
        self.pdf = pdf
        self.device = torch.device(device)

    def calc_all(self, x, q2):
        x = torch.as_tensor(x, device=self.device)
        q2 = torch.as_tensor(q2, device=self.device)

        xf2, xf3 = self.pdf.calc_all(x, q2)

        # q2fac is applied ONCE to every SF. Do NOT write this as
        # `W1 = xf2; W5 = xf2; W1 *= q2fac; W5 *= q2fac`: those alias the
        # SAME tensor, so the in-place multiplies land twice and W1/W5 come
        # out with q2fac**2 while W2/W3 get it once. That aliasing bug was
        # live from commit ee1505b until 2026-10-08 and silently defined the
        # toy cross section used by every campaign before that date
        # (tests/test_structure_functions.py::test_q2fac_applied_once pins
        # this). Out-of-place arithmetic keeps the five independent.
        q2fac = q2 / (q2 + 1)
        W1 = xf2 * q2fac
        W2_ = 4 * x / q2 * xf2 * q2fac
        W3 = 2 * xf3 * 0.1 * q2fac
        W4 = torch.zeros_like(x)
        W5 = xf2 * q2fac

        return W1, W2_, W3, W4, W5

    def calc_unique(self, x, q2, M=constants.M_A,
                    ml2=masses[pid.mu]**2, anti=False):
        """Single-lepton-mass unique combinations [W2, Wy, Wc] of the five
        SFs (the 3-SF basis paired with unique=True kinematic factors:
        (Enu/M)*W2, ytilde*Wy, 1*Wc). Built on calc_all so it inherits the
        current SF conventions. Restored for TruthCalc(unique=True) — the
        paper-1 tutorial path; the DUNE pipeline uses unique=False and never
        calls this. `anti` is accepted for the TruthCalc interface; the
        combination itself is flavor-symmetric here.
        """
        W1, W2_, W3, W4, W5 = self.calc_all(x, q2)
        W2 = M**2 * W2_
        Wy = W1 - (x / 2) * W2 - W3 / 2 + (ml2 / q2) * W4
        Wc = W3 - 2 * (ml2 / q2) * W5 - q2 / (2 * M**2 * x) * W2
        return torch.stack([W2_, Wy, Wc])
