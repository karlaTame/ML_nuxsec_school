import numpy as np
import torch
from tqdm.auto import tqdm

from .pdg import masses, pid
from .utils import grab, safe_make_dx
from .kinematics import KinematicMapping


def find_min_1d(all_p, ll):
    idx_min = grab(ll).argmin(axis=1)
    return np.take_along_axis(grab(all_p), idx_min, axis=0)


def find_min_2d(grid_x, grid_y, ll):
    assert grid_x.shape == grid_y.shape, (grid_x.shape, grid_y.shape)
    grid_shape = tuple(grid_x.shape)
    assert tuple(ll.shape[1:]) == grid_shape, (ll.shape, grid_shape)
    grid_x, grid_y = grid_x.ravel(), grid_y.ravel()
    ll = ll.flatten(1)
    idx_min = grab(ll.argmin(1))
    x_min = np.take_along_axis(grab(grid_x), idx_min, axis=0)
    y_min = np.take_along_axis(grab(grid_y), idx_min, axis=0)
    return x_min, y_min

def find_min_3d(grid_x, grid_y, grid_z, ll):
    assert grid_x.shape == grid_y.shape == grid_z.shape, (grid_x.shape, grid_y.shape, grid_z.shape)
    grid_shape = tuple(grid_x.shape)
    assert tuple(ll.shape[1:]) == grid_shape, (grid_shape, ll.shape)
    grid_x, grid_y, grid_z = grid_x.ravel(), grid_y.ravel(), grid_z.ravel()
    ll = ll.flatten(1)
    idx_min = grab(ll.argmin(1))
    x_min = np.take_along_axis(grab(grid_x), idx_min, axis=0)
    y_min = np.take_along_axis(grab(grid_y), idx_min, axis=0)
    z_min = np.take_along_axis(grab(grid_z), idx_min, axis=0)
    return x_min, y_min, z_min


def make_prior(x, x_true, width):
    return 1/2 * ((x - x_true)/width)**2


def get_grid_min_coords(G, x, y):
    ix, iy = np.where(G == G.min())
    ix, iy = ix[0], iy[0]
    return x[ix], y[iy]


# TODO: make independent of kinematic variables
# What is I? -- total cross section, "I"

def make_I(calculator, all_enu, all_var1, all_var2, flavor=pid.mu, kin_mapping=KinematicMapping()):
    ml2 = masses[abs(flavor)]**2
    dvar1 = all_var1[1] - all_var1[0]
    dvar2 = all_var2[1] - all_var2[0]
    grid_var1, grid_var2 = torch.meshgrid(all_var1, all_var2, indexing='ij')
    I = []
    for enu in tqdm(all_enu):
        ds = torch.nan_to_num(calculator(enu, grid_var1, grid_var2, flavor=flavor, kin_mapping=kin_mapping)) #enumax=enumax
        device = ds.device
        I.append(ds.sum() * dvar1.to(device) * dvar2.to(device))
    I = torch.stack(I)  # preserves device (CUDA if calculator is on GPU)
    return I

# TODO: make independent of kinematic variables
def make_dsigma_samp(calculator, all_enu, samp_v1, samp_v2,
                     flavor=pid.mu, kin_mapping=KinematicMapping()):
    dsigma_samp = []
    for enu in tqdm(all_enu):
        ds = torch.nan_to_num(calculator(enu, samp_v1, samp_v2, flavor=flavor, kin_mapping=kin_mapping))
        dsigma_samp.append(ds)
    dsigma_samp = torch.stack(dsigma_samp, dim=1)  # ~ (samp, Enu)
    return dsigma_samp


# WARNING: safe_log doesn't work well for grids, need safe_kl from make_exact_kl_curve
def safe_log(x):
    return torch.nan_to_num(x.log())

def make_log_likelihood(dsigma_samp, I, f, denu):
    device = dsigma_samp.device
    I    = I.to(device)
    f    = torch.as_tensor(f,    device=device)
    denu = torch.as_tensor(denu, device=device)
    norm = (I * f).sum() * denu
    samp = (dsigma_samp @ f) * denu
    return safe_log(samp) - safe_log(norm)  # NOTE: Not minus ll!
