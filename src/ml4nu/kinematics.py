import torch
import numpy as np
from dataclasses import dataclass
from .pdg import constants, pid, masses


def make_newk2(emu, cost):
    v1 = torch.sqrt(emu)
    v2 = torch.sqrt(emu * (1 - cost) / 2)
    return v1, v2


def unmake_newk2(v1, v2):
    emu = v1**2
    cost = 1 - 2 * v2**2 / emu
    return emu, cost


def unmake_enuxy(enu, x, y, ml2):
    nu = y * enu
    emu = enu - nu
    q2 = 2 * x * nu * constants.M_A
    cost = (enu * emu - (q2 + ml2) / 2) / (enu * torch.sqrt(emu**2 - ml2))
    return enu, emu, cost


def make_y(x, q2, enu):
    nu = q2 / (2 * constants.M_A * x)
    return nu / enu


def unmake_id(emu, cost):
    return emu, cost


def jacobian(emu, cost):
    return 1.0


def jacobian_newk2(v1, v2):
    return 8.0 * v2 / v1


def make_kinematic_factors(
    q2, x, y,
    unique=True,
    ml2=masses[pid.mu]**2,
    enumax=float("inf"),
    M=constants.M_A,
    Vud2=constants.Vud2,
    gf2=constants.gf2,
    anti=False
):
    device = q2.device
    dtype = q2.dtype

    enu, emu, cost = KinematicMapping.unmake_q2xy(q2, x, y, ml2=ml2)

    ytilde = y * (1 + ml2 / q2)

    all_prefactors = []
    if unique:
        all_prefactors.append(enu / M)
        all_prefactors.append(ytilde)
        all_prefactors.append(torch.ones_like(ytilde))
    else:
        all_prefactors.append(ytilde)
        all_prefactors.append(enu / M * (1 - y - (q2 + ml2) / (4 * enu**2)))
        all_prefactors.append((-1 + ytilde / 2) if anti else (1 - ytilde / 2))
        all_prefactors.append((ml2 / q2) * ytilde)
        all_prefactors.append(-2 * ml2 / q2)

    prefactor = (
        Vud2 * gf2 / torch.pi
        * torch.sqrt(emu**2 - ml2)
    )

    mask = KinematicMapping.make_kin_mask(
        enu=enu,
        emu=emu,
        cost=cost,
        x=x,
        y=y,
        ml=torch.sqrt(torch.tensor(ml2, device=device, dtype=dtype)),
        enumax=enumax
    ).to(device)

    prefactor = prefactor * mask
    all_prefactors = [torch.nan_to_num(k * prefactor) for k in all_prefactors]

    return tuple(all_prefactors)


def make_kinematic_factors_newk2(
    q2, x, y,
    make_kinematic_factors=make_kinematic_factors,
    unique=True,
    ml2=masses[pid.mu]**2,
    enumax=10,
    M=constants.M_A,
    Vud2=constants.Vud2,
    gf2=constants.gf2,
    anti=False
):
    kin_fact = make_kinematic_factors(
        q2, x, y,
        unique=unique,
        ml2=ml2,
        enumax=enumax,
        M=M,
        Vud2=Vud2,
        gf2=gf2,
        anti=anti
    )

    enu, emu, cost = KinematicMapping.unmake_q2xy(q2, x, y, ml2=ml2)
    v1, v2 = make_newk2(emu, cost)
    jac = jacobian_newk2(v1, v2)

    return tuple(torch.where(k == 0, k, k * jac) for k in kin_fact)


# ── (Pz, Pt) mapping for experiments binned in muon momentum components ──
# MINERvA CC-inclusive bins ARE rectangles in (Pz, Pt) space, so integrating
# there avoids the bounding-box approximation inherent to the (v1, v2) grids.
#
# Kinematic mapping: (Pz, Pt) → (E_mu, cos_theta)
# Jacobian for d(E_mu)d(cos_theta) → d(Pz)d(Pt):
#   |∂(E_mu, cos_theta)/∂(Pz, Pt)| = sin(theta) / E_mu = Pt / (E_mu * |p|)
#
# The lepton mass defaults to the muon (the only case measured this way so
# far); an electron-flavor variant would need a factory over ml2.
# `unique` passes straight through to make_kinematic_factors, so this works
# for both the 3-SF (unique=True, single-lepton) and 5-SF (unique=False,
# multi-lepton) parameterizations.

def unmake_pzpt(pz, pt):
    """Map (Pz, Pt) → (E_mu, cos_theta_mu).  Pt is the transverse-momentum magnitude."""
    ml = masses[pid.mu]
    p2   = pz**2 + pt**2
    p    = p2**0.5
    emu  = (p2 + ml**2)**0.5
    cost = torch.where(p > 0, pz / p, torch.ones_like(pz))
    return emu, cost


def make_kinematic_factors_pzpt(q2, x, y, unique=True, ml2=masses[pid.mu]**2,
                                enumax=float('inf'), M=constants.M_A,
                                Vud2=constants.Vud2, gf2=constants.gf2, anti=False):
    """Kinematic factors for Riemann integration in (Pz, Pt) bin space.

    Equals make_kinematic_factors * |∂(E_mu, cos_theta)/∂(Pz, Pt)|.
    The Jacobian factor sin(theta)/E_mu converts the (E_mu, cos_theta) cross-
    section to the (Pz, Pt) cross-section that MINERvA actually measures.
    """
    base = make_kinematic_factors(q2, x, y, unique=unique, ml2=ml2,
                                  enumax=enumax, M=M, Vud2=Vud2, gf2=gf2, anti=anti)
    _, emu, cost = KinematicMapping.unmake_q2xy(q2, x, y, ml2)
    p         = torch.as_tensor(emu**2 - ml2).clamp(min=0)**0.5   # |p_mu|
    sin_theta = torch.as_tensor(1 - cost**2).clamp(min=0)**0.5
    jac       = torch.where(p > 0,
                            sin_theta / torch.as_tensor(emu),
                            torch.zeros_like(p))
    return tuple(torch.as_tensor(k) * jac for k in base)


class KinematicMapping:
    def __init__(self, unmake_func=unmake_id,
                 make_kinematic_factors=make_kinematic_factors,
                 jacobian=jacobian):
        self.unmake_func = unmake_func
        self.make_kinematic_factors = make_kinematic_factors
        self.jacobian = jacobian

    @staticmethod
    def make_kin_mask(enu, emu, cost, x, y, ml, enumax=float("inf")):
        nu = enu - emu
        mask = (nu >= 1e-6)
        mask &= (enu <= enumax + 1e-6)
        mask &= (cost >= -1 + 1e-6) & (cost <= 1 - 1e-6)
        mask &= (x >= 1e-6) & (x <= 1 - 1e-6)
        mask &= (y >= 1e-6) & (y <= 1 - 1e-6)
        mask &= (enu >= ml)
        mask &= (emu >= ml)
        return mask

    @staticmethod
    def make_q2xy(enu, emu, cost, ml2):
        nu = enu - emu
        q2 = 2 * (enu * emu - cost * enu * torch.sqrt(emu**2 - ml2)) - ml2
        x = q2 / (2 * constants.M_A * nu)
        y = nu / enu
        return q2, x, y

    @staticmethod
    def unmake_q2xy(q2, x, y, ml2):
        nu = q2 / (2 * constants.M_A * x)
        enu = nu / y
        emu = enu - nu
        cost = (enu * emu - (q2 + ml2) / 2) / (enu * torch.sqrt(emu**2 - ml2))
        return enu, emu, cost


# Ready-made mapping for (Pz, Pt)-binned data (defined after KinematicMapping;
# usable for data expressed in longitudinal/transverse lepton momenta).
kin_mapping_pzpt = KinematicMapping(
    unmake_func=unmake_pzpt,
    make_kinematic_factors=make_kinematic_factors_pzpt)


class KinematicsGrid:
    def __init__(self, all_var1, all_var2, all_enu, flavor,
                 kin_mapping=KinematicMapping()):
        device = all_var1.device
        dtype = all_var1.dtype

        with torch.no_grad():
            ml2 = masses[abs(flavor)]**2
            self.dim1 = all_var1.shape[0]
            self.dim2 = all_var2.shape[0]
            self.dim_enu = all_enu.shape[0]
            # Keep 1D axis arrays on CPU for plotting / numpy interop;
            # meshgrids below live on whatever device the inputs are on.
            self.all_var1 = all_var1.cpu()
            self.all_var2 = all_var2.cpu()
            self.all_enu = all_enu.cpu()

            self.grid1, self.grid2, self.grid_enu = torch.meshgrid(
                all_var1, all_var2, all_enu, indexing="ij"
            )

            self.dvar1 = (all_var1[1] - all_var1[0]
                          if self.dim1 > 1 else torch.tensor(0.0, device=device, dtype=dtype))
            self.dvar2 = (all_var2[1] - all_var2[0]
                          if self.dim2 > 1 else torch.tensor(0.0, device=device, dtype=dtype))
            self.denu = all_enu[1] - all_enu[0]

            self.dV2 = self.dvar1 * self.dvar2
            self.dV3 = self.dV2 * self.denu

            self.grid_emu, self.grid_cost = kin_mapping.unmake_func(
                self.grid1, self.grid2
            )

            self.grid_q2, self.grid_x, self.grid_y = kin_mapping.make_q2xy(
                enu=self.grid_enu,
                emu=self.grid_emu,
                cost=self.grid_cost,
                ml2=ml2
            )

            kin_mask = kin_mapping.make_kin_mask(
                enu=self.grid_enu,
                emu=self.grid_emu,
                cost=self.grid_cost,
                x=self.grid_x,
                y=self.grid_y,
                ml=torch.sqrt(torch.tensor(ml2, device=device, dtype=dtype)),
                enumax=all_enu.max()
            )

            self.grid_x = torch.nan_to_num(self.grid_x)
            self.grid_y = torch.nan_to_num(self.grid_y)
            self.kin_mask = kin_mask.float()
