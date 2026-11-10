import torch
from .pdg import constants, pid, masses
from .kinematics import KinematicMapping
from .utils import get_default_device


# ===============================
# K · F utilities (GPU-safe)
# ===============================

def _to_tensor_list(xs, device):
    return [
        x.to(device) if isinstance(x, torch.Tensor)
        else torch.as_tensor(x, device=device)
        for x in xs
    ]


def K_times_F(kinematics, structure_function):
    """
    Pure torch, GPU-safe.
    Assumes kinematics and structure_function are iterable with same length.
    """

    device = (
        kinematics[0].device
        if isinstance(kinematics[0], torch.Tensor)
        else structure_function[0].device
    )

    K = torch.stack(_to_tensor_list(kinematics, device))
    F = torch.stack(_to_tensor_list(structure_function, device))

    # enforce finiteness
    F = torch.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    K = torch.nan_to_num(K, nan=0.0, posinf=0.0, neginf=0.0)

    result = K * F
    assert torch.isfinite(result).all()

    return result


def K_dot_F(kinematics, structure_function):
    return K_times_F(kinematics, structure_function).sum(dim=0)


# ===============================
# Truth calculator (GPU-safe)
# ===============================

class TruthCalc:
    def __init__(
        self,
        sf,
        unique=True,
        enumax=float("inf"),
        force_ds_pos=False,
        M=constants.M_A,
        anti=False,
        device=None,
    ):
        self._sf = sf
        self._unique = unique
        self._enumax = enumax
        self._force_ds_pos = force_ds_pos
        self._M = M
        self._anti = anti
        self.device = torch.device(device) if device is not None else get_default_device()

    def __call__(self, enu, var1, var2, kin_mapping=KinematicMapping(), flavor=pid.mu):

        enu  = torch.as_tensor(enu,  device=self.device)
        var1 = torch.as_tensor(var1, device=self.device)
        var2 = torch.as_tensor(var2, device=self.device)

        emu, cost = kin_mapping.unmake_func(var1, var2)

        ml2 = masses[abs(flavor)] ** 2
        q2, x, y = kin_mapping.make_q2xy(enu, emu, cost, ml2=ml2)

        if self._unique:
            F = self._sf.calc_unique(x, q2, M=self._M, ml2=ml2, anti=self._anti)
        else:
            F = self._sf.calc_all(x, q2)

        K = kin_mapping.make_kinematic_factors(
            q2, x, y,
            unique=self._unique,
            ml2=ml2,
            enumax=self._enumax,
            M=self._M,
            anti=self._anti,
        )

        dsigma = K_dot_F(K, F)

        if self._force_ds_pos:
            dsigma = torch.clamp(dsigma, min=0.0)

        return torch.nan_to_num(dsigma, nan=0.0, posinf=0.0, neginf=0.0)


# ===============================
# Model-based calculator (GPU-safe)
# ===============================

class ModelCalc:
    def __init__(
        self,
        model,
        unique=True,
        enumax=float("inf"),
        force_ds_pos=False,
        M=constants.M_A,
        anti=False,
        device=None,
    ):
        self.device = torch.device(device) if device is not None else get_default_device()
        self._model = model.to(self.device)
        self._unique = unique
        self._enumax = enumax
        # mirror of TruthCalc's flag: clamp dsigma at 0. Off for training
        # (bit-compat with the notebooks); enable at FD inference so the
        # per-event density integral over Enu only vanishes if the model is
        # nonpositive at EVERY energy for that event's kinematics.
        self._force_ds_pos = force_ds_pos
        self._M = M
        self._anti = anti

        self.X2 = None
        self.kun = None

    def __call__(self, enu, var1, var2, kin_mapping=KinematicMapping(), flavor=pid.mu):

        enu  = torch.as_tensor(enu,  device=self.device)
        var1 = torch.as_tensor(var1, device=self.device)
        var2 = torch.as_tensor(var2, device=self.device)

        emu, cost = kin_mapping.unmake_func(var1, var2)

        ml2 = masses[abs(flavor)] ** 2
        q2, x, y = kin_mapping.make_q2xy(enu, emu, cost, ml2=ml2)

        # kinematic prefactors
        kun = torch.stack(
            kin_mapping.make_kinematic_factors(
                q2, x, y,
                unique=self._unique,
                ml2=ml2,
                enumax=self._enumax,
                M=self._M,
                anti=self._anti,
            )
        )
        self.kun = torch.nan_to_num(kun, nan=0.0, posinf=0.0, neginf=0.0)

        # kinematic mask
        kin_mask = kin_mapping.make_kin_mask(
            enu=enu,
            emu=emu,
            cost=cost,
            x=x,
            y=y,
            ml=ml2**0.5,
            enumax=self._enumax,
        ).bool()

        X2 = torch.stack([x, q2], dim=-1)
        X2 = X2 * kin_mask[..., None]
        self.X2 = torch.nan_to_num(X2, nan=0.0, posinf=0.0, neginf=0.0)

        return self._finish(self._model(self.X2, self.kun))

    # cached call
    def dsigma(self):
        return self._finish(self._model(self.X2, self.kun))

    def _finish(self, dsigma):
        if self._force_ds_pos:
            dsigma = torch.clamp(dsigma, min=0.0)
        return dsigma
