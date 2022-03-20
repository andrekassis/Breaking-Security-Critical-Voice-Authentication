import numpy as np
from tqdm import tqdm
import torch

from . import pgd_attacks as base_attacks
from .reduce import sp


class CM_Attack:
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-arguments

    def __init__(
        self,
        est,
        first_layer,
        last_layer,
        mid_layers,
        itrs,
        alpha_factors,
        powers,
        r_c,
        stop_reduce=10,
        r_div=3,
        k_div=0.1,
        optim_sat=15,
        interval=1,
        stacks=2,
        stationary=False,
        apply_args_to_layers="all_but_last",
        verbose=0,
    ):
        r_c["device"] = est.device
        self.first_layer = (
            getattr(base_attacks, first_layer["type"])(
                est, **first_layer["args"], r_c=r_c
            )
            if first_layer is not None
            else None
        )

        self.last_layer = (
            getattr(base_attacks, last_layer["type"])(
                est, **last_layer["args"], r_c=r_c
            )
            if last_layer is not None
            else None
        )
        self.mid_layers = (
            {
                m["type"]: getattr(base_attacks, m["type"])(est, **m["args"], r_c=r_c)
                for m in mid_layers
            }
            if mid_layers is not None
            else []
        )

        self.mid_layers_conf = mid_layers if mid_layers is not None else []
        r_c["stationary"] = stationary
        self.spectral_gate = sp(**r_c)
        self.alpha_factors = alpha_factors
        self.powers = powers
        self.verbose = verbose
        self.itrs = itrs
        self.stop_reduce = itrs if stop_reduce is None else stop_reduce
        self.r_div = r_div
        self.k_div = k_div
        self.optim_sat = optim_sat
        self.m, self.p, self.r = 1, 1, 1
        self.estimator = est
        self.interval = interval
        self.num_stacks = stacks

        self.num_layers = 1 if self.first_layer else 0
        self.num_layers += 1 if self.last_layer else 0
        self.num_layers += len(self.mid_layers_conf) if self.mid_layers_conf else 0

        self._set_layers_with_args(apply_args_to_layers)

        r_c["stationary"] = False
        self.sp2 = sp(**r_c)

    def _get_idx_of_mid_layers(self):
        start_mid = 1 if self.first_layer else 0
        end_mid = self.num_layers if not self.last_layer else self.num_layers - 1
        return start_mid, end_mid

    def _set_layers_with_args(self, apply_args_to_layers):
        all_layers = list(range(self.num_layers))
        start_mid, end_mid = self._get_idx_of_mid_layers()
        if apply_args_to_layers is None:
            self.layers_with_args = []
        elif isinstance(apply_args_to_layers, str):
            if apply_args_to_layers not in [
                "all",
                "all_but_first",
                "all_but_last",
                "mid_layers",
                "first_only",
                "last_only",
                "first_and_last",
            ]:
                raise ValueError("invalid option for layers to apply args")
            if apply_args_to_layers == "all":
                self.layers_with_args = all_layers
            elif apply_args_to_layers == "all_but_first":
                self.layers_with_args = all_layers[start_mid:]
            elif apply_args_to_layers == "all_but_last":
                self.layers_with_args = all_layers[:end_mid]
            elif apply_args_to_layers == "mid_layers":
                self.layers_with_args = all_layers[start_mid:end_mid]
            elif apply_args_to_layers == "first_only":
                assert self.first_layer is not None
                self.layers_with_args = [0]
            elif apply_args_to_layers == "last_only":
                assert self.last_layer is not None
                self.layers_with_args = [self.num_layers - 1]
            else:
                assert self.last_layer is not None
                assert self.first_layer is not None
                self.layers_with_args = [0, self.num_layers - 1]
        else:
            if not isinstance(apply_args_to_layers, list):
                raise ValueError("invalid option for layers to apply args")
            if not (set(apply_args_to_layers) <= set(all_layers)):
                raise ValueError("invalid layers provided")
            self.layers_with_args = apply_args_to_layers

    def _mrp(self, k):
        k_curr = k // self.interval
        self.m = np.min([k_curr + 1, self.optim_sat])
        self.p = 1 / (k + 1 + self.k_div)
        self.r = (
            (1 / (1 - self.p * self.num_stacks) - 1) / self.r_div + 1
            if self.r_div
            else 1
        )

    def _configure_attacks(self, k):
        self._mrp(k)
        for m in self.mid_layers_conf:
            self.mid_layers[m["type"]].epsilon = np.power(
                m["args"]["epsilon"] / self.m, self.powers[m["type"]]
            )
            self.mid_layers[m["type"]].alpha = (
                self.mid_layers[m["type"]].epsilon / self.alpha_factors[m["type"]]
            )
            self.mid_layers[m["type"]].delta = m["args"]["delta"]
            self.mid_layers[m["type"]].max_iter = m["args"]["max_iter"]

    def _clip(self, adv, k):
        with torch.no_grad():
            adv = (adv * self.r).clip(-1, 1) if k < self.stop_reduce else adv
            adv = (adv * self.r).clip(-95, 95) if k == self.stop_reduce else adv
        return adv

    def _run_layers(self, adv, y, k, r_args):
        for i, m in enumerate(self.mid_layers_conf):
            adv = self.mid_layers[m["type"]].generate(adv, y, **r_args[i])
        with torch.no_grad():
            adv = self.spectral_gate(adv, p=self.p) if k <= self.stop_reduce else adv
        return adv

    def _log(self, adv, y, evalu=None):
        if self.verbose < 2:
            return

        with torch.no_grad():
            target = 1 - y
            wb = self.estimator.result(adv, target)
            if evalu is not None:
                res = [evalu[j].result(adv, target) for j in range(len(evalu))]
            else:
                res = ""
            print(str(res) + ", wb: " + str(wb))

    def get_r_args(self, **r_args):
        run_args = [
            r_args if l in self.layers_with_args else {} for l in range(self.num_layers)
        ]
        start_mid, end_mid = self._get_idx_of_mid_layers()
        return run_args, start_mid, end_mid

    def _run_edge_layer(self, adv, y, idx, run_args):
        if idx != 0 and idx != -1:
            return adv
        if idx == 0 and self.first_layer:
            return self.first_layer.generate(adv, y, **run_args[0])
        if idx == -1 and self.last_layer:
            return self.last_layer.generate(adv, y, **run_args[-1])
        return adv

    def generate(self, adv, y, evalu=None, **r_args):
        adv = torch.tensor(
            adv, requires_grad=True, device=self.estimator.device, dtype=torch.float
        )
        run_args, start_mid, end_mid = self.get_r_args(**r_args)

        self._log(adv, y, evalu)

        adv = self._run_edge_layer(adv, y, 0, run_args)

        self._log(adv, y, evalu)

        for k in tqdm(range(self.itrs), disable=self.verbose == 0):
            self._configure_attacks(k)

            for stack in range(self.num_stacks):
                adv = self._run_layers(adv, y, k, run_args[start_mid:end_mid])

            adv = self._clip(adv, k)

            self._log(adv, y, evalu)

        adv = self._run_edge_layer(adv, y, -1, run_args)

        self._log(adv, y, evalu)

        adv = adv.detach().cpu().numpy()
        return adv / np.max(np.abs(adv), axis=-1)[..., np.newaxis]
