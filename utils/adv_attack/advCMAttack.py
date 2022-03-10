import numpy as np
from tqdm import tqdm

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
        optim_sat=15,
        stationary=False,
        verbose=True,
    ):
        r_c["device"] = est.device
        self.first_layer = getattr(base_attacks, first_layer["type"])(
            est, **first_layer["args"], r_c=r_c
        )
        self.last_layer = getattr(base_attacks, last_layer["type"])(
            est, **last_layer["args"], r_c=r_c
        )
        self.mid_layers = {
            m["type"]: getattr(base_attacks, m["type"])(est, **m["args"], r_c=r_c)
            for m in mid_layers
        }
        self.mid_layers_conf = mid_layers
        r_c["stationary"] = stationary
        self.spectral_gate = sp(**r_c)
        self.alpha_factors = alpha_factors
        self.powers = powers
        self.verbose = verbose
        self.itrs = itrs
        self.stop_reduce = stop_reduce
        self.r_div = r_div
        self.optim_sat = optim_sat
        self.m, self.p, self.r = 1, 1, 1

    def set_ref(self, ref, device):
        self.first_layer.estimator.set_ref(ref, device)
        for l in self.mid_layers.keys():
            self.mid_layers[l].estimator.set_ref(ref, device)
        self.last_layer.estimator.set_ref(ref, device)

    def set_input_shape(self, shape):
        self.first_layer.estimator.set_input_shape(shape)
        for l in self.mid_layers.keys():
            self.mid_layers[l].estimator.set_input_shape(shape)
        self.last_layer.estimator.set_input_shape(shape)

    def _mrp(self, k):
        self.m = np.min([k + 1, self.optim_sat])
        self.p = 1 / (k + 1.1)
        self.r = (1 / (1 - self.p) - 1) / self.r_div + 1

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

    def _reduce_noise(self, adv, k):
        if k < self.stop_reduce:
            return np.clip(self.spectral_gate(adv, p=self.p) * self.r, -1, 1)
        if k == self.stop_reduce:
            return np.clip(self.spectral_gate(adv, p=self.p) * self.r, -0.95, 0.95)
        return adv

    def _first_pass(self, adv, y, **r_args):
        for m in self.mid_layers_conf:
            adv = self.mid_layers[m["type"]].generate(adv, y, **r_args)
        return adv

    def _second_pass(self, adv, y, **r_args):
        self.mid_layers_conf.reverse()
        adv = self._first_pass(adv, y, **r_args)
        self.mid_layers_conf.reverse()
        return adv

    def _log(self, adv, y, evalu=None):
        if self.verbose:
            print(
                [
                    evalu[j].result(
                        adv / np.max(np.abs(adv)), 1 - np.argmax(y, axis=1), eval=True
                    )[0]
                    for j in range(len(evalu))
                ]
            )

    def generate(self, adv, y, evalu=None, **r_args):
        adv = self.first_layer.generate(adv, y, **r_args)

        self._log(adv, y, evalu)

        for k in tqdm(range(self.itrs), disable=not self.verbose):
            self._configure_attacks(k)
            if k > 0:
                adv = self._first_pass(adv, y, **r_args)
            adv = self._reduce_noise(adv, k)
            adv = self._second_pass(adv, y, **r_args)
            self._log(adv, y, evalu)

        adv = self.last_layer.generate(adv, y, **{})  # r_args)

        self._log(adv, y, evalu)

        return adv / np.max(np.abs(adv))
