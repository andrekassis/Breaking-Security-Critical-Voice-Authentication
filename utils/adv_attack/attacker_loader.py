import os
import random
import numpy as np
import soundfile as sf

from art.attacks.evasion.carlini import CarliniL2Method
from art.attacks.evasion.boundary import BoundaryAttack
from art.attacks.evasion.brendel_bethge import BrendelBethgeAttack
from art.attacks.evasion.auto_projected_gradient_descent import (
    AutoProjectedGradientDescent,
)

from .estimators import ESTIMATOR
from .pgd_attacks import TIME_DOMAIN_ATTACK, FFT_Attack, STFT_Attack
from .advCMAttack import CM_Attack


class AttackerWrapper:
    # pylint: disable=R0903
    def __init__(self, attack, attack_type, input_dir, lengths):
        self.attack = attack
        self.attack_type = attack_type
        self.input_dir = input_dir
        self.lengths = lengths

    def _load_example(self, x, longer=False):
        with open(self.lengths, "r", encoding="utf8") as f:
            lengths = [line.strip().split(" ") for line in f]
        lines = [line for line in lengths if int(line[1]) == 1]

        if longer:
            lines = [line[0] for line in lines if int(line[2]) >= x.shape[1]]
        else:
            lines = [line[0] for line in lines]

        line = random.sample(lines, 1)[0]
        input_path = os.path.join(self.input_dir, "wavs/" + line + ".wav")
        init = sf.read(input_path)[0]
        return init

    def generate(self, x, y, evalu=None, **r_args):
        if len(x.shape) == 1:
            x = np.expand_dims(x, 0)
        if self.attack_type != "CM_Attack":
            self.attack.estimator.set_input_shape((x.shape[1],))
        else:
            self.attack.set_input_shape((x.shape[1],))

        if self.attack_type in ("FFT_Attack", "TIME_DOMAIN_ATTACK", "STFT_Attack"):
            return x, self.attack.generate(x, y)
        if self.attack_type == "CM_Attack":
            return x, self.attack.generate(x, y, evalu, **r_args)
        if self.attack_type in ("carlini", "auto_pgd"):
            return x, self.attack.generate(x, 1 - y)

        init = self._load_example(x, longer=True)[: x.shape[1]]
        init = np.expand_dims(init, 0)
        return x, self.attack.generate(x, x_adv_init=init)


def parse_config(config, dset, system):
    if system == "ADVJOINT":
        loader = "load_joint"
    elif system == "ADVCM":
        loader = "load_cm_asvspoof"
    else:
        loader = "load_asv_plda"

    model_cm = list(config["args"]["cm"]["selector"]) if system != "ADVSR" else []
    model_asv = list(config["args"]["asv"]["selector"]) if system != "ADVCM" else []

    model_cm = [
        os.path.join(
            os.path.join(
                os.path.join(os.path.join("configs", dset["cm"]), "cm"), model
            ),
            "config.json",
        )
        for model in model_cm
    ]
    model_asv = [
        os.path.join(
            os.path.join(
                os.path.join(os.path.join("configs", dset["asv"]), "asv"), model
            ),
            "config.json",
        )
        for model in model_asv
    ]

    if system == "ADVJOINT":
        lambdas = [config["args"]["cm"]["lambda"], config["args"]["asv"]["lambda"]]
        conf_ret = [
            {
                "cm": {
                    "loader": "load_cm_asvspoof",
                    "config": cm,
                    "lambda": lambdas[0],
                },
                "asv": {"loader": "load_asv_plda", "config": asv, "lambda": lambdas[1]},
                "system": "ADVJOINT",
            }
            for cm in model_cm
            for asv in model_asv
        ]
        return loader, conf_ret

    if system == "ADVCM":
        return loader, model_cm

    return loader, model_asv


def get_attacker(config, attack_type, system, device):
    loader, conf_ret = parse_config(
        config["discriminator_wb"], config["shadow"], system
    )

    est = ESTIMATOR(
        device=device, loader=loader, config=conf_ret[0], loss=config["loss"]
    )
    if (
        config["discriminator_wb"]["args"]["cm"]["selector"][0] == "comparative"
        or config["discriminator_wb"]["args"]["cm"]["selector"][0] == "RawDarts"
    ):
        if system == "ADVCM":
            est.attack.model.train()
        elif system == "ADVJOINT":
            est.attack.cm.model.train()

    if attack_type == "TIME_DOMAIN_ATTACK":
        Attacker = TIME_DOMAIN_ATTACK(est, **config["TIME_DOMAIN_ATTACK"])
    elif attack_type == "carlini":
        Attacker = CarliniL2Method(est, **config["carlini"])
    elif attack_type == "boundary":
        Attacker = BoundaryAttack(est, **config["boundary"])
    elif attack_type == "bb":
        Attacker = BrendelBethgeAttack(est, **config["bb"])
    elif attack_type == "FFT_Attack":
        Attacker = FFT_Attack(est, **config["FFT_Attack"])
    elif attack_type == "STFT_Attack":
        Attacker = STFT_Attack(est, **config["STFT_Attack"])
    elif attack_type == "CM_Attack":
        conf = config["CM_Attack"]
        conf["r_c"]["sr"] = config["sr"]
        Attacker = CM_Attack(est, **conf)
    elif attack_type == "auto_pgd":
        est.set_input_shape((16000 * 6,))
        Attacker = AutoProjectedGradientDescent(est, **config["auto_pgd"])

    return AttackerWrapper(
        Attacker, attack_type, config["input_dir"], config["lengths"]
    )
