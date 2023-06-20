# pylint: disable=W0406

import os
from pathlib import Path
import json
from collections import OrderedDict
import numpy as np
import torch

import parse_config
import components.cm_models.loss as module_loss
import components.cm_models as module_arch
from components.asv_models.gmm import FullGMM
from components.asv_models.ivector_extract import ivectorExtractor
from components.asv_models.plda import PLDA
from components.asv_models.SVsystem import SVsystem
from components.asv_models.xvector import xvectorModel
from utils.audio import feats


def load_cm(model, resume, state_dict, device):
    checkpoint = torch.load(resume, map_location=device)
    if state_dict is not None:
        state_dict = checkpoint[state_dict]
    else:
        state_dict = checkpoint

    ##Some pretrained models had these redundant keys. Remove manually
    for key in [
        "feature.lfcc_fb",
        "m_frontend.0.lfcc_fb",
        "fc_att.weight",
        "fc_att.bias",
    ]:
        try:
            del state_dict[key]
        except:
            pass
    model.load_state_dict(state_dict)
    return model


def load_cm_asvspoof(modelSpec, device, load_checkpoint=True, loss=None):
    if not isinstance(modelSpec.get("config"), dict):
        with Path(modelSpec["config"]).open("rt", encoding="utf8") as handle:
            config = json.load(handle, object_hook=OrderedDict)
    else:
        config = modelSpec["config"]

    system = config["system"]
    resume = modelSpec.get("path")

    if loss is None:
        loss = config["loss"]

    if loss["requires_device"]:
        loss_fn = getattr(module_loss, loss["type"])(device=device, **loss["args"])
    else:
        loss_fn = getattr(module_loss, loss["type"])(**loss["args"])

    if hasattr(loss_fn, "it"):
        loss_fn.it = np.inf

    if config["arch"]["requires_device"]:
        model = getattr(module_arch, config["arch"]["type"])(
            device=device, **config["arch"]["args"]
        )
    else:
        model = getattr(module_arch, config["arch"]["type"])(**config["arch"]["args"])

    loss = loss_fn
    extractor = getattr(feats, config["extractor"]["fn"])(
        device=device, **config["extractor"]["args"]
    )

    if load_checkpoint:
        model = load_cm(model, resume, config["state_dict"], device)
    model = model.to(device)

    if load_checkpoint:
        model.eval()

    return (
        "ADVCM",
        [modelSpec.get("eer_point")],
        config["logits"],
        {
            "model": model,
            "loss": loss,
            "extractor": extractor,
            "flip_label": config["flip_label"],
        },
    )


def load_asv_plda(modelSpec, device, loss=None, load_checkpoint=True):
    # pylint: disable=W0613
    if not isinstance(modelSpec.get("config"), dict):
        with Path(modelSpec.get("config")).open("rt", encoding="utf8") as handle:
            config = json.load(handle, object_hook=OrderedDict)
    else:
        config = modelSpec["config"]
    if config["asv_args"]["fgmmfile"] is not None:
        fgmm = FullGMM(
            os.path.join(modelSpec["path"], config["asv_args"]["fgmmfile"]),
            device,
        )
        extractor = ivectorExtractor(
            os.path.join(modelSpec["path"], config["asv_args"]["ivector_extractor"]),
            device,
        )
    else:
        fgmm = None
        extractor = xvectorModel(
            os.path.join(modelSpec["path"], config["asv_args"]["ivector_extractor"]),
            os.path.join(modelSpec["path"], config["asv_args"]["transform"]),
            device,
        )
    plda_mdl = PLDA(
        os.path.join(modelSpec["path"], config["asv_args"]["plda_mdl"]), device
    )

    SV_system = SVsystem(
        fgmm,
        extractor,
        plda_mdl,
        os.path.join(modelSpec["path"], config["asv_args"]["ivector_mean"]),
        device,
    )
    extractor = getattr(feats, config["extractor"]["fn"])(
        device=device, **config["extractor"]["args"]
    )

    return (
        "ADVSR",
        [modelSpec.get("eer_point")],
        config["logits"],
        {
            "model": SV_system,
            "extractor": extractor,
            "flip_label": config["flip_label"],
        },
    )


def load_joint(modelSpec, device, loss=None, load_checkpoint=True):
    _, cm_eer, cm_l, cm = load_cm_asvspoof(modelSpec["cm"]["spec"], device, loss=loss)
    _, asv_eer, asv_l, asv = load_asv_plda(modelSpec["asv"]["spec"], device, loss=loss)
    return (
        "ADVJOINT",
        asv_eer + cm_eer,
        [asv_l, cm_l],
        {
            "asv_args": asv,
            "cm_args": cm,
            "lambda_asv": modelSpec["asv"]["lambda"],
            "lambda_cm": modelSpec["cm"]["lambda"],
        },
    )
