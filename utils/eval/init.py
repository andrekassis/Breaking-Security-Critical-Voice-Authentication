import os
from .model_loaders import load_cm_asvspoof, load_asv_plda
from torch.utils.data.dataloader import DataLoader
from utils.audio.data import CMDataset, ASVDataset, CMRawBoost
import torch.nn.functional as F
import numpy as np
import torch


def init_system(config, system, test_device, load_checkpoint=True):
    loader = load_asv_plda if system == "ADVSR" else load_cm_asvspoof
    model_dict = loader(config, test_device, load_checkpoint=load_checkpoint)[3]
    Net = model_dict["model"].to(test_device)
    Net.eval()
    return Net, model_dict.get("loss", None)


def init_dataloader(
    config, system, data_base, section, loader_args, extract_device, aug=False
):
    protocol_file_path = os.path.join(
        os.path.join(data_base, "labels"), section + ".lab"
    )
    data_path = os.path.join(data_base, "wavs/")

    loader = load_asv_plda if system == "ADVSR" else load_cm_asvspoof
    init_fn = ASVDataset if system == "ADVSR" else CMDataset
    init_fn = CMRawBoost if aug else init_fn

    logits, model_dict = loader(config, extract_device, load_checkpoint=False)[2:]
    transform = (
        (lambda x: torch.flip(x, [-1]))
        if model_dict.get("flip_label")
        else (lambda x: x)
    )
    label_fn = (lambda x: 1 - x) if model_dict.get("flip_label") else (lambda x: x)
    final_layer = (lambda x: F.softmax(x, dim=1)) if logits is False else (lambda x: x)

    kwargs = {}
    test_set = init_fn(
        protocol_file_path,
        data_path,
        model_dict["extractor"],
        model_dict.get("flip_label"),
        extract_device,
        batch_size=loader_args["batch_size"],
        **kwargs
    )

    loader_args = {
        k: v
        for k, v in loader_args.items()
        if k not in ["utts_per_spkr", "eval", "hop_size", "window_length", "tisv_frame"]
    }
    if extract_device != "cpu" and loader_args.get("num_workers", 0) > 0:
        loader_args["multiprocessing_context"] = "spawn"
        loader_args["persistent_workers"] = True
    test_loader = DataLoader(test_set, **loader_args)

    return test_loader, transform, label_fn, final_layer
