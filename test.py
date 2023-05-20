import os
import argparse
import torch
from pathlib import Path
import json
from collections import OrderedDict
import torch.nn.functional as F
from tqdm import tqdm
from utils.eval.eer_tools import cal_roc_eer
from utils.eval.init import init_system, init_dataloader
import yaml
import numpy as np
import random


def to(vec, device):
    if isinstance(vec, list):
        return [v.to(device) for v in vec]
    else:
        return vec.to(device)


def asv_cal_accuracies(
    system,
    test_loader,
    net,
    transform,
    label_fn,
    final_layer,
    batch_size,
    device="cuda:0",
):
    # pylint: disable=R0913, W0621
    with torch.no_grad():
        probs = torch.empty(0, 3).to(device)
        for test_batch in tqdm(test_loader):
            test_sample, test_label = test_batch
            test_sample = to(test_sample, device)
            test_label = test_label.to(device)
            test_label = test_label.reshape((np.prod(test_label.shape),))
            infer = net(test_sample, eval=True)
            t1 = transform(final_layer(infer))
            t2 = label_fn(test_label.unsqueeze(-1))
            row = torch.cat((t1, t2), dim=1)
            probs = torch.cat((probs, row), dim=0)
    return probs.to("cpu")


def test(config, system, bs, base, devices, eer=None):
    test_device, extract_device = devices.split(",")
    Net = init_system(config, "ADV" + system.upper(), test_device)[0]
    loader_args = {
        "batch_size": bs,
        "shuffle": False,
        "num_workers": bs * 2,
        "pin_memory": True,
        "eval": True,
        "utts_per_spkr": 6,
        "tisv_frame": 180,
        "hop_size": 160,
        "window_length": 400,
    }
    labels = os.path.join(os.path.join(base, "labels"), "eval.lab")
    test_loader, transform, label_fn, final_layer = init_dataloader(
        config,
        "ADV" + system.upper(),
        base,
        "eval",
        loader_args,
        extract_device,
    )

    probabilities = asv_cal_accuracies(
        "ADV" + system.upper(),
        test_loader,
        Net,
        transform,
        label_fn,
        final_layer,
        bs,
        test_device,
    )
    eer = cal_roc_eer(probabilities) if eer is None else eval(eer)
    print("eer: " + str(eer))
    neg = len([prob for prob in probabilities if prob[-1] == 0])
    pos = len([prob for prob in probabilities if prob[-1] == 1])
    fpr = (
        len([prob for prob in probabilities if prob[-2] >= eer[1] and prob[-1] == 0])
        / neg
        if neg > 0
        else 1
    )
    fnr = (
        len([prob for prob in probabilities if prob[-2] < eer[1] and prob[-1] == 1])
        / pos
        if pos > 0
        else 1
    )
    acc = 1 - (fpr * neg + fnr * pos) / len(probabilities)
    fail = 1 - acc
    print(
        "fpr: "
        + str(fpr)
        + ", fnr: "
        + str(fnr)
        + ", accuracy: "
        + str(acc)
        + ", fail: "
        + str(fail)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--system")
    parser.add_argument("--subset", default="train")
    parser.add_argument("--task", default="cm")
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--base", default="datasets/asvspoofWavs")
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--eer")
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    test(
        config[args.task][args.subset][args.system],
        args.task,
        args.bs,
        args.base,
        args.devices,
        args.eer,
    )
