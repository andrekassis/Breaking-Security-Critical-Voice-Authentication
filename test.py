import os
import argparse
import torch
from pathlib import Path
import json
from collections import OrderedDict
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from utils.audio.data import ASVDataset, CMDataset
from utils.eval.eer_tools import cal_roc_eer
from utils.eval.model_loaders import load_cm_asvspoof, load_asv_plda


def to(vec, device):
    if isinstance(vec, list):
        return [v.to(device) for v in vec]
    else:
        return vec.to(device)


def asv_cal_accuracies(
    test_loader, net, transform, label_fn, final_layer, device="cuda:0"
):
    # pylint: disable=R0913, W0621
    with torch.no_grad():
        probs = torch.empty(0, 3).to(device)
        for test_batch in tqdm(test_loader):
            test_sample, test_label = test_batch
            test_sample = to(test_sample, device)
            test_label = test_label.to(device)
            infer = net(test_sample, eval=True)
            t1 = transform(final_layer(infer))
            t2 = label_fn(test_label.unsqueeze(-1))
            row = torch.cat((t1, t2), dim=1)
            print(row)
            probs = torch.cat((probs, row), dim=0)
    return probs.to("cpu")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--base", default="asvspoofWavs")
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    args = parser.parse_args()

    test_device, device = args.devices.split(",")
    protocol_file_path = os.path.join(os.path.join(args.base, "labels"), "eval.lab")
    data_path = os.path.join(args.base, "wavs/")

    with Path(args.config).open("rt", encoding="utf8") as handle:
        config = json.load(handle, object_hook=OrderedDict)

    if config["system"] == "ADVSR":
        loader = load_asv_plda
    else:
        loader = load_cm_asvspoof

    _, _, logits, model_dict = loader(args.config, device)
    _, _, _, model_dict1 = loader(args.config, test_device)
    Net = model_dict1["model"]
    extractor = model_dict["extractor"]
    flip_label = model_dict["flip_label"]
    Net.to(test_device)
    Net.eval()

    if flip_label:
        transform = lambda x: torch.flip(x, [-1])
        label_fn = lambda x: 1 - x
    else:
        transform = lambda x: x
        label_fn = lambda x: x

    if logits is False:
        final_layer = lambda x: F.softmax(x, dim=1)
    else:
        final_layer = lambda x: x

    if config["system"] == "ADVSR":
        test_set = ASVDataset(
            protocol_file_path, data_path, extractor, flip_label, device
        )
    else:
        test_set = CMDataset(
            protocol_file_path, data_path, extractor, flip_label, device
        )

    loader_args = {
        "batch_size": args.bs,
        "shuffle": False,
        "num_workers": args.bs * 2,
        "pin_memory": True,
    }
    if device != "cpu":
        loader_args["multiprocessing_context"] = "spawn"

    test_loader = DataLoader(test_set, **loader_args)

    probabilities = asv_cal_accuracies(
        test_loader,
        Net,
        transform,
        label_fn,
        final_layer,
        test_device,
    )
    eer = cal_roc_eer(probabilities)
    print("eer: " + str(eer))
    print("End of Program.")
