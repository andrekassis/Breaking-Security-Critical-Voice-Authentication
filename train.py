# pylint: disable=R0915, R0914

import argparse
import json
import os

from pathlib import Path
from collections import OrderedDict, defaultdict

from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader

from util.generic.setup import setup_seed
from utils.train import schedulers
from utils.audio.data import ASVDataset, SamplerBlockShuffleByLen, customize_collate
from utils.eval.eer_tools import cal_roc_eer
from utils.eval.model_loaders import load_cm_asvspoof


def unsqueeze_like(tensor: torch.Tensor, like: torch.Tensor):
    n_unsqueezes = like.ndim - tensor.ndim
    if n_unsqueezes < 0:
        raise ValueError(f"tensor.ndim={tensor.ndim} > like.ndim={like.ndim}")
    if n_unsqueezes == 0:
        return tensor
    return tensor[(...,) + (None,) * n_unsqueezes]


class OptimizerWrapper:
    def __init__(self, optimizer, sched):
        # pylint: disable=W0621
        self.optimizer = optimizer
        self.lr_scheduler = [
            getattr(schedulers, sched["type"])(opt, **sched["params"])
            for opt in self.optimizer
        ]

    def update(self, lr, epoch_num, step):
        # pylint: disable=W0621
        for s in self.lr_scheduler:
            s.update(lr, epoch_num, step)

    def zero_grad(self):
        for o in self.optimizer:
            o.zero_grad()

    def step(self):
        for o in self.optimizer:
            o.step()

    def increase_delta(self):
        for s in self.lr_scheduler:
            s.increase_delta()


def process_batch(test_batch, model, loss, **kwargs):
    # pylint: disable=W0621
    test_sample, test_label = test_batch
    test_sample = test_sample.to(device)
    test_label = test_label.to(device)
    out = model(test_sample, **kwargs)
    Loss = loss(out, test_label)
    return Loss, out


def calc_metric(val_metric, probs, lossDict):
    # pylint: disable=W0621
    if val_metric == "eer":
        res = cal_roc_eer(probs)
    elif val_metric == "acc":
        out = np.argmax(probs[:, :-1].numpy().reshape((probs.shape[0], 2)), axis=1)
        target = probs[:, -1].numpy()
        res = np.where(out == target)
        res = np.where(out == target)[0].shape[0] / out.shape[0]
    else:
        res = np.nanmean(lossDict["loss"])
    return res


def train(
    model,
    loss,
    optimizer,
    train_loader,
    val_loader,
    num_epochs,
    lr,
    out_f,
    eval_mode_for_validation,
    no_best_epoch_num,
    val_metric,
    drop_path_prob,
    transform,
    label_fn,
    final_layer,
    train_args,
    device,
):
    # pylint: disable=R0913,W0621

    for epoch_num in tqdm(range(num_epochs)):
        if drop_path_prob:
            model.drop_path_prob = drop_path_prob * epoch_num / num_epochs

        model.train()
        lossDict = defaultdict(list)

        for test_batch in tqdm(train_loader):
            Loss, out = process_batch(test_batch, model, loss, **train_args)
            optimizer.zero_grad()
            Loss.backward()
            optimizer.step()
            optimizer.update(lr, epoch_num, False)
        optimizer.update(lr, epoch_num, True)

        if eval_mode_for_validation:
            model.eval()

        with torch.no_grad():
            probs = torch.empty(0, 2).to(device)
            if val_metric == "acc":
                probs = torch.empty(0, 3).to(device)
            for test_batch in val_loader:
                Loss, out = process_batch(test_batch, model, loss)
                test_label = test_batch[1].to(device)
                t1 = transform(final_layer(out))
                t2 = label_fn(test_label.unsqueeze(-1))
                t1 = unsqueeze_like(t1, t2)
                if val_metric != "acc":
                    t1 = unsqueeze_like(t1[:, -1], t2)
                row = torch.cat((t1, t2), dim=-1)
                probs = torch.cat((probs, row), dim=0)
                lossDict["loss"].append(Loss.item())
            probs = probs.to("cpu")

        res = calc_metric(val_metric, probs, lossDict)

        if epoch_num == 0:
            best_res = res
            best_epoch = epoch_num
            best_epoch_tmp = epoch_num

        is_best = res < best_res if val_metric in ("eer", "loss") else res > best_res
        if is_best:
            model.save_state(os.path.join(out_f, "model.pth"))
            best_epoch = epoch_num
            best_epoch_tmp = epoch_num
            best_res = res

        Message = (
            "\nEpoch: "
            + str(epoch_num)
            + " - Val metric: "
            + val_metric
            + ", value: "
            + str(res)
            + ", best: "
            + str(best_res)
        )
        with open(os.path.join(out_f, "log.log"), "a", encoding="utf8") as log:
            log.write(Message + "\n")
        print(Message)

        model.save_state(os.path.join(out_f, "model" + str(epoch_num) + ".pth"))

        if epoch_num - best_epoch_tmp > 2:
            optimizer.increase_delta()
            best_epoch_tmp = epoch_num

        if (epoch_num - best_epoch) >= no_best_epoch_num:
            print("terminating - early stopping")
            break

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--base", default="asvspoofWavs")
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    args = parser.parse_args()

    with Path(args.config).open("rt", encoding="utf8") as handle:
        config = json.load(handle, object_hook=OrderedDict)

    device, device1 = args.devices.split(",")

    protocol_file_path = os.path.join(
        os.path.join(args.base, "labels"), config["training_parameters"]["labels"]
    )
    out = "/".join(config["path"].split("/")[:-1])
    Path(out).mkdir(parents=True, exist_ok=True)
    # pylint: disable=R1732
    f = open(os.path.join(out, "log.log"), "w", encoding="utf8")
    f.close()
    # pylint: enable=R1732
    data_path = os.path.join(args.base, "wavs/")

    _, _, logits, model_dict = load_cm_asvspoof(
        args.config, device1, load_checkpoint=False
    )
    _, _, _, model_dict1 = load_cm_asvspoof(args.config, device, load_checkpoint=False)
    model = model_dict1["model"]
    loss = model_dict1["loss"]
    extractor = model_dict["extractor"]
    flip_label = model_dict["flip_label"]
    loss = loss.to(device)
    model = model.to(device)

    if flip_label:
        transform = lambda x: torch.flip(x, [-1])
        label_fn = lambda x: 1 - x
    else:
        transform = lambda x: x
        label_fn = lambda x: x

    setup_seed(config["training_parameters"]["seed"])
    data_params = config["training_parameters"]["data_loader"]

    if logits is False:
        final_layer = lambda x: F.softmax(x, dim=1)
    else:
        final_layer = lambda x: x

    train_set = ASVDataset(protocol_file_path, data_path, extractor, flip_label)
    if config["training_parameters"]["sampler"] == "block_shuffle_by_length":
        with open(
            config["training_parameters"]["lens_file"], "r", encoding="utf8"
        ) as f:
            lengths = [int(line.strip().split(" ")[1]) for line in f]
        data_params["sampler"] = SamplerBlockShuffleByLen(
            lengths, data_params["batch_size"]
        )
        data_params["collate_fn"] = customize_collate
        data_params["shuffle"] = False
    train_loader = DataLoader(train_set, **data_params)

    val_set = ASVDataset(protocol_file_path, data_path, extractor, flip_label)
    data_params["shuffle"] = False
    val_loader = DataLoader(val_set, **data_params)

    optimizer = model.optimizer(
        config["training_parameters"]["optimizer"]["type"],
        **config["training_parameters"]["optimizer"]["params"],
    )
    optimizer_wrapper = OptimizerWrapper(
        optimizer, config["training_parameters"]["scheduler"]
    )
    num_epochs = config["training_parameters"]["num_epochs"]
    lr = config["training_parameters"]["optimizer"]["params"]["lr"]
    eval_mode_for_validation = config["training_parameters"]["eval_mode_for_validation"]
    no_best_epoch_num = config["training_parameters"]["no_best_epoch_num"]
    val_metric = config["training_parameters"]["val_metric"]

    try:
        drop_path_prob = config["training_parameters"]["drop_path_prob"]
    except:
        drop_path_prob = None

    try:
        train_args = config["training_parameters"]["train_args"]
    except:
        train_args = {}

    if config["arch"]["type"] == "DartsRaw":
        config["arch"]["args"]["is_mask"] = True

    model = train(
        model,
        loss,
        optimizer_wrapper,
        train_loader,
        val_loader,
        num_epochs,
        lr,
        out,
        eval_mode_for_validation,
        no_best_epoch_num,
        val_metric,
        drop_path_prob,
        transform,
        label_fn,
        final_layer,
        train_args,
        device,
    )
