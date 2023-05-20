# pylint: disable=R0915, R0914

import argparse
import json
import os
import yaml

from pathlib import Path
from collections import OrderedDict, defaultdict

from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F

from utils.generic.setup import setup_seed
from utils.train.schedulers import OptimizerWrapper
from utils.audio.data import (
    CMDataset,
    SamplerBlockShuffleByLen,
    customize_collate,
)
from utils.eval.eer_tools import cal_roc_eer
from utils.eval.model_loaders import load_cm
from utils.eval.init import init_system, init_dataloader


def to(vec, device):
    if isinstance(vec, list):
        return [v.to(device) for v in vec]
    else:
        return vec.to(device)


def custom_sampler(data_params, lens_file, labels, sampler):
    if sampler == "block_shuffle_by_length":
        with open(lens_file, "r", encoding="utf8") as f:
            lengths = [int(line.strip().split(" ")[1]) for line in f]
        data_params["sampler"] = SamplerBlockShuffleByLen(
            lengths, data_params["batch_size"]
        )
        data_params["collate_fn"] = customize_collate
        data_params["shuffle"] = False
    else:
        with open(labels) as f:
            lines = set([line.strip().split(" ")[2] for line in f])
        data_params["sampler"] = torch.utils.data.sampler.SubsetRandomSampler(
            range(len(lines) * 20)
        )
    return data_params


def get_loaders(spec, config, system, base, extract_device):
    data_params = config["training_parameters"]["data_loader"]
    if config["training_parameters"]["sampler"] is not None:
        sampler = config["training_parameters"]["sampler"]
        data_params = custom_sampler(
            data_params,
            config["training_parameters"].get("lens_file"),
            os.path.join(
                os.path.join(base, "labels"), config["training_parameters"]["labels"]
            ),
            sampler,
        )

    data_params["eval"] = False
    train_loader, transform, label_fn, final_layer = init_dataloader(
        spec,
        system,
        base,
        config["training_parameters"]["labels"].split(".")[0],
        data_params,
        extract_device,
        aug=config["arch"]["type"] == "WAV2VEC",
    )

    data_params["eval"] = True
    data_params["shuffle"] = False
    val_loader = init_dataloader(
        spec,
        system,
        base,
        config["training_parameters"]["labels"].split(".")[0],
        data_params,
        extract_device,
    )[0]
    return val_loader, train_loader, transform, label_fn, final_layer


def resume_state(path, model, optimizer, val_metric, lr, num_batch, state_dict, device):
    parent = Path(path).parent.absolute()
    results_f = os.path.join(parent, "log.log")
    with open(results_f, "r") as f:
        lines = [line for line in f]
    results = [float(line.strip().split(" ")[7][:-1]) for line in lines if line != "\n"]
    best = [float(line.strip().split(" ")[9]) for line in lines if line != "\n"][-1]
    idx = np.where(np.array(results) == best)[0][0]
    name = os.path.basename(path)
    itrs = idx + 1 if name == "model.pth" else int(name.split(".")[0][5:]) + 1
    optimizer.load(itrs, results, lr, num_batch, val_metric)
    model = load_cm(model, path, state_dict, device).to(device)
    model.train()
    lines = lines[:itrs]
    with open(results_f, "w") as f:
        f.writelines(lines)
    return model, optimizer, itrs


def make_model(
    spec, config, system, train_device, out, data_len, state_dict, resume=None
):
    if config["arch"]["type"] == "DartsRaw":
        config["arch"]["args"]["is_mask"] = True

    model, loss = init_system(spec, system, train_device, load_checkpoint=False)
    optimizer = model.optimizer(
        config["training_parameters"]["optimizer"]["type"],
        **config["training_parameters"]["optimizer"]["params"],
    )
    optimizer_wrapper = OptimizerWrapper(
        optimizer, config["training_parameters"]["scheduler"]
    )

    if resume:
        val_metric = config["training_parameters"]["val_metric"]
        lr = config["training_parameters"]["optimizer"]["params"]["lr"]
        model, optimizer_wrapper, itr = resume_state(
            os.path.join(out, resume),
            model,
            optimizer_wrapper,
            val_metric,
            lr,
            data_len,
            state_dict,
            train_device,
        )
    else:
        f = open(os.path.join(out, "log.log"), "w", encoding="utf8")
        f.close()
        itr = 0
    return model, optimizer_wrapper, loss, itr


def unsqueeze_like(tensor: torch.Tensor, like: torch.Tensor):
    n_unsqueezes = like.ndim - tensor.ndim
    if n_unsqueezes < 0:
        raise ValueError(f"tensor.ndim={tensor.ndim} > like.ndim={like.ndim}")
    if n_unsqueezes == 0:
        return tensor
    return tensor[(...,) + (None,) * n_unsqueezes]


def process_batch(test_batch, model, loss, device, **kwargs):
    # pylint: disable=W0621
    test_sample, test_label = test_batch
    test_sample = to(test_sample, device)
    test_label = test_label.to(device)
    out = model(test_sample, **kwargs)
    try:
        Loss = loss(out, test_label)
        return Loss, out
    except:
        raise
        return None, out


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


def train_epoch(
    system, epoch_num, optimizer, model, loss, train_loader, lr, device, **train_args
):
    model.train()
    for test_batch in tqdm(train_loader):
        Loss, out = process_batch(test_batch, model, loss, device, **train_args)
        optimizer.zero_grad()
        Loss.backward()
        optimizer.step()
        optimizer.update(lr, epoch_num, False)
    optimizer.update(lr, epoch_num, True)
    return optimizer, model


def validate_epoch(
    system,
    model,
    loss,
    val_loader,
    val_metric,
    transform,
    final_layer,
    label_fn,
    eval_mode_for_validation,
    batch_size,
    device,
):
    lossDict = defaultdict(list)
    if eval_mode_for_validation:
        model.eval()
    with torch.no_grad():
        probs = torch.empty(0, 2).to(device)
        if val_metric == "acc":
            probs = torch.empty(0, 3).to(device)
        for test_batch in val_loader:
            Loss, out = process_batch(test_batch, model, loss, device, eval=True)
            test_label = test_batch[1].to(device)
            t1 = transform(final_layer(out))
            t2 = label_fn(test_label.unsqueeze(-1))
            t1 = unsqueeze_like(t1, t2)
            if val_metric != "acc":
                t1 = unsqueeze_like(t1[:, -1], t2)
            row = torch.cat((t1, t2), dim=-1)
            probs = torch.cat((probs, row), dim=0)
            try:
                lossDict["loss"].append(Loss.item())
            except:
                pass
        probs = probs.to("cpu")

    res = calc_metric(val_metric, probs, lossDict)
    return res


def log_epoch(model, res, epoch_num, val_metric, is_best, best_res, out_f):
    if is_best:
        model.save_state(os.path.join(out_f, "model.pth"))
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


def train(
    system,
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
    batch_size,
    train_args,
    device,
    start=0,
):
    # pylint: disable=R0913,W0621

    for epoch_num in tqdm(range(start, num_epochs)):
        if drop_path_prob:
            model.drop_path_prob = drop_path_prob * epoch_num / num_epochs
        optimizer, model = train_epoch(
            system,
            epoch_num,
            optimizer,
            model,
            loss,
            train_loader,
            lr,
            device,
            **train_args,
        )
        res = validate_epoch(
            system,
            model,
            loss,
            val_loader,
            val_metric,
            transform,
            final_layer,
            label_fn,
            eval_mode_for_validation,
            batch_size,
            device,
        )

        is_best = epoch_num == 0 or (
            res < best_res if val_metric in ("eer", "loss") else res > best_res
        )
        if is_best:
            best_res, best_epoch, best_epoch_tmp = res, epoch_num, epoch_num

        log_epoch(model, res, epoch_num, val_metric, is_best, best_res, out_f)

        if epoch_num - best_epoch_tmp > 2:
            optimizer.increase_delta()
            best_epoch_tmp = epoch_num

        if (epoch_num - best_epoch) >= no_best_epoch_num:
            print("terminating - early stopping")
            break

    return model


def run_train(conf_path, base, resume, devices):
    with Path(conf_path["config"]).open("rt", encoding="utf8") as handle:
        config = json.load(handle, object_hook=OrderedDict)

    train_device, extract_device = devices.split(",")

    lr = config["training_parameters"]["optimizer"]["params"]["lr"]
    num_epochs = config["training_parameters"]["num_epochs"]
    eval_mode_for_validation = config["training_parameters"]["eval_mode_for_validation"]
    no_best_epoch_num = config["training_parameters"]["no_best_epoch_num"]
    val_metric = config["training_parameters"]["val_metric"]
    drop_path_prob = config["training_parameters"].get("drop_path_prob", None)
    train_args = config["training_parameters"].get("train_args", {})
    out = "/".join(config["path"].split("/")[:-1])
    batch_size = config["training_parameters"]["data_loader"]["batch_size"]

    Path(out).mkdir(parents=True, exist_ok=True)
    setup_seed(config["training_parameters"]["seed"])
    val_loader, train_loader, transform, label_fn, final_layer = get_loaders(
        conf_path,
        config,
        config["system"],
        base,
        extract_device,
    )
    model, optimizer, loss, itr = make_model(
        conf_path,
        config,
        config["system"],
        train_device,
        out,
        len(train_loader),
        config["state_dict"],
        resume=resume,
    )

    model = train(
        config["system"],
        model,
        loss,
        optimizer,
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
        batch_size,
        train_args,
        train_device,
        itr,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--system")
    parser.add_argument("--subset", default="dev")
    parser.add_argument("--task", default="cm")
    parser.add_argument("--base", default="datasets/asvspoofWavs")
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--resume")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    run_train(
        config[args.task][args.subset][args.system],
        args.base,
        args.resume,
        args.devices,
    )
