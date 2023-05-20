import os
import re
import sys
import random
import argparse
import shutil
import time
import warnings
import numpy as np
import soundfile as sf
import yaml
from tqdm import tqdm

from utils.adv_attack.estimators import ESTIMATOR
from utils.adv_attack.attacker_loader import get_attacker, parse_config

from utils.audio.feats import Pad
from utils.generic import *
from utils.audio.preprocessing import *
import scipy.signal


def sample_random_legit(wav_dir, sid, length, mode):
    with open(
        "datasets/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"
    ) as f:
        legits = [
            line.strip().split(" ")[1]
            for line in f
            if line.strip().split(" ")[4] == "bonafide"
            and line.strip().split(" ")[0] == sid
        ]
    random.shuffle(legits)
    maxLen = -1
    for idx, legit in enumerate(legits):
        ret = silence(os.path.join(wav_dir, legit + ".wav"), -35.0, mode)[1]
        if idx == 0 or maxLen < ret.shape[-1]:
            maxRet = np.copy(ret)
            maxLen = ret.shape[-1]

        if ret.shape[-1] >= length:
            return ret[:, :length]
    return pad_to_length(maxRet, length)


def get_res(line, expr):
    res = np.array(
        re.sub(
            "\\s+",
            " ",
            line[line.find(expr) + len(expr) + 1 : line.find("])", line.find(expr))],
        )
        .strip()
        .split(" ")
    ).astype(int)
    return res


def load_results(results_file):
    with open(results_file, "r") as f:
        line = f.readlines()[-1]
    results = get_res(line, "bb - (")
    results_asv = get_res(line, "bb_asv - (")
    return results, results_asv


def get_available_cuda():
    cmd = (
        "nvidia-smi | grep -v [\+V] | tail -n+3 |"
        " sed '/Processes/,$d' | sed 's/ \+/\t/g' |"
        " cut -f 2,13 | sed 's/N\/A\t//g' |"
        " sed '$!N;s/\\n/ /' | grep 0\% | cut -c1 | head -1"
    )
    device = "cuda:" + popen(cmd).read()
    return device


def delay(x, d):
    sr = 16000
    d = int(sr * d / 1000)
    ret = np.zeros(x.shape, dtype=x.dtype)
    for i in range(len(x) - d):
        ret[i + d] = x[i]
    return ret


def echo(x, start, end):
    x = x.squeeze()
    mid = x[start:end]
    y = np.copy(x)
    x = mid
    delays = range(100)
    for idx, i in enumerate(delays):
        x = x + delay(x, i) / (8 * idx + 1)
    x = x / np.max(np.abs(x))

    x = np.concatenate((y[:start], x, y[end:]))
    return np.expand_dims(x, 0)


def load_input(path_label, sid, wav_dir, ref_dir, attack_type, device):
    orig = np.expand_dims(sf.read(os.path.join(wav_dir, path_label + ".wav"))[0], 0)
    if attack_type != "CM_Attack":
        return orig, np.array([[1, 0]]), orig, 0, None
    try:
        trimmed = trim(os.path.join(wav_dir, path_label + ".wav"), -25.0, -10, 350)
        trimmed_one_sided = trim_l(
            os.path.join(wav_dir, path_label + ".wav"), -25.0, -10
        )
        processed = trim_mid(
            trimmed, frame_duration_ms=30, aggressive=3, sample_rate=16000
        )
        remnant = trimmed_one_sided[:, trimmed.shape[-1] :]
        processed = np.concatenate((processed, remnant), 1)
        pad_l, pad_r = orig.shape[-1] - processed.shape[-1], remnant.shape[-1]
        pad_left = sample_random_legit(ref_dir, sid, pad_l, "leading")
        pad_right = sample_random_legit(ref_dir, sid, pad_r, "leading")
        processed = np.concatenate((pad_left, processed), 1) + np.pad(
            pad_right, [[0, 0], [orig.shape[-1] - pad_r, 0]]
        )
        processed = processed / np.max(np.abs(processed))
        processed = freq_increase(processed, 1000, 6000, fs=16000, factor=1.5, order=20)
        processed = echo(processed, pad_l, orig.shape[-1] - pad_r)
        return processed, np.array([[1, 0]]), orig, pad_l, orig.shape[-1] - pad_r
    except:
        return orig, np.array([[1, 0]]), orig, 0, None


def calc_succ(evalu, x, y):
    return np.array(
        [1 if evalu.result(x, y)[i][0] != "FAIL" else 0 for i in range(x.shape[0])]
    ).sum()


def iter_stats(x, adv, y, sample, exp, ref):
    target = 1 - np.argmax(y, axis=1)
    distance = np.max(np.abs(np.squeeze(adv - x[:, : adv.shape[-1]])), -1)
    exp["bb_asv"] += np.array([calc_succ(e, adv, target) for e in exp["eval_asv"]])
    exp["bb_cm"] += np.array([calc_succ(e, adv, target) for e in exp["eval_cm"]])

    if exp["Attacker"]:
        exp["Attacker"].set_ref(ref)
        target = 1 - y
        orig_res = exp["Attacker"].result(x, target)
        result = exp["Attacker"].result(adv, target)
    else:
        orig_res, result = None, None

    wb = (
        "orig: " + str(orig_res) + ", wb - (" + str(result) + "), "
        if exp["Attacker"]
        else ""
    )
    to_write = (
        str(sample[0])
        + ": "
        + wb
        + "bb - ("
        + str(exp["bb_cm"])
        + "), bb_asv - ("
        + str(exp["bb_asv"])
        + "), total: "
        + str(exp["ctr"] + 1)
        + ", distance - ("
        + str(distance)
        + ")"
    )
    exp["ctr"] = exp["ctr"] + 1
    return to_write


def load_exp(exp, cont):
    with open(os.path.join("experiments", os.path.join(cont, "results.txt")), "r") as f:
        start = len(f.readlines())
    shutil.copy(
        os.path.join("experiments", os.path.join(cont, "results.txt")), exp["out_file"]
    )
    exp["ctr"] += start
    exp["bb_cm"], exp["bb_asv"] = load_results(exp["out_file"])
    if os.path.exists(os.path.join("experiments", os.path.join(cont, "failed.txt"))):
        with open(
            os.path.join("experiments", os.path.join(cont, "failed.txt")), "r"
        ) as f:
            exp["failed_lst"] = [line.strip() for line in f]
            exp["failed"] += len(exp["failed_lst"])
    if (
        os.path.exists(os.path.join("experiments", os.path.join(cont, "wavs")))
        and exp["write_wavs"]
    ):
        os.rmdir(exp["out_wav"])
        shutil.move(
            os.path.join("experiments", os.path.join(cont, "wavs")), exp["out_wav"]
        )
    if (
        os.path.exists(os.path.join("experiments", os.path.join(cont, "plots")))
        and exp["write_plots"]
    ):
        os.rmdir(exp["pltdir"])
        shutil.move(
            os.path.join("experiments", os.path.join(cont, "plots")), exp["pltdir"]
        )
    return start, exp


def init_outdir(exp, args):
    os.makedirs("experiments", exist_ok=True)
    exp["id"] = time.strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join("experiments", exp["id"])
    os.makedirs(out_dir, exist_ok=True)
    shutil.copy(args.conf, os.path.join(out_dir, args.conf))
    shutil.copy(os.path.abspath(__file__), os.path.join(out_dir, "attack.py"))
    shutil.copy(
        "utils/adv_attack/pgd_attacks.py", os.path.join(out_dir, "pgd_attacks.py")
    )
    shutil.copy(
        "utils/adv_attack/advCMAttack.py", os.path.join(out_dir, "advCMAttack.py")
    )
    exp["out_file"] = os.path.join(out_dir, "results.txt")
    exp["log_file"] = os.path.join(out_dir, "failed.txt")
    if exp["write_wavs"]:
        exp["out_wav"] = os.path.join(out_dir, "wavs")
        os.makedirs(exp["out_wav"], exist_ok=True)

    if exp["write_plots"]:
        exp["pltdir"] = os.path.join(out_dir, "plots")
        os.makedirs(exp["pltdir"], exist_ok=True)
    return exp


def init_systems(exp, config, device):
    with open(config["model_maps"], encoding="utf8") as f:
        configMap = yaml.load(f, Loader=yaml.Loader)
    exp["Attacker"] = get_attacker(
        config,
        configMap,
        attack_type=config["attack_type"],
        system=config["system"],
        device=device,
    )
    if config["attack_type"] == "CM_Attack":
        exp["sv_optimizer"] = get_attacker(
            config,
            configMap,
            attack_type="TIME_DOMAIN_ATTACK",
            system="ADVSR",
            device=device,
        )
    else:
        exp["sv_optimizer"] = None

    loader, conf_ret = parse_config(
        config["discriminators"], configMap, config["target"], system="ADVCM"
    )
    exp["eval_cm"] = [
        ESTIMATOR(device=device, loader=loader, config=cf) for cf in conf_ret
    ]

    loader, conf_ret = parse_config(
        config["discriminators"], configMap, config["target"], system="ADVSR"
    )
    exp["eval_asv"] = [
        ESTIMATOR(device=device, loader=loader, config=cf) for cf in conf_ret
    ]
    return exp


def init(config, device, args, cont):
    exp = {
        "sr": config["sr"],
        "perf": config["perf_log"],
        "padder": Pad(0, device=device),
        "num_samples": config["num_samples"],
        "wav_dir": os.path.join(config["input_dir"], "wavs"),
        "log_interval": config["log_interval"],
        "attack_type": config["attack_type"],
        "system": config["system"],
        "print_iter_out": config["print_iter_out"],
        "write_wavs": config["write_wavs"],
        "write_plots": config["write_plots"],
        "silence_threshold": float(config["silence_threshold"]),
        "ref_dir": os.path.join(config.get("ref_dir", config["input_dir"]), "wavs"),
        "failed": 0,
        "failed_lst": [],
        "ctr": 0,
    }

    exp["r_args"] = config["r_args"] if exp["attack_type"] == "CM_attack" else {}

    try:
        exp["SD"] = SortedDict.fromfile(exp["perf"], Score.reader())
    except FileNotFoundError:
        exp["SD"] = SortedDict()

    exp = init_outdir(exp, args)

    with open(config["inputs"], encoding="utf8") as in_file:
        exp["inputs"] = [line.strip().split(" ")[1:] for line in in_file.readlines()]
    with open(config["inputs"], encoding="utf8") as in_file:
        exp["ids"] = [line.strip().split(" ")[0] for line in in_file.readlines()]
    exp = init_systems(exp, config, device)
    exp["bb_cm"] = np.array([0] * len(exp["eval_cm"]))
    exp["bb_asv"] = np.array([0] * len(exp["eval_asv"]))

    if cont is not None:
        start, exp = load_exp(exp, cont)
    else:
        start = 0

    return start, exp


def prepare_iter(sample, exp):
    ref_adv = sample[-100:]

    ref_test = list(set(sample[1:]) - set(ref_adv))
    ref_fin = random.sample(ref_test, len(ref_test) * 2 // 3)
    ref_test = list(set(ref_test) - set(ref_fin))

    with open(
        os.path.join(os.path.join("experiments", exp["id"]), "ref_fin.txt"), "a"
    ) as f:
        f.write(" ".join([sample[0]] + ref_fin) + "\n")

    ref_fin = [
        np.expand_dims(sf.read(os.path.join(exp["ref_dir"], s + ".wav"))[0], 0)
        for s in ref_fin
    ]
    exp["padder"].set_max_len(np.min([r.shape[-1] for r in ref_fin]))
    ref_fin = np.array([exp["padder"](r).squeeze() for r in ref_fin])

    ref = [
        np.expand_dims(sf.read(os.path.join(exp["ref_dir"], s + ".wav"))[0], 0)
        for s in ref_test
    ]
    exp["padder"].set_max_len(np.min([r.shape[-1] for r in ref]))
    ref = np.array([exp["padder"](r).squeeze() for r in ref])

    for e_asv in exp["eval_asv"]:
        e_asv.set_ref(ref_fin)
    if exp["attack_type"] != "eval_only" and exp["system"] != "ADVCM":
        exp["Attacker"].set_ref(ref)
        if exp["sv_optimizer"] is not None:
            exp["sv_optimizer"].set_ref(ref)
    return exp, ref_fin


def log_failure(exp, s_id, e):
    exp["failed"] = exp["failed"] + 1
    exp["failed_lst"].append(s_id)
    print(s_id + "! message: " + str(e) + ". skipping")


def terminate(exp):
    result = (
        "bb - ("
        + str(exp["bb_cm"])
        + "), bb_asv - ("
        + str(exp["bb_asv"])
        + "), total: "
        + str(exp["ctr"])
        + ")"
    )

    if exp["failed"] > 0:
        result = result + ", failed: (" + str(exp["failed"]) + ")"

    print("All done! Experiment: " + exp["id"] + ". Stats: " + result)
    if exp["failed"] > 0:
        print("Failed samples written to: " + str(exp["log_file"]))
        with open(exp["log_file"], "w") as f:
            f.writelines([line + "\n" for line in exp["failed_lst"]])
    sys.exit(0)


def write_iter(x, adv, exp, sample, to_write):
    if exp["ctr"] % exp["log_interval"] == 0:
        exp["SD"][exp["id"]] = Score(
            exp["bb_cm"] / (exp["ctr"] + 1),
            exp["bb_asv"] / (exp["ctr"] + 1),
            exp["ctr"] + 1,
        )
        exp["SD"].tofile(exp["perf"])

    if exp["write_wavs"]:
        outAdv = exp["out_wav"] + "/" + sample[0] + "-" + exp["id"] + "-adv.wav"
        sf.write(exp["out_wav"] + "/" + sample[0] + "-orig.wav", x.squeeze(), exp["sr"])
        sf.write(outAdv, adv.squeeze(), 16000)
        shutil.move(outAdv, exp["out_wav"] + "/" + sample[0] + "-adv.wav")

    with open(exp["out_file"], "a+", buffering=1, encoding="utf8") as out_file:
        if exp["print_iter_out"]:
            print(to_write)
        out_file.write(to_write + "\n")

    if exp["write_plots"]:
        plot(sample[0], exp["out_wav"], exp["pltdir"], exp["sr"])


def normalize(x):
    return x / np.max(np.abs(x))


def run_iter(sample, sid, exp, device):
    adv, y, orig, start, end = load_input(
        sample[0], sid, exp["wav_dir"], exp["ref_dir"], exp["attack_type"], device
    )
    exp, ref = prepare_iter(sample, exp)
    if exp["attack_type"] != "eval_only":
        try:
            if exp["attack_type"] == "CM_Attack":
                adv = normalize(preemphasize(adv, coef=0.5, device=device))
            adv = exp["Attacker"].generate(
                adv, y, start=start, end=end, evalu=exp["eval_cm"], **exp["r_args"]
            )[1]
            if exp["sv_optimizer"] is not None:
                adv = exp["sv_optimizer"].generate(
                    adv,
                    y,
                    start=start,
                    end=end,
                    evalu=exp["eval_cm"],
                    preemphasize=False,
                    **exp["r_args"]
                )[1]
        except KeyboardInterrupt:
            terminate(exp)
        except Exception as e:
            log_failure(exp, sample[0], e)

    to_write = iter_stats(orig, adv, y, sample, exp, ref)
    write_iter(orig, adv, exp, sample, to_write)


def main(config, device, args, cont):
    start, exp = init(config, device, args, cont)
    for s, sample in enumerate(tqdm(exp["inputs"])):
        if s < start:
            continue
        run_iter(sample, exp["ids"][s], exp, device)
    terminate(exp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", default="conf.yaml")
    parser.add_argument("--device", default="")
    parser.add_argument("--inputs")
    parser.add_argument("--resume")
    parser.add_argument("--attack_type")
    parser.add_argument("--input_dir")

    arguments = parser.parse_args()
    if arguments.device == "":
        device = get_available_cuda()
    else:
        device = arguments.device
    with open(arguments.conf, encoding="utf8") as f:
        conf_map = yaml.load(f, Loader=yaml.Loader)

    setup_seed(conf_map["seed"])
    if arguments.inputs:
        conf_map["inputs"] = arguments.inputs
    if arguments.attack_type:
        conf_map["attack_type"] = arguments.attack_type
    if arguments.input_dir:
        conf_map["input_dir"] = arguments.input_dir
    main(conf_map, device, arguments, arguments.resume)
