import os
import random
import argparse
import shutil
import time
import warnings
import numpy as np
import soundfile as sf
import yaml

from tqdm import tqdm

# pylint: disable=C0413
warnings.filterwarnings("ignore")
# pylint: enable=C0413

from pydub import AudioSegment
from pydub.silence import detect_leading_silence as dls

from utils.adv_attack.estimators import ESTIMATOR
from utils.adv_attack.attacker_loader import get_attacker, parse_config

from utils.audio.feats import Pad
from utils.generic.sortedDict import SortedDict
from utils.generic.score import Score
from utils.generic.plot import plot
from utils.generic.setup import setup_seed

trim_leading: AudioSegment = lambda x, threshold: x[
    dls(x, silence_threshold=threshold) :
]
trim_trailing: AudioSegment = lambda x, threshold: trim_leading(
    x.reverse(), threshold
).reverse()
strip_silence: AudioSegment = lambda x, threshold: trim_trailing(
    trim_leading(x, threshold), threshold
)


def trim(audio, silence_threshold=-25.0):
    sound = AudioSegment.from_file(audio)
    stripped = strip_silence(sound, silence_threshold)
    ret = np.array(stripped.get_array_of_samples())
    ret = ret / np.max(np.abs(ret))
    return np.expand_dims(ret, 0)


def get_available_cuda():
    cmd = (
        "nvidia-smi | grep -v [\+V] | tail -n+3 |"
        " sed '/Processes/,$d' | sed 's/ \+/\t/g' |"
        " cut -f 2,13 | sed 's/N\/A\t//g' |"
        " sed '$!N;s/\\n/ /' | grep 0\% | cut -c1 | head -1"
    )
    device = "cuda:" + popen(cmd).read()
    return device


def load_input(path_label, wav_dir):
    return np.expand_dims(
        sf.read(os.path.join(wav_dir, path_label + ".wav"))[0], 0
    ), np.array([[1, 0]])


def calc_succ(evalu, x, y):
    return np.array(
        [1 if evalu.result(x, y)[i][0] != "FAIL" else 0 for i in range(x.shape[0])]
    ).sum()


def iter_stats(x, adv, y, sample, exp):
    target = 1 - np.argmax(y, axis=1)
    distance = np.max(np.abs(np.squeeze(adv - x[:, : adv.shape[-1]])))
    exp["bb_asv"] += np.array([calc_succ(e, adv, target) for e in exp["eval_asv"]])
    exp["bb_cm"] += np.array([calc_succ(e, adv, target) for e in exp["eval_cm"]])

    ref = np.expand_dims(
        sf.read(os.path.join(exp["wav_dir"], sample[-1] + ".wav"))[0], 0
    )
    exp["Attacker"].set_ref(ref)
    target = 1 - y
    orig_res = exp["Attacker"].result(x, target)
    result = exp["Attacker"].result(adv, target)

    to_write = (
        str(sample[0])
        + ": orig: "
        + str(orig_res)
        + ", wb - ("
        + str(result)
        + "), bb - ("
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


def init_outdir(exp, args):
    os.makedirs("expirements", exist_ok=True)
    exp["id"] = time.strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join("expirements", exp["id"])
    os.makedirs(out_dir)
    shutil.copy(args.conf, os.path.join(out_dir, args.conf))
    shutil.copy(os.path.abspath(__file__), os.path.join(out_dir, "attack.py"))
    shutil.copy(
        "utils/adv_attack/pgd_attacks.py", os.path.join(out_dir, "pgd_attacks.py")
    )
    exp["out_file"] = os.path.join(out_dir, "results.txt")

    if exp["write_wavs"]:
        exp["out_wav"] = os.path.join(out_dir, "wavs")
        os.makedirs(exp["out_wav"])

    if exp["write_plots"]:
        exp["pltdir"] = os.path.join(out_dir, "plots")
        os.makedirs(exp["pltdir"])
        os.makedirs(os.path.join(exp["pltdir"], "raw"))
        os.makedirs(os.path.join(exp["pltdir"], "fft"))

    return exp


def init_systems(exp, config, device):
    exp["Attacker"] = get_attacker(
        config,
        attack_type=config["attack_type"],
        system=config["system"],
        device=device,
    )

    loader, conf_ret = parse_config(
        config["discriminators"], config["target"], system="ADVCM"
    )
    exp["eval_cm"] = [
        ESTIMATOR(device=device, loader=loader, config=cf) for cf in conf_ret
    ]

    loader, conf_ret = parse_config(
        config["discriminators"], config["target"], system="ADVSR"
    )
    exp["eval_asv"] = [
        ESTIMATOR(device=device, loader=loader, config=cf) for cf in conf_ret
    ]
    return exp


def init(config, device, args):
    exp = {
        "sr": config["sr"],
        "perf": config["perf_log"],
        "padder": Pad(config["length"]),
        "num_samples": config["num_samples"],
        "wav_dir": os.path.join(config["input_dir"], "wavs"),
        "log_interval": config["log_interval"],
        "attack_type": config["attack_type"],
        "system": config["system"],
        "print_iter_out": config["print_iter_out"],
        "write_wavs": config["write_wavs"],
        "write_plots": config["write_plots"],
        "failed": 0,
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

    exp = init_systems(exp, config, device)

    exp["bb_cm"] = np.array([0] * len(exp["eval_cm"]))
    exp["bb_asv"] = np.array([0] * len(exp["eval_asv"]))

    return exp


def prepare_iter(sample, exp):
    if exp["system"] != "ADVCM":
        ref = random.sample(sample[1:-1], exp["num_samples"])
        ref = [trim(os.path.join(exp["wav_dir"], r + ".wav")) for r in ref]
        exp["padder"].set_max_len(np.min([r.shape[-1] for r in ref]))
        ref = np.array([exp["padder"](r).squeeze() for r in ref])
        exp["Attacker"].set_ref(ref)

    ref = np.expand_dims(
        sf.read(os.path.join(exp["wav_dir"], sample[-1] + ".wav"))[0], 0
    )

    for e_asv in exp["eval_asv"]:
        e_asv.set_ref(ref)

    return exp


def run_iter(sample, exp, **r_args):
    x, y = load_input(sample[0], exp["wav_dir"])
    exp = prepare_iter(sample, exp)
    try:
        adv = exp["Attacker"].generate(x, y, evalu=exp["eval_cm"], **r_args)[1]
    # pylint: disable=W0703
    except Exception as e:
        exp["failed"] = exp["failed"] + 1
        print("attack_failed! message: " + str(e) + ". skipping")
        return x, x, y, False
    # pylint: enable=W0703
    return x, adv, y, True


def write_iter(x, adv, exp, sample, to_write):
    if exp["ctr"] % exp["log_interval"] == 0:
        exp["SD"][exp["id"]] = Score(
            exp["bb_cm"] / (exp["ctr"] + 1),
            exp["bb_asv"] / (exp["ctr"] + 1),
            exp["ctr"] + 1,
        )
        exp["SD"].tofile(exp["perf"])

    if exp["write_wavs"]:
        sf.write(
            exp["out_wav"] + "/" + sample[0] + "-adv.wav", adv.squeeze(), exp["sr"]
        )
        sf.write(exp["out_wav"] + "/" + sample[0] + "-orig.wav", x.squeeze(), exp["sr"])

    with open(exp["out_file"], "a+", buffering=1, encoding="utf8") as out_file:
        if exp["print_iter_out"]:
            print(to_write)
        out_file.write(to_write + "\n")

    if exp["write_plots"]:
        plot(sample[0], exp["out_wav"], exp["pltdir"], exp["sr"])


def main(config, device, args):
    exp = init(config, device, args)
    for sample in tqdm(exp["inputs"]):
        x, adv, y, ret = run_iter(sample, exp, **exp["r_args"])
        if ret:
            to_write = iter_stats(x, adv, y, sample, exp)
            write_iter(x, adv, exp, sample, to_write)
    print("done! # of failed samples: " + str(exp["failed"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf")
    parser.add_argument("--device", default="")

    arguments = parser.parse_args()
    if arguments.device == "":
        device = get_available_cuda()
    else:
        device = arguments.device
    with open(arguments.conf, encoding="utf8") as f:
        conf_map = yaml.load(f, Loader=yaml.Loader)

    setup_seed(1234)
    main(conf_map, device, arguments)
