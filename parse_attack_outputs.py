import argparse
import numpy as np
import re
import os, tempfile
import yaml


def fix_format(input_file, results_file):
    cmd = (
        "sed ':a;N;$!ba;s/\\n/ /g' | sed -E 's/distance - "
        "\\(([[:digit:]]*\\.?[[:digit:]]*)\\)*/distance - \\1\\n/g' "
        " | head -n -1 "
    )

    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp.close()
    splits = cmd.split("|")
    splits[0] = splits[0] + input_file
    splits[-1] = splits[-1] + "> " + results_file
    cmd = "|".join(splits)
    os.system(cmd)
    return results_file


def get_successful_samples(scores, ids, sys):
    ret = []
    for i, score in enumerate(scores):
        indcs = np.where(score == 1)[0]
        if sys in indcs:
            ret.append(ids[i])
    return ret, len(ret) / scores.shape[0]


def samples_and_scores(results_file_raw):
    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp.close()
    results_file = fix_format(results_file_raw, tmp.name)

    with open(results_file, "r") as f:
        results = np.array(
            [
                re.sub(
                    "\\s+",
                    " ",
                    line[
                        line.find("bb - ([") + 7 : line.find("])", line.find("bb - (["))
                    ],
                )
                .strip()
                .split(" ")
                for line in f
            ]
        ).astype(int)

    scores = np.copy(results)
    for i in range(1, len(results)):
        scores[i] = results[i] - results[i - 1]

    with open(results_file, "r") as f:
        ids = [line.strip().split(" ")[0][:-1] for line in f]

    os.unlink(results_file)
    return ids, scores


def write_results(exp, results_map, outdir):
    log = os.path.join(outdir, "log") if outdir else None
    messages = ["******************* Experiment - " + exp + " *******************"]

    for k, v in results_map.items():
        messages.append("System - " + k + ": Success rate - " + str(v["success_rate"]))
        if outdir:
            out_file = os.path.join(outdir, k + ".out")
            with open(out_file, "w") as f:
                f.writelines([line + "\n" for line in v["samples"]])
    if log:
        with open(log, "w") as f:
            f.writelines([message + "\n" for message in messages])
    for message in messages:
        print(message)

    if outdir:
        print("Output written to: '" + outdir + "'")


def get_results(results_file_raw, sys_map, sys):
    ids, scores = samples_and_scores(results_file_raw)
    results_map = {}

    if outdir:
        assert sys == "all"
    else:
        assert sys == "all" or sys in sys_map
    sut = sys_map if sys == "all" else [sys]
    for system in sut:
        samples, success_rate = get_successful_samples(
            scores, ids, sys_map.index(system)
        )
        results_map[system] = {"success_rate": success_rate, "samples": samples}
    return results_map


def geneate_logs(exp, results_file_raw, sys_map, sys, outdir):
    results_map = get_results(results_file_raw, sys_map, sys)
    write_results(exp, results_map, outdir)


def get_experiment_inputs(exp, write_logs=False):
    with open(os.path.join("expirements", os.path.join(exp, "conf.yaml"))) as f:
        conf = yaml.load(f, Loader=yaml.Loader)
    sys_map = conf["discriminators"]["args"]["cm"]["selector"]
    results_file_raw = os.path.join("expirements", os.path.join(exp, "results.txt"))

    if write_logs:
        outdir = os.path.join("expirements", os.path.join(exp, "results_per_system"))
        os.makedirs(outdir, exist_ok=True)
    else:
        outdir = None
    return exp, results_file_raw, sys_map, outdir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp")
    parser.add_argument("--sys", default="all")
    parser.add_argument("--write_logs", action="store_true")
    args = parser.parse_args()
    name, results_file_raw, sys_map, outdir = get_experiment_inputs(
        args.exp, args.write_logs
    )
    geneate_logs(name, results_file_raw, sys_map, args.sys, outdir)
