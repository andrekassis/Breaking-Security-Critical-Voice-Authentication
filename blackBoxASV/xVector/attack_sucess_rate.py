import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", "-d")
    parser.add_argument("--eer", "-e", default='["6.421%", "-3.75052"]')
    args = parser.parse_args()
    eer = float(eval(args.eer)[1])
    print(
        "eer: " + str(eval(args.eer)[0]) + ", at threshold: " + str(eval(args.eer)[1])
    )
    with open(args.data_dir + "scores") as f:
        lines = [line.strip().split(" ") for line in f]
    lines = [
        [line[i].split("-")[-1] for i in range(len(line) - 1)] + [line[-1]]
        for line in lines
    ]
    dict = {}
    for line in lines:
        if dict.get(line[0]) is None:
            dict[line[0]] = [line[-1]]
        else:
            dict[line[0]].append(line[-1])
    for key in dict.keys():
        dict[key] = [float(v) for v in dict[key]]
    for key in dict.keys():
        dict[key] = np.array(dict[key]).mean()
    for key in dict.keys():
        if dict[key] >= (eer):
            dict[key] = 1
        else:
            dict[key] = 0
    print(np.array(list(dict.values())).mean())
