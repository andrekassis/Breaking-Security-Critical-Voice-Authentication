import numpy as np
import os
import argparse
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("--exp")
parser.add_argument("--outdir", default="inputs")
parser.add_argument("--ids", default="inputs/ids")
args = parser.parse_args()

wav_dir = os.path.join(args.exp, "wavs")
orig_dir = os.path.join(args.outdir, "ABX/orig")
adv_dir = os.path.join(args.outdir, "ABX/adv")

os.makedirs(os.path.join(args.inputs, "ABX"))
os.makedirs(orig_dir)
os.makedirs(adv_dir)

with open(args.ids) as f:
    files = [line.strip() for line in f]

with open(os.path.join(args.outdir, "ABX/test.conf"), "w") as out_f:
    for f in files:
        shutil.copy(
            os.path.join(wav_dir, f + "-orig.wav"), os.path.join(orig_dir, f + ".wav")
        )
        shutil.copy(
            os.path.join(wav_dir, f + "-adv.wav"), os.path.join(adv_dir, f + ".wav")
        )
        flip = np.random.randint(0, 2)
        if flip == 1:
            message = "A: orig, B: adv, "
        else:
            message = "A: adv, B: orig, "
        flip = np.random.randint(0, 2)
        if flip == 1:
            message += "x: A"
        else:
            message += "x: B"
        out_f.write(message + "\n")
