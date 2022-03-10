import os
import argparse
import matplotlib.pyplot as plt
from numpy.fft import rfft, rfftfreq
import soundfile as sf


def plot(ID, wavdir, pltdir, sr):
    for ext in ["-orig", "-adv"]:
        x, _ = sf.read(os.path.join(wavdir, ID + ext + ".wav"))
        X_F = rfft(x)
        freq = rfftfreq(len(x), d=1 / sr)
        for out in [(range(len(x)), x, "raw"), (freq, X_F, "fft")]:
            plt.plot(out[0], out[1])
            plt.savefig(os.path.join(os.path.join(pltdir, out[2]), ID + ext + ".png"))
            plt.clf()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--i")
    parser.add_argument("--w")
    parser.add_argument("--p")
    parser.add_argument("--sr")
    args = parser.parse_args()
    plot(args.i, args.w, args.p, args.sr)
