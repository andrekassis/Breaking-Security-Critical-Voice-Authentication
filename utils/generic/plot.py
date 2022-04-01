import os
import argparse
from pathlib import Path
from numpy.fft import rfft, rfftfreq
import matplotlib.pyplot as plt
import soundfile as sf


def figures(outs, ID, pltdir):
    fig, ax = plt.subplots(2, 2)
    for idx, out in enumerate(outs):
        i = idx // 2
        j = idx % 2
        ax[i, j].plot(out[0], out[1])
        ax[i, j].set_title(out[2])
    plt.savefig(os.path.join(pltdir, ID + ".png"))
    plt.clf()


def plot(ID, wavdir, pltdir, sr=16000):
    x_orig = sf.read(os.path.join(wavdir, ID + "-orig" + ".wav"))[0]
    x_adv = sf.read(os.path.join(wavdir, ID + "-adv" + ".wav"))[0]
    X_F_orig = abs(rfft(x_orig))
    X_F_adv = abs(rfft(x_adv))
    freq = rfftfreq(len(x_orig), d=1 / sr)

    outs = [
        (range(len(x_orig)), x_orig, "orig"),
        (range(len(x_orig)), x_adv, "adv"),
        (freq, X_F_orig, "orig FFT"),
        (freq, X_F_adv, "adv FFT"),
    ]

    figures(outs, ID, pltdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--i")
    parser.add_argument("--w")
    parser.add_argument("--p")
    parser.add_argument("--sr", type=int)
    args = parser.parse_args()
    plot(args.i, args.w, args.p, args.sr)
