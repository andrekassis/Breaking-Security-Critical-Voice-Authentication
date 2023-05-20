import os
import argparse
from pathlib import Path
from numpy.fft import rfft, rfftfreq
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import soundfile as sf
import scipy.signal as signal
import numpy as np


def figures(outs, ID, pltdir):
    fig, ax = plt.subplots(3, 2)
    for idx, out in enumerate(outs):
        i = idx // 2
        j = idx % 2
        getattr(ax[i, j], out[-2])(*out[0:-2])
        ax[i, j].set_title(out[-1])
    fig.tight_layout()
    plt.savefig(os.path.join(pltdir, ID + ".png"))
    plt.clf()


def plot(ID, wavdir, pltdir, sr=16000, ext=""):
    x_orig = sf.read(os.path.join(wavdir, ID + "-orig.wav"))[0]
    x_adv = sf.read(os.path.join(wavdir, ID + "-" + ext + "-adv.wav"))[0]
    X_F_orig = abs(rfft(x_orig))
    X_F_adv = abs(rfft(x_adv))
    freq = rfftfreq(len(x_orig), d=1 / sr)
    freq_a = rfftfreq(len(x_adv), d=1 / sr)
    f_orig, t_orig, Zxx_orig = signal.stft(x_orig, fs=sr)
    f_adv, t_adv, Zxx_adv = signal.stft(x_adv, fs=sr)

    outs = [
        (range(len(x_orig)), x_orig, "plot", "orig"),
        (range(len(x_adv)), x_adv, "plot", "adv"),
        (freq, X_F_orig, "plot", "orig FFT"),
        (freq_a, X_F_adv, "plot", "adv FFT"),
        (t_orig, f_orig, np.abs(Zxx_orig), "pcolormesh", "orig STFT"),
        (t_adv, f_adv, np.abs(Zxx_adv), "pcolormesh", "adv STFT"),
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
