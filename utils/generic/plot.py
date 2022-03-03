import matplotlib.pyplot as plt
from numpy.fft import rfft as fft
import os
import soundfile as sf

def plot(id, wavdir, pltdir):
    for ext in ['-orig', '-adv']:
        x, _ = sf.read(os.path.join(wavdir, id + ext + '.wav'))
        xf = fft(x)
        for out in [(x, 'raw'), (xf, 'fft')]:
            plt.plot(range(len(out[0])), out[0])
            plt.savefig(os.path.join(os.path.join(pltdir, out[1]), id + ext + '.png'))
            plt.clf()
