import torch
import numpy as np
from scipy import signal
import torch.nn.functional as F
import librosa


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    need_expand = data.ndim != 1
    b, a = signal.butter(order, [lowcut, highcut], fs=fs, btype="band")
    y = signal.lfilter(b, a, data.squeeze()).copy()
    y = np.expand_dims(y, 0) if need_expand else y
    return y


def freq_response(lowcut, highcut, pts, fs=16000, order=5):
    b, a = signal.butter(order, [lowcut, highcut], fs=fs, btype="band")
    w, h = signal.freqz(b, a, fs=fs, worN=pts)
    return w, h


def freq_increase(data, lowcut, highcut, factor=1.5, fs=16000, order=5):
    need_expand = data.ndim != 1
    is_array = isinstance(data, np.ndarray)
    fft = torch.fft if not is_array else np.fft

    data = fft.rfft(data.squeeze())
    b, a = signal.butter(order, [lowcut, highcut], fs=fs, btype="band")
    _, h = signal.freqz(b, a, fs=fs, worN=data.shape[-1])

    if not is_array:
        h = torch.tensor(h, device=data.device)
    h = h * (factor - 1) + 1
    data = fft.irfft(data * h)
    if need_expand:
        data = np.expand_dims(data, 0) if is_array else data.unsqueeze(0)

    if is_array:
        return data / np.max(np.abs(data))
    return data / torch.max(torch.abs(data), -1).values


def freq_increase_raw(data, lowcut, highcut, factor=1.5, fs=16000, order=5):
    is_array = isinstance(data, np.ndarray)

    b, a = signal.butter(order, [lowcut, highcut], fs=fs, btype="band")
    _, h = signal.freqz(b, a, fs=fs, worN=data.shape[-1])
    h = np.reshape(h, data.shape)
    h = torch.tensor(h, device=data.device) if not is_array else h

    h = h * (factor - 1) + 1
    return data * h


def preemphasize(x, device="cuda:0"):
    need_expand = True if x.ndim == 1 else False

    isarray = isinstance(x, np.ndarray)
    if isarray:
        x = torch.tensor(x, device=device)
    x = x.unsquezze(0) if need_expand else x

    win = torch.tensor(
        np.array([-0.97, 1], dtype=np.float64).reshape((1, 1, 1, 2)),
        device=x.device,
    ).type(x.type())
    x = torch.reshape(
        F.pad(x, [1, 1], "constant"), (x.shape[0], 1, 1, int(x.shape[1]) + 2)
    )
    x = F.conv2d(x, win).squeeze(1).squeeze(1)[:, :-1].type(x.type())
    x = x.squeeze() if need_expand else x
    if isarray:
        x = x.detach().cpu().numpy()
    return x


def deemphasize(x):
    isarray = isinstance(x, np.ndarray)
    if not isarray:
        device = x.device
        x = x.detach().cpu().numpy()
    x = librosa.effects.deemphasis(x.squeeze()).reshape(x.shape)
    if not isarray:
        x = torch.tensor(x, device=device, requires_grad=True)
    return x
