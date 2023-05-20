import torch
import warnings
import numpy as np
import contextlib
import wave
import webrtcvad
import tempfile
import os
from scipy import signal
import torch.nn.functional as F
import librosa
import soundfile as sf
from scipy.signal import correlate

# pylint: disable=C0413
warnings.filterwarnings("ignore")
# pylint: enable=C0413

from pydub import AudioSegment
from pydub.silence import detect_leading_silence as dls

import random


def _seg_to_array(s):
    ret = np.array(s.get_array_of_samples())
    # ret = ret / np.max(np.abs(ret))
    return np.expand_dims(ret, 0)


def shift_pitch(x, fs=16000, factor=1.1, frac=0.005):
    x = x.squeeze()
    splits = np.split(
        x[: x.shape[-1] // int(fs * frac) * int(fs * frac)],
        x.shape[-1] // int(fs * frac),
    )
    rem = x[-(x.shape[-1] % int(fs * frac)) :]
    if len(rem) > 0:
        splits.append(rem)
    for i, s in enumerate(splits):
        freq = np.fft.rfftfreq(s.shape[-1], d=1 / fs)
        sf = np.fft.rfft(s)
        y = np.zeros(shape=freq.shape, dtype=np.complex128)
        for j in range(len(y)):
            idx = int(np.floor(factor * j))
            if idx >= len(y):
                break
            y[idx] = sf[j]
        res = np.fft.irfft(y)
        splits[i] = res
    # print(str(i) + ' ' + str(len(splits)))
    out = np.array([item for arr in splits for item in arr])
    return np.expand_dims(out, 0)


def _to_bytes(x, sample_rate=16000):
    tmp = tempfile.NamedTemporaryFile(suffix=".wav")
    tmp.close()
    path = tmp.name

    sf.write(path, x.squeeze(), sample_rate)
    with contextlib.closing(wave.open(path, "rb")) as wf:
        x_bytes = wf.readframes(wf.getnframes())
    os.unlink(path)
    return x_bytes


def calcVadRate(wav):
    x = trim(wav, silence_threshold=-25.0)
    x_bytes = _to_bytes(x, 16000)
    vad = webrtcvad.Vad(3)
    n = int(16000 * (30 / 1000.0) * 2)
    frames = [x_bytes[offset : offset + n] for offset in range(0, len(x_bytes) - n, n)]
    silent = [
        idx for idx, frame in enumerate(frames) if not vad.is_speech(frame, 16000)
    ]
    return float(len(silent)) / float(len(frames))


def _vad(x, frame_duration_ms=30, aggressive=2, sample_rate=16000):
    x_bytes = _to_bytes(x, sample_rate)
    vad = webrtcvad.Vad(aggressive)
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    frames = [x_bytes[offset : offset + n] for offset in range(0, len(x_bytes) - n, n)]

    voiced = [
        idx for idx, frame in enumerate(frames) if vad.is_speech(frame, sample_rate)
    ]

    win_sz = frame_duration_ms / 1000 * sample_rate
    num_frames = len(frames) + 1 if win_sz * len(frames) < x.shape[-1] else len(frames)
    x = np.pad(x.squeeze(), [0, int(num_frames * win_sz) - x.shape[-1]])

    silent = list(set(list(range(num_frames))) - set(voiced))
    silent.sort()
    voiced.sort()
    splits = np.split(x, num_frames)
    return np.stack(splits), voiced, silent


trim_leading: AudioSegment = lambda x, threshold, margin=0: x[
    np.max([0, dls(x, silence_threshold=threshold) - margin]) :
]
trim_trailing: AudioSegment = lambda x, threshold, margin=0: trim_leading(
    x.reverse(), threshold, -margin
).reverse()
strip_silence: AudioSegment = (
    lambda x, threshold, margin_l=0, margin_r=0: trim_trailing(
        trim_leading(x, threshold, margin_l), threshold, margin_r
    )
)


def silent_idx(mal, frame_duration_ms=30, aggressive=2, sample_rate=16000):
    splits, _, silent = _vad(mal, frame_duration_ms, aggressive, sample_rate)
    idx = np.array(
        [
            np.arange(i * len(splits[i]), (i + 1) * len(splits[i]))
            for i in range(len(splits))
            if i in silent
        ]
    )
    if silent[-1] == len(splits) - 1:
        idx = idx[:-1]
        rem = [np.arange((len(splits) - 1) * len(splits[0]), mal.shape[-1])]
        idx = np.concatenate((np.reshape(idx, np.prod(idx.shape)), rem[0]))
    return idx


def trim_mid(mal, frame_duration_ms=30, aggressive=2, sample_rate=16000):
    orig_len = mal.shape[-1]
    splits_m, voiced_m, silent_m = _vad(mal, frame_duration_ms, aggressive, sample_rate)
    """
    silent_m = np.tile(np.expand_dims(np.array(silent_m), -1),2).reshape((len(silent_m)*2)).tolist()
    voiced_m = voiced_m + silent_m
    voiced_m.sort()
    return splits_m[voiced_m]
    """
    return np.reshape(splits_m[voiced_m], (1, len(voiced_m) * splits_m.shape[-1]))[
        :, :orig_len
    ]


def subst_silence(mal, legit, frame_duration_ms=30, aggressive=2, sample_rate=16000):
    orig_len = mal.shape[-1]
    splits_m, _, silent_m = _vad(mal, frame_duration_ms, aggressive, sample_rate)
    splits_l, _, silent_l = _vad(
        legit, frame_duration_ms, aggressive=2, sample_rate=sample_rate
    )
    legit_silence = splits_l[silent_l]
    legit_silence = np.tile(
        legit_silence, (len(silent_m) // len(silent_l) + 1, 1)
    ).tolist()
    random.shuffle(legit_silence)
    splits_m[silent_m] = np.array(legit_silence)[: len(silent_m)]
    return np.reshape(splits_m, (1, len(splits_m) * splits_m.shape[-1]))[:, :orig_len]


def manip(x, fs=16000, factor=1.5, start_f=2000, end_f=5000, start_t=1, end_t=2):
    shape = x.shape[-1]
    f, t, Zxx = signal.stft(x, fs=fs)
    start_f = np.argmin(np.abs(f - start_f))
    end_f = np.argmin(np.abs(f - end_f))
    start_t = np.argmin(np.abs(t - start_t))
    end_t = np.argmin(np.abs(t - end_t))
    Zxx[:, start_f:end_f, start_t:end_t] *= factor
    x = signal.istft(Zxx, fs=fs)[1]
    x = x / np.max(np.abs(x))
    return x[:, :shape]


def trim_l(audio, silence_threshold=-25.0, margin=0):
    sound = AudioSegment.from_file(audio)
    stripped = trim_leading(sound, silence_threshold, margin)
    return _seg_to_array(stripped)


def trim_r(audio, silence_threshold=-25.0, margin=0):
    sound = AudioSegment.from_file(audio)
    stripped = trim_trailing(sound, silence_threshold, margin)
    return _seg_to_array(stripped)


def trim(audio, silence_threshold=-25.0, margin_l=0, margin_r=0):
    sound = AudioSegment.from_file(audio)
    stripped = strip_silence(sound, silence_threshold, margin_l, margin_r)
    return _seg_to_array(stripped)


def silence(sample, threshold=-25.0, type="leading"):
    trim_f = trim_leading if type == "leading" else trim_trailing
    padding = (
        lambda arr, start, type: (start, arr[:, :start])
        if type == "leading"
        else (-start, arr[:, -start:])
    )
    sample = AudioSegment.from_file(sample)
    start = (
        _seg_to_array(sample).shape[-1]
        - _seg_to_array(trim_f(sample, threshold=threshold)).shape[-1]
    )
    return padding(_seg_to_array(sample), start, type)


def pad_to_length(x, length):
    return np.tile(x, [1, length // x.shape[-1] + 1])[:, :length]


def freq_increase(data, lowcut, highcut, factor=1.5, fs=16000, order=5):
    need_expand = data.ndim != 1
    is_array = isinstance(data, np.ndarray)
    fft = torch.fft if not is_array else np.fft
    orig_len = data.shape[-1]

    data = fft.rfft(data.squeeze())
    b, a = signal.butter(order, [lowcut, highcut], fs=fs, btype="band")
    _, h = signal.freqz(b, a, fs=fs, worN=data.shape[-1])

    if not is_array:
        h = torch.tensor(h, device=data.device)
    h = h * (factor - 1) + 1
    data = fft.irfft(data * h, n=orig_len)
    if need_expand:
        data = np.expand_dims(data, 0) if is_array else data.unsqueeze(0)

    if is_array:
        return data  # / np.max(np.abs(data))
    return data  # / torch.max(torch.abs(data), -1).values


def butter_pass(data, lowcut, highcut, fs=16000, order=5):
    need_expand = data.ndim != 1
    is_array = isinstance(data, np.ndarray)
    fft = torch.fft if not is_array else np.fft
    orig_len = data.shape[-1]

    data = fft.rfft(data.squeeze())
    b, a = signal.butter(order, [lowcut, highcut], fs=fs, btype="band")
    _, h = signal.freqz(b, a, fs=fs, worN=data.shape[-1])

    if not is_array:
        h = torch.tensor(h, device=data.device)

    data = fft.irfft(data * h, n=orig_len)
    if need_expand:
        data = np.expand_dims(data, 0) if is_array else data.unsqueeze(0)

    if is_array:
        return data  # / np.max(np.abs(data))
    return data  # / torch.max(torch.abs(data), -1).values


def parabolic(f, x):
    xv = 1 / 2.0 * (f[x - 1] - f[x + 1]) / (f[x - 1] - 2 * f[x] + f[x + 1]) + x
    yv = f[x] - 1 / 4.0 * (f[x - 1] - f[x + 1]) * (xv - x)
    return (xv, yv)


def freq_from_autocorr(sig, fs):
    """
    Estimate frequency using autocorrelation
    """
    corr = correlate(sig, sig, mode="full")
    corr = corr[len(corr) // 2 :]
    d = np.diff(corr)
    start = np.nonzero(d > 0)[0][0]
    peak = np.argmax(corr[start:]) + start
    px, py = parabolic(corr, peak)

    return fs / px


def preemphasize(x, device="cuda:0", coef=0.97):
    need_expand = True if x.ndim == 1 else False

    isarray = isinstance(x, np.ndarray)
    if isarray:
        x = torch.tensor(x, device=device)
    x = x.unsquezze(0) if need_expand else x

    win = torch.tensor(
        np.array([-coef, 1], dtype=np.float64).reshape((1, 1, 1, 2)),
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


def deemphasize(x, coef=0.97):
    isarray = isinstance(x, np.ndarray)
    if not isarray:
        device = x.device
        x = x.detach().cpu().numpy()
    x = librosa.effects.deemphasis(x.squeeze(), coef=coef).reshape(x.shape)
    if not isarray:
        x = torch.tensor(x, device=device, requires_grad=True)
    return x
