import tensorflow as tf
import librosa
import numpy as np
import functools
from scipy.signal import get_window
import soundfile as sf


def hamming(M, dtype):
    win = get_window("hamming", int(M))
    return tf.constant(win, dtype=dtype)


def power_to_db(S, ref=1.0, amin=1e-10, top_db=80.0):
    ref_value = np.abs(ref)
    log_spec = 10.0 * tf.math.log(tf.maximum(amin, S)) / np.log(10.0)
    log_spec = log_spec - 10.0 * tf.cast(
        tf.math.log(tf.maximum(amin, ref_value)) / np.log(10.0), tf.float64
    )
    log_spec = tf.maximum(log_spec, tf.math.reduce_max(log_spec) - top_db)
    return log_spec


def tf_stft(x, n_fft, hop_size):
    window_length = n_fft
    x = tf.pad(x, [[0, 0], [n_fft // 2, n_fft // 2]], "REFLECT")
    result = tf.signal.stft(
        x,
        frame_length=window_length,
        frame_step=hop_size,
        fft_length=n_fft,
        window_fn=functools.partial(hamming),
        pad_end=False,
    )
    return tf.transpose(tf.squeeze(result), perm=(1, 0))


def _spectrogram(y=None, S=None, n_fft=400, hop_length=160, power=1):
    S = tf.math.abs(tf_stft(y, n_fft=n_fft, hop_size=hop_length)) ** power
    return S, n_fft


def melspectrogram(
    y=None, sr=16000, S=None, n_fft=400, hop_length=160, power=2.0, **kwargs
):
    S, n_fft = _spectrogram(y=y, S=S, n_fft=n_fft, hop_length=hop_length, power=power)
    mel_basis = librosa.filters.mel(
        sr, n_fft, n_mels=30, fmin=20.0, fmax=7600, **kwargs
    ).astype(np.float64)
    return tf.tensordot(mel_basis, S, axes=((1,), (0,)))


def tf_mfcc(y, sr=16000, n_mfcc=24, dct_type=2, norm="ortho", **kwargs):
    S = tf.squeeze(power_to_db(melspectrogram(y=y, sr=sr, **kwargs)))
    return tf.transpose(
        tf.transpose(tf.signal.dct(tf.transpose(S), type=dct_type, norm=norm, axis=-1))[
            :n_mfcc
        ]
    )


def mfcc(wav_path):
    x = tf.Variable(sf.read(wav_path)[0])
    win = np.array([-0.97, 1], dtype=np.float64).reshape((1, 2, 1, 1))
    x = tf.reshape(
        tf.pad(x, tf.constant([[1, 1]]), "CONSTANT", constant_values=0),
        (1, 1, int(x.shape[0]) + 2, 1),
    )
    x = tf.squeeze(tf.nn.conv2d(x, win, [1, 1, 1, 1], "VALID"), axis=[0, 1, -1])[:-1]
    x = x.numpy()
    input = [x]

    audio = tf.Variable(input, dtype=tf.float64)
    feats = np.squeeze(tf_mfcc(audio).numpy())
    return feats


def tf_mfcc_pad(audio):
    win = np.array([-0.97, 1], dtype=np.float64).reshape((1, 2, 1, 1))
    x = tf.squeeze(audio)
    x = tf.reshape(
        tf.pad(x, tf.constant([[1, 1]]), "CONSTANT", constant_values=0),
        (1, 1, int(x.shape[0]) + 2, 1),
    )
    x = tf.squeeze(tf.nn.conv2d(x, win, [1, 1, 1, 1], "VALID"), axis=[0, 1, -1])[:-1]
    x = tf.expand_dims(x, 0)

    return tf_mfcc(x)
