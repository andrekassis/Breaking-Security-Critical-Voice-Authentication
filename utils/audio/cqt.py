# pylint: disable=E1102

import math
from functools import lru_cache
import numpy as np
import torch
import scipy
import resampy
from librosa import get_fftlib


def _get_sinc_resample_kernel(
    orig_freq: int,
    new_freq: int,
    gcd: int,
    lowpass_filter_width: int,
    rolloff: float,
    resampling_method: str,
    beta: float,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = None,
):

    if resampling_method not in ["sinc_interpolation", "kaiser_window"]:
        raise ValueError("Invalid resampling method: {}".format(resampling_method))

    orig_freq = int(orig_freq) // gcd
    new_freq = int(new_freq) // gcd

    assert lowpass_filter_width > 0
    kernels = []
    base_freq = min(orig_freq, new_freq)

    width = math.ceil(lowpass_filter_width * orig_freq / base_freq)
    idx_dtype = dtype if dtype is not None else torch.float64
    idx = torch.arange(-width, width + orig_freq, device=device, dtype=idx_dtype)

    for i in range(new_freq):
        t = (-i / new_freq + idx / orig_freq) * base_freq
        t = t.clamp_(-lowpass_filter_width, lowpass_filter_width)

        if resampling_method == "sinc_interpolation":
            window = torch.cos(t * math.pi / lowpass_filter_width / 2) ** 2
        else:
            # kaiser_window
            if beta is None:
                beta = 14.769656459379492
            beta_tensor = torch.tensor(float(beta))
            window = torch.i0(
                beta_tensor * torch.sqrt(1 - (t / lowpass_filter_width) ** 2)
            ) / torch.i0(beta_tensor)
        t *= math.pi
        kernel = torch.where(t == 0, torch.tensor(1.0).to(t), torch.sin(t) / t)
        kernel.mul_(window)
        kernels.append(kernel)

    scale = base_freq / orig_freq
    kernels = torch.stack(kernels).view(new_freq, 1, -1).mul_(scale)
    if dtype is None:
        kernels = kernels.to(dtype=torch.float32)
    return kernels, width


def _apply_sinc_resample_kernel(
    waveform: torch.Tensor,
    orig_freq: int,
    new_freq: int,
    gcd: int,
    kernel: torch.Tensor,
    width: int,
):
    orig_freq = int(orig_freq) // gcd
    new_freq = int(new_freq) // gcd

    # pack batch
    shape = waveform.size()
    waveform = waveform.view(-1, shape[-1])

    num_wavs, length = waveform.shape
    waveform = torch.nn.functional.pad(waveform, (width, width + orig_freq))
    resampled = torch.nn.functional.conv1d(waveform[:, None], kernel, stride=orig_freq)
    resampled = resampled.transpose(1, 2).reshape(num_wavs, -1)
    target_length = int(math.ceil(new_freq * length / orig_freq))
    resampled = resampled[..., :target_length]

    # unpack batch
    resampled = resampled.view(shape[:-1] + resampled.shape[-1:])
    return resampled


def resample_tensor(
    waveform,
    orig_freq,
    new_freq,
    lowpass_filter_width=6,
    rolloff=0.99,
    resampling_method="sinc_interpolation",
    beta=None,
):

    assert orig_freq > 0.0 and new_freq > 0.0

    if orig_freq == new_freq:
        return waveform

    gcd = math.gcd(int(orig_freq), int(new_freq))

    kernel, width = _get_sinc_resample_kernel(
        orig_freq,
        new_freq,
        gcd,
        lowpass_filter_width,
        rolloff,
        resampling_method,
        beta,
        waveform.device,
        waveform.dtype,
    )
    resampled = _apply_sinc_resample_kernel(
        waveform, orig_freq, new_freq, gcd, kernel, width
    )
    return resampled


def pad(x, leng):
    l_orig = x.shape[-1]
    x1 = x[:, 1:]
    x2 = torch.flip(x, [-1])[:, 1:]
    len1 = x1.shape[-1]
    len2 = x2.shape[-1]

    while len1 - l_orig - leng + 1 < 0:
        x1 = torch.cat((x, x2[:, :leng]), axis=-1)[:, 1 : l_orig + leng]
        x2 = torch.cat((torch.flip(x, [-1]), x1), axis=-1)[:, 1 : l_orig + leng]
        len1 = x1.shape[-1]

    res = torch.cat((torch.flip(x2, [-1])[:, :leng], x, x1[:, -leng:]), axis=-1)
    return res


def tiny(x):
    x = np.asarray(x)

    # Only floating types generate a tiny
    if np.issubdtype(x.dtype, np.floating) or np.issubdtype(
        x.dtype, np.complexfloating
    ):
        dtype = x.dtype
    else:
        dtype = np.float32

    return np.finfo(dtype).tiny


def get_window(window, Nx, fftbins=True):
    if callable(window):
        return window(Nx)

    if isinstance(window, (str, tuple)) or np.isscalar(window):
        return scipy.signal.get_window(window, Nx, fftbins=fftbins)

    if isinstance(window, (np.ndarray, list)):
        if len(window) == Nx:
            return np.asarray(window)


def pad_center(data, size, axis=-1, **kwargs):
    kwargs.setdefault("mode", "constant")

    n = data.shape[axis]

    lpad = int((size - n) // 2)

    lengths = [(0, 0)] * data.ndim
    lengths[axis] = (lpad, int(size - n - lpad))

    return np.pad(data, lengths, **kwargs)


def normalize(S, norm=np.inf, axis=0, threshold=None, fill=None):
    if threshold is None:
        threshold = tiny(S)

    # All norms only depend on magnitude, let's do that first
    mag = np.abs(S).astype(np.float)

    # For max/min norms, filling with 1 works
    fill_norm = 1

    if norm == np.inf:
        length = np.max(mag, axis=axis, keepdims=True)

    elif norm == -np.inf:
        length = np.min(mag, axis=axis, keepdims=True)

    elif norm == 0:
        length = np.sum(mag > 0, axis=axis, keepdims=True, dtype=mag.dtype)

    elif np.issubdtype(type(norm), np.number) and norm > 0:
        length = np.sum(mag**norm, axis=axis, keepdims=True) ** (1.0 / norm)

        if axis is None:
            fill_norm = mag.size ** (-1.0 / norm)
        else:
            fill_norm = mag.shape[axis] ** (-1.0 / norm)

    elif norm is None:
        return S

    small_idx = length < threshold

    Snorm = np.empty_like(S)
    if fill is None:

        length[small_idx] = 1.0
        Snorm[:] = S / length

    elif fill:
        length[small_idx] = np.nan
        Snorm[:] = S / length
        Snorm[np.isnan(Snorm)] = fill_norm
    else:
        length[small_idx] = np.inf
        Snorm[:] = S / length

    return Snorm


def __float_window(window_spec):
    def _wrap(n, *args, **kwargs):
        """The wrapped window"""
        n_min, n_max = int(np.floor(n)), int(np.ceil(n))

        window = get_window(window_spec, n_min)

        if len(window) < n_max:
            window = np.pad(window, [(0, n_max - len(window))], mode="constant")

        window[n_min:] = 0.0

        return window

    return _wrap


def constant_q(
    sr,
    fmin=None,
    n_bins=84,
    bins_per_octave=12,
    window="hann",
    filter_scale=1,
    pad_fft=True,
    norm=1,
    dtype=np.complex64,
    gamma=0,
    **kwargs,
):

    # Pass-through parameters to get the filter lengths
    lengths = constant_q_lengths(
        sr,
        fmin,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        window=window,
        filter_scale=filter_scale,
        gamma=gamma,
    )

    freqs = fmin * (2.0 ** (np.arange(n_bins, dtype=float) / bins_per_octave))

    # Build the filters
    filters = []
    for ilen, freq in zip(lengths, freqs):
        # Build the filter: note, length will be ceil(ilen)
        sig = np.exp(
            np.arange(-ilen // 2, ilen // 2, dtype=float) * 1j * 2 * np.pi * freq / sr
        )

        # Apply the windowing function
        sig = sig * __float_window(window)(len(sig))
        sig = normalize(sig, norm=norm)

        filters.append(sig)

    # Pad and stack
    max_len = max(lengths)
    if pad_fft:
        max_len = int(2.0 ** (np.ceil(np.log2(max_len))))
    else:
        max_len = int(np.ceil(max_len))

    filters = np.asarray(
        [pad_center(filt, max_len, **kwargs) for filt in filters], dtype=dtype
    )

    return filters, np.asarray(lengths)


WINDOW_BANDWIDTHS = {
    "bart": 1.3334961334912805,
    "barthann": 1.4560255965133932,
    "bartlett": 1.3334961334912805,
    "bkh": 2.0045975283585014,
    "black": 1.7269681554262326,
    "blackharr": 2.0045975283585014,
    "blackman": 1.7269681554262326,
    "blackmanharris": 2.0045975283585014,
    "blk": 1.7269681554262326,
    "bman": 1.7859588613860062,
    "bmn": 1.7859588613860062,
    "bohman": 1.7859588613860062,
    "box": 1.0,
    "boxcar": 1.0,
    "brt": 1.3334961334912805,
    "brthan": 1.4560255965133932,
    "bth": 1.4560255965133932,
    "cosine": 1.2337005350199792,
    "flat": 2.7762255046484143,
    "flattop": 2.7762255046484143,
    "flt": 2.7762255046484143,
    "halfcosine": 1.2337005350199792,
    "ham": 1.3629455320350348,
    "hamm": 1.3629455320350348,
    "hamming": 1.3629455320350348,
    "han": 1.50018310546875,
    "hann": 1.50018310546875,
    "hanning": 1.50018310546875,
    "nut": 1.9763500280946082,
    "nutl": 1.9763500280946082,
    "nuttall": 1.9763500280946082,
    "ones": 1.0,
    "par": 1.9174603174603191,
    "parz": 1.9174603174603191,
    "parzen": 1.9174603174603191,
    "rect": 1.0,
    "rectangular": 1.0,
    "tri": 1.3331706523555851,
    "triang": 1.3331706523555851,
    "triangle": 1.3331706523555851,
}


def sparsify_rows(x, quantile=0.01, dtype=None):
    if x.ndim == 1:
        x = x.reshape((1, -1))

    if dtype is None:
        dtype = x.dtype

    x_sparse = scipy.sparse.lil_matrix(x.shape, dtype=dtype)

    mags = np.abs(x)
    norms = np.sum(mags, axis=1, keepdims=True)

    mag_sort = np.sort(mags, axis=1)
    cumulative_mag = np.cumsum(mag_sort / norms, axis=1)

    threshold_idx = np.argmin(cumulative_mag < quantile, axis=1)

    for i, j in enumerate(threshold_idx):
        idx = np.where(mags[i] >= mag_sort[i, j])
        x_sparse[i, idx] = x[i, idx]

    return x_sparse.tocsr()


def fix_length(data, size, axis=-1, **kwargs):
    kwargs.setdefault("mode", "constant")

    n = data.shape[axis]

    if n > size:
        slices = [slice(None)] * data.ndim
        slices[axis] = slice(0, size)
        return data[tuple(slices)]

    if n < size:
        lengths = [(0, 0)] * data.ndim
        lengths[axis] = (0, size - n)
        return np.pad(data, lengths, **kwargs)

    return data


def resample(
    y, orig_sr, target_sr, res_type="kaiser_best", fix=True, scale=False, **kwargs
):
    if orig_sr == target_sr:
        return y

    ratio = float(target_sr) / orig_sr
    n_samples = int(np.ceil(y.shape[-1] * ratio))
    if res_type == "kaiser_best":
        y_hat = resample_tensor(
            y,
            orig_sr,
            target_sr,
            lowpass_filter_width=64,
            rolloff=0.9475937167399596,
            resampling_method="kaiser_window",
            beta=14.769656459379492,
        )
    elif res_type == "kaiser_fast":
        y_hat = resample_tensor(
            y,
            orig_sr,
            target_sr,
            lowpass_filter_width=16,
            rolloff=0.85,
            resampling_method="kaiser_window",
            beta=8.555504641634386,
        )

    y_hat = fix_length(y_hat, n_samples, **kwargs)
    y_hat /= np.sqrt(ratio)

    return y_hat


def window_bandwidth(window, n=1000):
    if hasattr(window, "__name__"):
        key = window.__name__
    else:
        key = window

    return WINDOW_BANDWIDTHS[key]


def constant_q_lengths(
    sr, fmin, n_bins=84, bins_per_octave=12, window="hann", filter_scale=1, gamma=0
):
    alpha = 2.0 ** (1.0 / bins_per_octave) - 1.0
    Q = float(filter_scale) / alpha

    # Compute the frequencies
    freq = fmin * (2.0 ** (np.arange(n_bins, dtype=float) / bins_per_octave))
    # Convert frequencies to filter lengths
    lengths = Q * sr / (freq + gamma / alpha)

    return lengths


def __num_two_factors(x):
    if x <= 0:
        return 0
    num_twos = 0
    while x % 2 == 0:
        num_twos += 1
        x //= 2

    return num_twos


def __early_downsample_count(nyquist, filter_cutoff, hop_length, n_octaves):
    BW_FASTEST = resampy.filters.get_filter("kaiser_fast")[2]

    downsample_count1 = max(
        0, int(np.ceil(np.log2(BW_FASTEST * nyquist / filter_cutoff)) - 1) - 1
    )

    num_twos = __num_two_factors(hop_length)
    downsample_count2 = max(0, num_twos - n_octaves + 1)

    return min(downsample_count1, downsample_count2)


def __early_downsample(
    y, sr, hop_length, res_type, n_octaves, nyquist, filter_cutoff, scale
):

    downsample_count = __early_downsample_count(
        nyquist, filter_cutoff, hop_length, n_octaves
    )

    if downsample_count > 0 and res_type == "kaiser_fast":
        downsample_factor = 2 ** (downsample_count)

        hop_length //= downsample_factor
        new_sr = sr / float(downsample_factor)
        y = resample(y, sr, new_sr, res_type=res_type, scale=True)

        if not scale:
            y *= np.sqrt(downsample_factor)

        sr = new_sr

    return y, sr, hop_length


def cqt_frequencies(n_bins, fmin, bins_per_octave=12, tuning=0.0):
    correction = 2.0 ** (float(tuning) / bins_per_octave)
    frequencies = 2.0 ** (np.arange(0, n_bins, dtype=float) / bins_per_octave)

    return correction * fmin * frequencies


def __trim_stack(cqt_resp, n_bins):
    cqt_resp = [
        [cqt_resp[i][j] for i in range(len(cqt_resp))] for j in range(len(cqt_resp[0]))
    ]
    max_col = [min(c_i.shape[-1] for c_i in c) for c in cqt_resp]
    cqt_out = [
        torch.zeros(
            (n_bins, max_col[i]),
            dtype=cqt_resp[0][0].dtype,
            device=cqt_resp[0][0].device,
        )
        for i in range(len(max_col))
    ]

    for i, _ in enumerate(cqt_resp):
        end = n_bins
        for c_i in cqt_resp[i]:
            n_oct = c_i.shape[0]
            if end < n_oct:
                cqt_out[i][:end] = c_i[-end:, : max_col[i]]
            else:
                cqt_out[i][end - n_oct : end] = c_i[:, : max_col[i]]

            end -= n_oct
    cqt_out = torch.stack(cqt_out)
    return cqt_out


@lru_cache(maxsize=None)
def __cqt_filter_fft(
    sr,
    fmin,
    n_bins,
    bins_per_octave,
    filter_scale,
    norm,
    sparsity,
    hop_length=None,
    window="hann",
    gamma=0.0,
    dtype=np.complex,
):
    """Generate the frequency domain constant-Q filter basis."""
    basis, lengths = constant_q(
        sr,
        fmin=fmin,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        filter_scale=filter_scale,
        norm=norm,
        pad_fft=True,
        window=window,
        gamma=gamma,
    )

    # Filters are padded up to the nearest integral power of 2
    n_fft = basis.shape[1]

    if hop_length is not None and n_fft < 2.0 ** (1 + np.ceil(np.log2(hop_length))):

        n_fft = int(2.0 ** (1 + np.ceil(np.log2(hop_length))))

    # re-normalize bases with respect to the FFT window length
    basis *= lengths[:, np.newaxis] / float(n_fft)

    # FFT and retain only the non-negative frequencies
    fft = get_fftlib()
    fft_basis = fft.fft(basis, n=n_fft, axis=1)[:, : (n_fft // 2) + 1]

    # sparsify the basis
    fft_basis = sparsify_rows(fft_basis, quantile=sparsity, dtype=dtype)

    return fft_basis, n_fft, lengths


def __cqt_response(y, n_fft, hop_length, fft_basis, mode, dtype=None):
    """Compute the filter response with a target STFT hop."""

    y = pad(y, n_fft // 2)
    # Compute the STFT matrix
    D = torch.stft(
        y,
        n_fft,
        hop_length=hop_length,
        window=None,
        center=False,
        pad_mode=mode,
        normalized=False,
        onesided=None,
        return_complex=True,
    )
    # And filter response energy
    fft_basis = fft_basis.toarray()
    fft_basis = torch.tensor(fft_basis, device=D.device)
    return torch.matmul(fft_basis, D)


def cqt(
    y,
    sr=22050,
    hop_length=512,
    fmin=None,
    n_bins=84,
    bins_per_octave=12,
    tuning=0.0,
    filter_scale=1,
    norm=1,
    sparsity=0.01,
    window="hann",
    scale=True,
    pad_mode="reflect",
):
    dtype = np.complex128
    gamma = 0
    BW_FASTEST = resampy.filters.get_filter("kaiser_fast")[2]

    # How many octaves are we dealing with?
    n_octaves = int(np.ceil(float(n_bins) / bins_per_octave))
    n_filters = min(bins_per_octave, n_bins)

    # Relative difference in frequency between any two consecutive bands
    alpha = 2.0 ** (1.0 / bins_per_octave) - 1

    # First thing, get the freqs of the top octave
    freqs = cqt_frequencies(n_bins, fmin, bins_per_octave=bins_per_octave)[
        -bins_per_octave:
    ]

    fmin_t = np.min(freqs)
    fmax_t = np.max(freqs)

    # Determine required resampling quality
    Q = float(filter_scale) / alpha
    filter_cutoff = fmax_t * (1 + 0.5 * window_bandwidth(window) / Q) + 0.5 * gamma
    nyquist = sr / 2.0

    auto_resample = True
    if filter_cutoff < BW_FASTEST * nyquist:
        res_type = "kaiser_fast"
    else:
        res_type = "kaiser_best"

    y, sr, hop_length = __early_downsample(
        y, sr, hop_length, res_type, n_octaves, nyquist, filter_cutoff, scale
    )

    vqt_resp = []

    # Skip this block for now
    if auto_resample and res_type != "kaiser_fast":

        # Do the top octave before resampling to allow for fast resampling
        fft_basis, n_fft, _ = __cqt_filter_fft(
            sr,
            fmin_t,
            n_filters,
            bins_per_octave,
            filter_scale,
            norm,
            sparsity,
            window=window,
            gamma=gamma,
            dtype=dtype,
        )

        # Compute the VQT filter response and append it to the stack
        vqt_resp.append(
            __cqt_response(y, n_fft, hop_length, fft_basis, pad_mode, dtype=dtype)
        )

        fmin_t /= 2
        fmax_t /= 2
        n_octaves -= 1

        filter_cutoff = fmax_t * (1 + 0.5 * window_bandwidth(window) / Q)

        res_type = "kaiser_fast"

    # Make sure our hop is long enough to support the bottom octave
    num_twos = __num_two_factors(hop_length)

    # Now do the recursive bit
    my_y, my_sr, my_hop = y, sr, hop_length

    # Iterate down the octaves
    for i in range(n_octaves):
        # Resample (except first time)
        if i > 0:
            my_y = resample(my_y, 2, 1, res_type=res_type, scale=True)
            my_sr /= 2.0
            my_hop //= 2

        fft_basis, n_fft, _ = __cqt_filter_fft(
            my_sr,
            fmin_t * 2.0**-i,
            n_filters,
            bins_per_octave,
            filter_scale,
            norm,
            sparsity,
            window=window,
            gamma=gamma,
            dtype=dtype,
        )

        fft_b = fft_basis[:] * np.sqrt(2**i)
        vqt_resp.append(
            __cqt_response(my_y, n_fft, my_hop, fft_b, pad_mode, dtype=dtype)
        )

    V = __trim_stack(vqt_resp, n_bins)

    if scale:
        lengths = constant_q_lengths(
            sr,
            fmin,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
            window=window,
            filter_scale=filter_scale,
            gamma=gamma,
        )
        V = V / torch.sqrt(torch.tensor(lengths, device=V.device)[:, None])
    return V
