# pylint: disable=E1102, E1111
from abc import ABC
import numpy as np
import torch
import torch.nn.functional as F

torch.set_flush_denormal(True)


class SpectralGate(ABC):
    # pylint: disable=R0902, R0903, R0913
    def __init__(
        self,
        sr=16000,
        time_constant_s=2.0,
        freq_mask_smooth_hz=500,
        time_mask_smooth_ms=50,
        padding=30000,
        n_fft=1024,
        win_length=None,
        hop_length=None,
        win="hann_window",
        device="cuda:0",
    ):
        self.sr = sr
        self.device = device
        self.padding = padding

        self._n_fft = n_fft
        if win_length is None:
            self._win_length = self._n_fft
        else:
            self._win_length = win_length
        if hop_length is None:
            self._hop_length = self._win_length // 4
        else:
            self._hop_length = hop_length

        win_args = {
            "window_length": self._win_length,
            "device": self.device,
            "requires_grad": False,
        }
        if win is not None:
            self.win = getattr(torch, win)(**win_args)
        else:
            self.win = None
        self._time_constant_s = time_constant_s
        self._generate_mask_smoothing_filter(freq_mask_smooth_hz, time_mask_smooth_ms)

        self.length = None

    def _set_len(self, length):
        self.length = length

    def _filter(self, n_grad_freq, n_grad_time):
        smoothing_filter = np.outer(
            np.concatenate(
                [
                    np.linspace(0, 1, n_grad_freq + 1, endpoint=False),
                    np.linspace(1, 0, n_grad_freq + 2),
                ]
            )[1:-1],
            np.concatenate(
                [
                    np.linspace(0, 1, n_grad_time + 1, endpoint=False),
                    np.linspace(1, 0, n_grad_time + 2),
                ]
            )[1:-1],
        )
        smoothing_filter = smoothing_filter / np.sum(smoothing_filter)
        return torch.tensor(np.copy(smoothing_filter[::-1, ::-1]), device=self.device)

    def _generate_mask_smoothing_filter(self, freq_mask_smooth_hz, time_mask_smooth_ms):
        n_grad_freq = int(freq_mask_smooth_hz / (self.sr / (self._n_fft / 2)))
        n_grad_time = int(time_mask_smooth_ms / ((self._hop_length / self.sr) * 1000))

        if (n_grad_time == 1) & (n_grad_freq == 1):
            self.smooth_mask = False
        else:
            self.smooth_mask = True
            self._smoothing_filter = self._filter(n_grad_freq, n_grad_time)

    def _stft(self, x):
        return torch.stft(
            x,
            n_fft=self._n_fft,
            hop_length=self._hop_length,
            win_length=self._win_length,
            window=self.win,
            center=True,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )

    def _istft(self, x):
        return torch.istft(
            x,
            n_fft=self._n_fft,
            hop_length=self._hop_length,
            win_length=self._win_length,
            window=self.win,
            center=True,
            length=self.length + self.padding * 2,
            normalized=False,
            onesided=True,
            return_complex=False,
        )

    def _smoothen(self, x):
        if self.smooth_mask:
            padding = (
                self._smoothing_filter.shape[1] // 2,
                self._smoothing_filter.shape[1] // 2
                - (self._smoothing_filter.shape[1] + 1) % 2,
                self._smoothing_filter.shape[0] // 2,
                self._smoothing_filter.shape[0] // 2
                - (self._smoothing_filter.shape[0] + 1) % 2,
            )
            x = F.pad(x, padding)
            x = F.conv2d(
                x.unsqueeze(1), self._smoothing_filter.unsqueeze(0).unsqueeze(0)
            ).squeeze(1)
        return x

    def _calc_mask(self, x, n_std_thresh_stationary=None):
        pass

    def _apply_mask(self, sig_mask, p):
        pass

    def _do_filter(self, x, p, n_std_thresh_stationary=1.5):
        sig_stft = self._stft(x)
        abs_sig_stft = torch.abs(sig_stft)
        sig_mask = self._calc_mask(
            abs_sig_stft, n_std_thresh_stationary=n_std_thresh_stationary
        )
        sig_mask = self._apply_mask(sig_mask, p)
        return self._istft(sig_stft * sig_mask)

    def __call__(self, x, p=1.0, n_std_thresh_stationary=1.5):
        if p is None:
            return x

        rev = False
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, device=self.device)
            rev = True

        if self.length != x.shape[-1]:
            self._set_len(x.shape[-1])

        padded = F.pad(x, (self.padding, self.padding))
        filtered_padded_chunk = self._do_filter(
            padded, p, n_std_thresh_stationary=n_std_thresh_stationary
        )
        ret = filtered_padded_chunk[:, self.padding : x.shape[-1] + self.padding]
        # ret = ret / torch.max(torch.abs(ret))

        if rev:
            return ret.cpu().numpy()

        return ret


class SpectralGateNonStationary(SpectralGate):
    # pylint: disable=R0903, R0913
    def __init__(
        self,
        sr=16000,
        time_constant_s=2.0,
        freq_mask_smooth_hz=500,
        time_mask_smooth_ms=50,
        thresh_n_mult_nonstationary=2,
        sigmoid_slope_nonstationary=10,
        padding=30000,
        n_fft=1024,
        win_length=None,
        hop_length=None,
        win="hann_window",
        device="cuda:0",
    ):

        super().__init__(
            sr=sr,
            time_constant_s=time_constant_s,
            freq_mask_smooth_hz=freq_mask_smooth_hz,
            time_mask_smooth_ms=time_mask_smooth_ms,
            padding=padding,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            win=win,
            device=device,
        )

        self._thresh_n_mult_nonstationary = thresh_n_mult_nonstationary
        self._sigmoid_slope_nonstationary = sigmoid_slope_nonstationary
        self.mat = None

    def _set_len(self, length):
        super()._set_len(length)

        t_frames = self._time_constant_s * self.sr / float(self._hop_length)
        b = (np.sqrt(1 + 4 * t_frames**2) - 1) / (2 * t_frames**2)
        dim = int((length + 2 * self.padding) // self._hop_length) + 1
        ones = torch.ones((1, 1, dim, dim))
        self.mat = torch.triu(ones)
        rows = torch.tensor(
            [
                1 / np.power(1 - b, k) if k == 0 else b / np.power(1 - b, k)
                for k in range(dim)
            ]
        )
        rows = rows.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        cols = torch.tensor([(1 - b) ** k for k in range(dim)])
        cols = cols.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        self.mat = cols * self.mat * rows

        # self.mat = torch.transpose(self.mat, 2,3)
        self.mat = self.mat.to(self.device)

    def _run_filter(self, feat):
        # out = torch.matmul(self.mat, feat.unsqueeze(-1)).squeeze(-1)
        out = (feat.unsqueeze(-1) * self.mat).sum(axis=2)
        out = torch.flip(out, [-1])
        return out

    def _calc_mask(self, x, n_std_thresh_stationary=None):
        sig_stft_smooth = self._run_filter(self._run_filter(x))
        sig_mult_above_thresh = (x - sig_stft_smooth) / sig_stft_smooth
        return 1 / (
            1
            + torch.exp(
                -(sig_mult_above_thresh - self._thresh_n_mult_nonstationary)
                * self._sigmoid_slope_nonstationary
            )
        )

    def _apply_mask(self, sig_mask, p):
        sig_mask = self._smoothen(sig_mask)
        return sig_mask * p + torch.ones(sig_mask.shape, device=self.device) * (1.0 - p)


class SpectralGateStationary(SpectralGate):
    # pylint: disable=R0903, R0913
    def __init__(
        self,
        sr=16000,
        time_constant_s=2.0,
        freq_mask_smooth_hz=500,
        time_mask_smooth_ms=50,
        padding=30000,
        n_fft=1024,
        win_length=None,
        hop_length=None,
        win="hann_window",
        device="cuda:0",
    ):

        super().__init__(
            sr=sr,
            time_constant_s=time_constant_s,
            freq_mask_smooth_hz=freq_mask_smooth_hz,
            time_mask_smooth_ms=time_mask_smooth_ms,
            padding=padding,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            win=win,
            device=device,
        )

    def _calc_mask(self, x, n_std_thresh_stationary=1.5):
        sig_stft_db = 10 * torch.log10(
            torch.maximum(torch.tensor(1e-40, device=x.device), torch.abs(x) ** 2)
        )
        sig_stft_db = torch.maximum(sig_stft_db, sig_stft_db.max() - 80.0)
        return (
            sig_stft_db
            > (
                torch.mean(sig_stft_db, -1)
                + torch.std(sig_stft_db, -1) * n_std_thresh_stationary
            ).t()
        )

    def _apply_mask(self, sig_mask, p):
        sig_mask = sig_mask * p + torch.ones(
            sig_mask.shape, device=self.device
        ).double() * (1.0 - p)
        return self._smoothen(sig_mask)


def sp(stationary, **kwargs):
    if stationary:
        return SpectralGateStationary(**kwargs)
    return SpectralGateNonStationary(**kwargs)
