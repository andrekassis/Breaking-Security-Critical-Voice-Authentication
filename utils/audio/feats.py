# pylint: disable=E1102

import torch
from torch import Tensor
import torch.nn.functional as F
import librosa
import numpy as np
from .cqt import cqt


class _AxisMasking(torch.nn.Module):
    __constants__ = ["mask_param", "axis", "iid_masks"]

    def __init__(self, mask_param: int, axis: int, iid_masks: bool) -> None:

        super(_AxisMasking, self).__init__()
        self.mask_param = mask_param
        self.axis = axis
        self.iid_masks = iid_masks

    def forward(self, specgram: Tensor, mask_value: float = 0.0) -> Tensor:
        if self.iid_masks and specgram.dim() == 4:
            return F.mask_along_axis_iid(
                specgram, self.mask_param, mask_value, self.axis + 1
            )
        return F.mask_along_axis(specgram, self.mask_param, mask_value, self.axis)


class FrequencyMasking(_AxisMasking):
    def __init__(self, freq_mask_param: int, iid_masks: bool = False) -> None:
        super(FrequencyMasking, self).__init__(freq_mask_param, 1, iid_masks)


def dct(x, norm=None):
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)
    Vc = torch.fft.fft(v)
    Vc = torch.stack([Vc.real, Vc.imag], dim=-1)

    k = -torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)
    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == "ortho":
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2
    V = 2 * V.view(*x_shape)

    return V


class Null(torch.nn.Module):
    def __init__(self, device="cuda:0"):
        super(Null, self).__init__()

    def forward(self, x):
        return x


class LinearDCT(torch.nn.Linear):
    def __init__(self, in_features, norm=None, bias=False, device="cuda:0"):
        self.N = in_features
        self.norm = norm
        self.device = device
        super(LinearDCT, self).__init__(in_features, in_features, bias=bias)

    def reset_parameters(self):
        I = torch.eye(self.N, device=self.device)
        self.weight.data = dct(I, norm=self.norm).data.t()
        self.weight.requires_grad = False


class LFCC(torch.nn.Module):
    def __init__(
        self,
        fl,
        fs,
        fn,
        sr,
        filter_num,
        num_ceps,
        skip=True,
        with_energy=False,
        with_emphasis=False,
        compress=True,
        device="cuda:0",
    ):
        super(LFCC, self).__init__()
        self.fl = fl
        self.fs = fs
        self.fn = fn
        self.sr = sr
        self.filter_num = filter_num
        self.skip = skip
        self.num_ceps = num_ceps
        self.compress = compress
        self.mask = FrequencyMasking(filter_num // 5).to(device)

        f = (sr / 2) * torch.linspace(0, 1, fn // 2 + 1)
        filter_bands = torch.linspace(min(f), max(f), filter_num + 2)

        filter_bank = torch.zeros([fn // 2 + 1, filter_num])
        for idx in range(filter_num):
            filter_bank[:, idx] = self.trimf(
                f, [filter_bands[idx], filter_bands[idx + 1], filter_bands[idx + 2]]
            )
        self.lfcc_fb = torch.nn.Parameter(filter_bank, requires_grad=False).to(device)

        self.l_dct = LinearDCT(filter_num, norm="ortho", device=device)

        self.with_energy = with_energy
        self.with_emphasis = with_emphasis

    @staticmethod
    def delta(x):
        length = x.shape[1]
        output = torch.zeros_like(x)
        x_temp = F.pad(x.unsqueeze(1), (0, 0, 1, 1), "replicate").squeeze(1)
        output = -1 * x_temp[:, 0:length] + x_temp[:, 2:]
        return output

    @staticmethod
    def trimf(x, params):
        if len(params) != 3:
            exit(1)
        a = params[0]
        b = params[1]
        c = params[2]
        if a > b or b > c:
            exit(1)
        y = torch.zeros_like(x, dtype=torch.float32)
        if a < b:
            index = torch.logical_and(a < x, x < b)
            y[index] = (x[index] - a) / (b - a)
        if b < c:
            index = torch.logical_and(b < x, x < c)
            y[index] = (c - x[index]) / (c - b)
        y[x == b] = 1
        return y

    def forward(self, x, is_mask=False):
        if self.compress:
            x = x.squeeze(1)

        if self.with_emphasis:
            x_copy = torch.clone(x)
            x_copy[:, 1:] = x[:, 1:] - 0.97 * x[:, 0:-1]
        else:
            x_copy = x
        x_stft = torch.stft(
            x_copy,
            self.fn,
            self.fs,
            self.fl,
            window=torch.hamming_window(self.fl).to(x.device),
            onesided=True,
            pad_mode="constant",
            return_complex=False,
        )

        sp_amp = torch.norm(x_stft, 2, -1).pow(2).permute(0, 2, 1).contiguous()

        if self.skip:
            sp_amp = sp_amp[:, -(x.shape[-1] // self.fs) :, :]

        fb_feature = torch.log10(
            torch.matmul(sp_amp, self.lfcc_fb) + torch.finfo(torch.float32).eps
        )

        lfcc = self.l_dct(fb_feature)[:, :, : self.num_ceps]

        if self.with_energy:
            power_spec = sp_amp / self.fn
            energy = torch.log10(
                power_spec.sum(axis=2) + torch.finfo(torch.float32).eps
            )
            lfcc[:, :, 0] = energy

        lfcc_delta = self.delta(lfcc)
        lfcc_delta_delta = self.delta(lfcc_delta)
        lfcc_output = torch.cat((lfcc, lfcc_delta, lfcc_delta_delta), 2)

        if is_mask:
            lfcc_output = self.mask(lfcc_output.permute(0, 2, 1)).permute(0, 2, 1)

        return lfcc_output.unsqueeze(1)


class LFCC_Pad(LFCC, torch.nn.Module):
    def __init__(
        self,
        fl,
        fs,
        fn,
        sr,
        filter_num,
        num_ceps,
        with_energy=False,
        with_emphasis=False,
        skip=True,
        compress=True,
        device="cuda:0",
        max_len=750,
    ):
        torch.nn.Module.__init__(self)
        LFCC.__init__(
            self,
            fl=fl,
            fs=fs,
            fn=fn,
            sr=sr,
            filter_num=filter_num,
            num_ceps=num_ceps,
            skip=skip,
            with_energy=with_energy,
            with_emphasis=with_emphasis,
            compress=compress,
            device=device,
        )
        self.padder = Pad(max_len, device)

    def forward(self, x):
        x = LFCC.forward(self, x).squeeze(1)
        out = self.padder(x).squeeze(1)
        return out.permute(0, 2, 1).unsqueeze(1)


class Pad(torch.nn.Module):
    def __init__(self, max_len=64600, device="cuda:0"):
        super(Pad, self).__init__()
        self.max_len = max_len
        self.device = device

    def forward(self, x):
        need_convert = isinstance(x, np.ndarray)

        if need_convert:
            # pylint: disable=E1102
            x = torch.tensor(x, device=self.device)
            # pylint: disable=E1102

        x_len = x.shape[1]
        repeats = self.max_len // x_len + 1
        x = x.repeat([1, repeats] + ([1] * (len(x.shape) - 2)))[:, : self.max_len]
        x = x.unsqueeze(1)

        if need_convert:
            return x.cpu().numpy()
        return x


class lps(torch.nn.Module):
    def __init__(
        self,
        n_fft=128,
        hop_size=64,
        window_length=128,
        ref=1.0,
        amin=1e-30,
        power=2,
        device="cuda:0",
    ):
        super(lps, self).__init__()
        self.device = device
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.window_length = window_length
        self.amin = torch.tensor(amin, device=self.device)

        self.win = torch.blackman_window(
            self.window_length, device=device, requires_grad=False
        )
        self.ref_value = torch.tensor(np.abs(ref), device=self.device)
        self.power = 2

    def _transform(self, S):
        return torch.transpose(S, 2, 1).float()

    def _calc(self, x):
        spec = torch.stft(
            x,
            self.n_fft,
            hop_length=self.hop_size,
            win_length=self.window_length,
            window=self.win,
            center=True,
            pad_mode="reflect",
            normalized=False,
            onesided=None,
            return_complex=True,
        )
        S = torch.abs(spec) ** self.power
        S = self._transform(S)

        ret = 10 * torch.log10(torch.maximum(self.amin, S))
        ret = (
            ret
            - 10.0 * (torch.log10(torch.maximum(self.amin, self.ref_value))).double()
        )
        return ret

    def forward(self, x):
        ret = self._calc(x)
        return ret


class ASVLps(lps, torch.nn.Module):
    def __init__(
        self,
        n_fft=128,
        hop_size=64,
        window_length=128,
        ref=1.0,
        amin=1e-30,
        power=2,
        device="cuda:0",
    ):
        torch.nn.Module.__init__(self)
        lps.__init__(
            self,
            n_fft=n_fft,
            hop_size=hop_size,
            window_length=window_length,
            ref=ref,
            amin=amin,
            power=power,
            device=device,
        )

    def forward(self, x):
        ret = lps.forward(self, x)
        idx = [torch.where(torch.sum(x, axis=1) > -60 * x.shape[1]) for x in ret]
        ret = torch.stack([ret[i][idx[i]] for i in range(len(idx))])
        return ret


class CMLps(lps, torch.nn.Module):
    def __init__(
        self,
        n_fft=1724,
        hop_size=130,
        window_length=1724,
        ref=1.0,
        amin=1e-30,
        max_len=600,
        power=2,
        device="cuda:0",
    ):
        torch.nn.Module.__init__(self)
        lps.__init__(
            self,
            n_fft=n_fft,
            hop_size=hop_size,
            window_length=window_length,
            ref=ref,
            amin=amin,
            power=power,
            device=device,
        )
        self.padder = Pad(max_len)

    def forward(self, x):
        ret = lps.forward(self, x)
        ret = self.padder(ret)
        return ret


class mfcc(lps, torch.nn.Module):
    def __init__(
        self,
        sampling_rate=16000,
        n_fft=400,
        hop_size=160,
        window_length=400,
        num_mels=30,
        fmin=20.0,
        fmax=7600,
        ref=1.0,
        amin=1e-30,
        top_db=80.0,
        n_mfcc=24,
        power=2,
        device="cuda:0",
    ):
        torch.nn.Module.__init__(self)
        lps.__init__(
            self,
            n_fft=n_fft,
            hop_size=hop_size,
            window_length=window_length,
            ref=ref,
            amin=amin,
            power=power,
            device=device,
        )
        self.top_db = top_db
        self.n_mfcc = n_mfcc
        self.mel_basis = torch.tensor(
            librosa.filters.mel(
                sampling_rate, self.n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
            ).astype(np.float32),
            device=device,
        )

        self.win = torch.hamming_window(
            self.window_length, device=device, requires_grad=False
        )

        self.top_db = top_db
        self.n_mfcc = n_mfcc

    def _transform(self, S):
        return torch.tensordot(self.mel_basis, S, ([1], [1]))

    def _mfcc(self, y):
        S = self._calc(y).permute(1, 2, 0)
        ret = torch.maximum(S, torch.max(S) - self.top_db).float()
        ret = dct(ret, norm="ortho")
        return ret[:, :, : self.n_mfcc]

    def forward(self, audio):
        win = torch.tensor(
            np.array([-0.97, 1], dtype=np.float64).reshape((1, 1, 1, 2)),
            device=audio.device,
        ).type(torch.float32)
        x = audio.type(torch.float32)
        x = torch.reshape(
            F.pad(x, [1, 1], "constant"), (x.shape[0], 1, 1, int(x.shape[1]) + 2)
        )
        x = F.conv2d(x, win).squeeze(1).squeeze(1)[:, :-1]
        return self._mfcc(x)


class CQT(torch.nn.Module):
    def __init__(
        self,
        sr=16000,
        hop_length=256,
        n_bins=432,
        bins_per_octave=48,
        window="hann",
        fmin=15,
        truncate_len=400,
        ref=1.0,
        amin=1e-30,
        device="cuda:0",
    ):
        super(CQT, self).__init__()
        self.sr = sr
        self.hop_length = hop_length
        self.n_bins = n_bins
        self.bins_per_octave = bins_per_octave
        self.window = window
        self.fmin = fmin
        self.device = device
        self.truncate_len = truncate_len

        self.amin = torch.tensor(amin, device=self.device)
        self.ref_value = torch.tensor(np.abs(ref), device=self.device)

    def _tensor_cnn_utt(self, x):
        mat = torch.transpose(x, 2, 1)
        max_len = self.truncate_len * int(np.ceil(mat.shape[-1] / self.truncate_len))
        repetition = int(max_len / mat.shape[-1])

        tensor = mat.repeat([1, 1, repetition])
        repetition = max_len % mat.shape[-1]

        rest = mat[:, :, :repetition]
        tensor = torch.cat((tensor, rest), axis=-1)
        return tensor

    def _construct_slide_tensor(self, x):
        tensor = self._tensor_cnn_utt(x)
        sub_tensor = tensor[:, :, : self.truncate_len]
        return sub_tensor

    def forward(self, x):
        win = torch.tensor(
            np.array([-0.97, 1], dtype=np.float64).reshape((1, 1, 1, 2)),
            device=x.device,
        ).type(torch.float32)
        x = torch.reshape(
            F.pad(x, [1, 1], "constant"), (x.shape[0], 1, 1, int(x.shape[1]) + 2)
        )
        x = F.conv2d(x, win).squeeze(1).squeeze(1)[:, :-1].type(torch.float64)
        ret = self._construct_slide_tensor(self._cqt(x))
        return ret.type(torch.float32).unsqueeze(1)

    def _cqt(self, x):
        y = cqt(
            x,
            sr=self.sr,
            hop_length=self.hop_length,
            n_bins=self.n_bins,
            bins_per_octave=self.bins_per_octave,
            window=self.window,
            fmin=self.fmin,
        ).permute(0, 2, 1)
        y = torch.abs(y) ** 2
        m = torch.maximum(self.amin, y)
        log_spec = 10.0 * torch.log10(m)
        log_spec -= 10.0 * torch.log10(torch.maximum(self.amin, self.ref_value))
        return log_spec
