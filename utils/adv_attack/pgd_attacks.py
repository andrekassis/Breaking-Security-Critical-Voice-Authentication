# pylint: disable=E1102, E1111, R0913, W0702

from abc import ABC
import numpy as np
import torch
from .reduce import sp
from utils.audio.preprocessing import preemphasize, deemphasize
import scipy.signal


class TypedOps:
    def __init__(self, dtype, sr=16000, device="cuda:0"):
        self.dtype = dtype
        self.device = device
        self.sr = sr

    @staticmethod
    def is_array(x):
        return isinstance(x, np.ndarray)

    @staticmethod
    def argmin(x):
        return np.argmin(x) if TypedOps.is_array(x) else torch.argmin(x)

    @staticmethod
    def rfft(x, **kwargs):
        return (
            np.fft.rfft(x, **kwargs)
            if TypedOps.is_array(x)
            else torch.fft.rfft(x, **kwargs)
        )

    @staticmethod
    def irfft(x, **kwargs):
        return (
            np.fft.irfft(x, **kwargs)
            if TypedOps.is_array(x)
            else torch.fft.irfft(x, **kwargs)
        )

    @staticmethod
    def rfftfreq(x, d):
        return np.fft.rfftfreq(x, d=d)

    def sgn(self, x):
        if self.is_array(x):
            if self.dtype != np.complex128:
                return np.sign(x)
            abs_x = np.abs(x)
            abs_x[abs_x == 0] = 1
            return x / abs_x
        return torch.sgn(x)

    @staticmethod
    def abs(x):
        return np.abs(x) if TypedOps.is_array(x) else torch.abs(x)

    def zeros(self, shape, is_array):
        if not is_array:
            dtype = torch.complex128 if self.dtype == np.complex128 else torch.float64
        else:
            dtype = self.dtype
        return (
            np.zeros(shape).astype(dtype)
            if is_array
            else torch.zeros(shape, device=self.device).type(dtype)
        )

    def ones(self, shape, is_array):
        if not is_array:
            dtype = torch.complex128 if self.dtype == np.complex128 else torch.float64
        else:
            dtype = self.dtype
        return (
            np.ones(shape).astype(dtype)
            if is_array
            else torch.ones(shape, device=self.device).type(dtype)
        )

    def random_noise(self, shape, epsilon, is_array):
        if is_array:
            delta = np.random.normal(size=shape).astype(self.dtype)
        else:
            dtype = torch.complex128 if self.dtype == np.complex128 else torch.float64
            delta = torch.normal(0, 1.0, size=shape, device=self.device).type(dtype)
        return self.normalize(delta) * epsilon

    @staticmethod
    def minimum(x, y):
        return np.minimum(x, y) if TypedOps.is_array(x) else torch.minimum(x, y)

    @staticmethod
    def norm(x, **kwargs):
        return (
            np.linalg.norm(x, **kwargs)
            if TypedOps.is_array(x)
            else torch.norm(x, **kwargs)
        )

    @staticmethod
    def normalize(x):
        return (
            x / np.max(np.abs(x), -1, keepdims=True)
            if TypedOps.is_array(x)
            else x / torch.max(torch.abs(x), -1).values.unsqueeze(-1)
        )

    def make_var(self, x):
        var = (
            torch.tensor(x, requires_grad=True, device=self.device)
            if self.is_array(x)
            else x.detach().clone()
        )

        var.requires_grad = True
        var.grad = None
        return var

    def mask(self, length, fromm=None, to=None, is_array=False):
        freq = self.rfftfreq(length, d=1 / self.sr)
        idx1 = self.argmin(TypedOps.abs(freq - fromm)) if fromm else 0
        idx2 = self.argmin(TypedOps.abs(freq - to)) if to else len(freq) - 1

        mask = self.zeros((1, freq.shape[-1]), is_array)
        mask[:, idx1:idx2] = 1
        return mask

    @staticmethod
    def raw(x, is_array):
        return x.detach().cpu().numpy() if is_array else x


class stft:
    def __init__(self, nfft, win_length, hop_length, window, device):
        self.nfft = nfft
        self.win_length = win_length
        if self.win_length is None:
            self.win_length = self.nfft
        self.hop_length = hop_length
        win_args = {
            "window_length": self.win_length,
            "device": device,
            "requires_grad": False,
        }

        if window is None:
            self.win = None
        else:
            self.win = getattr(torch, window)(**win_args)

    def stft(self, x):
        return torch.stft(
            x,
            n_fft=self.nfft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.win,
            center=True,
            pad_mode="reflect",
            normalized=True,
            onesided=False,
            return_complex=True,
        )

    def istft(self, x, length):
        return torch.istft(
            x,
            n_fft=self.nfft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.win,
            center=True,
            normalized=True,
            onesided=False,
            length=length,
            return_complex=False,
        )


class PGD(ABC):
    # pylint: disable=R0902
    def __init__(
        self,
        estimator,
        epsilon,
        max_iter,
        dtype,
        delta=None,
        r_c=None,
        norm=None,
        sr=16000,
        restarts=1,
    ):
        self.estimator = estimator
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.delta = delta
        self.alpha = None
        self.length = None
        self.norm = norm
        self.ops = TypedOps(dtype=dtype, sr=sr, device=self.estimator.device)
        self.restarts = restarts

        if r_c is not None:
            r_c["device"] = self.estimator.device
            self.sg = sp(**r_c)
        else:
            self.sg = None

    def _transform(self, x, requires_grad=True):
        pass

    def _inverse_transform(self, x, requires_grad=True):
        pass

    def _project(self, delta):
        pass

    def _clip(self, x):
        return self.ops.sgn(x) * self.ops.abs(x).clip(0, self.epsilon)

    def _gradient(self, x, y, **r_args):
        var = self.ops.make_var(x)
        x_t = self._inverse_transform(var)

        x_t = self.sg(x_t, **r_args) if self.sg else x_t

        res = self.estimator.compute_loss(x_t, y)
        res.backward(retain_graph=True)

        return self.ops.raw(var.grad, self.ops.is_array(x))

    def _set_len(self, length, is_array):
        self.length = length

    def _filter_delta(self, delta, start=0, end=None):
        start = np.max([20, start])
        end = (
            delta.shape[-1] - 20 if end is None else np.min([delta.shape[-1] - 20, end])
        )
        delta[:, :start] = 0
        delta[:, end:] = 0
        return delta

    def _itr(self, X, y, start=0, end=None, **r_args):
        self._set_len(X.shape[-1], self.ops.is_array(X))
        X_F = self._transform(X, requires_grad=False)
        alpha = self.alpha if self.alpha is not None else self.epsilon / 10

        delta = (
            self._project(
                self.ops.random_noise(X.shape, self.epsilon, self.ops.is_array(X))
            )
            if self.delta is None
            else self.ops.ones(shape=X_F.shape, is_array=self.ops.is_array(X))
            * self.delta
        )

        # pylint: disable=W0612
        for t in range(self.max_iter):
            gradient = self._gradient(X_F + delta, y, **r_args)
            delta = delta - alpha * self.ops.sgn(gradient)
            delta = self._clip(self._project(delta))
        # pylint: enable=W0612
        return self._inverse_transform(delta, requires_grad=False)

    def generate(self, x, y, preemphasis=False, start=0, end=None, **r_args):
        label = 1 - y
        y = label
        label = np.argmax(np.array(label[0]))

        madv = x
        for rest in range(self.restarts):
            if preemphasis:
                delta = self._itr(
                    preemphasize(x, self.ops.device), y, start=start, end=end, **r_args
                )
                adv = deemphasize(
                    preemphasize(x, self.ops.device)[:, : delta.shape[-1]] + delta
                )
            else:
                delta = self._itr(x, y, start=start, end=end, **r_args)
                adv = x[:, : delta.shape[-1]] + delta
            loss = self.estimator.compute_loss(adv, y)
            if rest == 0 or loss < minloss:
                minloss = loss
                madv = adv
        return madv


class TIME_DOMAIN_ATTACK(PGD):
    def __init__(
        self,
        estimator,
        epsilon,
        max_iter,
        delta=None,
        r_c=None,
        norm=None,
        sr=16000,
        restarts=1,
    ):
        super().__init__(
            estimator, epsilon, max_iter, np.float64, delta, r_c, norm, sr, restarts
        )

    def _transform(self, x, requires_grad=True):
        return x

    def _inverse_transform(self, x, requires_grad=True):
        return x

    def _project(self, delta):
        return delta

    def _to_range(self, delta, X, start=0, end=None):
        delta = self._filter_delta(delta, start=start, end=end)
        X_r = delta + X[:, : delta.shape[-1]]
        X_r = self.ops.normalize(X_r)
        return X_r - X[:, : X_r.shape[-1]]


class Spectral_Attack(PGD):
    def __init__(
        self,
        estimator,
        epsilon,
        max_iter,
        delta=None,
        r_c=None,
        factor=None,
        thresh=None,
        sr=16000,
        norm=None,
        restarts=1,
    ):
        super().__init__(
            estimator, epsilon, max_iter, np.complex128, delta, r_c, norm, sr, restarts
        )

        self.factor, self.thresh, self.mask = factor, thresh, None

        if self.factor:
            assert self.thresh is not None

    # def _itr(self, X, y, start=0, end=None, **r_args):
    #    X = X * (1-self.epsilon)
    #    return super()._itr(X, y, start, end, **r_args)
    def _to_range(self, delta, X, start=0, end=None):
        delta = self._filter_delta(delta, start=start, end=end)
        X_r = delta + X[:, : delta.shape[-1]]
        # return X_r.clip(-1, 1) - X[:, : X_r.shape[-1]]
        X_r = self.ops.normalize(X_r)
        return X_r - X[:, : X_r.shape[-1]]

    def _set_len(self, length, is_array):
        if self.length == length and self.mask is not None:
            if is_array != self.ops.is_array(self.mask):
                if is_array:
                    self.mask = self.mask.detach().cpu().numpy()
                else:
                    self.mask = torch.tensor(self.mask, device=self.estimator.device)
            return

        self.length = length

        if self.factor is None:
            return

        self.mask = self.ops.mask(length=length, fromm=self.thresh, is_array=is_array)

    def _project(self, delta):
        if self.mask is None:
            return delta

        return self._transform(
            self.ops.irfft(self.ops.rfft(self._inverse_transform(delta)) * self.mask)
        )


class STFT_Attack(Spectral_Attack):
    def __init__(
        self,
        estimator,
        epsilon,
        max_iter,
        delta=None,
        nfft=512,
        window=None,
        hop_length=None,
        win_length=None,
        r_c=None,
        factor=None,
        thresh=None,
        sr=16000,
        norm=None,
        restarts=1,
    ):
        super().__init__(
            estimator, epsilon, max_iter, delta, r_c, factor, thresh, sr, norm, restarts
        )
        self.stft = stft(nfft, win_length, hop_length, window, self.estimator.device)

    def _transform(self, x, requires_grad=True):
        y = torch.tensor(x, device=self.estimator.device) if self.ops.is_array(x) else x
        y = y.detach().clone() if not requires_grad else y
        ret = self.stft.stft(y)
        return self.ops.raw(ret, self.ops.is_array(x))

    def _inverse_transform(self, x, requires_grad=True):
        y = torch.tensor(x, device=self.estimator.device) if self.ops.is_array(x) else x
        y = y.detach().clone() if not requires_grad else y
        ret = self.stft.istft(y, self.length)
        return self.ops.raw(ret, self.ops.is_array(x))


class FFT_Attack(Spectral_Attack):
    def __init__(
        self,
        estimator,
        epsilon,
        max_iter,
        delta=None,
        r_c=None,
        factor=None,
        thresh=None,
        sr=16000,
        norm=None,
        restarts=1,
    ):
        super().__init__(
            estimator, epsilon, max_iter, delta, r_c, factor, thresh, sr, norm, restarts
        )

    def _transform(self, x, requires_grad=True):
        if requires_grad is False and not self.ops.is_array(x):
            x = x.detach().clone()
        return self.ops.rfft(x)

    def _inverse_transform(self, x, requires_grad=True):
        if requires_grad is False and not self.ops.is_array(x):
            x = x.detach().clone()
        return self.ops.irfft(x, n=self.length)
