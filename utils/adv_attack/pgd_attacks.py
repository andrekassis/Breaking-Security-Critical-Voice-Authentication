# pylint: disable=E1102, E1111, R0913, W0702

from abc import ABC
import numpy as np
import torch
from .reduce import sp


class PGD(ABC):
    # pylint: disable=R0902
    def __init__(self, estimator, epsilon, max_iter, dtype, delta=None, r_c=None):
        self.estimator = estimator
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.type = dtype
        self.delta = delta
        self.alpha = None
        self.eps = None
        self.length = None

        if r_c is not None:
            self.sg = sp(**r_c)
        else:
            self.sg = None

    def _transform(self, x):
        pass

    def _inverse_transform(self, x):
        pass

    def _project(self, delta):
        pass

    @staticmethod
    def _sgn(x):
        pass

    def clip(self, x):
        return self._transform(self._inverse_transform(x).clip(-1, 1))

    def _clip(self, x):
        pass

    def _gradient(self, x, y, **r_args):
        var = torch.tensor(x, requires_grad=True, device=self.estimator.device)
        x_t = self._inverse_transform(var)

        if self.sg:
            x_t = self.sg(x_t, **r_args)

        grad = self.estimator.loss_gradient(x_t.detach().cpu().numpy(), y)
        grad = torch.tensor(grad, device=self.estimator.device)

        res = x_t * grad
        res = torch.sum(res)
        res.backward()

        grad = var.grad.detach().cpu().numpy()
        return grad

    def set_len(self, length):
        self.length = length

    def _itr(self, X, y, **r_args):
        self.set_len(X.shape[-1])

        X_F = self._transform(X)
        self.eps = np.ones(X_F.shape) * self.epsilon
        if self.alpha is not None:
            alpha = self.alpha
        else:
            alpha = self.eps / 10

        delta = np.random.uniform(size=X_F.shape).astype(self.type)
        delta = 2 * self.eps * delta - self.eps
        delta = self._project(delta)

        if self.delta is not None:
            delta = self.delta

        # pylint: disable=W0612
        for t in range(self.max_iter):
            gradient = self._gradient(X_F + delta, y, **r_args)
            delta = delta - alpha * self._sgn(gradient)
            delta = self._clip(self._project(delta))
            delta = self.clip(X_F + delta) - X_F
        # pylint: enable=W0612

        X_r = self._inverse_transform(X_F + delta)
        delt = X_r[:, : X.shape[-1]] - X[:, : X_r.shape[-1]]
        return delt

    def generate(self, x, y, **r_args):
        label = 1 - y
        y = label
        label = np.argmax(np.array(label[0]))
        delta = self._itr(x, y, **r_args)

        ret = x[:, : delta.shape[-1]] + delta
        return ret


class TIME_DOMAIN_ATTACK(PGD):
    def __init__(self, estimator, epsilon, max_iter, delta=None, r_c=None):
        super().__init__(estimator, epsilon, max_iter, np.float64, delta, r_c)

    def _transform(self, x):
        return x

    def _inverse_transform(self, x):
        return x

    def _project(self, delta):
        return delta

    def _clip(self, x):
        return x.clip(-self.epsilon, self.epsilon)

    @staticmethod
    def _sgn(x):
        return np.sign(x)


class Spectral_Attack(PGD):
    def __init__(
        self,
        estimator,
        epsilon,
        max_iter,
        delta=None,
        r_c=None,
        factor=0.25,
        thresh=2500,
        sr=16000,
    ):
        super().__init__(estimator, epsilon, max_iter, np.complex128, delta, r_c)

        self.factor = factor
        self.thresh = thresh
        self.sr = sr

        self.mask = None

        if self.factor:
            assert self.sr is not None
            assert self.thresh is not None

    def set_len(self, length):
        if self.length == length:
            return

        self.length = length

        if self.factor is None:
            return

        freq = np.fft.rfftfreq(self.length, d=1.0 / self.sr)
        idx = (np.abs(freq - self.thresh)).argmin()

        self.mask = np.ones((1, freq.shape[-1]), dtype=np.complex)
        self.mask[:, idx:] *= self.factor

    def _clip(self, x):
        sgn = np.sign(x.real) + 1j * np.sign(x.imag)
        mag = (
            np.minimum(np.abs(x.real), np.abs(self.eps))
            + np.minimum(np.abs(x.imag), np.abs(self.eps)) * 1j
        )
        ret = sgn.real * mag.real + sgn.imag * mag.imag * 1j
        return ret

    @staticmethod
    def _sgn(x):
        return np.sign(x.real) + 1j * np.sign(x.imag)

    def _project(self, delta):
        if self.mask is None:
            return delta

        return self._transform(
            np.fft.irfft(np.fft.rfft(self._inverse_transform(delta)) * self.mask)
        )


class STFT_Attack(Spectral_Attack):
    def __init__(
        self,
        estimator,
        epsilon,
        max_iter,
        delta=None,
        nfft=512,
        window="hann_window",
        hop_length=None,
        win_length=None,
        r_c=None,
        factor=0.25,
        thresh=2500,
        sr=16000,
    ):
        super().__init__(estimator, epsilon, max_iter, delta, r_c, factor, thresh, sr)
        self.nfft = nfft
        self.win_length = win_length
        if self.win_length is None:
            self.win_length = self.nfft
        self.hop_length = hop_length
        win_args = {
            "window_length": self.win_length,
            "device": self.estimator.device,
            "requires_grad": False,
        }

        if window is None:
            self.win = None
        else:
            self.win = getattr(torch, window)(**win_args)

    def _transform(self, x):
        if isinstance(x, np.ndarray):
            y = torch.tensor(x, device=self.estimator.device)
        else:
            y = x
        ret = torch.stft(
            y,
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
        if isinstance(x, np.ndarray):
            return ret.cpu().numpy()
        return ret

    def _inverse_transform(self, x):
        if isinstance(x, np.ndarray):
            y = torch.tensor(x, device=self.estimator.device)
        else:
            y = x

        ret = torch.istft(
            y,
            n_fft=self.nfft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.win,
            center=True,
            normalized=True,
            onesided=False,
            length=self.length,
            return_complex=False,
        )
        if isinstance(x, np.ndarray):
            return ret.cpu().numpy()
        return ret


class FFT_Attack(Spectral_Attack):
    def __init__(
        self,
        estimator,
        epsilon,
        max_iter,
        delta=None,
        r_c=None,
        factor=0.25,
        thresh=2500,
        sr=16000,
    ):
        super().__init__(estimator, epsilon, max_iter, delta, r_c, factor, thresh, sr)

    def _transform(self, x):
        if isinstance(x, np.ndarray):
            return np.fft.rfft(x)
        return torch.fft.rfft(x)

    def _inverse_transform(self, x):
        if isinstance(x, np.ndarray):
            return np.fft.irfft(x)
        return torch.fft.irfft(x)
