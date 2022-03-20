# pylint: disable=E1102, E1111, R0913, W0702

from abc import ABC
import numpy as np
import torch
from .reduce import sp


class PGD(ABC):
    # pylint: disable=R0902
    def __init__(
        self, estimator, epsilon, max_iter, dtype, delta=None, r_c=None, norm=None
    ):
        self.estimator = estimator
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.dtype = dtype
        self.delta = delta
        self.alpha = None
        self.eps = None
        self.length = None
        self.norm = norm

        if r_c is not None:
            self.sg = sp(**r_c)
        else:
            self.sg = None

    def _convert_type(self, is_array):
        if is_array:
            return self.dtype
        if self.dtype == np.complex128:
            return torch.complex128
        return torch.float64

    def _transform(self, x, requires_grad=True):
        pass

    def _inverse_transform(self, x, requires_grad=True):
        pass

    def _project(self, delta):
        pass

    @staticmethod
    def _sgn(x):
        pass

    def _clip(self, x):
        pass

    @staticmethod
    def _retrieve_grad(var, x):
        grad = var.grad
        grad = grad.detach().cpu().numpy() if isinstance(x, np.ndarray) else grad
        return grad

    @staticmethod
    def _x_to_var(x, requires_grad, device):
        var = (
            torch.tensor(x, requires_grad=requires_grad, device=device)
            if isinstance(x, np.ndarray)
            else x
        )
        if requires_grad and var.requires_grad is False:
            var.requires_grad = True
            var.grad = None
        return var

    def _gradient(self, x, y, **r_args):
        if not isinstance(x, np.ndarray):
            x = x.detach().clone()

        var = self._x_to_var(x, requires_grad=True, device=self.estimator.device)
        x_t = self._inverse_transform(var)

        x_t = self.sg(x_t, **r_args) if self.sg else x_t

        res = self.estimator.compute_loss(x_t, y)
        res.backward(retain_graph=True)

        return self._retrieve_grad(var, x)

    def _set_len(self, length, is_array):
        self.length = length

    def _set_eps(self, shape, is_array):
        if self.eps is None or self.eps.shape != shape:
            if not is_array:
                self.eps = torch.ones(shape, device=self.estimator.device)
            else:
                self.eps = np.ones(shape)

    def _itr(self, X, y, **r_args):
        self._set_len(X.shape[-1], isinstance(X, np.ndarray))
        X_F = self._transform(X, requires_grad=False)
        self._set_eps(X_F.shape, isinstance(X, np.ndarray))

        if self.alpha is not None:
            alpha = self.alpha
        else:
            alpha = self.eps * self.epsilon / 10

        if self.delta is None:
            if isinstance(X, np.ndarray):
                delta = np.random.uniform(size=X_F.shape).astype(
                    self._convert_type(True)
                )
            else:
                delta = torch.rand(size=X_F.shape, device=self.estimator.device).type(
                    self._convert_type(False)
                )
            delta = 2 * self.eps * delta - (self.eps * self.epsilon)
            delta = self._project(delta)
        else:
            delta = self.delta

        # pylint: disable=W0612
        for t in range(self.max_iter):
            gradient = self._gradient(X_F + delta, y, **r_args)
            if self.norm == "fro":
                delta = gradient
                delta = self._clip(self._project(delta))
            else:
                delta = delta - alpha * self._sgn(gradient)
                delta = self._clip(self._project(delta))
        # pylint: enable=W0612

        X_r = self._inverse_transform(X_F + delta, requires_grad=False)
        X_r = (
            X_r / np.max(np.abs(X_r), -1)[..., np.newaxis]
            if isinstance(X_r, np.ndarray)
            else X_r / torch.max(torch.abs(X_r), -1).values.unsqueeze(-1)
        )
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
    def __init__(self, estimator, epsilon, max_iter, delta=None, r_c=None, norm=None):
        super().__init__(estimator, epsilon, max_iter, np.float64, delta, r_c, norm)

    def _transform(self, x, requires_grad=True):
        return x

    def _inverse_transform(self, x, requires_grad=True):
        return x

    def _project(self, delta):
        return delta

    def _clip(self, x):
        if self.norm == "fro":
            if isinstance(x, np.ndarray):
                return x / np.linalg.norm(x) * self.epsilon
            return x / torch.norm(x) * self.epsilon
        return x.clip(-self.epsilon, self.epsilon)

    @staticmethod
    def _sgn(x):
        if isinstance(x, np.ndarray):
            return np.sign(x)
        return torch.sign(x)


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
        norm=None,
    ):
        super().__init__(estimator, epsilon, max_iter, np.complex128, delta, r_c, norm)

        self.factor = factor
        self.thresh = thresh
        self.sr = sr

        self.mask = None

        if self.factor:
            assert self.sr is not None
            assert self.thresh is not None

    def _set_len(self, length, is_array):
        if self.length == length:
            if is_array != isinstance(self.mask, np.ndarray):
                if is_array:
                    self.mask = self.mask.detach().cpu().numpy()
                else:
                    self.mask = torch.tensor(self.mask, device=self.estimator.device)
            return

        self.length = length

        if self.factor is None:
            return

        freq = np.fft.rfftfreq(self.length, d=1.0 / self.sr)
        idx = (np.abs(freq - self.thresh)).argmin()

        self.mask = np.ones((1, freq.shape[-1]), dtype=np.complex)
        self.mask[:, idx:] *= self.factor
        if not is_array:
            self.mask = torch.tensor(self.mask, device=self.estimator.device)

    def _clip(self, x):
        if isinstance(x, np.ndarray):
            return self._clip_np(x)
        return self._clip_torch(x)

    def _clip_torch(self, x):
        if self.norm == "fro":
            return x / torch.norm(x) * self.epsilon
        sgn = torch.sign(x.real) + 1j * torch.sign(x.imag)
        mag = (
            torch.minimum(torch.abs(x.real), torch.abs(self.eps * self.epsilon))
            + torch.minimum(torch.abs(x.imag), torch.abs(self.eps * self.epsilon)) * 1j
        )
        ret = sgn.real * mag.real + sgn.imag * mag.imag * 1j
        return ret

    def _clip_np(self, x):
        if self.norm == "fro":
            return x / np.linalg.norm(x) * self.epsilon
        sgn = np.sign(x.real) + 1j * np.sign(x.imag)
        mag = (
            np.minimum(np.abs(x.real), np.abs(self.eps * self.epsilon))
            + np.minimum(np.abs(x.imag), np.abs(self.eps * self.epsilon)) * 1j
        )
        ret = sgn.real * mag.real + sgn.imag * mag.imag * 1j
        return ret

    @staticmethod
    def _sgn(x):
        if isinstance(x, np.ndarray):
            return np.sign(x.real) + 1j * np.sign(x.imag)
        return torch.sign(x.real) + 1j * torch.sign(x.imag)

    def _project(self, delta):
        if self.mask is None:
            return delta

        if not isinstance(delta, np.ndarray):
            return self._transform(
                torch.fft.irfft(
                    torch.fft.rfft(self._inverse_transform(delta)) * self.mask
                )
            )

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
        norm=None,
    ):
        super().__init__(
            estimator, epsilon, max_iter, delta, r_c, factor, thresh, sr, norm
        )
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

    def _transform(self, x, requires_grad=True):
        y = self._x_to_var(x, requires_grad=requires_grad, device=self.estimator.device)
        if requires_grad is False:
            y = y.detach().clone()

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

        ret = ret.detach().cpu().numpy() if isinstance(x, np.ndarray) else ret
        return ret

    def _inverse_transform(self, x, requires_grad=True):
        y = self._x_to_var(x, requires_grad=requires_grad, device=self.estimator.device)
        if requires_grad is False:
            y = y.detach().clone()

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

        ret = ret.detach().cpu().numpy() if isinstance(x, np.ndarray) else ret
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
        norm=None,
    ):
        super().__init__(
            estimator, epsilon, max_iter, delta, r_c, factor, thresh, sr, norm
        )

    def _transform(self, x, requires_grad=True):
        if requires_grad is False and not isinstance(x, np.ndarray):
            x = x.detach().clone()
        return np.fft.rfft(x) if isinstance(x, np.ndarray) else torch.fft.rfft(x)

    def _inverse_transform(self, x, requires_grad=True):
        if requires_grad is False and not isinstance(x, np.ndarray):
            x = x.detach().clone()
        return np.fft.irfft(x) if isinstance(x, np.ndarray) else torch.fft.irfft(x)
