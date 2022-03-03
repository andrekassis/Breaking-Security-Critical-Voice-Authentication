import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import lfilter_zi
torch.set_flush_denormal(True)

def sigmoid(x, shift, mult):
    return 1 / (1 + torch.exp(-(x + shift) * mult))

def lfilter(b, x, zi=None):
    #return x
    y = torch.zeros(x.shape,device = x.device)
    y[..., 0] = b*x[..., 0] 
    if zi != None:
        y[..., 0] += zi.squeeze(-1)
    for itr in range(1, x.shape[-1]):
        y[..., itr] = b*x[..., itr] - (b-1)*y[..., itr-1]
    return y
    
def filtfilt(b, a, x):
    zi = lfilter_zi(b, a)
    zi_shape = [1]*x.ndim
    zi_shape[-1] = zi.size
    zi = np.reshape(zi, zi_shape)
    zi = torch.tensor(zi, device=x.device)

    x0 = x[...,0].unsqueeze(-1)
    y = lfilter(b[0], x, zi=zi * x0)

    y0 = y[...,-1].unsqueeze(-1)
    y = lfilter(b[0], torch.flip(y, [-1]), zi=zi * y0)
    return torch.flip(y, [-1])

def get_time_smoothed_representation(x, samplerate, hop_length, time_constant_s=0.001):
    t_frames = time_constant_s * samplerate / float(hop_length)
    b = [(np.sqrt(1 + 4 * t_frames ** 2) - 1) / (2 * t_frames ** 2)]
    a = [1, b[0]-1]
    return filtfilt(b, a, x)

class SpectralGateNonStationary:
    def __init__(
        self,
        sr=16000, 
        prop_decrease = 1.0,
        time_constant_s=2.0,
        freq_mask_smooth_hz=500,
        time_mask_smooth_ms=50,
        thresh_n_mult_nonstationary=2,
        sigmoid_slope_nonstationary=10,
        padding=30000,
        n_fft=1024,
        win_length=None,
        hop_length=None,
        win='hann_window',
        device='cuda:0'
    ):

        self.sr = sr
        self.device = device
        self.n_channels = 1
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
        
        win_args = {'window_length': self._win_length, 'device': self.device, 'requires_grad': False}
        self.win =getattr(torch, win)(**win_args)
        self._time_constant_s = time_constant_s
        self._prop_decrease = prop_decrease
        self._generate_mask_smoothing_filter(freq_mask_smooth_hz, time_mask_smooth_ms)

        self._thresh_n_mult_nonstationary = thresh_n_mult_nonstationary
        self._sigmoid_slope_nonstationary = sigmoid_slope_nonstationary

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
            #n_grad_freq = nfft/64
            #n_grad_time = 800/hop_length
            self._smoothing_filter = self._filter(n_grad_freq, n_grad_time)


    def _do_filter(self, x):
        orig_len = x.shape[-1]
        sig_stft = torch.stft(x, n_fft = self._n_fft, hop_length=self._hop_length, 
                                         win_length=self._win_length, window=self.win, center=True,
                                         pad_mode ="reflect", 
                                         #normalized = False, onesided=True, return_complex=True)
                                         normalized=True, onesided=False, return_complex=True)
        abs_sig_stft = torch.abs(sig_stft)
        print(abs_sig_stft)
        sig_stft_smooth = get_time_smoothed_representation(abs_sig_stft, self.sr, 
                                           self._hop_length, time_constant_s=self._time_constant_s)
        print(sig_stft_smooth)
        sig_mult_above_thresh = (abs_sig_stft - sig_stft_smooth) / sig_stft_smooth
        sig_mask = sigmoid(sig_mult_above_thresh, -self._thresh_n_mult_nonstationary, 
                                                                 self._sigmoid_slope_nonstationary)
        if self.smooth_mask:
            padding = (self._smoothing_filter.shape[1]//2, 
                       self._smoothing_filter.shape[1]//2-(self._smoothing_filter.shape[1]+1)%2, 
                       self._smoothing_filter.shape[0]//2, 
                       self._smoothing_filter.shape[0]//2-(self._smoothing_filter.shape[0]+1)%2)
            sig_mask = F.pad(sig_mask, padding)
            sig_mask = F.conv2d(sig_mask.unsqueeze(1), 
                                    self._smoothing_filter.unsqueeze(0).unsqueeze(0)).squeeze(1)
        
        sig_mask = sig_mask * self._prop_decrease + torch.ones(sig_mask.shape, device=self.device) * (1.0 - self._prop_decrease)
        sig_stft_denoised = sig_stft * sig_mask
        denoised_signal = torch.istft(sig_stft_denoised, n_fft = self._n_fft, hop_length=self._hop_length,
                                                win_length=self._win_length, window=self.win, center=True,
                                                length=orig_len,
                                                #normalized = False, onesided=True, return_complex=False)
                                                normalized=True, onesided=False, return_complex=False)
        return denoised_signal

    def __call__(self, x, p):
        if p == None:
            return x
        rev = False
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, device = self.device)
            rev = True
        self._prop_decrease = p
        filtered_padded_chunk = self._do_filter(F.pad(x, (self.padding, self.padding)))
        ret = filtered_padded_chunk[:, self.padding : x.shape[-1] + self.padding]
        if rev:
            return ret.cpu().numpy()
        return ret
