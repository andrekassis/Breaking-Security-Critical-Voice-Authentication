# pylint: disable=E1102
import os
import sys
import random
import soundfile as sf
import numpy as np
from scipy import signal
import torch
import torch.utils.data.sampler as torch_sampler
from torch.utils.data.dataloader import Dataset
import numpy as np
from random import randrange
import copy


def normWav(x, always):
    if always:
        x = x / np.amax(abs(x))
    elif np.amax(abs(x)) > 1:
        x = x / np.amax(abs(x))
    return x


def genNotchCoeffs(
    nBands, minF, maxF, minBW, maxBW, minCoeff, maxCoeff, minG, maxG, fs
):
    b = 1
    for i in range(0, nBands):
        fc = randRange(minF, maxF, 0)
        bw = randRange(minBW, maxBW, 0)
        c = randRange(minCoeff, maxCoeff, 1)

        if c / 2 == int(c / 2):
            c = c + 1
        f1 = fc - bw / 2
        f2 = fc + bw / 2
        if f1 <= 0:
            f1 = 1 / 1000
        if f2 >= fs / 2:
            f2 = fs / 2 - 1 / 1000
        b = np.convolve(
            signal.firwin(c, [float(f1), float(f2)], window="hamming", fs=fs), b
        )

    G = randRange(minG, maxG, 0)
    _, h = signal.freqz(b, 1, fs=fs)
    b = pow(10, G / 20) * b / np.amax(abs(h))
    return b


def filterFIR(x, b):
    N = b.shape[0] + 1
    xpad = np.pad(x, (0, N), "constant")
    y = signal.lfilter(b, 1, xpad)
    y = y[int(N / 2) : int(y.shape[0] - N / 2)]
    return y


def randRange(x1, x2, integer):
    y = np.random.uniform(low=x1, high=x2, size=(1,))
    if integer:
        y = int(y)
    return y


def LnL_convolutive_noise(
    x,
    N_f,
    nBands,
    minF,
    maxF,
    minBW,
    maxBW,
    minCoeff,
    maxCoeff,
    minG,
    maxG,
    minBiasLinNonLin,
    maxBiasLinNonLin,
    fs,
):
    y = [0] * x.shape[0]
    for i in range(0, N_f):
        if i == 1:
            minG = minG - minBiasLinNonLin
            maxG = maxG - maxBiasLinNonLin
        b = genNotchCoeffs(
            nBands, minF, maxF, minBW, maxBW, minCoeff, maxCoeff, minG, maxG, fs
        )
        y = y + filterFIR(np.power(x, (i + 1)), b)
    y = y - np.mean(y)
    y = normWav(y, 0)
    return y


# Impulsive signal dependent noise
def ISD_additive_noise(x, P, g_sd):
    beta = randRange(0, P, 0)

    y = copy.deepcopy(x)
    x_len = x.shape[0]
    n = int(x_len * (beta / 100))
    p = np.random.permutation(x_len)[:n]
    f_r = np.multiply(
        ((2 * np.random.rand(p.shape[0])) - 1), ((2 * np.random.rand(p.shape[0])) - 1)
    )
    r = g_sd * x[p] * f_r
    y[p] = x[p] + r
    y = normWav(y, 0)
    return y


# Stationary signal independent noise


def SSI_additive_noise(
    x,
    SNRmin,
    SNRmax,
    nBands,
    minF,
    maxF,
    minBW,
    maxBW,
    minCoeff,
    maxCoeff,
    minG,
    maxG,
    fs,
):
    noise = np.random.normal(0, 1, x.shape[0])
    b = genNotchCoeffs(
        nBands, minF, maxF, minBW, maxBW, minCoeff, maxCoeff, minG, maxG, fs
    )
    noise = filterFIR(noise, b)
    noise = normWav(noise, 1)
    SNR = randRange(SNRmin, SNRmax, 0)
    noise = (
        noise / np.linalg.norm(noise, 2) * np.linalg.norm(x, 2) / 10.0 ** (0.05 * SNR)
    )
    x = x + noise
    return x


def _pad_sequence(batch, padding_value=0.0):
    dim_size = batch[0].size()
    trailing_dims = dim_size[1:]
    max_len = max([s.size(0) for s in batch])

    if all(x.shape[0] == max_len for x in batch):
        return batch

    out_dims = (max_len,) + trailing_dims
    output_batch = []
    for tensor in batch:
        out_tensor = tensor.new_full(out_dims, padding_value)
        out_tensor[: tensor.size(0), ...] = tensor
        output_batch.append(out_tensor)
    return output_batch


def customize_collate(batch):
    elem = batch[0][0]
    t, l = zip(*batch)
    batch_new = _pad_sequence(list(t))
    out = None
    if torch.utils.data.get_worker_info() is not None:
        numel = max([x.numel() for x in batch_new]) * len(batch_new)
        # pylint: disable=W0212
        storage = elem.storage()._new_shared(numel)
        # pylint: enable=W0212
        out = elem.new(storage)
    batch_new = torch.stack(batch_new, 0, out=out)
    return batch_new, torch.stack(l)


def f_shuffle_slice_inplace(input_list, slice_start=None, slice_stop=None):
    if slice_start is None or slice_start < 0:
        slice_start = 0
    if slice_stop is None or slice_stop > len(input_list):
        slice_stop = len(input_list)

    idx = slice_start
    while idx < slice_stop - 1:
        idx_swap = random.randrange(idx, slice_stop)
        tmp = input_list[idx_swap]
        input_list[idx_swap] = input_list[idx]
        input_list[idx] = tmp
        idx += 1


def f_shuffle_in_block_inplace(input_list, block_size):
    if block_size <= 1:
        return

    list_length = len(input_list)
    for iter_idx in range(-(-list_length // block_size)):
        f_shuffle_slice_inplace(
            input_list, iter_idx * block_size, (iter_idx + 1) * block_size
        )


def f_shuffle_blocks_inplace(input_list, block_size):
    tmp_list = input_list.copy()

    block_number = len(input_list) // block_size

    shuffle_block_idx = list(range(block_number))
    random.shuffle(shuffle_block_idx)

    new_idx = None
    for iter_idx in range(block_size * block_number):
        block_idx = iter_idx // block_size
        in_block_idx = iter_idx % block_size
        new_idx = shuffle_block_idx[block_idx] * block_size + in_block_idx
        input_list[iter_idx] = tmp_list[new_idx]


class SamplerBlockShuffleByLen(torch_sampler.Sampler):
    # pylint: disable=W0231
    def __init__(self, buf_dataseq_length, batch_size):
        if batch_size == 1:
            print("Sampler block shuffle by length requires batch-size>1")
            sys.exit(1)

        self.m_block_size = batch_size * 4
        self.m_idx = np.argsort(buf_dataseq_length)

    def __iter__(self):
        tmp_list = list(self.m_idx.copy())
        f_shuffle_in_block_inplace(tmp_list, self.m_block_size)
        f_shuffle_blocks_inplace(tmp_list, self.m_block_size)
        return iter(tmp_list)

    def __len__(self):
        return len(self.m_idx)


class CMDataset(Dataset):
    def __init__(self, protocol, path_data, extractor, flip_label, device, **kwargs):
        super().__init__()
        with open(protocol, "r", encoding="utf8") as f:
            self.protocol = [line.strip().split(" ") for line in f]
            self.data_path = path_data
        self.extractor = extractor
        self.device = device
        self.flip = flip_label

    def _label(self, x):
        if self.flip:
            return 1 - x
        return x

    def __len__(self):
        return len(self.protocol)

    def len(self):
        return self.__len__()

    def _obtain(self, index):
        return np.expand_dims(
            sf.read(os.path.join(self.data_path, self.protocol[index][0] + ".wav"))[0],
            0,
        )

    def __getitem__(self, index):
        test_sample = self._obtain(index)
        test_sample = torch.tensor(test_sample, dtype=torch.float, device=self.device)
        with torch.no_grad():
            test_sample = self.extractor(test_sample).squeeze(1).cpu()
        test_label = self._label(torch.tensor(int(self.protocol[index][1])))
        return test_sample, test_label


def CMRawBoost(CMDataset):
    def process_Rawboost_feature(self, feature):
        feature = LnL_convolutive_noise(
            feature, 5, 5, 20, 8000, 100, 1000, 10, 100, 0, 0, 5, 20, 16000
        )
        feature = ISD_additive_noise(feature, 10, 2)
        return feature

    def _obtain(self, index):
        x = sf.read(os.path.join(self.data_path, self.protocol[index][0] + ".wav"))[0]
        x = self.process_Rawboost_feature(x)
        return np.expand_dims(x, 0)


class ASVDataset(Dataset):
    def __init__(self, protocol, path_data, extractor, flip_label, device, **kwargs):
        super().__init__()
        with open(protocol, "r", encoding="utf8") as f:
            self.protocol = [line.strip().split(" ") for line in f]
            self.data_path = path_data
        self.extractor = extractor
        self.device = device
        self.flip = flip_label

    def _label(self, x):
        if self.flip:
            return 1 - x
        return x

    def __len__(self):
        return len(self.protocol)

    def len(self):
        return self.__len__()

    def __getitem__(self, index):
        enroll = os.path.join(self.data_path, self.protocol[index][-1])
        test = os.path.join(self.data_path, self.protocol[index][-2])
        label = self.protocol[index][0]

        enroll = np.expand_dims(sf.read(enroll)[0], 0)
        test = np.expand_dims(sf.read(test)[0], 0)

        enroll = torch.tensor(enroll, dtype=torch.float, device=self.device)
        test = torch.tensor(test, dtype=torch.float, device=self.device)

        with torch.no_grad():
            enroll = self.extractor(enroll)
            test = self.extractor(test)

        test_sample = [enroll.squeeze().cpu(), test.squeeze().cpu()]
        test_label = self._label(torch.tensor(int(label)))
        return test_sample, test_label
