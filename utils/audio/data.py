# pylint: disable=E1102
import os
import sys
import random
import soundfile as sf
import numpy as np
import torch
import torch.utils.data.sampler as torch_sampler
from torch.utils.data.dataloader import Dataset


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
    def __init__(self, protocol, path_data, extractor, flip_label, device):
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
        test_sample = np.expand_dims(
            sf.read(os.path.join(self.data_path, self.protocol[index][0] + ".wav"))[0],
            0,
        )
        test_sample = torch.tensor(test_sample, dtype=torch.float, device=self.device)
        with torch.no_grad():
            test_sample = self.extractor(test_sample).squeeze(1).cpu()
        test_label = self._label(torch.tensor(int(self.protocol[index][1])))
        return test_sample, test_label


class ASVDataset(Dataset):
    def __init__(self, protocol, path_data, extractor, flip_label, device):
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
