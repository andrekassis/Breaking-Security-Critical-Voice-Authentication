import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import math
from collections import namedtuple

Genotype = namedtuple("Genotype", "normal normal_concat reduce reduce_concat")

OPS = {
    "none": lambda C, stride, affine: Zero(stride),
    "max_pool_3": lambda C, stride, affine: nn.MaxPool1d(3, stride=stride, padding=1),
    "avg_pool_3": lambda C, stride, affine: nn.AvgPool1d(
        3, stride=stride, padding=1, count_include_pad=False
    ),
    "skip_connect": lambda C, stride, affine: Identity()
    if stride == 1
    else FactorizedReduce(C, C, affine=affine),
    "std_conv_3": lambda C, stride, affine: StdConv1d(
        C, C, 3, stride, 1, affine=affine
    ),
    "std_conv_5": lambda C, stride, affine: StdConv1d(
        C, C, 5, stride, 2, affine=affine
    ),
    "std_conv_7": lambda C, stride, affine: StdConv1d(
        C, C, 7, stride, 3, affine=affine
    ),
    "dil_conv_3": lambda C, stride, affine: DilConv(
        C, C, 3, stride, 2, 2, affine=affine
    ),
    "dil_conv_5": lambda C, stride, affine: DilConv(
        C, C, 5, stride, 4, 2, affine=affine
    ),
    "dil_conv_7": lambda C, stride, affine: DilConv(
        C, C, 7, stride, 6, 2, affine=affine
    ),
}


class ReLUConvBN_half(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN_half, self).__init__()
        self.op = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.3),
            nn.Conv1d(
                C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False
            ),
            nn.MaxPool1d(2),
            nn.BatchNorm1d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class ReLUConvBN_same(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN_same, self).__init__()
        self.op = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.3),
            nn.Conv1d(
                C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False
            ),
            nn.BatchNorm1d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class DilConv(nn.Module):
    def __init__(
        self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True
    ):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.3),
            nn.Conv1d(
                C_in,
                C_in,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=C_in,
                bias=False,
            ),
            nn.Conv1d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm1d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class StdConv1d(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(StdConv1d, self).__init__()
        self.op = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.3),
            nn.Conv1d(
                C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False
            ),
            nn.BatchNorm1d(C_in, affine=affine),
        )

    def forward(self, x):

        return self.op(x)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):
    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.0)
        return x[:, :, :: self.stride].mul(0.0)


class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.LeakyReLU(negative_slope=0.3)

        self.conv_1 = nn.Conv1d(C_in, C_out // 2, 1, stride=1, padding=0, bias=False)
        self.conv_2 = nn.Conv1d(C_in, C_out // 2, 1, stride=1, padding=0, bias=False)
        self.pl = nn.MaxPool1d(2)
        self.bn = nn.BatchNorm1d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x)], dim=1)
        out = self.pl(out)
        out = self.bn(out)
        return out


class P2SActivationLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(P2SActivationLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        return

    def forward(self, input_feat):
        w = self.weight.renorm(2, 1, 1e-5).mul(1e5)
        x_modulus = input_feat.pow(2).sum(1).pow(0.5)
        w_modulus = w.pow(2).sum(0).pow(0.5)
        inner_wx = input_feat.mm(w)
        cos_theta = inner_wx / x_modulus.view(-1, 1)
        cos_theta = cos_theta.clamp(-1, 1)
        return cos_theta


class Conv_0(nn.Module):
    def __init__(
        self,
        out_channels,
        kernel_size,
        stride=1,
        padding=2,
        dilation=1,
        bias=False,
        groups=1,
        is_mask=False,
    ):
        super(Conv_0, self).__init__()
        self.conv = nn.Conv1d(
            1, out_channels, kernel_size, stride, padding, dilation, groups
        )
        self.channel_number = out_channels
        self.is_mask = is_mask

    def forward(self, x, is_training):
        x = self.conv(x)
        if is_training and self.is_mask:
            v = self.channel_number
            f = np.random.uniform(low=0.0, high=16)
            f = int(f)
            f0 = np.random.randint(0, v - f)
            x[:, f0 : f0 + f, :] = 0

        return x


class SincConv_fast(nn.Module):
    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(
        self,
        out_channels,
        kernel_size,
        sample_rate=16000,
        in_channels=1,
        stride=1,
        padding=2,
        dilation=1,
        bias=False,
        groups=1,
        min_low_hz=50,
        min_band_hz=50,
        freq_scale="mel",
        is_trainable=False,
        is_mask=False,
    ):

        super(SincConv_fast, self).__init__()

        if in_channels != 1:
            msg = (
                "SincConv only support one input channel (here, in_channels = {%i})"
                % (in_channels)
            )
            raise ValueError(msg)

        self.out_channels = out_channels + 4
        self.kernel_size = kernel_size
        self.is_mask = is_mask

        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1

        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        if bias:
            raise ValueError("SincConv does not support bias.")
        if groups > 1:
            raise ValueError("SincConv does not support groups.")

        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        low_hz = 0
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)

        if freq_scale == "mel":
            mel = np.linspace(
                self.to_mel(low_hz), self.to_mel(high_hz), self.out_channels + 1
            )
            hz = self.to_hz(mel)
        elif freq_scale == "lem":
            mel = np.linspace(
                self.to_mel(low_hz), self.to_mel(high_hz), self.out_channels + 1
            )
            hz = self.to_hz(mel)
            hz = np.abs(np.flip(hz) - 1)
        elif freq_scale == "linear":
            hz = np.linspace(low_hz, high_hz, self.out_channels + 1)

        self.low_hz_ = nn.Parameter(
            torch.Tensor(hz[:-1]).view(-1, 1), requires_grad=is_trainable
        )

        self.band_hz_ = nn.Parameter(
            torch.Tensor(np.diff(hz)).view(-1, 1), requires_grad=is_trainable
        )

        n_lin = torch.linspace(
            0, (self.kernel_size / 2) - 1, steps=int((self.kernel_size / 2))
        )
        self.window_ = 0.54 - 0.46 * torch.cos(2 * math.pi * n_lin / self.kernel_size)

        n = (self.kernel_size - 1) / 2.0
        self.n_ = 2 * math.pi * torch.arange(-n, 0).view(1, -1) / self.sample_rate

    def forward(self, waveforms, is_training=False):
        self.n_ = self.n_.to(waveforms.device)

        self.window_ = self.window_.to(waveforms.device)

        low = self.min_low_hz + torch.abs(self.low_hz_)

        high = torch.clamp(
            low + self.min_band_hz + torch.abs(self.band_hz_),
            self.min_low_hz,
            self.sample_rate / 2,
        )
        band = (high - low)[:, 0]

        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)

        band_pass_left = (
            (torch.sin(f_times_t_high) - torch.sin(f_times_t_low)) / (self.n_ / 2)
        ) * self.window_
        band_pass_center = 2 * band.view(-1, 1)
        band_pass_right = torch.flip(band_pass_left, dims=[1])

        band_pass = torch.cat(
            [band_pass_left, band_pass_center, band_pass_right], dim=1
        )

        band_pass = band_pass / (2 * band[:, None])

        self.filters = (band_pass).view(self.out_channels, 1, self.kernel_size)
        self.filters = self.filters[: self.out_channels - 4, :, :]

        if is_training and self.is_mask:
            v = self.filters.shape[0]
            f = np.random.uniform(low=0.0, high=16)
            f = int(f)
            f0 = np.random.randint(0, v - f)
            self.filters[f0 : f0 + f, :, :] = 0
        output = F.conv1d(
            waveforms,
            self.filters,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=None,
            groups=1,
        )
        return output


def drop_path(x, drop_prob):
    if drop_prob > 0.0:
        keep_prob = 1.0 - drop_prob
        mask = torch.cuda.FloatTensor(x.size(0), 1, 1).bernoulli_(keep_prob)
        x.div_(keep_prob)
        x.mul_(mask)
    return x


class Cell(nn.Module):
    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        self.reduction = reduction

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN_half(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN_same(C_prev, C, 1, 1, 0, affine=False)

        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
        self._compile(C, op_names, indices, concat, reduction)
        self.pooling_layer = nn.MaxPool1d(2)

    def _compile(self, C, op_names, indices, concat, reduction):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 1
            op = OPS[name](C, stride, True)
            self._ops += [op]
        self._indices = indices

    def forward(self, s0, s1, drop_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2 * i]]
            h2 = states[self._indices[2 * i + 1]]
            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]
            h1 = op1(h1)
            h2 = op2(h2)

            if self.training and drop_prob > 0.0:
                if not isinstance(op1, Identity):
                    h1 = drop_path(h1, drop_prob)
                if not isinstance(op2, Identity):
                    h2 = drop_path(h2, drop_prob)
            s = h1 + h2
            states += [s]
        out = torch.cat([states[i] for i in self._concat], dim=1)
        out = self.pooling_layer(out)
        return out


class Network(nn.Module):
    def __init__(
        self,
        C,
        layers,
        num_classes,
        genotype,
        gru_hsize=1024,
        gru_layers=3,
        is_mask=False,
    ):
        super(Network, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        genotype = eval(genotype)

        self.sinc = SincConv_fast(
            C, kernel_size=128, freq_scale="mel", is_mask=is_mask, is_trainable=False
        )

        self.mp = nn.MaxPool1d(3)
        self.bn = nn.BatchNorm1d(C)
        self.lrelu = nn.LeakyReLU(negative_slope=0.3)

        self.stem = nn.Sequential(
            nn.Conv1d(C, C, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(C),
            nn.LeakyReLU(negative_slope=0.3),
        )

        C_prev_prev, C_prev, C_curr = C, C, C

        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False

            cell = Cell(
                genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev
            )
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr

        self.gru = nn.GRU(
            input_size=C_prev,
            hidden_size=gru_hsize,
            num_layers=gru_layers,
            batch_first=True,
        )
        self.fc_gru = nn.Linear(gru_hsize, gru_hsize)
        self.l_layer = P2SActivationLayer(gru_hsize, out_dim=2)

    def forward(self, input, eval=False):

        # input = input.unsqueeze(1)
        s0 = self.sinc(input, self.training)
        s0 = self.mp(s0)
        s0 = self.bn(s0)
        s0 = self.lrelu(s0)
        s1 = self.stem(s0)

        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)

        v = s1
        v = v.permute(0, 2, 1)
        self.gru.flatten_parameters()
        v, _ = self.gru(v)
        v = v[:, -1, :]
        embeddings = self.fc_gru(v)
        logits = self.l_layer(embeddings)
        return logits

    def forward_classifier(self, embeddings):
        logits = self.l_layer(embeddings)
        return logits

    def save_state(self, path):
        state_dict = self.state_dict()
        torch.save(state_dict, path)

    def optimizer(self, opt, **params):
        optimizer = getattr(torch.optim, opt)(self.parameters(), **params)
        return [optimizer]


def DartsRaw(drop_path_prob=0.0, **kwargs):
    model = Network(**kwargs)
    model.drop_path_prob = drop_path_prob
    return model
