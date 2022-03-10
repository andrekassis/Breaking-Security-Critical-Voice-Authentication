import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class xvecTDNN(nn.Module):
    def __init__(self, p_dropout=0):
        super(xvecTDNN, self).__init__()
        self.tdnn1 = nn.Conv1d(
            in_channels=30, out_channels=512, kernel_size=5, dilation=1
        )
        self.bn_tdnn1 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.dropout_tdnn1 = nn.Dropout(p=p_dropout)

        self.tdnn2 = nn.Conv1d(
            in_channels=512, out_channels=512, kernel_size=5, dilation=2
        )
        self.bn_tdnn2 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.dropout_tdnn2 = nn.Dropout(p=p_dropout)

        self.tdnn3 = nn.Conv1d(
            in_channels=512, out_channels=512, kernel_size=7, dilation=3
        )
        self.bn_tdnn3 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.dropout_tdnn3 = nn.Dropout(p=p_dropout)

        self.tdnn4 = nn.Conv1d(
            in_channels=512, out_channels=512, kernel_size=1, dilation=1
        )
        self.bn_tdnn4 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.dropout_tdnn4 = nn.Dropout(p=p_dropout)

        self.tdnn5 = nn.Conv1d(
            in_channels=512, out_channels=1500, kernel_size=1, dilation=1
        )
        self.bn_tdnn5 = nn.BatchNorm1d(1500, momentum=0.1, affine=False)
        self.dropout_tdnn5 = nn.Dropout(p=p_dropout)

        self.fc1 = nn.Linear(3000, 512)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.dropout_tdnn1(self.bn_tdnn1(F.relu(self.tdnn1(x))))
        x = self.dropout_tdnn2(self.bn_tdnn2(F.relu(self.tdnn2(x))))
        x = self.dropout_tdnn3(self.bn_tdnn3(F.relu(self.tdnn3(x))))
        x = self.dropout_tdnn4(self.bn_tdnn4(F.relu(self.tdnn4(x))))
        x = self.dropout_tdnn5(self.bn_tdnn5(F.relu(self.tdnn5(x))))

        stats = torch.cat((x.mean(dim=2), x.std(dim=2)), dim=1)
        return self.fc1(stats)


class xvectorModel(nn.Module):
    def __init__(self, resume, transform, device="cuda:0"):
        super(xvectorModel, self).__init__()
        self.net = xvecTDNN()
        checkpoint = torch.load(resume, map_location=device)
        self.net.load_state_dict(checkpoint)
        self.net = self.net.to(device)
        self.net = self.net.eval()

        with open(transform, "r") as f:
            l = [line.strip() for line in f][1:]
        l[-1] = l[-1][:-2]
        self.matrix = torch.tensor(
            np.array([eval("[ " + ",".join(line.split(" ")) + " ]") for line in l]),
            device=device,
        ).float()

    def forward(self, x):
        return self.net(x)

    def Extractivector(self, x):
        return self.forward(x)

    def LengthNormalization(self, ivector, expected_length):
        input_norm = torch.norm(ivector, dim=-1)
        if torch.prod(input_norm) == 0:
            print("Zero ivector!")
            exit(0)
        radio = 200 / input_norm
        ivector = ivector * radio[:, None]

        return ivector

    def SubtractGlobalMean(self, ivector, mean):
        x = ivector - mean.to(ivector.device)
        vec_out = torch.matmul(self.matrix[:, : x.shape[-1]], x.unsqueeze(-1)).squeeze(
            -1
        )
        if self.matrix.shape[-1] == x.shape[-1] + 1:
            vec_out += self.matrix[:, -1]

        if (
            self.matrix.shape[-1] != x.shape[-1] + 1
            and self.matrix.shape[-1] != x.shape[-1]
        ):
            raise ValueError("shape is invalid")
        return vec_out
