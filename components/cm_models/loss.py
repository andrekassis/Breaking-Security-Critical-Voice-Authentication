import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class P2SGradLoss(nn.Module):
    def __init__(self):
        super(P2SGradLoss, self).__init__()
        self.m_loss = nn.MSELoss()

    def forward(self, input_score, target):
        target = target.long()
        # with torch.no_grad():
        index = torch.zeros_like(input_score).to(target.device)
        index.scatter_(1, target.data.view(-1, 1), 1)
        loss = self.m_loss(input_score, index)
        return loss


class OCLoss(nn.Module):
    def __init__(self, r_real=0.9, r_fake=0.5, alpha=20.0):
        super(OCLoss, self).__init__()
        self.r_real = r_real
        self.r_fake = r_fake
        self.alpha = alpha
        self.softplus = nn.Softplus()

    def forward(self, output, labels):
        output = 1 - output.unsqueeze(1)
        output[labels == 0] = self.r_real - output[labels == 0]
        output[labels == 1] = output[labels == 1] - self.r_fake
        loss = self.softplus(self.alpha * output).mean()
        return loss


class NllLoss(nn.Module):
    def __init__(self):
        super(NllLoss, self).__init__()
        self.loss = nn.NLLLoss()

    def forward(self, output, target):
        try:
            return self.loss(output, target)
        except:
            return self.loss(output[0], target)


class CEL(nn.Module):
    def __init__(self, weights, device="cuda:0"):
        super(CEL, self).__init__()
        # print(device)
        weights = torch.tensor(np.array(weights), dtype=torch.float32, device=device)
        self.loss = nn.CrossEntropyLoss(weight=weights)

    def forward(self, output, target):
        # print(target.device)
        # print(output.device)
        # print(weights.device)
        return self.loss(output, target)


class AngleLoss(nn.Module):
    def __init__(self, gamma=0):
        super(AngleLoss, self).__init__()
        self.gamma = gamma
        self.it = 0
        self.LambdaMin = 5.0
        self.LambdaMax = 1500.0
        self.lamb = 1500.0

    def forward(self, input, target, eval=False):
        self.it += 1
        cos_theta, psi_theta = input
        target = target.view(-1, 1)  # size=(B,1)

        index = cos_theta.data * 0.0  # size=(B,Classnum)
        index.scatter_(1, target.data.view(-1, 1), 1)
        # index = index.byte()     #to uint8
        index = index.bool()
        index = Variable(index)

        self.lamb = max(self.LambdaMin, self.LambdaMax / (1 + 0.1 * self.it))
        output = cos_theta * 1.0  # size=(B,Classnum)

        output[index] -= cos_theta[index] * (1.0 + 0) / (1 + self.lamb)
        output[index] += psi_theta[index] * (1.0 + 0) / (1 + self.lamb)

        logpt = F.log_softmax(output, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if not eval:
            loss = loss.mean()

        return loss


class SSNet_loss(nn.Module):
    def __init__(self, protocol, device):
        super(SSNet_loss, self).__init__()
        with open(protocol, "r") as f:
            label_info = 1 - np.array([int(line.strip().split(" ")[1]) for line in f])
            num_zero_class = (label_info == 0).sum()
            num_one_class = (label_info == 1).sum()
            weights = torch.tensor(
                [num_one_class, num_zero_class], dtype=torch.float32, device=device
            )
            self.weights = weights / (weights.sum())

    def forward(self, output, target):
        return F.cross_entropy(output, target, weight=self.weights.to(output.device))


############################
class AMLoss(nn.Module):
    def __init__(self, num_classes, s=20, m=0.9):
        super(AMLoss, self).__init__()
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.loss = nn.CrossEntropyLoss()

    def forward(self, output, label):
        batch_size = output.shape[0]
        y_onehot = torch.FloatTensor(batch_size, self.num_classes)
        y_onehot.zero_()
        y_onehot = torch.tensor(y_onehot).to(output.device)
        y_onehot.scatter_(1, torch.unsqueeze(label, dim=-1), self.m)
        margin_logits = self.s * (output - y_onehot)
        loss = self.loss(margin_logits, label)
        return loss
