# pylint: disable=E1102
import torch
import torch.nn.functional as F
import numpy as np


class FullGMM(object):
    def __init__(self, mdlfile, device, random=False):
        self.device = device
        if random is True:
            self.num_gaussians = 2048
            self.dim = 60
            self.gconsts = torch.ones(self.num_gaussians, device=device)
            self.weights = torch.ones(self.num_gaussians, device=device)
            self.means_invcovars = torch.ones(
                self.num_gaussians, self.dim, device=device
            )
            self.invcovars = torch.ones(
                self.num_gaussians, self.dim, self.dim, device=device
            )
        else:
            rdfile = open(mdlfile, "r")
            line = rdfile.readline()
            while line != "":
                if "<GCONSTS>" in line:
                    gconsts = line.split()[2:-1]
                    self.num_gaussians = len(gconsts)
                    for i in range(self.num_gaussians):
                        gconsts[i] = float(gconsts[i])
                    self.gconsts = torch.tensor(gconsts, device=device)
                    line = rdfile.readline()
                elif "<WEIGHTS>" in line:
                    weights = line.split()[2:-1]
                    for i in range(self.num_gaussians):
                        weights[i] = float(weights[i])
                    self.weights = torch.tensor(weights, device=device)
                    line = rdfile.readline()
                elif "<MEANS_INVCOVARS>" in line:
                    line = rdfile.readline()
                    means_invcovars = []
                    for i in range(self.num_gaussians):
                        data = line.split(" ")[2:-1]
                        for j in range(len(data)):
                            data[j] = float(data[j])
                        means_invcovars.append(data)
                        line = rdfile.readline()
                    self.dim = len(data)
                    self.means_invcovars = torch.tensor(means_invcovars, device=device)
                elif "<INV_COVARS>" in line:
                    self.invcovars = np.zeros(
                        (self.num_gaussians, self.dim, self.dim), dtype=np.float32
                    )
                    for i in range(self.num_gaussians):
                        line = rdfile.readline()
                        for j in range(self.dim):
                            data = line.split(" ")[:-1]
                            for k in range(len(data)):
                                self.invcovars[i][j][k] = float(data[k])
                                self.invcovars[i][k][j] = float(data[k])
                            line = rdfile.readline()
                    self.invcovars = torch.tensor(self.invcovars, device=device)
                else:
                    line = rdfile.readline()
            rdfile.close()
        self.Means()

    def Means(self):
        self.means = torch.zeros(self.num_gaussians, self.dim, device=self.device)
        self.means = torch.matmul(
            torch.inverse(self.invcovars), self.means_invcovars.unsqueeze(-1)
        ).squeeze(-1)

    def ComponentLogLikelihood(self, data):
        loglike = torch.matmul(self.means_invcovars, data.unsqueeze(-1)).squeeze(-1)
        mul = torch.matmul(self.invcovars, data.permute((0, 2, 1)).unsqueeze(1))
        loglike -= 0.5 * torch.matmul(
            mul.permute((0, 3, 1, 2)), data.unsqueeze(-1)
        ).squeeze(-1)
        loglike += self.gconsts
        return loglike

    def Posterior(self, data):
        post = F.softmax(self.ComponentLogLikelihood(data), -1)
        return post

    def Zeroth_First_Stats(self, data_seq):
        post = self.Posterior(data_seq)
        zeroth_stats = post.sum(1)
        first_stats = torch.matmul(post.permute(0, 2, 1), data_seq)
        return zeroth_stats, first_stats
