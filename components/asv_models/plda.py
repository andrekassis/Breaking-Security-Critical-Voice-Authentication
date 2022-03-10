# pylint: disable=E1102
import torch


class PLDA(object):
    def __init__(self, mdlfile, device, random=False):
        self.device = device
        if random is True:
            self.dim = 600
            self.mean = torch.ones(self.dim, device=device)
            self.transform = torch.ones(self.dim, self.dim, device=device)
            self.psi = torch.ones(self.dim, device=device)
        else:
            rdfile = open(mdlfile, "r")
            line = rdfile.readline()
            data = line.split()[2:-1]
            self.dim = len(data)
            for i in range(self.dim):
                data[i] = float(data[i])
            self.mean = torch.tensor(data, device=device)

            line = rdfile.readline()
            line = rdfile.readline()
            transform_matrix = []
            for i in range(self.dim):
                data = line.split(" ")[2:-1]
                for j in range(self.dim):
                    data[j] = float(data[j])
                transform_matrix.append(data)
                line = rdfile.readline()
            self.transform = torch.tensor(transform_matrix, device=device)

            data = line.split()[1:-1]
            for i in range(self.dim):
                data[i] = float(data[i])
            self.psi = torch.tensor(data, device=device)

            rdfile.close()

    def TransformIvector(
        self, ivector, num_examples, simple_length_norm, normalize_length
    ):
        trans_ivector = torch.matmul(
            self.transform, (ivector - self.mean).unsqueeze(-1)
        ).squeeze(-1)
        factor = 1.0
        if simple_length_norm is True:
            factor = torch.sqrt(self.dim) / torch.norm(trans_ivector, 2, dim=-1)
        elif normalize_length is True:
            factor = self.GetNormalizaionFactor(trans_ivector, num_examples)

        trans_ivector = trans_ivector * factor[:, None]

        return trans_ivector

    def GetNormalizaionFactor(self, trans_ivector, num_examples):
        trans_ivector_sq = torch.pow(trans_ivector, 2)
        inv_covar = 1.0 / (self.psi + 1.0 / num_examples)
        factor = torch.sqrt(
            self.dim / torch.matmul(inv_covar, trans_ivector_sq.unsqueeze(-1))
        ).squeeze(-1)

        return factor

    def ComputeScores(self, trans_trainivector, num_examples, trans_testivector):
        mean = (
            num_examples * self.psi / (num_examples * self.psi + 1.0)
        ) * trans_trainivector
        variance = 1.0 + self.psi / (num_examples * self.psi + 1.0)

        logdet = torch.sum(torch.log(variance))

        sqdiff = torch.pow(trans_testivector - mean, 2)

        variance = 1.0 / variance

        loglike_given_class = -0.5 * (
            logdet
            + torch.log(2 * torch.tensor(3.1415926, device=self.device)) * self.dim
            + torch.matmul(sqdiff, variance)
        )

        # work out loglike_without_class
        sqdiff = torch.pow(trans_testivector, 2)
        variance = self.psi + 1.0
        logdet = torch.sum(torch.log(variance))
        variance = 1.0 / variance

        loglike_without_class = -0.5 * (
            logdet
            + torch.log(2 * torch.tensor(3.1415926, device=self.device)) * self.dim
            + torch.matmul(sqdiff, variance)
        )

        loglike_ratio = loglike_given_class - loglike_without_class
        return loglike_ratio
