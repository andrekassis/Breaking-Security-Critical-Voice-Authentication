import torch
import numpy as np

class ivectorExtractor(object):
    def __init__(self, mdlfile, device, random=False):
        self.device = device
        if random == True:
            self.num_gaussian = 2048
            self.dim = 60
            self.ivector_dim = 600
            self.extractor_matrix = torch.ones(
                self.num_gaussian, self.dim, self.ivector_dim, device=device)
            self.sigma_inv = torch.ones(
                self.num_gaussian, self.dim, self.dim, device=device)
            self.offset = torch.tensor(1.0, device=device)
        else:
            rdfile = open(mdlfile, 'r')
            line = rdfile.readline()
            while line != '':
                if '<w_vec>' in line:
                    data = line.split()[2:-1]
                    self.num_gaussian = len(data)
                    line = rdfile.readline()
                elif '<M>' in line:
                    extractor_matrix = []
                    for i in range(self.num_gaussian):
                        line = rdfile.readline()
                        component_extractor_matrix = []
                        while ']' not in line:
                            data = line.split()
                            for j in range(len(data)):
                                data[j] = float(data[j])
                            component_extractor_matrix.append(data)
                            line = rdfile.readline()
                        data = line.split()[:-1]
                        for j in range(len(data)):
                            data[j] = float(data[j])
                        component_extractor_matrix.append(data)
                        line = rdfile.readline()
                        extractor_matrix.append(component_extractor_matrix)
                    self.extractor_matrix = torch.tensor(
                        extractor_matrix, device=device)  # C*F*D
                elif '<SigmaInv>' in line:
                    self.dim = self.extractor_matrix.size()[1]
                    self.ivector_dim = self.extractor_matrix.size()[2]
                    self.sigma_inv = np.zeros((
                        self.num_gaussian, self.dim, self.dim), dtype=np.float32)
                    for i in range(self.num_gaussian):
                        line = rdfile.readline()
                        for j in range(self.dim):
                            data = line.split()
                            for k in range(j+1):
                                self.sigma_inv[i][j][k] = float(data[k])
                                self.sigma_inv[i][k][j] = float(data[k])
                            line = rdfile.readline()
                    self.sigma_inv = torch.tensor(
                        self.sigma_inv, device=device)
                elif '<IvectorOffset>' in line:
                    self.offset = torch.tensor(
                        float(line.split()[1]), device=device)
                    line = rdfile.readline()
                else:
                    line = rdfile.readline()
            rdfile.close()

        self.aux = torch.matmul(
            self.extractor_matrix.permute((0, 2, 1)), self.sigma_inv)
        self.aux2 = torch.matmul(torch.matmul(self.extractor_matrix.permute(
            (0, 2, 1)), self.sigma_inv), self.extractor_matrix)

    def Extractivector(self, zeroth_stats, first_stats):
        L = (zeroth_stats.unsqueeze(-1).unsqueeze(-1) * self.aux2).sum(1) + torch.eye(self.ivector_dim, device=self.device)
        linear = (torch.matmul(self.aux, first_stats.unsqueeze(-1))
                  ).sum(1).squeeze(-1)
        linear[:, 0] += self.offset
        L_inv = torch.inverse(L)
        
        ivector = torch.matmul(L_inv, linear.unsqueeze(-1)).squeeze(-1)
        ivector[:, 0] -= self.offset
        return ivector, L_inv, linear

    def LengthNormalization(self, ivector, expected_length):
        input_norm = torch.norm(ivector, dim = -1)
        if torch.prod(input_norm) == 0:
            print('Zero ivector!')
            exit(0)
        radio = expected_length/input_norm
        ivector = ivector*radio[:, None]

        return ivector

    def SubtractGlobalMean(self, ivector, mean):
        ivector = ivector-mean.to(ivector.device)
        return ivector
