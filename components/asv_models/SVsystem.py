# pylint: disable=E1102

import torch


class SVsystem(object):
    def __init__(self, fgmm, extractor, plda, ivector_meanfile, device):
        self.device = device
        self.fgmm = fgmm
        self.extractor = extractor
        self.plda = plda
        rfile = open(ivector_meanfile, "r")
        line = rfile.readline()
        data = line.split()[1:-1]
        self.ivector_dim = len(data)
        for i in range(len(data)):
            data[i] = float(data[i])
        self.ivector_mean = torch.tensor(data, device=device)
        rfile.close()

    def Getivector(self, acstc_data):
        if self.fgmm is not None:
            zeroth_stats, first_stats = self.fgmm.Zeroth_First_Stats(acstc_data)
            ivector, _, _ = self.extractor.Extractivector(zeroth_stats, first_stats)
        else:
            ivector = self.extractor.Extractivector(acstc_data)

        ivector = self.extractor.SubtractGlobalMean(ivector, self.ivector_mean)
        ivector = self.extractor.LengthNormalization(
            ivector,
            torch.sqrt(
                torch.tensor(self.ivector_dim, dtype=torch.float, device=self.device)
            ),
        )
        return ivector

    def TransformIvectors(
        self, ivector, num_utt, simple_length_norm=False, normalize_length=False
    ):
        return self.plda.TransformIvector(
            ivector, num_utt, simple_length_norm, normalize_length
        )

    def ComputePLDAScore(self, enrollivector, testivector):
        score = self.plda.ComputeScores(enrollivector, 1, testivector)
        return score
