##
import time

import kaldi_io
import torch
import torch.nn.functional as F

from local.gmm import FullGMM
from local.ivector_extract import ivectorExtractor
from local.data_prepare import load_data
from local.plda import PLDA

# torch.cuda.set_device(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cuda:1"


class SVsystem(object):
    def __init__(self, fgmm, extractor, plda, ivector_meanfile):
        self.fgmm = fgmm
        self.extractor = extractor
        self.plda = plda
        # self.enrollkeys, self.enrollivectors = plda.ReadIvectors(enrollscpfile)
        # self.enrollivectors = torch.tensor(self.enrollivectors)
        self.ivector_dim = extractor.ivector_dim
        rfile = open(ivector_meanfile, "r")
        line = rfile.readline()
        data = line.split()[1:-1]
        for i in range(self.ivector_dim):
            data[i] = float(data[i])
        self.ivector_mean = torch.tensor(data, device=device)
        rfile.close()

    def Getivector(self, acstc_data):
        # time0 = time.time()
        zeroth_stats, first_stats = self.fgmm.Zeroth_First_Stats(acstc_data)
        # time1 = time.time()
        ivector, L_inv, linear = self.extractor.Extractivector(
            zeroth_stats, first_stats
        )
        # time2 = time.time()
        # ivector = self.extractor.LengthNormalization(ivector, torch.sqrt(torch.tensor(self.ivector_dim, dtype=torch.float, device=device)))
        ivector = self.extractor.SubtractGlobalMean(ivector, self.ivector_mean)
        # time3 = time.time()
        ivector = self.extractor.LengthNormalization(
            ivector,
            torch.sqrt(
                torch.tensor(self.ivector_dim, dtype=torch.float, device=device)
            ),
        )
        # time4 = time.time()
        # print("get ivector stats")
        # print(str(time1-time0) + " zeroth")
        # print(str(time2-time1) + " Extract")
        # print(str(time3-time2) + " sub mean")
        # print(str(time4-time3) + " normalize")
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
