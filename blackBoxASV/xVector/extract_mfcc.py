import os, sys
import numpy as np
import argparse
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from functools import partial

from tqdm import tqdm

import kaldi_io
from tf_mfcc import mfcc


def build_from_path(wavlist, out_dir, rstfilename, num_workers=1):
    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []
    utt_list = []
    for wav_info in tqdm(wavlist):
        items = wav_info.split()
        utt_idx = items[0]
        utt_list.append(utt_idx)
        channel = 0
        wav_path = items[1]
        futures.append(_process_utterance(wav_path))

    ark_scp_output = (
        "ark:| copy-feats ark:- ark,scp:"
        + out_dir
        + "/"
        + rstfilename
        + ".ark,"
        + out_dir
        + "/"
        + rstfilename
        + ".scp"
    )
    write_matrix(utt_list, [future for future in futures], ark_scp_output)


def _process_utterance(wav_path):
    mc = mfcc(wav_path)
    return mc


def preprocess(wavlist, out_dir, rstfilename, nj):
    os.makedirs(out_dir, exist_ok=True)
    build_from_path(wavlist, out_dir, rstfilename, nj)


def write_matrix(utt_list, matrix_list, filename):
    with kaldi_io.open_or_fd(filename, "wb") as f:
        for key, matrix in zip(utt_list, matrix_list):
            kaldi_io.write_mat(f, matrix, key=key)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--train-wav-file', type=str, default='/scratch/xli/kaldi/egs/voxceleb_xli/v1/data/voxceleb1_train/wav.scp')
    parser.add_argument(
        "--test-wav-file",
        type=str,
        default="/scratch/xli/kaldi/egs/voxceleb_xli/v1/data/voxceleb1_test/wav.scp",
    )
    parser.add_argument("--num-workers", type=int, default=5)
    parser.add_argument("--out-dir", type=str, default="./data/LPMS")
    # parser.add_argument('--train-rstfilename', type=str, default='train')
    parser.add_argument("--test-rstfilename", type=str, default="test")
    args = parser.parse_args()

    args.num_workers = 5
    print("number of workers: ", args.num_workers)

    # print("Preprocess train data ...")
    # rfile = open(args.train_wav_file, 'r')
    # wavlist = rfile.readlines()
    # rfile.close()
    # preprocess(wavlist, args.out_dir, args.train_rstfilename, args.num_workers)

    print("Preprocess test data ...")
    rfile = open(args.test_wav_file, "r")
    wavlist = rfile.readlines()
    rfile.close()
    preprocess(wavlist, args.out_dir, args.test_rstfilename, args.num_workers)

    print("DONE!")
    sys.exit(0)
