from resemblyzer import preprocess_wav, VoiceEncoder
from tqdm import tqdm
import numpy as np
import os
import sys

sys.path.append("../../")
from utils.eval.eer_tools import cal_roc_eer
import torch
import random
import argparse
import soundfile as sf
import tempfile


def eer(speakers, encoder):
    final = []
    for s in tqdm(speakers.keys()):
        utts = speakers[s]
        for utt in range(len(utts)):
            spk_embeds_a = np.array([encoder.embed_speaker([utts[utt]])])
            spkr = random.sample(list(speakers.keys()), 1)[0]
            while spkr == s:
                spkr = random.sample(list(speakers.keys()), 1)[0]
            spkr_utts = speakers[spkr]
            spk_embeds_b = np.array(
                [
                    encoder.embed_speaker(
                        [utts[u] for u in range(len(utts)) if u != utt][:1]
                    )
                ]
                + [encoder.embed_speaker(spkr_utts)][:1]
            )
            spk_sim_matrix = np.inner(spk_embeds_a, spk_embeds_b).squeeze()
            # print(spk_sim_matrix)
            final.append([spk_sim_matrix[0], 1])
            final.append([spk_sim_matrix[1], 0])
    final = np.array(final)
    results = cal_roc_eer(torch.tensor(final))
    return results


def verify(speakers, encoder, eer_val):
    final = []
    outs = []
    ctr = 0
    for s in tqdm(speakers.keys()):
        utts = speakers[s]
        spk_embeds_a = np.array([encoder.embed_speaker([utts[0]])])
        # spk_embeds_b = [np.array([encoder.embed_speaker([utts[u]])]) for u in range(1, len(utts))]
        # spk_sim_matrix = np.array([np.inner(spk_embeds_a, embed).squeeze() for embed in spk_embeds_b ]).mean()
        spk_embeds_b = np.array([encoder.embed_speaker(utts[1:])])
        spk_sim_matrix = np.inner(spk_embeds_a, spk_embeds_b).squeeze()
        final.append(spk_sim_matrix)
        if spk_sim_matrix >= eer_val:
            ctr += 1
            outs.append(s)
    return ctr, outs  # final


def labeledDict(lPath, label):
    with open(lPath) as f:
        lines = [line.strip().split(" ") for line in f]
    lines = [[line[0], line[2]] for line in lines if line[1] == label]
    spkrsDict = {}
    for line in lines:
        if spkrsDict.get(line[1]) is None:
            spkrsDict[line[1]] = [line[0]]
        elif len(spkrsDict[line[1]]) >= 10:
            continue
        else:
            spkrsDict[line[1]].append(line[0])
    return spkrsDict


def loadTest(lPath):
    with open(lPath) as f:
        lines = [line.strip().split(" ") for line in f]
    # lines = [ line[1:] for line in lines ]#[:20]
    lines = [line for line in lines]  # [:20]
    spkrsDict = {}
    for i, line in enumerate(lines):
        spkrsDict[i] = line
    return spkrsDict


def calc_eer(labels_file, wav_dir, encoder):
    spkrsBen = labeledDict(labels_file, "1")
    speaker_wavs_ben = {
        speaker: [
            preprocess_wav(os.path.join(wav_dir, wav + ".wav"))
            for wav in spkrsBen[speaker]
        ]
        for speaker in spkrsBen.keys()
    }
    eer_val = eer(speaker_wavs_ben, encoder)
    return eer_val


def calc_success_spoof(spkrsMal, wav_dir, eer_val):
    speaker_wavs_spoof = {
        speaker: [
            preprocess_wav(os.path.join(wav_dir, wav + ".wav"), source_sr=16000)
            for wav in spkrsMal[speaker]
        ]
        for speaker in tqdm(list(spkrsMal.keys()))
    }
    score, utts = verify(speaker_wavs_spoof, encoder, eer_val)
    return score, utts


def calc_success_adv(spkrsMal, wav_dir, exp, eer_val):
    speaker_wavs_adv = {
        speaker: [
            preprocess_wav(
                sf.read(
                    os.path.join(
                        os.path.join(os.path.join("../../experiments", exp), "wavs"),
                        spkrsMal[speaker][0] + "-adv.wav",
                    )
                )[0],
                source_sr=16000,
            )
        ]
        + [
            preprocess_wav(
                os.path.join(wav_dir, spkrsMal[speaker][wav] + ".wav"), source_sr=16000
            )
            for wav in range(1, len(spkrsMal[speaker]))
        ]
        for s, speaker in tqdm(enumerate(list(spkrsMal.keys())))
    }
    score, utts = verify(speaker_wavs_adv, encoder, eer_val)
    return score, utts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labels_file", "-l", default="../../datasets/asvspoofWavs/labels/eval.lab"
    )
    parser.add_argument("--wav_dir", "-w", default="../../datasets/asvspoofWavs/wavs/")
    parser.add_argument("--exp")
    parser.add_argument("--ref", "-r")
    parser.add_argument(
        "--eer", "-e"
    )  # default="[0.026865671641791045, 0.7704025506973267]")
    parser.add_argument("--device", "-d", default="cuda:0")
    parser.add_argument("--outdir")
    args = parser.parse_args()

    encoder = VoiceEncoder(args.device)

    if args.eer is None:
        eer_val = calc_eer(args.labels_file, args.wav_dir, encoder)
    else:
        eer_val = eval(args.eer)
    print("eer: " + str(eer_val[0]) + ", at threshold: " + str(eer_val[1]))

    if args.exp is not None:
        ref = (
            os.path.join(os.path.join("../../experiments", args.exp), "ref_fin.txt")
            if args.ref is None
            else args.ref
        )
        spkrsMal = loadTest(ref)
        # suc_spoof, spoofed = calc_success_spoof(spkrsMal, args.wav_dir, float(eer_val[1]))
        suc_adv, adv = calc_success_adv(
            spkrsMal, args.wav_dir, args.exp, float(eer_val[1])
        )
        # print("spoofed samples success rate: " + str(suc_spoof))
        print("adv samples success rate: " + str(suc_adv))
        if args.outdir:
            # with open(os.path.join(os.path.join(args.outdir, 'spoofAttack'), "resemblyzer.txt"), "w") as f:
            #    f.writelines([ str(line) + '\n' for line in spoofed])
            with open(
                os.path.join(os.path.join(args.outdir, "advAttack"), "resemblyzer.txt"),
                "w",
            ) as f:
                f.writelines([str(line) + "\n" for line in adv])
