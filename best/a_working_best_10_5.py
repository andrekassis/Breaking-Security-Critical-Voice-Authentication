import yaml
import soundfile as sf
import numpy as np
import os
import random
import argparse
import shutil
import torch
import time
import json
from collections import OrderedDict
import pickle
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import noisereduce as nr
from pydub import AudioSegment
from pydub.silence import detect_leading_silence

from art.attacks.evasion.carlini import CarliniL2Method
from art.attacks.evasion.boundary import BoundaryAttack
from art.attacks.evasion.brendel_bethge import BrendelBethgeAttack
from art.attacks.evasion.auto_projected_gradient_descent import AutoProjectedGradientDescent

from utils.adv_attack.estimators import ESTIMATOR_NON_ADAPTIVE
from utils.adv_attack.pgd_attacks import TIME_DOMAIN_ATTACK, FFT_Attack, STFT_Attack
from utils.audio.feats import Pad
from utils.generic.sortedDict import SortedDict
from utils.generic.score import Score
from utils.adv_attack.reduce import sp

trim_leading_silence: AudioSegment = lambda x, silence_threshold: x[detect_leading_silence(x, silence_threshold = silence_threshold) :]
trim_trailing_silence: AudioSegment = lambda x, silence_threshold: trim_leading_silence(x.reverse(), silence_threshold).reverse()
strip_silence: AudioSegment = lambda x, silence_threshold: trim_trailing_silence(trim_leading_silence(x, silence_threshold), silence_threshold)

def trim(audio, silence_threshold = -25.0):
    sound = AudioSegment.from_file(audio)
    stripped = strip_silence(sound, silence_threshold)
    ret = np.array(stripped.get_array_of_samples())
    ret = ret / np.max(np.abs(ret))
    return ret

def parse_config(config, dset, system):
    if system == "ADVJOINT":
        loader = "load_joint"
    elif system == "ADVCM":
        loader = "load_cm_asvspoof"
    else:
        loader = "load_asv_plda"
    
    model_cm = [ cm for cm in config["args"]["cm"]["selector"] ] if system != "ADVSR" else []
    model_asv = [ asv for asv in config["args"]["asv"]["selector"] ] if system != "ADVCM" else []
    model_cm = [ os.path.join(os.path.join(os.path.join(os.path.join("configs", dset['cm']), "cm"), model), "config.json") for model in model_cm ]
    model_asv = [ os.path.join(os.path.join(os.path.join(os.path.join("configs", dset['asv']), "asv"), model), "config.json") for model in model_asv ]
    
    if system == "ADVJOINT":
        lambdas = [config["args"]["cm"]["lambda"], config["args"]["asv"]["lambda"]]
        conf_ret = [ {"cm": {"loader": "load_cm_asvspoof", "config": cm, "lambda": lambdas[0]},
                   "asv": {"loader": "load_asv_plda", "config": asv, "lambda": lambdas[1]},
                   "system": "ADVJOINT"} for cm in model_cm for asv in model_asv ]
        return loader, conf_ret
        
    if system == "ADVCM":
        return loader, model_cm
        
    return loader, model_asv
    
    
class AttackerWrapper:
    def __init__(self, attack, attack_type, input_dir, lengths):
        self.attack = attack
        self.attack_type = attack_type
        self.input_dir = input_dir
        self.lengths = lengths

    def _load_example(self, x, longer=False):
            with open(self.lengths, "r") as f:
                lengths = [line.strip().split(' ') for line in f]
            lines = [line for line in lengths if int(line[1]) == 1 ]

            if longer:
                lines = [line[0] for line in lines if int(line[2]) >= x.shape[1]]
            else:
                lines = [ line[0] for line in lines ]

            line = random.sample(lines, 1)[0]
            input_path = os.path.join(self.input_dir, "wavs/" + line + ".wav")
            init = sf.read(input_path)[0]
            return init

    def generate(self, x, y, **r_args):
        if len(x.shape) == 1:
            x = np.expand_dims(x, 0)
        self.attack.estimator.set_input_shape((x.shape[1], ))
        if self.attack_type == "FFT_Attack" or self.attack_type == "TIME_DOMAIN_ATTACK" or self.attack_type == "STFT_Attack":
            return x, self.attack.generate(x, y, **r_args)
        if self.attack_type == "carlini" or self.attack_type == "auto_pgd":
            return x, self.attack.generate(x, 1 - y)

        init = self._load_example(x, longer = True)[:x.shape[1]]
        init = np.expand_dims(init, 0)
        return x, self.attack.generate(x, x_adv_init = init)

def get_attacker(config, attack_type, system, device, r_c=None):
    loader, conf_ret = parse_config(config["discriminator_wb"], config["shadow"], system)

    est = ESTIMATOR_NON_ADAPTIVE(device = device, loader = loader, config = conf_ret[0], loss = config['loss'])
    if config['discriminator_wb']['args']['cm']['selector'][0] == 'comparative' or config['discriminator_wb']['args']['cm']['selector'][0] == 'RawDarts':
        if system == 'ADVCM':
            est.attack.model.train()
        elif system == 'ADVJOINT':
            est.attack.cm.model.train()

    if attack_type == "TIME_DOMAIN_ATTACK":
        Attacker = TIME_DOMAIN_ATTACK(est, **config["TIME_DOMAIN_ATTACK"], r_c = r_c)
    elif attack_type == "carlini":
        Attacker = CarliniL2Method(est, **config["carlini"])
    elif attack_type == "boundary":
        Attacker = BoundaryAttack(est, **config["boundary"])
    elif attack_type == "bb":
        Attacker = BrendelBethgeAttack(est, **config["bb"])
    elif attack_type == "FFT_Attack":
        Attacker = FFT_Attack(est, **config["FFT_Attack"], r_c = r_c)
    elif attack_type == "STFT_Attack":
            Attacker = STFT_Attack(est, **config["STFT_Attack"], r_c = r_c)
    elif attack_type == "auto_pgd":
        est.set_input_shape((16000*6,))
        Attacker = AutoProjectedGradientDescent(est, **config["auto_pgd"])
    
    return AttackerWrapper(Attacker, attack_type, config["input_dir"], config["lengths"])

def load_input(path_label, wav_dir):
    #return trim(os.path.join(wav_dir, path_label + '.wav')), np.array([[1, 0]])
    return sf.read(os.path.join(wav_dir, path_label + '.wav'))[0], np.array([[1, 0]])

def configure_attack(A, start, m, power, delta, alpha_factor, max_iter):
    A.attack.epsilon =  np.power(start / m, power)
    A.attack.alpha = A.attack.epsilon / alpha_factor
    A.attack.delta = delta
    A.attack.max_iter = max_iter
    return A

def generate(adv, y, TFD, FD, TD, itrs, sg, evalu=None, verbose = False):
    TFD.attack.length = adv.shape[1]
    TD = configure_attack(TD, start=0.005, m=1, power=1, delta=0, alpha_factor=10, max_iter=30)
    adv = TD.generate(adv, y, **{'p': 1, 'n_std_thresh_stationary': 1.5})[1]
    
    if verbose:
        print([ evalu[j].result(adv, 1 - np.argmax(y, axis=1), eval=True)[0] for j in range(len(evalu))])

    for k in tqdm(range(itrs)):   
        m =  np.min([k+1, 15])
        p= 1 / (k+1.1)
        r= (1/(1-p) - 1)/ 3 + 1
        n_std_thresh_stationary = 1.5
        r_args = {'p': 1, 'n_std_thresh_stationary': n_std_thresh_stationary}

        FD = configure_attack(FD, start=0.2, m=m, power=1.2, delta=0, alpha_factor=1, max_iter=1)
        TFD = configure_attack(TFD, start=0.0002, m=m, power=0.9, delta=0, alpha_factor=1, max_iter=1)

        adv = TFD.generate(adv, y, **r_args)[1]
        adv = FD.generate(adv, y, **r_args)[1]
         
        if k < 5: #True: #k < 10:#5:
            adv = sg(adv, p, n_std_thresh_stationary= 1.5)      
            adv = np.clip(adv*r, -1, 1)
        
        adv = FD.generate(adv, y, **r_args)[1] 
        adv = TFD.generate(adv, y, **r_args)[1]

        if verbose:
            print([ evalu[j].result(adv / np.max(np.abs(adv)), 1 - np.argmax(y, axis=1), eval=True)[0] for j in range(len(evalu))])

    TD = configure_attack(TD, start=0.0002, m=1, power=1, delta=0, alpha_factor=10, max_iter=30)
    #adv = TD.generate(adv, y, **{'p': 1, 'n_std_thresh_stationary': 2.5})[1]
    adv = TD.generate(adv, y, **{'p': None})[1]

    
    if verbose:
        print([ evalu[j].result(adv / np.max(np.abs(adv)), 1 - np.argmax(y, axis=1), eval=True)[0] for j in range(len(evalu))])
        
    return adv / np.max(np.abs(adv))
    
def log_stats(orig, adv, y, eval_asv, eval_cm, bb_asv, bb_cm, tester, id, input, idx, SD, log_interval):
    orig_res = tester.attack.estimator.result(orig, 1 - np.argmax(y, axis=1))
    result = tester.attack.estimator.result(adv, 1 - np.argmax(y, axis=1))
    distance = np.max(abs(np.squeeze(adv-orig[:, :adv.shape[-1]])))
    bb_asv += np.array([ int(eval_asv[j].result(adv, 1 - np.argmax(y, axis=1),
                                           eval=True)[0][0] != "FAIL") for j in range(len(eval_asv)) ])
    c_res = [ eval_cm[j].result(adv, 1 - np.argmax(y, axis=1),
                                        eval=True)[0] for j in range(len(eval_cm))]
    
    bb_cm += np.array([ int(c_res[j][0] != "FAIL") for j in range(len(c_res)) ])
    to_write = str(input[0]) + ': orig: ' + str(orig_res) + ', wb - (' + str(result) + \
                                        '), bb - (' + str(bb_cm) + '), bb_asv - (' + str(bb_asv) + '), total: ' + str(idx+1) + \
                                                   ', distance - (' + str(distance) + ')'

    if idx > 0 and idx % log_interval == 0:
        SD[id] = Score(bb_cm / (idx + 1), bb_asv / (idx + 1), idx + 1)
        SD.tofile('perf.txt')

    return bb_asv, bb_cm, to_write
    
def init(config, device):
    config["STFT_Attack"]["length"] = config["length"]
    reduce_conf = {'stationary': True, 'win': 'kaiser_window', 'n_fft': 1024, 'hop_length': 256, 'win_length': 1024, 'device': device, 'freq_mask_smooth_hz': 500, 'time_mask_smooth_ms': 50, 'padding':0}

    TD = get_attacker(config, attack_type = "TIME_DOMAIN_ATTACK", system = "ADVSR", r_c = reduce_conf, device = device) 
    FD = get_attacker(config, attack_type = "FFT_Attack", system = "ADVSR", r_c = reduce_conf, device = device) 
    TFD = get_attacker(config, attack_type = "STFT_Attack", system = "ADVSR", r_c = reduce_conf, device = device)
    tester = get_attacker(config, attack_type = "TIME_DOMAIN_ATTACK", system = "ADVJOINT", r_c = reduce_conf, device = device)
    
    labels_file = os.path.join(config["input_dir"], 'labels/eval.lab')
    wav_dir = os.path.join(config["input_dir"], 'wavs')

    with open(config["inputs"]) as f:
        inputs = [ line.strip().split(' ')[1:] for line in f.readlines() ]

    os.makedirs("expirements", exist_ok = True)
    id = time.strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join('expirements', id)
    os.makedirs(out_dir)
    shutil.copy(arguments.conf, os.path.join(out_dir, arguments.conf))
    shutil.copy(os.path.abspath(__file__), os.path.join(out_dir, 'attack.py'))
    shutil.copy('utils/adv_attack/pgd_attacks.py', os.path.join(out_dir, 'pgd_attacks.py'))
    out_file = os.path.join(out_dir, 'results.txt')

    out_wav = os.path.join(out_dir, 'wavs')
    os.makedirs(out_wav) 

    try:
        SD = SortedDict.fromfile('perf.txt', Score.reader())
    except Exception as e:
        SD = SortedDict()

    loader, conf_ret = parse_config(config["discriminators"], config["target"], system ="ADVCM")
    eval_cm = [ ESTIMATOR_NON_ADAPTIVE(device = device, loader = loader, config = cf) for cf in conf_ret ]
    
    loader, conf_ret = parse_config(config["discriminators"], config["target"], system = "ADVSR")
    eval_asv = [ ESTIMATOR_NON_ADAPTIVE(device = device, loader = loader, config = cf) for cf in conf_ret ]
    return FD, TFD, TD, tester, eval_asv, eval_cm, inputs, out_file, wav_dir, out_wav, id, SD, reduce_conf
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf')
    parser.add_argument('--device')
    
    arguments = parser.parse_args()
    device = arguments.device
    
    with open(arguments.conf) as f:
        config = yaml.load(f, Loader=yaml.Loader)    
     
    FD, TFD, TD, tester, eval_asv, eval_cm, inputs, out_file, wav_dir, out_wav, id, SD, r_c = init(config, device)    
    r_c['stationary'] = False
    sg = sp(**r_c) 
    bb_cm = np.array([0] * len(eval_cm))
    bb_asv = np.array([0] * len(eval_asv))
    padder = Pad(config["length"])
    with open(out_file, 'w', buffering=1) as f:
        for i, input in enumerate(tqdm(inputs)):
            x, y = load_input(input[0], wav_dir)
            ref = random.sample(input[1:-1], config['num_samples'])
            #ref = [ sf.read(os.path.join(wav_dir, r + '.wav'))[0] for r in ref]
            ref = [ trim(os.path.join(wav_dir, r + '.wav')) for r in ref]
            padder.max_len = np.min([len(r) for r in ref])
            ref = np.array([ np.squeeze(np.squeeze(padder(torch.tensor(np.expand_dims(r, 0))).numpy(), 0), 0) for r in ref ])


            for a in [TD, FD, TFD]:
                a.attack.estimator.set_ref(ref, device)

            ref = np.expand_dims(sf.read(os.path.join(wav_dir, input[-1] + '.wav'))[0], 0)
            tester.attack.estimator.set_ref(ref, device)
            [ e.set_ref(ref, device) for e in eval_asv ]
            orig = np.expand_dims(x, 0)            
            adv = np.copy(orig)

            adv = generate(adv, y, TFD, FD, TD, 50, sg, evalu = eval_cm, verbose = True)

            bb_asv, bb_cm, to_write = log_stats(orig, adv, y, eval_asv, eval_cm, bb_asv, bb_cm, tester, id, input, i, SD, config['log_interval'])
            sf.write(out_wav + "/" + input[0] + "-adv.wav", adv[0], config["sr"])
            sf.write(out_wav + "/" + input[0] + "-orig.wav", orig[0], config["sr"])
            print(to_write)
            f.write(to_write + '\n')

