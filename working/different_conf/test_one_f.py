import yaml
import soundfile as sf
import numpy as np
import os
import random
import argparse
import shutil
import torch
import time
from tqdm import tqdm
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

trim_leading_silence: AudioSegment = lambda x, silence_threshold: x[detect_leading_silence(x, silence_threshold = silence_threshold) :]
trim_trailing_silence: AudioSegment = lambda x, silence_threshold: trim_leading_silence(x.reverse(), silence_threshold).reverse()
strip_silence: AudioSegment = lambda x, silence_threshold: trim_trailing_silence(trim_leading_silence(x, silence_threshold), silence_threshold)

def trim(audio, silence_threshold = -30.0):
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
    model_cm = [ os.path.join(os.path.join(os.path.join(os.path.join("configs", dset), "cm"), model), "config.json") for model in model_cm ]
    model_asv = [ os.path.join(os.path.join(os.path.join(os.path.join("configs", dset), "asv"), model), "config.json") for model in model_asv ]
    
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

    def generate(self, x, y):
        if len(x.shape) == 1:
            x = np.expand_dims(x, 0)
        self.attack.estimator.set_input_shape((x.shape[1], ))
        if self.attack_type == "FFT_Attack" or self.attack_type == "TIME_DOMAIN_ATTACK" or self.attack_type == "STFT_Attack":
            return x, self.attack.generate(x, y)
        if self.attack_type == "carlini" or self.attack_type == "auto_pgd":
            return x, self.attack.generate(x, 1 - y)

        init = self._load_example(x, longer = True)[:x.shape[1]]
        init = np.expand_dims(init, 0)
        return x, self.attack.generate(x, x_adv_init = init)

def get_attacker(config, attack_type, system, device):
    loader, conf_ret = parse_config(config["discriminator_wb"], config["shadow"], system)
    est = ESTIMATOR_NON_ADAPTIVE(device = device, loader = loader, config = conf_ret[0], loss = config["loss"])
    if config['discriminator_wb']['args']['cm']['selector'][0] == 'comparative':
        if system == 'ADVCM':
            est.attack.model.train()
        elif system == 'ADVJOINT':
            est.attack.cm.model.train()

    if attack_type == "TIME_DOMAIN_ATTACK":
        Attacker = TIME_DOMAIN_ATTACK(est, **config["TIME_DOMAIN_ATTACK"])
    elif attack_type == "carlini":
        Attacker = CarliniL2Method(est, **config["carlini"])
    elif attack_type == "boundary":
        Attacker = BoundaryAttack(est, **config["boundary"])
    elif attack_type == "bb":
        Attacker = BrendelBethgeAttack(est, **config["bb"])
    elif attack_type == "FFT_Attack":
        Attacker = FFT_Attack(est, **config["FFT_Attack"])
    elif attack_type == "STFT_Attack":
            Attacker = STFT_Attack(est, **config["STFT_Attack"])
    elif attack_type == "auto_pgd":
        est.set_input_shape((16000*6,))
        Attacker = AutoProjectedGradientDescent(est, **config["auto_pgd"])
    
    return AttackerWrapper(Attacker, attack_type, config["input_dir"], config["lengths"])

def load_input(path_label, wav_dir):
    return sf.read(os.path.join(wav_dir, path_label + '.wav'))[0], np.array([[1, 0]])

def generate(adv, y, TFD, FD, TD, sr=16000, nfft=400, itrs=450, prop_decrease=0.7, evalu=None): 
    TFD.attack.length = adv.shape[1]
    itrs = 600
    
    TD.attack.delta = 0 
    TFD.attack.delta = None
    FD.attack.delta = 0

    TFD.attack.max_iter = 15
    adv = TFD.generate(adv, y)[1]
    TFD.attack.delta = 0

    for k in tqdm(range(850)):
        FD.attack.epsilon = np.power(0.3 / (k+2), 0.98)
        m = 0.05 * (k+1) * np.log10(itrs  + 1 - np.min([k, 7.5*itrs/9])) ** 2 
        TFD.attack.epsilon =  0.000003 / 10 #m #0.03 / np.power(k+1, 0.5)
        
        FD.attack.alpha = FD.attack.epsilon 
        FD.attack.max_iter = 1 

        #TFD.attack.epsilon =  np.power(0.5 /(k+2), 0.98) / 500
        TFD.attack.alpha = TFD.attack.epsilon 
        TFD.attack.max_iter = 1
        p= 1 / (k+1.5)

        if True:
            adv = nr.reduce_noise(y = adv[0], sr=sr, stationary= False, n_fft=200, prop_decrease=p)
            adv *= (1 / (1-p) - 1) / 3 + 1
            adv = adv.clip(-1, 1)     
            adv = np.expand_dims(adv, 0)
            
        adv = TFD.generate(adv, y)[1]
        if True: #(k % 5 == 1 or k % 5 == 3 or k % 5 == 4):
            adv = FD.generate(adv, y)[1]
        
        ret = adv
        ret = ret / np.max(np.abs(ret))
        c_res = [ evalu[j].result(ret, 1 - np.argmax(y, axis=1),
                                                    eval=True)[0] for j in range(len(evalu))]
        print(c_res)
        if k == 25:
            break

    TFD.attack.epsilon, TFD.attack.alpha, TFD.attack.max_iter = 0.00005, 0.000005, 25

    adv = adv / np.max(np.abs(adv))
    adv = TFD.generate(adv, y)[1]
    
    ret = adv / np.max(np.abs(adv))
    c_res = [ evalu[j].result(ret, 1 - np.argmax(y, axis=1),
                                                            eval=True)[0] for j in range(len(evalu))]
    print(c_res)
    return adv
    
def log_stats(orig, adv, y, eval_asv, eval_cm, bb_asv, bb_cm, tester, input):
    orig_res = tester.attack.estimator.result(orig, 1 - np.argmax(y, axis=1))
    result = tester.attack.estimator.result(adv, 1 - np.argmax(y, axis=1))
    distance = np.max(abs(np.squeeze(adv-orig[:, :adv.shape[-1]])))
    bb_asv += np.array([ int(eval_asv[j].result(adv, 1 - np.argmax(y, axis=1),
                                           eval=True)[0][0] != "FAIL") for j in range(len(eval_asv)) ])
    c_res = [ eval_cm[j].result(adv, 1 - np.argmax(y, axis=1),
                                        eval=True)[0] for j in range(len(eval_cm))]
    
    bb_cm += np.array([ int(c_res[j][0] != "FAIL") for j in range(len(c_res)) ])
    to_write = str(input[0]) + ': orig: ' + str(orig_res) + ', wb - (' + str(result) + \
                                        '), bb - (' + str(bb_cm) + '), bb_asv - (' + str(bb_asv) + '), total: ' + str(i+1) + \
                                                   ', distance - (' + str(distance) + ')'
    return bb_asv, bb_cm, to_write
    
def init(config, device):
    config["STFT_Attack"]["length"] = config["length"]

    TD = get_attacker(config, attack_type = "TIME_DOMAIN_ATTACK", system = "ADVSR", device = device)
    FD = get_attacker(config, attack_type = "FFT_Attack", system = "ADVSR", device = device) 
    TFD = get_attacker(config, attack_type = "STFT_Attack", system = "ADVSR", device = device) 

    tester = get_attacker(config, attack_type = "TIME_DOMAIN_ATTACK", system = "ADVJOINT", device = device)
    
    labels_file = os.path.join(config["input_dir"], 'labels/eval.lab')
    wav_dir = os.path.join(config["input_dir"], 'wavs')

    with open(config["inputs"]) as f:
        inputs = [ line.strip().split(' ')[1:] for line in f.readlines() ]
    
    os.makedirs("expirements", exist_ok = True)
    out_dir = os.path.join('expirements', time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(out_dir)
    shutil.copy(arguments.conf, os.path.join(out_dir, arguments.conf))
    out_file = os.path.join(out_dir, 'results.txt')
    out_wav = os.path.join(out_dir, 'wavs')
    os.makedirs(out_wav)
        
    loader, conf_ret = parse_config(config["discriminators"], config["target"], system ="ADVCM")
    eval_cm = [ ESTIMATOR_NON_ADAPTIVE(device = device, loader = loader, config = cf) for cf in conf_ret ]
    
    loader, conf_ret = parse_config(config["discriminators"], config["target"], system = "ADVSR")
    eval_asv = [ ESTIMATOR_NON_ADAPTIVE(device = device, loader = loader, config = cf) for cf in conf_ret ]
    return FD, TFD, TD, tester, eval_asv, eval_cm, inputs, out_file, wav_dir, out_wav
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf')
    parser.add_argument('--device')
    arguments = parser.parse_args()
    device = arguments.device
    with open(arguments.conf) as f:
        config = yaml.load(f, Loader=yaml.Loader)    

    FD, TFD, TD, tester, eval_asv, eval_cm, inputs, out_file, wav_dir, out_wav = init(config, device)   
    
    bb_cm = np.array([0] * len(eval_cm))
    bb_asv = np.array([0] * len(eval_asv))
    padder = Pad(config["length"])
    with open(out_file, 'w', buffering=1) as f:
        for i, input in enumerate(tqdm(inputs)):
            x, y = load_input(input[0], wav_dir)
            ref = [ sf.read(os.path.join(wav_dir, input[k] + '.wav'))[0] for k in range(1, 20)]
            #ref = [ trim(os.path.join(wav_dir, input[k] + '.wav')) for k in range(1, 15)]
            ref = np.array([ np.squeeze(np.squeeze(padder(torch.tensor(np.expand_dims(r, 0))).numpy(), 0), 0) for r in ref ])
            for a in [TD, FD, TFD]:
                a.attack.estimator.set_ref(ref, device)

            ref = np.expand_dims(sf.read(os.path.join(wav_dir, input[-1] + '.wav'))[0], 0)
            tester.attack.estimator.set_ref(ref, device)
            [ e.set_ref(ref, device) for e in eval_asv ]
            orig = np.expand_dims(x, 0)            
           
            adv = generate(orig, y, TFD, FD, TD, sr=config["sr"], nfft=config["nr_nfft"], itrs=config["optimization_itrs"], prop_decrease=config["prop_decrease"], evalu = eval_cm)

            bb_asv, bb_cm, to_write = log_stats(orig, adv, y, eval_asv, eval_cm, bb_asv, bb_cm, tester, input)
            sf.write(out_wav + "/" + input[0] + "-adv.wav", adv[0], config["sr"])
            sf.write(out_wav + "/" + input[0] + "-orig.wav", orig[0], config["sr"])
            print(to_write)
            f.write(to_write + '\n')
