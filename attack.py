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

import warnings
warnings.filterwarnings("ignore")

from pydub import AudioSegment
from pydub.silence import detect_leading_silence as dls

from utils.adv_attack.estimators import ESTIMATOR
from utils.adv_attack.pgd_attacks import TIME_DOMAIN_ATTACK
from utils.adv_attack.attacker_loader import get_attacker, parse_config

from utils.audio.feats import Pad
from utils.generic.sortedDict import SortedDict
from utils.generic.score import Score
from utils.generic.plot import plot
from utils.generic.setup import setup_seed

trim_leading: AudioSegment = lambda x, threshold: x[dls(x, silence_threshold = threshold) :]
trim_trailing: AudioSegment = lambda x, threshold: trim_leading(x.reverse(), threshold).reverse()
strip_silence: AudioSegment = lambda x, threshold: trim_trailing(trim_leading(x, threshold), threshold)

def trim(audio, silence_threshold = -25.0):
    sound = AudioSegment.from_file(audio)
    stripped = strip_silence(sound, silence_threshold)
    ret = np.array(stripped.get_array_of_samples())
    ret = ret / np.max(np.abs(ret))
    return np.expand_dims(ret, 0)

def load_input(path_label, wav_dir):
    #return trim(os.path.join(wav_dir, path_label + '.wav')), np.array([[1, 0]])
    return np.expand_dims(sf.read(os.path.join(wav_dir, path_label + '.wav'))[0], 0), np.array([[1, 0]])
 
def iter_stats(x, adv, y, input, exp, ret):
    if not ret:
        return ''
    orig_res = exp['tester'].attack.estimator.result(x, 1 - np.argmax(y, axis=1))
    result = exp['tester'].attack.estimator.result(adv, 1 - np.argmax(y, axis=1))
    
    distance = np.max(abs(np.squeeze(adv-x[:, :adv.shape[-1]])))
    exp['bb_asv'] += np.array([ int(exp['eval_asv'][j].result(adv, 1 - np.argmax(y, axis=1),
                                           eval=True)[0][0] != "FAIL") for j in range(len(exp['eval_asv'])) ])
    
    c_res = [ exp['eval_cm'][j].result(adv, 1 - np.argmax(y, axis=1),
                                        eval=True)[0] for j in range(len(exp['eval_cm']))]
    if exp['verbose']:
        print(c_res)
        
    exp['bb_cm'] += np.array([ int(c_res[j][0] != "FAIL") for j in range(len(c_res)) ])
    
    to_write = str(input[0]) + ': orig: ' + str(orig_res) + ', wb - (' + str(result) + \
                            '), bb - (' + str(exp['bb_cm']) + '), bb_asv - (' + str(exp['bb_asv']) + \
                            '), total: ' + str(exp['ctr']+1) + ', distance - (' + str(distance) \
                            + ')'
    return to_write
    
def init_outdir(exp, config, arguments):
    os.makedirs("expirements", exist_ok = True)
    exp['id'] = time.strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join('expirements', exp['id'])
    os.makedirs(out_dir)
    shutil.copy(arguments.conf, os.path.join(out_dir, arguments.conf))
    shutil.copy(os.path.abspath(__file__), os.path.join(out_dir, 'attack.py'))
    shutil.copy('utils/adv_attack/pgd_attacks.py', os.path.join(out_dir, 'pgd_attacks.py'))
    exp['out_file'] = os.path.join(out_dir, 'results.txt')

    exp['out_wav'] = os.path.join(out_dir, 'wavs')
    os.makedirs(exp['out_wav']) 
    
    exp['pltdir'] = os.path.join(out_dir, 'plots')
    os.makedirs(exp['pltdir'])
    os.makedirs(os.path.join(exp['pltdir'], 'raw'))
    os.makedirs(os.path.join(exp['pltdir'], 'fft'))
    return exp
    
def init_systems(exp, config):
    exp['Attacker'] = get_attacker(config, attack_type = config['attack_type'], system = config["system"], device = device) 
    exp['tester'] = get_attacker(config, attack_type = "TIME_DOMAIN_ATTACK", system = "ADVJOINT", device = device)
    
    loader, conf_ret = parse_config(config["discriminators"], config["target"], system ="ADVCM")
    exp['eval_cm'] = [ ESTIMATOR(device = device, loader = loader, config = cf) for cf in conf_ret ]
    
    loader, conf_ret = parse_config(config["discriminators"], config["target"], system = "ADVSR")
    exp['eval_asv'] = [ ESTIMATOR(device = device, loader = loader, config = cf) for cf in conf_ret ]
    return exp
    
def init(config, device, arguments):
    exp = {'sr': config['sr'], 'perf': config['perf_log'], 'padder': Pad(config["length"]), 'num_samples': config['num_samples'],
      'wav_dir': os.path.join(config["input_dir"], 'wavs'), 'log_interval': config['log_interval'], 'verbose': config['verbose'],
                    'attack_type': config['attack_type'], 'system': config['system'], 'print_iter_out': config['print_iter_out'], 
                                                        'write_wavs': config['write_wavs'], 'write_plots': config['write_plots']}
    if exp['attack_type'] == 'CM_attack':
        exp['r_args'] = config['r_args']
    else:
        exp['r_args'] = {}
        
    try:
        exp['SD'] = SortedDict.fromfile(exp['perf'], Score.reader())
    except Exception as e:
        exp['SD'] = SortedDict()    
    exp['ctr'] = 0
    exp = init_outdir(exp, config, arguments)

    with open(config["inputs"]) as f:
        exp['inputs'] = [ line.strip().split(' ')[1:] for line in f.readlines() ]

    exp = init_systems(exp, config)
    
    exp['bb_cm'] = np.array([0] * len(exp['eval_cm']))
    exp['bb_asv'] = np.array([0] * len(exp['eval_asv']))

    return exp
    
def prepare_iter(input, exp, device):
    if exp['system'] != 'ADVCM':
        ref = random.sample(input[1:-1], exp['num_samples'])
        ref = [ trim(os.path.join(exp['wav_dir'], r + '.wav')) for r in ref]
        exp['padder'].max_len = np.min([r.shape[-1] for r in ref])
        ref = np.array([ exp['padder'](torch.tensor(r)).numpy().squeeze() for r in ref ])
        
        if exp['attack_type'] != 'CM_Attack':
            exp['Attacker'].attack.estimator.set_ref(ref, device)
        else:
            exp['Attacker'].attack.set_ref(ref, device)
            
    ref = np.expand_dims(sf.read(os.path.join(exp['wav_dir'], input[-1] + '.wav'))[0], 0)
    exp['tester'].attack.estimator.set_ref(ref, device)
    [ e.set_ref(ref, device) for e in exp['eval_asv'] ]
    
    return exp
    
def run_iter(input, exp, device, **r_args):
    x, y = load_input(input[0], exp['wav_dir'])
    exp = prepare_iter(input, exp, device)
    try:
        adv = exp['Attacker'].generate(x, y, evalu = exp['eval_cm'], **r_args)[1]
    except:
        print("attack_failed! skipping")
        return x, x, y, False
    return x, adv, y, True

def write_iter(x, adv, exp, idx, input, to_write):
    if idx % exp['log_interval'] == 0:
        exp['SD'][exp['id']] = Score(exp['bb_cm'] / (exp['ctr'] + 1), exp['bb_asv'] / (exp['ctr'] + 1), exp['ctr'] + 1)
        exp['SD'].tofile(exp['perf'])

    if exp['write_wavs']:
        sf.write(exp['out_wav'] + "/" + input[0] + "-adv.wav", adv.squeeze(), exp["sr"])
        sf.write(exp['out_wav'] + "/" + input[0] + "-orig.wav", x.squeeze(), exp["sr"])

    with open(exp['out_file'], 'a+', buffering=1) as f:
        if exp['print_iter_out']:
            print(to_write)
        f.write(to_write + '\n')
    
    if exp['write_plots']:
        plot(input[0], exp['out_wav'], exp['pltdir'])

def main(config, device, arguments):
    exp = init(config, device, arguments)
    for i, input in enumerate(tqdm(exp['inputs'])):
        x, adv, y, ret = run_iter(input, exp, device, **exp['r_args'])
        to_write = iter_stats(x, adv, y, input, exp, ret)        
        write_iter(x, adv, exp, i, input, to_write)
   
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf')
    parser.add_argument('--device')
    
    arguments = parser.parse_args()
    device = arguments.device
    
    with open(arguments.conf) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    
    setup_seed(1234)
    main(config, device, arguments)
