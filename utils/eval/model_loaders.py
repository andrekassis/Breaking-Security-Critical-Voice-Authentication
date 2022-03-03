import components.cm_models.loss as module_loss
import components.cm_models as module_arch
from components.asv_models.gmm import FullGMM
from components.asv_models.ivector_extract import ivectorExtractor
from components.asv_models.plda import PLDA
from components.asv_models.SVsystem import SVsystem
from components.asv_models.xvector import xvectorModel
from utils.audio import feats
import os
from pathlib import Path
import json
from collections import OrderedDict
import torch
import parse_config
from . import model_loaders
import numpy as np

def load_cm_asvspoof(config, device, load_checkpoint = True, loss = None):
        with Path(config).open('rt') as handle:
            config = json.load(handle, object_hook=OrderedDict)
        resume = config["path"]
        
        if loss == None:
            loss = config['loss']

        if loss['requires_device']:
            loss_fn = getattr(module_loss, loss['type'])(device = device, **loss['args'])
        else:
            loss_fn = getattr(module_loss, loss['type'])(**loss['args'])
            
        if hasattr(loss_fn, 'it'):
            loss_fn.it = np.inf
        
        if config['arch']['requires_device']:
            model = getattr(module_arch, config['arch']['type'])(device = device, **config['arch']['args'])
        else:
            model = getattr(module_arch, config['arch']['type'])(**config['arch']['args'])
        
        loss = loss_fn
        extractor = getattr(feats, config['extractor']['fn'])(device = device, **config['extractor']['args'])
         
        if load_checkpoint:
            checkpoint = torch.load(resume, map_location = device)
            if config['state_dict'] != None:
                state_dict = checkpoint[config['state_dict']]  
            else:
                state_dict = checkpoint   

            for key in ['feature.lfcc_fb', 'm_frontend.0.lfcc_fb', 'fc_att.weight', 'fc_att.bias']:
                try:
                    del state_dict[key]
                except:
                    pass

            model.load_state_dict(state_dict)
        
        model = model.to(device)

        if load_checkpoint:
            model.eval()

        return config['system'], [config['eer']], config['logits'], {'model': model, 'loss': loss, 'extractor': extractor, 'flip_label': config['flip_label']}

def load_asv_plda(config, device, loss = None):
        with Path(config).open('rt') as handle:
            config = json.load(handle, object_hook=OrderedDict)
        if config['asv_args']['fgmmfile'] != None:
            fgmm = FullGMM(os.path.join(config['asv_args']['base'], config['asv_args']['fgmmfile']), device)
            extractor = ivectorExtractor(os.path.join(config['asv_args']['base'], config['asv_args']['ivector_extractor']), device)
        else:
            fgmm = None
            extractor = xvectorModel(os.path.join(config['asv_args']['base'], config['asv_args']['ivector_extractor']), os.path.join(config['asv_args']['base'], config['asv_args']['transform']) ,device) 
        plda_mdl = PLDA(os.path.join(config['asv_args']['base'], config['asv_args']['plda_mdl']), device)
        
        SV_system = SVsystem(fgmm, extractor, plda_mdl, os.path.join(config['asv_args']['base'], config['asv_args']['ivector_mean']), device)  
        extractor = getattr(feats, config['extractor']['fn'])(device = device, **config['extractor']['args'])
        
        return config['system'], [config['eer']], config['logits'], {'model': SV_system, 'extractor': extractor, 'flip_label': config['flip_label']}
        
def load_joint(config, device, loss = None):
    #with Path(config).open('rt') as handle:
    #    config = json.load(handle, object_hook=OrderedDict)
    _, cm_eer, cm_l, cm = getattr(model_loaders, config['cm']['loader'])(config['cm']['config'], device, loss = loss)
    _, asv_eer, asv_l, asv = getattr(model_loaders, config['asv']['loader'])(config['asv']['config'], device, loss = loss)
    return config['system'], asv_eer + cm_eer, [asv_l, cm_l], {'asv_args': asv, 'cm_args': cm, 'lambda_asv': config['asv']['lambda'], 'lambda_cm': config['cm']['lambda']}
    
