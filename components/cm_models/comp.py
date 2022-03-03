import numpy as np
import random
import os
import torch
from torch.nn import Parameter
from utils.audio.feats import LFCC

def set_random_seed(random_seed):                                   
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)

    cudnn_deterministic = True
    cudnn_benchmark = False
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = cudnn_benchmark
    return
    
def parse_length(input_str):
    return int(input_str.split(',')[3])

def parse_filename(input_str):
    return input_str.split(',')[1]

class MaxFeatureMap2D(torch.nn.Module):
    def __init__(self, max_dim = 1):
        super().__init__()
        self.max_dim = max_dim
        
    def forward(self, inputs):
        shape = list(inputs.size())
        if self.max_dim >= len(shape):
            exit(1)
        if shape[self.max_dim] // 2 * 2 != shape[self.max_dim]:
            exit(1)
        shape[self.max_dim] = shape[self.max_dim]//2
        shape.insert(self.max_dim, 2)
        m, i = inputs.view(*shape).max(self.max_dim)
        return m
        
class BLSTMLayer(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BLSTMLayer, self).__init__()
        if output_dim % 2 != 0:
            exit(1)

        self.l_blstm = torch.nn.LSTM(input_dim, output_dim // 2, \
                                     bidirectional=True)
    def forward(self, x):
        blstm_data, _ = self.l_blstm(x.permute(1, 0, 2))
        return blstm_data.permute(1, 0, 2)
        
class P2SActivationLayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(P2SActivationLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        self.weight = Parameter(torch.Tensor(in_dim, out_dim))
        self.weight.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        return

    def forward(self, input_feat):
        w = self.weight.renorm(2, 1, 1e-5).mul(1e5)

        x_modulus = input_feat.pow(2).sum(1).pow(0.5)
        w_modulus = w.pow(2).sum(0).pow(0.5)
        inner_wx = input_feat.mm(w)
        cos_theta = inner_wx / x_modulus.view(-1, 1)
        cos_theta = cos_theta.clamp(-1, 1)

        return cos_theta
        
def protocol_parse(protocol_filepath):
    data_buffer = {}
    temp_buffer = np.loadtxt(protocol_filepath, dtype='str')
    for row in temp_buffer:
        if row[-1] == 'bonafide':
            data_buffer[row[1]] = 1
        else:
            data_buffer[row[1]] = 0
    return data_buffer

class Model(torch.nn.Module):
    def __init__(self, in_dim, out_dim, mean_std=None, device='cuda:0'):
        super(Model, self).__init__()

        in_m, in_s, out_m, out_s = self.prepare_mean_std(in_dim,out_dim,\
                                                         mean_std)
        self.input_mean = torch.nn.Parameter(in_m, requires_grad=False)
        self.input_std = torch.nn.Parameter(in_s, requires_grad=False)
        self.output_mean = torch.nn.Parameter(out_m, requires_grad=False)
        self.output_std = torch.nn.Parameter(out_s, requires_grad=False)
        
        #protocol_file = prj_conf.optional_argument[0]
        #self.protocol_parser = protocol_parse(protocol_file)
        
        self.m_target_sr = 16000

        self.frame_hops = [160]
        self.frame_lens = [320]
        self.fft_n = [512]

        self.lfcc_dim = [20]
        self.lfcc_with_delta = True

        self.win = torch.hann_window
        self.amp_floor = 0.00001

        self.v_truncate_lens = [None for x in self.frame_hops]
        self.v_submodels = len(self.frame_lens)        
        self.v_emd_dim = 64

        self.v_out_class = 2
        self.m_transform = []
        self.m_before_pooling = []
        self.m_output_act = []
        self.m_frontend = []
        self.m_angle = []
        
        for idx, (trunc_len, fft_n, lfcc_dim) in enumerate(zip(
                self.v_truncate_lens, self.fft_n, self.lfcc_dim)):
            
            fft_n_bins = fft_n // 2 + 1
            if self.lfcc_with_delta:
                lfcc_dim = lfcc_dim * 3
            
            self.m_transform.append(
                torch.nn.Sequential(
                    torch.nn.Conv2d(1, 64, [5, 5], 1, padding=[2, 2]),
                    MaxFeatureMap2D(),
                    torch.nn.MaxPool2d([2, 2], [2, 2]),

                    torch.nn.Conv2d(32, 64, [1, 1], 1, padding=[0, 0]),
                    MaxFeatureMap2D(),
                    torch.nn.BatchNorm2d(32, affine=False),
                    torch.nn.Conv2d(32, 96, [3, 3], 1, padding=[1, 1]),
                    MaxFeatureMap2D(),

                    torch.nn.MaxPool2d([2, 2], [2, 2]),
                    torch.nn.BatchNorm2d(48, affine=False),

                    torch.nn.Conv2d(48, 96, [1, 1], 1, padding=[0, 0]),
                    MaxFeatureMap2D(),
                    torch.nn.BatchNorm2d(48, affine=False),
                    torch.nn.Conv2d(48, 128, [3, 3], 1, padding=[1, 1]),
                    MaxFeatureMap2D(),

                    torch.nn.MaxPool2d([2, 2], [2, 2]),

                    torch.nn.Conv2d(64, 128, [1, 1], 1, padding=[0, 0]),
                    MaxFeatureMap2D(),
                    torch.nn.BatchNorm2d(64, affine=False),
                    torch.nn.Conv2d(64, 64, [3, 3], 1, padding=[1, 1]),
                    MaxFeatureMap2D(),
                    torch.nn.BatchNorm2d(32, affine=False),

                    torch.nn.Conv2d(32, 64, [1, 1], 1, padding=[0, 0]),
                    MaxFeatureMap2D(),
                    torch.nn.BatchNorm2d(32, affine=False),
                    torch.nn.Conv2d(32, 64, [3, 3], 1, padding=[1, 1]),
                    MaxFeatureMap2D(),
                    torch.nn.MaxPool2d([2, 2], [2, 2]),
                    
                    torch.nn.Dropout(0.7)
                )
            )

            self.m_before_pooling.append(
                torch.nn.Sequential(
                    BLSTMLayer((lfcc_dim//16) * 32, (lfcc_dim//16) * 32),
                    BLSTMLayer((lfcc_dim//16) * 32, (lfcc_dim//16) * 32)
                )
            )

            self.m_output_act.append(
                torch.nn.Linear((lfcc_dim // 16) * 32, self.v_emd_dim)
            )

            self.m_angle.append(
                P2SActivationLayer(self.v_emd_dim, self.v_out_class)
            )
            
            self.m_frontend.append(
                LFCC(self.frame_lens[idx],
                                   self.frame_hops[idx],
                                   self.fft_n[idx],
                                   self.m_target_sr,
                                   self.lfcc_dim[idx],
                                   self.lfcc_dim[idx],
                                   skip = False,
                                   with_energy=True,
                                   with_emphasis=True,
                                   compress = True,
                                   device = device)
            )

        self.m_frontend = torch.nn.ModuleList(self.m_frontend)
        self.m_transform = torch.nn.ModuleList(self.m_transform)
        self.m_output_act = torch.nn.ModuleList(self.m_output_act)
        self.m_angle = torch.nn.ModuleList(self.m_angle)
        self.m_before_pooling = torch.nn.ModuleList(self.m_before_pooling)
        # done
        return
    
    def prepare_mean_std(self, in_dim, out_dim, data_mean_std=None):
        if data_mean_std is not None:
            in_m = torch.from_numpy(data_mean_std[0])
            in_s = torch.from_numpy(data_mean_std[1])
            out_m = torch.from_numpy(data_mean_std[2])
            out_s = torch.from_numpy(data_mean_std[3])
            if in_m.shape[0] != in_dim or in_s.shape[0] != in_dim:
                print("Input dim: {:d}".format(in_dim))
                print("Mean dim: {:d}".format(in_m.shape[0]))
                print("Std dim: {:d}".format(in_s.shape[0]))
                print("Input dimension incompatible")
                sys.exit(1)
            if out_m.shape[0] != out_dim or out_s.shape[0] != out_dim:
                print("Output dim: {:d}".format(out_dim))
                print("Mean dim: {:d}".format(out_m.shape[0]))
                print("Std dim: {:d}".format(out_s.shape[0]))
                print("Output dimension incompatible")
                sys.exit(1)
        else:
            in_m = torch.zeros([in_dim])
            in_s = torch.ones([in_dim])
            out_m = torch.zeros([out_dim])
            out_s = torch.ones([out_dim])
            
        return in_m, in_s, out_m, out_s
        
    def normalize_input(self, x):
        return (x - self.input_mean) / self.input_std

    def normalize_target(self, y):
        return (y - self.output_mean) / self.output_std

    def denormalize_output(self, y):
        return y * self.output_std + self.output_mean


    def _front_end(self, wav, idx, trunc_len, datalength):        
        #with torch.no_grad():
        x_sp_amp = self.m_frontend[idx](wav.squeeze(-1))
        return x_sp_amp

    def _compute_embedding(self, x, datalength):
        batch_size = x.shape[0]
        output_emb = torch.zeros([batch_size * self.v_submodels, 
                                  self.v_emd_dim], 
                                  device=x.device, dtype=x.dtype)

        for idx, (fs, fl, fn, trunc_len, m_trans, m_be_pool, m_output) in \
            enumerate(
                zip(self.frame_hops, self.frame_lens, self.fft_n, 
                    self.v_truncate_lens, self.m_transform, 
                    self.m_before_pooling, self.m_output_act)):
            
            x_sp_amp = self._front_end(x, idx, trunc_len, datalength)
            hidden_features = m_trans(x_sp_amp)
            hidden_features = hidden_features.permute(0, 2, 1, 3).contiguous()
            frame_num = hidden_features.shape[1]
            hidden_features = hidden_features.view(batch_size, frame_num, -1)
            hidden_features_lstm = m_be_pool(hidden_features)
            tmp_emb = m_output((hidden_features_lstm + hidden_features).sum(1))
            
            output_emb[idx * batch_size : (idx+1) * batch_size] = tmp_emb

        return output_emb

    def _compute_score(self, x, inference=False):
        batch_size = x.shape[0]
        out_score = torch.zeros(
            [batch_size * self.v_submodels, self.v_out_class], 
            device=x.device, dtype=x.dtype)

        for idx, m_score in enumerate(self.m_angle):
            tmp_score = m_score(x[idx * batch_size : (idx+1) * batch_size])
            out_score[idx * batch_size : (idx+1) * batch_size] = tmp_score

        return out_score

    def _get_target(self, filenames):
        try:
            return [self.protocol_parser[x] for x in filenames]
        except KeyError:
            print("Cannot find target data for %s" % (str(filenames)))
            exit(1)

    def forward(self, x, fileinfo = None, eval = False):
        datalength = [ y.shape[0] for y in x ]
        feature_vec = self._compute_embedding(x, datalength)
        scores = self._compute_score(feature_vec, eval)
        return scores
            

    def save_state(self, path):
        state_dict = self.state_dict()
        torch.save(state_dict, path)

    def optimizer(self, opt, **params):
        optimizer = getattr(torch.optim, opt)(self.parameters(), **params)
        return [optimizer]

def Comp(**kwargs):
    set_random_seed(1)
    return Model(**kwargs)
    
