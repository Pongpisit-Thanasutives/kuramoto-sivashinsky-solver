import numpy as np
from numpy import linalg
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable, grad
import pickle
from sympy import Symbol, Integer, Float, Add, Mul, Lambda, simplify
from sympy.parsing.sympy_parser import parse_expr
from sympy.core import evaluate
from sympytorch import SymPyModule
import sympytorch

def cat(*args):
    return torch.cat(args, dim=-1)

def cpu_load(a_path):
    return torch.load(a_path, map_location="cpu")

def gpu_load(a_path):
    return torch.load(a_path, map_location="cuda")

## Saving ###
def pickle_save(obj, path):
    for i in range(2):
        with open(path, 'wb') as f:
            pickle.dump(obj, f)
    print('Saved to', str(path))
    data = pickle_load(path)
    print('Test loading passed')
    del data

### Loading ###
def pickle_load(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    print('Loaded from', str(path))
    return obj

def scientific2string(x):
    return format(x, '.1e')

def dimension_slicing(a_tensor):
    c = a_tensor.shape[-1]
    out = []
    for i in range(1, c+1): out.append(a_tensor[:, i-1:i])
    return out

def diff_order(dterm):
    return dterm.split("_")[-1][::-1]

def diff_flag(index2feature):
    dd = {0:[], 1:[]}
    for t in index2feature:
        if '_' not in t: dd[0].append(t)
        else: dd[1].append(diff_order(t))
    return dd

def diff(func, inp, device):
    return grad(func, inp, create_graph=True, retain_graph=True, grad_outputs=torch.ones(func.shape, dtype=func.dtype).to(device))[0]

def complex_diff(func, inp, device, return_complex=True):
    if return_complex: return diff(func.real, inp, device)+1j*diff(func.imag, inp, device)
    else: return cat(diff(func.real, inp), diff(func.imag, inp))

def gradients_dict(u, x, t, feature_names):
    grads_dict = {}
    df = diff_flag(feature_names)
    if feature_names[0].split('_')[0] == 'h': h = u
    
    for e in df[0]:
        grads_dict[e] = eval(e)
        
    for e in df[1]:
        out = u
        for c in e: out = diff(out, eval(c))
        grads_dict['u_'+e[::-1]] = out
        
    return grads_dict

def save(a_model, path):
    return torch.save(a_model.state_dict(), path)

def load_weights(a_model, a_path, mode="cpu"):
    if mode=="cpu": sd = cpu_load(a_path)
    elif mode=="gpu": sd = gpu_load(a_path)
    try:
        a_model.load_state_dict(sd, strict=True)
        print("Loaded the model's weights properly")
    except: 
        try: 
            a_model.load_state_dict(sd, strict=False)
            print("Loaded the model's weights with strict=False")
        except:
            print("Cannot load the model' weights properly.")
    return a_model

def is_nan(a_tensor):
    return torch.isnan(a_tensor).any().item()

def to_column_vector(arr):
    return arr.flatten()[:, None]

def to_tensor(arr, g=True):
    return torch.tensor(arr).float().requires_grad_(g)

def to_complex_tensor(arr, g=True):
    return torch.tensor(arr, dtype=torch.cfloat).requires_grad_(g)

def to_numpy(a_tensor):
    return a_tensor.detach().numpy()

def relative_l2_error(sig, ground):
    return linalg.norm((sig-ground), 2)/linalg.norm(ground, 2)

def perturb(a_array, intensity=0.01, noise_type="normal", overwrite=True):
    if intensity <= 0.0: return a_array
    if noise_type == "normal": 
        noise = intensity*np.std(a_array)*np.random.randn(a_array.shape[0], a_array.shape[1])
    elif noise_type == "uniform": 
        # This is hard...
        noise = intensity*np.std(a_array)*np.random.uniform(a_array.shape[0], a_array.shape[1])
    elif noise_type == "sparse": 
        noise = np.random.randn(a_array.shape[0], a_array.shape[1])
        mask = np.random.uniform(0, 1, (a_array.shape[0], a_array.shape[1]))
        sparsemask = np.where(mask>0.9, 1, 0)
        noise = intensity*np.std(u)*noise*sparsemask
    else: 
        print("Not recognized noise_type")
        noise = 0.0
    if overwrite: return a_array + noise
    else: return noise

# This function assumes that each dimension (variable) is independent from each other.
def perturb2d(a_array, intensity):
    for i in range(a_array.shape[1]):
        a_array[:, i:i+1] = perturb(a_array[:, i:i+1], intensity=intensity)
    return a_array

def build_exp(program, trainable_one=True):
    x = Symbol("x"); y = Symbol("y")
    
    local_dict = {
        "add": Add,
        "mul": Mul,
        "sub": Lambda((x, y), x - y),
        "div": Lambda((x, y), x/y),
    }
    
    exp = simplify(parse_expr(str(program), local_dict=local_dict))
    if trainable_one:
        exp = exp.subs(Integer(-1), Float(-1.0, precision=53))
        exp = exp.subs(Integer(+1), Float(1.0, precision=53))
    variables = exp.atoms(Symbol)
    
    return exp, variables

# My version of sympytorch.SymPyModule
class SympyTorch(nn.Module):
    def __init__(self, expressions):
        super(SympyTorch, self).__init__()
        self.mod = sympytorch.SymPyModule(expressions=expressions)                                                                      
    def forward(self, gd):
        return torch.squeeze(self.mod(**gd), dim=-1)
    
def string2sympytorch(a_string):
    expr, variables = build_exp(a_string)
    return SympyTorch(expressions=[expr]), variables

class TorchMLP(nn.Module):
    def __init__(self, dimensions, bias=True, activation_function=nn.Tanh(), bn=None, dropout=None):
        super(TorchMLP, self).__init__()
        self.model  = nn.ModuleList()

        for i in range(len(dimensions)-1):
            self.model.append(nn.Linear(dimensions[i], dimensions[i+1], bias=bias))
            if bn is not None and i!=len(dimensions)-2:
                self.model.append(bn(dimensions[i+1]))
                if dropout is not None:
                    self.model.append(dropout)
            if i==len(dimensions)-2: break
            self.model.append(activation_function)

    def xavier_init(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        for i, l in enumerate(self.model): 
            x = l(x)
        return x

# class TorchMLP(nn.Module):
#     def __init__(self, dimensions, bias=True,activation_function=nn.Tanh, bn=None, dropout=None, inp_drop=False, final_activation=None):
#         super(TorchMLP, self).__init__()
#         print("Using old implementation of TorchMLP in utils.py. See models.py for more new model-related source code.")
#         self.model  = nn.ModuleList()
#         # Can I also use the LayerNorm with elementwise_affine=True
#         # This should be a callable module.
#         self.activation_function = activation_function()
#         self.bn = bn
#         if dropout is not None and inp_drop: self.inp_dropout = dropout
#         else: self.inp_dropout = None
#         for i in range(len(dimensions)-1):
#             self.model.append(nn.Linear(dimensions[i], dimensions[i+1], bias=bias))
#             if self.bn is not None and i!=len(dimensions)-2:
#                 self.model.append(self.bn(dimensions[i+1]))
#                 if dropout is not None:
#                     self.model.append(dropout)
#             if i==len(dimensions)-2: break
#             self.model.append(activation_function())
#         if final_activation is not None:
#             self.model.append(final_activation())
#         self.model.apply(self.xavier_init)

#     def xavier_init(self, m):
#         if type(m) == nn.Linear:
#             torch.nn.init.xavier_uniform_(m.weight)
#             m.bias.data.fill_(0.01)

#     def forward(self, x):
#         if hasattr(self, 'inp_dropout'):
#             if self.inp_dropout is not None:
#                 x = self.inp_dropout(x)
#         for i, l in enumerate(self.model): 
#             x = l(x)
#         return x

# pytorch version of fft denoisg algorithm.
def fft1d_denoise(signal, thres=None, c=0, return_real=True):
    signal = signal.flatten()
    n = len(signal)
    fhat = torch.fft.fft(signal, n)
    PSD = (fhat.real**2 + fhat.imag**2) / n
    if thres is None: thres = (PSD.mean() + c*PSD.std()).item()
    indices = PSD > thres
    fhat = indices * fhat
    out = torch.fft.ifft(fhat)
    if return_real: out = out.real
    return out.reshape(-1, 1), fhat, PSD

# numpy version of fft denoising algorithm.
def fft1d_denoise_numpy(signal, thres=None, c=0, return_real=True):
    signal = signal.flatten()
    n = len(signal)
    fhat = np.fft.fft(signal, n)
    PSD = (fhat.real**2 + fhat.imag**2) / n
    if thres is None: thres = (PSD.mean() + c*PSD.std())
    indices = PSD > thres
    fhat = indices * fhat
    out = np.fft.ifft(fhat)
    if return_real: out = out.real
    return out.reshape(-1, 1), fhat, PSD

class FFTTh(nn.Module):
    def __init__(self, c=0.9, minmax=(-5.0, 5.0), func=lambda x:x):
        super(FFTTh, self).__init__()
        self.c = nn.Parameter(data=torch.FloatTensor([float(c)]))
        self.mini = minmax[0]
        self.maxi = minmax[1]
        # self.func = lambda x:(torch.exp(-F.relu(x)))
        # self.func can return a negative value
        self.func = func

    def forward(self, PSD):
        m, s = PSD.mean(), PSD.std()
        normalized_PSD = (PSD-m)/s
        th = F.relu(m+torch.clamp(self.func(self.c)*normalized_PSD.max(), min=self.mini, max=self.maxi)*s)
        indices = F.relu(PSD-th)
        d = torch.ones_like(indices)
        d[indices>0] = indices[indices>0]
        indices = indices / d
        return indices
