# coding: utf-8
import torch
from torch.autograd import grad, Variable
import torch.nn.functional as F

import os; from os.path import exists
from collections import OrderedDict
from scipy import io
from utils import *
from preprocess import *
from models import *

# Let's do facy optimizers
from madgrad import MADGRAD
from lbfgsnew import LBFGSNew

# Tracking
from tqdm import trange

import sympy
import sympytorch


# torch device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("You're running on", device)

DATA_PATH = './data/harmonic_osc.mat'
data = io.loadmat(DATA_PATH)

xlimit = 512
tlimit = 161

x = data['x'][0][:xlimit]
t = data['t'][:,0][:tlimit]

spatial_dim = x.shape[0]
time_dim = t.shape[0]

potential = np.vstack([0.5*np.power(x,2).reshape((1,spatial_dim)) for _ in range(time_dim)])
X, T = np.meshgrid(x, t)
Exact = data['usol'][:tlimit, :xlimit]

def fn(e): return e.flatten()[:, None]

Exact_u = np.real(Exact)
Exact_v = np.imag(Exact)

# Converting in a feature vector for each feature
X_star = np.hstack((fn(X), fn(T)))
h_star = fn(Exact)
potential = fn(potential)

# Doman bounds
lb = X_star.min(axis=0)
ub = X_star.max(axis=0)

force_save = False
idx_path = "./qho_weights/pub/idx.npy"
X_train_noise_path = "./qho_weights/pub/X_train_noise.npy"
u_train_noise_path = "./qho_weights/pub/u_train_noise.npy"
v_train_noise_path = "./qho_weights/pub/v_train_noise.npy"

N = 30000 # N = X_star.shape[0] 
N = min(N, X_star.shape[0])
idx = np.random.choice(X_star.shape[0], N, replace=False)
if not force_save:
    if exists(idx_path): idx = np.load(idx_path)
    else: np.save(idx_path, idx)
else: np.save(idx_path, idx)
print("Training with", N, "samples...")

lb = to_tensor(lb, False).to(device)
ub = to_tensor(ub, False).to(device)

X_train = X_star[idx, :]
h_train = h_star[idx, :]
u_train = np.real(h_train)
v_train = np.imag(h_train)
V = potential[idx, :]

# adding noise
denoise = True
if denoise:
    modelname = "dft"
else:
    modelname = "nodft"
print(modelname)

noise_intensity = 0.01/np.sqrt(2)
noisy_xt = True; noisy_labels = True

if noisy_labels:
    u_train_noise = perturb(u_train, noise_intensity, overwrite=False)
    v_train_noise = perturb(v_train, noise_intensity, overwrite=False)

    if not force_save:
        if exists(u_train_noise_path): u_train_noise = np.load(u_train_noise_path)
        else: np.save(u_train_noise_path, u_train_noise)
    else: np.save(u_train_noise_path, u_train_noise)

    if not force_save:
        if exists(v_train_noise_path): v_train_noise = np.load(v_train_noise_path)
        else: np.save(v_train_noise_path, v_train_noise)
    else: np.save(v_train_noise_path, v_train_noise)

    u_train = u_train + u_train_noise
    v_train = v_train + v_train_noise
    h_train = u_train+1j*v_train

    del u_train_noise, v_train_noise
    print("Noisy labels")
else: print("Clean labels")
if noisy_xt:
    X_train_noise = perturb2d(X_train, noise_intensity, overwrite=False)
    if not force_save:
        if exists(X_train_noise_path): X_train_noise = np.load(X_train_noise_path)
        else: np.save(X_train_noise_path, X_train_noise)
    else: np.save(X_train_noise_path, X_train_noise)
    X_train = X_train + X_train_noise
    del X_train_noise
    print("Noisy (x, t)")
else: print("Clean X_train")

# Converting to tensor
X_star = to_tensor(X_star, True)
h_star = to_complex_tensor(h_star, False)
X_train = to_tensor(X_train, True)
u_train = to_tensor(u_train, False)
v_train = to_tensor(v_train, False)
h_train = torch.tensor(h_train, dtype=torch.cfloat, requires_grad=False)
V = to_tensor(V, False).to(device)

feature_names = ['hf', 'h_xx', 'V']

noise_x, x_fft, x_PSD = fft1d_denoise(to_tensor(X_train[:, 0:1]), c=-0.05, return_real=True)
noise_x = X_train[:, 0:1] - noise_x
noise_t, t_fft, t_PSD = fft1d_denoise(to_tensor(X_train[:, 1:2]), c=-0.05, return_real=True)
noise_t = X_train[:, 1:2] - noise_t
X_train_S = cat(noise_x, noise_t)

h_train_S, h_train_fft, h_train_PSD = fft1d_denoise(h_train, c=-0.05, return_real=False)
h_train_S = h_train - h_train_S

del noise_x, noise_t

# 1st stage results
# clean all
# PDE derived using STRidge
# u_t = (-0.000337 +0.497526i)h_xx
#     + (-0.001670 -0.997429i)hf V
# 161x512
# u_t = (-0.000722 +0.499001)h_xx
#     + (-0.002967 -1.000228i)hf V
# noisy1
# PDE derived using STRidge
# u_t = (0.000702 +0.495803i)h_xx
#     + (0.000641 -0.994030i)hf V
# noisy2
# PDE derived using STRidge
# u_t = (-0.001146 +0.487772i)h_xx
#     + (-0.001516 -0.989395i)hf V

mode = int(noisy_xt)+int(noisy_labels)

if mode == 0:
    cn1 = (-0.002272-0.999772*1j)
    cn2 = (-0.000547+0.499581*1j)
    # new
    cn1 = (-0.001970-1.000545*1j)
    cn2 = (-0.000233+0.499484*1j)
    name = "cleanall"
elif mode == 1:
    cn1 = (-0.002839-0.998631*1j)
    cn2 = (-0.000211+0.499448*1j)
    name = "noisy1"
elif mode == 2:
    cn1 = (-0.002149-0.996097*1j)
    cn2 = (-0.000531+0.497700*1j)
    name = "noisy2"
    
cns = [cn1, cn2]

# Type the equation got from the symbolic regression step
# No need to save the eq save a pickle file before
program1 = "X0*X2"
pde_expr1, variables1 = build_exp(program1); print(pde_expr1, variables1)

program2 = "X1"
pde_expr2, variables2 = build_exp(program2); print(pde_expr2, variables2)

mod = ComplexSymPyModule(expressions=[pde_expr1, pde_expr2], complex_coeffs=cns); mod.train()

class PDEExpression(nn.Module):
    def __init__(self, terms, values, symbolic_module=True):
        super(PDEExpression, self).__init__()
        self.terms = terms
        self.values = [complex(e) for e in values]
        self.diff_dict = dict(zip(self.terms, self.values))
        self.string_expression = '+'.join([str(v)+'*'+str(k) for k, v in self.diff_dict.items()])
        pde_expr, self.variables = build_exp(self.string_expression)
        print("Constructing", pde_expr, self.variables)
        self.pde_expr = None
        if symbolic_module:
            self.pde_expr = sympytorch.SymPyModule(expressions=[pde_expr])
            
    # Computing the approx u_t
    def forward(self, e): return self.pde_expr(e)
    # Get a coeff
    def get_coeff(self, t): return self.diff_dict[t]

mod = PDEExpression(["hf*V", "h_xx"], cns, False)

class ComplexPINN(nn.Module):
    def __init__(self, model, loss_fn, index2features, scale=False, lb=None, ub=None, init_cs=(0.01, 0.01), init_betas=(0.0, 0.0)):
        super(ComplexPINN, self).__init__()
        self.model = model
        
        # Setting the parameters up
        self.initial_param0 = loss_fn.get_coeff("hf*V")
        self.initial_param1 = loss_fn.get_coeff("h_xx")
        self.param0_real = nn.Parameter(torch.FloatTensor([self.initial_param0.real]))
        self.param0_imag = nn.Parameter(torch.FloatTensor([self.initial_param0.imag]))
        self.param1_real = nn.Parameter(torch.FloatTensor([self.initial_param1.real]))
        self.param1_imag = nn.Parameter(torch.FloatTensor([self.initial_param1.imag]))
        
        global N
        # self.in_fft_nn = FFTTh(c=init_cs[0], func=lambda x: (torch.exp(-F.relu(x))))
        # self.out_fft_nn = FFTTh(c=init_cs[1], func=lambda x: (torch.exp(-F.relu(x))))
        self.in_fft_nn = FFTTh(c=init_cs[0])
        self.out_fft_nn = FFTTh(c=init_cs[1])
        # Beta-Robust PCA
        self.inp_rpca = RobustPCANN(beta=init_betas[0], is_beta_trainable=True, inp_dims=2, hidden_dims=32)
        self.out_rpca = RobustPCANN(beta=init_betas[1], is_beta_trainable=True, inp_dims=2, hidden_dims=32)
        
        self.index2features = index2features; self.feature2index = {}
        for idx, fn in enumerate(self.index2features): self.feature2index[fn] = str(idx)
        
        self.scale = scale; self.lb, self.ub = lb, ub
        if self.scale and (self.lb is None or self.ub is None): 
            print("Please provide thw lower and upper bounds of your PDE.")
            print("Otherwise, there will be error(s)")
        
        self.diff_flag = diff_flag(self.index2features)
        
    def forward(self, x, t):
        H = torch.cat([x, t], dim=1)
        if self.scale: H = self.neural_net_scale(H)
        return self.model(H)
    
    def loss(self, X_input, X_input_S, y_input, y_input_S, update_pde_params=True, pure_imag=False, denoise=False):
        total_loss = []
        
        if denoise:
            # Denoising FFT on (x, t)
            X_input_S = cat(torch.fft.ifft(self.in_fft_nn(X_input_S[1])*X_input_S[0]).real.reshape(-1, 1), 
                     torch.fft.ifft(self.in_fft_nn(X_input_S[3])*X_input_S[2]).real.reshape(-1, 1))
            X_input_S = X_input - X_input_S
            X_input = self.inp_rpca(X_input, X_input_S, normalize=False, center=False, is_clamp=False, axis=0, apply_tanh=True)

            # Denoising FFT on y_input
            y_input_S = y_input-torch.fft.ifft(self.out_fft_nn(y_input_S[1])*y_input_S[0]).reshape(-1, 1)
            y_input = self.out_rpca(cat(y_input.real, y_input.imag), 
                                    cat(y_input_S.real, y_input_S.imag), 
                                    normalize=False, center=False, is_clamp=False, axis=0, apply_tanh=True)
            y_input = torch.complex(y_input[:, 0:1], y_input[:, 1:2])
        
        # Compute losses
        grads_dict, u_t = self.grads_dict(X_input[:, 0:1], X_input[:, 1:2])
        X0 = cplx2tensor(grads_dict['X0'])
        
        # MSE Loss
        total_loss.append(complex_mse(X0, y_input))
            
        # PDE Loss
        param0 = torch.complex(self.param0_real, self.param0_imag)
        param1 = torch.complex(self.param1_real, self.param1_imag)
        if not update_pde_params: param0, param1 = param0.detach(), param1.detach()
        u_t_pred = (param0*X0*grads_dict['X2'])+(param1*grads_dict['X1'])
        total_loss.append(complex_mse(u_t_pred, u_t))
        
        if pure_imag:
            total_loss.append(torch.linalg.norm(param0.real, 1)+torch.linalg.norm(param1.real, 1))
            
        return total_loss
    
    def grads_dict(self, x, t):
        uf = self.forward(x, t)
        u_t = complex_diff(uf, t, device)
        
        ### PDE Loss calculation ###
        # Without calling grad
        derivatives = {}
        derivatives['X0'] = uf
        derivatives['X1'] = complex_diff(complex_diff(uf, x, device), x, device)
        derivatives['X2'] = 0.5*torch.pow(x, 2)
        
        return derivatives, u_t
    
    def gradients(self, func, x):
        return grad(func, x, create_graph=True, retain_graph=True, grad_outputs=torch.ones(func.shape))

    def neural_net_scale(self, inp): 
        return 2*(inp-self.lb)/(self.ub-self.lb)-1

inp_dimension = 2
act = CplxToCplx[torch.tanh]
complex_model = CplxSequential(
                            CplxLinear(100, 100, bias=True),
                            act(),
                            CplxLinear(100, 100, bias=True),
                            act(),
                            CplxLinear(100, 100, bias=True),
                            act(),
                            CplxLinear(100, 100, bias=True),
                            act(),
                            CplxLinear(100, 1, bias=True),
                            )

complex_model = torch.nn.Sequential(
                                    torch.nn.Linear(inp_dimension, 200),
                                    RealToCplx(),
                                    complex_model
                                    )


# Pretrained model
if mode == 2:
    semisup_model_state_dict = cpu_load("./qho_weights/jointtrained_noisy2_semisup_model_lambda1_0.03.pth")
elif mode == 1:
    semisup_model_state_dict = cpu_load("./qho_weights/jointtrained_noisy1_semisup_model_lambda1_0.03.pth")
elif mode == 0:
    # semisup_model_state_dict = cpu_load("./qho_weights/jointtrained_semisup_model_lambda1_0.03_work.pth")
    semisup_model_state_dict = cpu_load("./qho_weights/jointtrained_semisup_model_lambda1_0.02.pth")

parameters = OrderedDict()
# Filter only the parts that I care about renaming (to be similar to what defined in TorchMLP).
inner_part = "network.model."
for p in semisup_model_state_dict:
    if inner_part in p:
        parameters[p.replace(inner_part, "")] = semisup_model_state_dict[p]
complex_model.load_state_dict(parameters)

pinn = ComplexPINN(model=complex_model, loss_fn=mod, index2features=feature_names, 
                   scale=True, lb=lb, ub=ub, 
                   init_cs=(0.1, 0.1), init_betas=(-1e-5, 1e-5)).to(device)

pinn.param0_real.requires_grad_(True)
pinn.param0_imag.requires_grad_(True)
pinn.param1_real.requires_grad_(True)
pinn.param1_imag.requires_grad_(True)

# pure_imag = (mode == 0)
pure_imag = False
update_pde_params = True

def closure():
    global X_train, h_train, update_pde_params, pure_imag
    if torch.is_grad_enabled(): optimizer2.zero_grad(set_to_none=True)
    losses = pinn.loss(X_train, (x_fft, x_PSD, t_fft, t_PSD), 
                      h_train, (h_train_fft, h_train_PSD), 
                      update_pde_params=update_pde_params, pure_imag=pure_imag, denoise=denoise)
    l = sum(losses)
    if l.requires_grad: l.backward(retain_graph=True)
    return l

X_train, h_train = (X_train).to(device), (h_train).to(device)
x_fft, x_PSD = (x_fft).to(device), (x_PSD).to(device)
t_fft, t_PSD = (t_fft).to(device), (t_PSD).to(device)
h_train_fft, h_train_PSD = (h_train_fft).to(device), (h_train_PSD).to(device)

epochs1, epochs2 = 0, 200
if mode > 0: optimizer2 = torch.optim.LBFGS(pinn.parameters(), lr=1e-1, max_iter=500, max_eval=int(500*1.25), history_size=300, line_search_fn='strong_wolfe')
else: optimizer2 = torch.optim.LBFGS(pinn.parameters(), lr=1e-1, max_iter=300, max_eval=int(300*1.25), history_size=150, line_search_fn='strong_wolfe')

print('2nd Phase optimization using LBFGS')
for i in range(epochs2):
    optimizer2.step(closure)
    l = closure()
    if (i % 5) == 0 or i == epochs2-1:
        print("Epoch {}: ".format(i), l.item())

#print(pinn.param0_real)
#print(pinn.param0_imag)
#print(pinn.param1_real)
#print(pinn.param1_imag)

save(pinn, f"./qho_weights/pub/{name}_161x512_{modelname}_pinn_learned.pth")

X_star, h_star = X_star.to(device), h_star.to(device)
print("Test MSE:", complex_mse(pinn(X_star[:, 0:1], X_star[:, 1:2]), h_star).item())

true_norm = torch.sqrt(h_star.real**2 + h_star.imag**2).detach().cpu().numpy()
pred_norm = pinn(X_star[:, 0:1], X_star[:, 1:2])
pred_norm = torch.sqrt(pred_norm.real**2 + pred_norm.imag**2).detach().cpu().numpy()
print("Test relative l2 error:", relative_l2_error(true_norm, pred_norm))

g1, g2 = -1j, 0.5j
est1 = pinn.param0_real.item() + 1j*pinn.param0_imag.item()
est2 = pinn.param1_real.item() + 1j*pinn.param1_imag.item()
errs = np.array([np.abs(est1-g1)/np.abs(g1), np.abs(est2-g2)/np.abs(g2)])*100
print(est1)
print(est2)
print(errs.mean(), errs.std())
