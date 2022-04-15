# coding: utf-8
import torch; device = torch.device("cuda"); print(device)
from torch.autograd import grad, Variable
import torch.nn.functional as F

import os
from collections import OrderedDict
from scipy.io import loadmat
from scipy.signal import wiener
from utils import *
from preprocess import *
from models import RobustPCANN

# Let's do facy optimizers
from madgrad import MADGRAD
from lbfgsnew import LBFGSNew

# Tracking
from tqdm import trange

import sympy
import sympytorch

data = pickle_load('./data/KdV_simple2.pkl')
t = data['t'].flatten()[:,None]
x = data['x'].flatten()[:,None]
Exact = np.real(data['u'])
X_sol, T_sol = np.meshgrid(x, t)
x_star = X_sol.flatten()[:,None]
t_star = T_sol.flatten()[:,None]
X_star = np.hstack((x_star, t_star))
if Exact.shape[1]==X_sol.shape[0] and Exact.shape[0]==X_sol.shape[1]:
    Exact = Exact.T
u_star = Exact.flatten()[:,None]
# Bound
ub = X_star.max(axis=0)
lb = X_star.min(axis=0)

# For identification
load_idx = True
N = 20000
idx = np.random.choice(X_star.shape[0], N, replace=False)
if load_idx: 
    idx = np.load("./kdv_weights/idx.npy")
    print("Loaded indices...")
else: 
    np.save("./kdv_weights/idx.npy", idx)
    print("Saved indices...")
idx = np.sort(idx)
X_train = X_star[idx,:]
u_train = u_star[idx,:]
print("Training with", N, "samples")

noise_intensity = 0.01
noisy_xt, noisy_labels = True, True
state = int(noisy_xt)+int(noisy_labels)
if noisy_xt: 
    X_noise = perturb2d(X_train, noise_intensity/np.sqrt(2), overwrite=False)
    X_noise = np.load("./kdv_weights/X_noise.npy")
    X_train = X_train + X_noise
    print("Noisy (x, t)")
else: print("Clean (x, t)")
if noisy_labels: 
    u_noise = perturb(u_train, noise_intensity, overwrite=False)
    u_noise = np.load("./kdv_weights/u_noise.npy")
    u_train = u_train + u_noise
    print("Noisy labels")
else: print("Clean labels")

# Convert to torch.tensor
X_train = to_tensor(X_train, True).to(device)
u_train = to_tensor(u_train, False).to(device)
X_star = to_tensor(X_star, True).to(device)
u_star = to_tensor(u_star, False).to(device)

# lb and ub are used in adversarial training
scaling_factor = 1.0
lb = scaling_factor*to_tensor(lb, False).to(device)
ub = scaling_factor*to_tensor(ub, False).to(device)

# Feature names
feature_names=('uf', 'u_x', 'u_xx', 'u_xxx')

class RobustPINN(nn.Module):
    def __init__(self, model, loss_fn, index2features, scale=False, lb=None, ub=None, pretrained=False, noiseless_mode=True, init_cs=(0.5, 0.5), init_betas=(0.0, 0.0), learnable_pde_coeffs=True):
        super(RobustPINN, self).__init__()
        self.model = model
        if not pretrained: self.model.apply(self.xavier_init)
        
        self.noiseless_mode = noiseless_mode
        if self.noiseless_mode: print("No denoising")
        else: print("With denoising method")
        
        self.in_fft_nn = None; self.out_fft_nn = None
        self.inp_rpca = None; self.out_rpca = None
        if not self.noiseless_mode:
            # FFTNN
            self.in_fft_nn = FFTTh(c=init_cs[0])
            self.out_fft_nn = FFTTh(c=init_cs[1])

            # Robust Beta-PCA
            self.inp_rpca = RobustPCANN(beta=init_betas[0], is_beta_trainable=True, inp_dims=2, hidden_dims=32)
            self.out_rpca = RobustPCANN(beta=init_betas[1], is_beta_trainable=True, inp_dims=1, hidden_dims=32)
        
        self.callable_loss_fn = loss_fn
        self.learn = learnable_pde_coeffs
        self.coeff_buffer = None
        if self.callable_loss_fn is not None:
            self.param0 = nn.Parameter(torch.FloatTensor([loss_fn.get_coeff("uf*u_x")])).requires_grad_(self.learn)
            self.param1 = nn.Parameter(torch.FloatTensor([loss_fn.get_coeff("u_xxx")])).requires_grad_(self.learn)
        else:
            self.param0, self.param1 = None, None
        del self.callable_loss_fn
        
        self.index2features = index2features; self.feature2index = {}
        for idx, fn in enumerate(self.index2features): self.feature2index[fn] = str(idx)
        self.scale = scale; self.lb, self.ub = lb, ub
        self.diff_flag = diff_flag(self.index2features)
        
    def set_learnable_coeffs(self, condition):
        self.learn = condition
        if self.learn: print("Grad updates to PDE coeffs.")
        else: print("NO Grad updates to PDE coeffs, use the unbiased estimation.")
        self.param0.requires_grad_(self.learn)
        self.param1.requires_grad_(self.learn)
        
    def xavier_init(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        
    def forward(self, x, t):
        H = torch.cat([x, t], dim=1)
        if self.scale: H = self.neural_net_scale(H)
        return self.model(H)
    
    def loss(self, X_input, X_input_noise, y_input, y_input_noise, update_network_params=True, update_pde_params=True):
        # Denoising process
        if not self.noiseless_mode:
            # (1) Denoising FFT on (x, t)
            # This line returns the approx. recon.
            X_input_noise = cat(torch.fft.ifft(self.in_fft_nn(X_input_noise[1])*X_input_noise[0]).real.reshape(-1, 1), 
                                torch.fft.ifft(self.in_fft_nn(X_input_noise[3])*X_input_noise[2]).real.reshape(-1, 1))
            X_input_noise = X_input-X_input_noise
            X_input = self.inp_rpca(X_input, X_input_noise, normalize=False, center=False, is_clamp=False, axis=0, apply_tanh=True)
            
            # (2) Denoising FFT on y_input
            y_input_noise = y_input-torch.fft.ifft(self.out_fft_nn(y_input_noise[1])*y_input_noise[0]).real.reshape(-1, 1)
            y_input = self.out_rpca(y_input, y_input_noise, normalize=False, center=False, is_clamp=False, axis=None, apply_tanh=True)
        
        grads_dict, u_t = self.grads_dict(X_input[:, 0:1], X_input[:, 1:2])
        
        total_loss = []
        # MSE Loss
        if update_network_params:
            mse_loss = F.mse_loss(grads_dict["uf"], y_input)
            total_loss.append(mse_loss)
            
        # PDE Loss
        if update_pde_params:
            if not self.learn:
                H = cat(grads_dict["uf"]*grads_dict["u_x"], grads_dict["u_xxx"])
                self.coeff_buffer = torch.linalg.lstsq(H, u_t).solution.detach()
                l_eq = F.mse_loss(H@(self.coeff_buffer), u_t)
            else:
                u_t_pred = (self.param0*grads_dict["uf"]*grads_dict["u_x"])+(self.param1*grads_dict["u_xxx"])
                l_eq = F.mse_loss(u_t_pred, u_t)
            total_loss.append(l_eq)
            
        return total_loss
    
    def grads_dict(self, x, t):
        uf = self.forward(x, t)
        u_t = self.gradients(uf, t)[0]
        u_x = self.gradients(uf, x)[0]
        u_xx = self.gradients(u_x, x)[0]
        u_xxx = self.gradients(u_xx, x)[0]
        return {"uf":uf, "u_x":u_x, "u_xxx":u_xxx}, u_t
    
    def get_selector_data(self, x, t):
        uf = self.forward(x, t)
        u_t = self.gradients(uf, t)[0]
        
        ### PDE Loss calculation ###
        u_x = self.gradients(uf, x)[0]
        u_xx = self.gradients(u_x, x)[0]
        u_xxx = self.gradients(u_xx, x)[0]
        derivatives = []
        derivatives.append(uf)
        derivatives.append(u_x)
        derivatives.append(u_xx)
        derivatives.append(u_xxx)
        
        return torch.cat(derivatives, dim=1), u_t
    
    def gradients(self, func, x):
        return grad(func, x, create_graph=True, retain_graph=True, grad_outputs=torch.ones(func.shape).to(device))
    
    def neural_net_scale(self, inp): 
        return -1.0+2.0*(inp-self.lb)/(self.ub-self.lb)

model = TorchMLP(dimensions=[2, 50, 50, 50 ,50, 50, 1], bn=nn.LayerNorm, dropout=None)
# Pretrained model
key = ''
name = "cleanall"
if state == 1: 
	key = 'noisy1_'
	name = "noisy1"
elif state == 2: 
	key = 'noisy2_'
	name = "noisy2"
num_train_samples = 1000
# pretrained_weiights = f"./kdv_weights/{key}simple2_semisup_model_state_dict_{num_train_samples}labeledsamples{num_train_samples}unlabeledsamples_tanhV2.pth"
pretrained_weiights = f"./kdv_lambdas_study_weights/2e-06_{key}2000samples.pth"
semisup_model_state_dict = cpu_load(pretrained_weiights)
parameters = OrderedDict()
# Filter only the parts that I care about renaming (to be similar to what defined in TorchMLP).
inner_part = "network.model."
for p in semisup_model_state_dict:
    if inner_part in p:
        parameters[p.replace(inner_part, "")] = semisup_model_state_dict[p]
model.load_state_dict(parameters)

cs = 0.1; betas = 1e-3
noiseless_mode = True
if noiseless_mode: model_name = "nodft"
else: model_name = "dft"
print(model_name)

pinn = RobustPINN(model=model, loss_fn=None, 
                  index2features=feature_names, scale=True, lb=lb, ub=ub, 
                  pretrained=True, noiseless_mode=noiseless_mode, 
                  init_cs=(cs, cs), init_betas=(betas, betas), learnable_pde_coeffs=False).to(device)

NUMBER = 128*1000
NUMBER = min(NUMBER, X_star.shape[0])
xx, tt = X_star[:NUMBER, 0:1], X_star[:NUMBER, 1:2]

uf = pinn(xx, tt)
u_t = pinn.gradients(uf, tt)[0]
u_x = pinn.gradients(uf, xx)[0]
u_xx = pinn.gradients(u_x, xx)[0]
u_xxx = pinn.gradients(u_xx, xx)[0]

derivatives = []
derivatives.append(uf*u_x)
derivatives.append(u_xxx)
derivatives = torch.cat(derivatives, dim=1)

terms = ["uf*u_x", "u_xxx"]
values = torch.linalg.lstsq(derivatives, u_t).solution.detach().cpu().numpy().flatten()

class PDEExpression(nn.Module):
    def __init__(self, terms, values):
        super(PDEExpression, self).__init__()
        self.terms = terms
        self.values = [float(e) for e in values]
        self.diff_dict = dict(zip(self.terms, self.values))
        self.string_expression = '+'.join([str(v)+'*'+str(k) for k, v in self.diff_dict.items()])
        pde_expr, self.variables = build_exp(self.string_expression)
        print("Constructing", pde_expr, self.variables)
        self.pde_expr = sympytorch.SymPyModule(expressions=[pde_expr])
            
    # Computing the approx u_t
    def forward(self, e): return self.pde_expr(e)
    # Get a coeff
    def get_coeff(self, t): return self.diff_dict[t]

mod = PDEExpression(terms, values)
del pinn, X_star, u_star

pinn = RobustPINN(model=model, loss_fn=mod, 
                  index2features=feature_names, scale=True, lb=lb, ub=ub, 
                  pretrained=True, noiseless_mode=noiseless_mode, 
                  init_cs=(0.1, 0.1), init_betas=(1e-3, 1e-3), learnable_pde_coeffs=True).to(device)

_, x_fft, x_PSD = fft1d_denoise(X_train[:, 0:1], c=-5, return_real=True)
_, t_fft, t_PSD = fft1d_denoise(X_train[:, 1:2], c=-5, return_real=True)
_, u_train_fft, u_train_PSD = fft1d_denoise(u_train, c=-5, return_real=True)

u_train_fft, u_train_PSD = u_train_fft.to(device), u_train_PSD.to(device)
x_fft, x_PSD = x_fft.detach().to(device), x_PSD.detach().to(device)
t_fft, t_PSD = t_fft.detach().to(device), t_PSD.detach().to(device)

AAA = 1
BBB = 1

def closure1():
    if torch.is_grad_enabled():
        optimizer1.zero_grad()
    losses = pinn.loss(X_train, (x_fft, x_PSD, t_fft, t_PSD), u_train, (u_train_fft, u_train_PSD), update_network_params=True, update_pde_params=True)
    l = AAA*losses[0] + BBB*losses[1]
    if l.requires_grad:
        l.backward(retain_graph=True)
    return l

def closure2():
    if torch.is_grad_enabled():
        optimizer2.zero_grad()
    losses = pinn.loss(X_train, (x_fft, x_PSD, t_fft, t_PSD), u_train, (u_train_fft, u_train_PSD), update_network_params=True, update_pde_params=True)
    l = AAA*losses[0] + BBB*losses[1]
    if l.requires_grad:
        l.backward(retain_graph=True)
    return l

# save_weights_at = f"./kdv_weights/kdv_pretrained{num_train_samples}samples_{model_name}_learnedcoeffs_{name}.pth"
save_weights_at = f"./kdv_weights/pub_dPINNs/{model_name}_{name}.pth"

epochs1, epochs2 = 30, 30
max_iters, max_evals = 20000, 20000 # 2: (20000, 20000, 1000)
optimizer1 = torch.optim.LBFGS(pinn.parameters(), lr=1e-1, max_iter=max_iters, max_eval=max_evals, history_size=1000, line_search_fn='strong_wolfe')
pinn.train(); pinn.set_learnable_coeffs(True)
print('1st Phase')
for i in range(epochs1):
    optimizer1.step(closure1)
    if (i % 10) == 0 or i == epochs1-1:
        l = closure1()
        print("Epoch {}: ".format(i), l.item())
        pred_params = np.array([pinn.param0.item(), pinn.param1.item()]); print(pred_params)
        errs = 100*np.abs(np.array([(pred_params[0]+6)/6.0, pred_params[1]+1])); print(errs.mean(), errs.std())

if epochs2 > 0:
    pinn.set_learnable_coeffs(False)
    optimizer2 = torch.optim.LBFGS(pinn.parameters(), lr=1e-1, max_iter=max_iters, max_eval=max_evals, history_size=1000, line_search_fn='strong_wolfe')
    print('2nd Phase')
    for i in range(epochs2):
        optimizer2.step(closure2)
        if (i % 10) == 0 or i == epochs2-1:
            l = closure2()
            print("Epoch {}: ".format(i), l.item())
            pred_params = pinn.coeff_buffer.cpu().flatten().numpy()
            print(pred_params)
            errs = 100*np.abs(np.array([(pred_params[0]+6)/6.0, pred_params[1]+1])); print(errs.mean(), errs.std())

save(pinn, save_weights_at)
