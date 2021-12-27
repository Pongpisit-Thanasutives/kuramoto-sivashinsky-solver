# coding: utf-8
import torch; device = torch.device("cuda"); print(device)
from torch.autograd import grad, Variable
import torch.nn.functional as F

import os
from collections import OrderedDict
from scipy.io import loadmat
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

DATA_PATH = './data/kuramoto_sivishinky.mat'
X, T, Exact = space_time_grid(data_path=DATA_PATH, real_solution=True)
X_star, u_star = get_trainable_data(X, T, Exact)

# Doman bounds
lb = X_star.min(axis=0)
ub = X_star.max(axis=0)

N = 20000 # 20000, 30000, 60000
print(f"Fine-tuning with {N} samples")
# idx = np.random.choice(X_star.shape[0], N, replace=False)
idx = np.arange(N)
X_u_train = X_star[idx, :]
u_train = u_star[idx,:]

noise_intensity = 0.01
noisy_xt = False; noisy_labels = False; state = int(noisy_xt)+int(noisy_labels)
if noisy_xt: X_u_train = perturb(X_u_train, noise_intensity); print("Noisy (x, t)")
else: print("Clean (x, t)")
if noisy_labels: u_train = perturb(u_train, noise_intensity); print("Noisy labels")
else: print("Clean labels")

# Convert to torch.tensor
X_u_train = to_tensor(X_u_train, True)
u_train = to_tensor(u_train, False)
X_star = to_tensor(X_star, True)
u_star = to_tensor(u_star, False)

# lb and ub are used in adversarial training
scaling_factor = 1.0
lb = scaling_factor*to_tensor(lb, False).to(device)
ub = scaling_factor*to_tensor(ub, False).to(device)

# Feature names, base on the symbolic regression results (only the important features)
feature_names=('uf', 'u_x', 'u_xx', 'u_xxxx'); feature2index = {}

### 1-st results ###
# Noisy (x, t) and noisy labels
# PDE derived using STRidge to NN diff features
# u_t = (-0.912049 +0.000000i)u_xx
#     + (-0.909050 +0.000000i)u_xxxx
#     + (-0.951584 +0.000000i)uf*u_x

# Clean (x, t) but noisy labels
# PDE derived using STRidge to NN diff features
# u_t = (-0.942656 +0.000000i)u_xx
#     + (-0.900600 +0.000000i)u_xxxx
#     + (-0.919862 +0.000000i)uf*u_x

# Clean all
# PDE derived using STRidge to fd_derivatives
# u_t = (-0.995524 +0.000000i)uu_{x}
#     + (-1.006815 +0.000000i)u_{xx}
#     + (-1.005177 +0.000000i)u_{xxxx}

program = None
if state == 0:
    program = [-1.0161751508712769, -0.9876205325126648, -0.9817131161689758]
    program = f'''
    {program[1]}*u_xx{program[0]}*u_xxxx{program[2]}*uf*u_x
    '''
elif state == 1:
    program = [-0.9498964548110962, -0.9398553967475891, -1.0064408779144287]
    program = f'''
    {program[1]}*u_xx{program[0]}*u_xxxx{program[2]}*uf*u_x
    '''
elif state == 2:
    program = [-1.025016188621521, -0.8882913589477539, -0.9990026354789734]
    program = f'''
    {program[1]}*u_xx{program[0]}*u_xxxx{program[2]}*uf*u_x
    '''

pde_expr, variables = build_exp(program); print(pde_expr, variables)
mod = sympytorch.SymPyModule(expressions=[pde_expr]); mod.train()

class RobustPINN(nn.Module):
    def __init__(self, model, loss_fn, index2features, scale=False, lb=None, ub=None, pretrained=False, noiseless_mode=True, init_cs=(0.5, 0.5), init_betas=(0.0, 0.0)):
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
        self.init_parameters = [nn.Parameter(torch.tensor(x.item())) for x in loss_fn.parameters()]

        # Be careful of the indexing you're using here. Need more systematic way of dealing with the parameters.
        self.param0 = float(self.init_parameters[0])
        self.param1 = float(self.init_parameters[1])
        self.param2 = float(self.init_parameters[2])
        print("Please check the following parameters.")
        print("Initial parameters", (self.param0, self.param1, self.param2))
        del self.callable_loss_fn, self.init_parameters
        
        self.index2features = index2features; self.feature2index = {}
        for idx, fn in enumerate(self.index2features): self.feature2index[fn] = str(idx)
        self.scale = scale; self.lb, self.ub = lb, ub
        self.diff_flag = diff_flag(self.index2features)
        
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
            X_input = self.inp_rpca(X_input, X_input_noise, normalize=True)
            
            # (2) Denoising FFT on y_input
            y_input_noise = y_input-torch.fft.ifft(self.out_fft_nn(y_input_noise[1])*y_input_noise[0]).real.reshape(-1, 1)
            y_input = self.out_rpca(y_input, y_input_noise, normalize=True)
        
        grads_dict, u_t = self.grads_dict(X_input[:, 0:1], X_input[:, 1:2])
        
        total_loss = []
        # MSE Loss
        if update_network_params:
            mse_loss = F.mse_loss(grads_dict["uf"], y_input)
            total_loss.append(mse_loss)
            
        # PDE Loss
        if update_pde_params:
            u_t_pred = (self.param2*grads_dict["uf"]*grads_dict["u_x"])+(self.param1*grads_dict["u_xx"])+(self.param0*grads_dict["u_xxxx"])
            l_eq = F.mse_loss(u_t_pred, u_t)
            total_loss.append(l_eq)
            
        return total_loss
    
    def grads_dict(self, x, t):
        uf = self.forward(x, t)
        u_t = self.gradients(uf, t)[0]
        u_x = self.gradients(uf, x)[0]
        u_xx = self.gradients(u_x, x)[0]
        u_xxx = self.gradients(u_xx, x)[0]
        u_xxxx = self.gradients(u_xxx, x)[0]
        return {"uf":uf, "u_x":u_x, "u_xx":u_xx, "u_xxxx":u_xxxx}, u_t
    
    def get_selector_data(self, x, t):
        uf = self.forward(x, t)
        u_t = self.gradients(uf, t)[0]
        
        ### PDE Loss calculation ###
        # 'uf', 'u_x', 'u_xx', 'u_xxxx', 'u_xxx'
        u_x = self.gradients(uf, x)[0]
        u_xx = self.gradients(u_x, x)[0]
        u_xxx = self.gradients(u_xx, x)[0]
        u_xxxx = self.gradients(u_xxx, x)[0]
        u_xxxxx = self.gradients(u_xxxx, x)[0]
        derivatives = []
        derivatives.append(uf)
        derivatives.append(u_x)
        derivatives.append(u_xx)
        derivatives.append(u_xxx)
        derivatives.append(u_xxxx)
        derivatives.append(u_xxxxx)
        
        return torch.cat(derivatives, dim=1), u_t
    
    def gradients(self, func, x):
        return grad(func, x, create_graph=True, retain_graph=True, grad_outputs=torch.ones(func.shape).to(device))
    
    def neural_net_scale(self, inp): 
        return -1.0+2.0*(inp-self.lb)/(self.ub-self.lb)

noiseless_mode = False
model = TorchMLP(dimensions=[2, 50, 50, 50 ,50, 50, 1], bn=nn.LayerNorm, dropout=None)

# Pretrained model
load_fn = gpu_load
if not next(model.parameters()).is_cuda:
    load_fn = cpu_load

semisup_model_state_dict = load_fn("./weights/rudy_KS_chaotic_semisup_model_with_LayerNormDropout_without_physical_reg_trained60000labeledsamples_trained0unlabeledsamples.pth")
parameters = OrderedDict()
# Filter only the parts that I care about renaming (to be similar to what defined in TorchMLP).
inner_part = "network.model."
for p in semisup_model_state_dict:
    if inner_part in p:
        parameters[p.replace(inner_part, "")] = semisup_model_state_dict[p]
model.load_state_dict(parameters)

pinn = RobustPINN(model=model, loss_fn=mod, 
                  index2features=feature_names, scale=True, lb=lb, ub=ub, 
                  pretrained=True, noiseless_mode=noiseless_mode).to(device)
# pinn = load_weights(pinn, "./weights/...")

_, x_fft, x_PSD = fft1d_denoise(X_u_train[:, 0:1], c=-5, return_real=True)
_, t_fft, t_PSD = fft1d_denoise(X_u_train[:, 1:2], c=-5, return_real=True)
_, u_train_fft, u_train_PSD = fft1d_denoise(u_train, c=-5, return_real=True)
u_train_fft = u_train_fft.to(device)
u_train_PSD = u_train_PSD.to(device)

x_fft, x_PSD = x_fft.detach().to(device), x_PSD.detach().to(device)
t_fft, t_PSD = t_fft.detach().to(device), t_PSD.detach().to(device)

X_u_train = X_u_train.to(device)
u_train = u_train.to(device)

WWW = 1e-4

def closure():
    if torch.is_grad_enabled():
        optimizer2.zero_grad()
    losses = pinn.loss(X_u_train, (x_fft, x_PSD, t_fft, t_PSD), u_train, (u_train_fft, u_train_PSD), update_network_params=True, update_pde_params=True)
    l = losses[0]+WWW*losses[1]
    if l.requires_grad:
        l.backward(retain_graph=True)
    return l

def mtl_closure():
    if torch.is_grad_enabled():
        optimizer1.zero_grad()
    losses = pinn.loss(X_u_train, (x_fft, x_PSD, t_fft, t_PSD), u_train, (u_train_fft, u_train_PSD), update_network_params=True, update_pde_params=True)
    l = losses[0]+WWW*losses[1]
    if l.requires_grad:
        l.backward(retain_graph=True)
    return l

epochs1, epochs2 = 50, 50
# TODO: Save best state dict and training for more epochs.
optimizer1 = MADGRAD(pinn.parameters(), lr=1e-5, momentum=0.9)
pinn.train(); best_loss = 1e6; saved_weights = "./weights/dft_fixedcoeffs_cleanall.pth"

print('1st Phase optimization using Adam with PCGrad gradient modification')
for i in range(epochs1):
    optimizer1.step(mtl_closure)
    if (i % 1) == 0 or i == epochs1-1:
        l = mtl_closure()
        print("Epoch {}: ".format(i), l.item())
        print(pinn.param0, pinn.param1, pinn.param2)
        track = F.mse_loss(pinn(X_star[:, 0:1], X_star[:, 1:2]).detach().cpu(), u_star).item()
        print(track)
        if track < best_loss:
            best_loss = track
            save(pinn, saved_weights)
        
optimizer2 = torch.optim.LBFGS(pinn.parameters(), lr=1e-1, max_iter=500, max_eval=int(500*1.25), history_size=300, line_search_fn='strong_wolfe')
print('2nd Phase optimization using LBFGS')
for i in range(epochs2):
    optimizer2.step(closure)
    if (i % 5) == 0 or i == epochs2-1:
        l = closure()
        print("Epoch {}: ".format(i), l.item())
        track = F.mse_loss(pinn(X_star[:, 0:1], X_star[:, 1:2]).detach().cpu(), u_star).item()
        print(track)
        if track < best_loss:
            best_loss = track
            save(pinn, saved_weights)

pred_params = [pinn.param0, pinn.param1, pinn.param2]
print(pred_params)
