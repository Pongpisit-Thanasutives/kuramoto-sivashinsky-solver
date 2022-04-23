# coding: utf-8
import random; random.seed(0)
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

DATA_PATH = './data/kuramoto_sivishinky.mat'
X, T, Exact = space_time_grid(data_path=DATA_PATH, real_solution=True)
X_star, u_star = get_trainable_data(X, T, Exact)

# Doman bounds
lb = X_star.min(axis=0)
ub = X_star.max(axis=0)

force_save = False

N = 1024*21 # 20000, 30000, 60000
N = min(N, X_star.shape[0])
print(f"Fine-tuning with {N} samples")
idx = np.random.choice(X_star.shape[0], N, replace=False)
idx = np.arange(N)
if force_save: np.save("./weights/final/idx.npy", idx)
else: idx = np.load("./weights/final/idx.npy")

X_u_train = X_star[idx, :]
u_train = u_star[idx,:]

noise_intensity = 0.01; double = 2
double = int(double)
print(f"double = {double}")
noisy_xt = True; noisy_labels = True; state = int(noisy_xt)+int(noisy_labels)
if noisy_xt: 
    print("Noisy (x, t)")
    X_noise = perturb2d(X_u_train, noise_intensity/np.sqrt(2), overwrite=False)
    if force_save: np.save("./weights/final/X_noise.npy", X_noise)
    else: X_noise = np.load("./weights/final/X_noise.npy")
    X_noise = X_noise * double
    X_u_train = X_u_train + X_noise
else: print("Clean (x, t)")
if noisy_labels: 
    print("Noisy labels")
    u_noise = perturb(u_train, noise_intensity, overwrite=False)
    if force_save: np.save("./weights/final/u_noise.npy", u_noise)
    else: u_noise = np.load("./weights/final/u_noise.npy")
    u_noise = u_noise * double
    u_train = u_train + u_noise
else: print("Clean labels")

u_noise_wiener = to_tensor(u_train-wiener(u_train, noise=1e-5), False).to(device)
X_noise_wiener = to_tensor(X_u_train-wiener(X_u_train, noise=1e-2), False).to(device)

noiseless_mode = False
if noiseless_mode: model_name = "nodft"
else: model_name = "dft"
print(model_name)

# Convert to torch.tensor
X_u_train = to_tensor(X_u_train, True)
u_train = to_tensor(u_train, False)
X_star = to_tensor(X_star, True).to(device)
u_star = to_tensor(u_star, False)

# lb and ub are used in adversarial training
scaling_factor = 1.0
lb = scaling_factor*to_tensor(lb, False).to(device)
ub = scaling_factor*to_tensor(ub, False).to(device)

# Feature names, base on the symbolic regression results (only the important features)
feature_names=('uf', 'u_x', 'u_xx', 'u_xxxx'); feature2index = {}

program = None; name = None
if state == 0:
    program = [-1.031544, -0.976023, -0.973498]
    name = "cleanall"
elif state == 2:
    program = [-0.898254, -0.808380, -0.803464]
    if double > 1: program = [-random.uniform(0, 1)*(1e-6) for _ in range(3)]
    name = "noisy2"
    print(X_noise.max(), X_noise.min())
elif state == 1:
    program = [-0.846190, -0.766933, -0.855584]
    if double > 1: program = [-random.uniform(0, 1)*(1e-6) for _ in range(3)]
    name = "noisy1"
program = f'''
{program[0]}*u_xx{program[1]}*u_xxxx{program[2]}*uf*u_x
'''
print("name =", name)
pde_expr, variables = build_exp(program); print(pde_expr, variables)
mod = sympytorch.SymPyModule(expressions=[pde_expr]); mod.train()

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
        self.init_parameters = [nn.Parameter(torch.tensor(x.item())) for x in loss_fn.parameters()]

        # Be careful of the indexing you're using here. Need more systematic way of dealing with the parameters.
        self.learn = learnable_pde_coeffs
        self.param0 = (self.init_parameters[0]).requires_grad_(self.learn)
        self.param1 = (self.init_parameters[1]).requires_grad_(self.learn)
        self.param2 = (self.init_parameters[2]).requires_grad_(self.learn)
        print("Please check the following parameters.")
        print("Initial parameters", (self.param0, self.param1, self.param2))
        print("u_xxxx, u_xx, uu_x")
        del self.callable_loss_fn, self.init_parameters
        self.coeff_buffer = None
        
        self.index2features = index2features; self.feature2index = {}
        for idx, fn in enumerate(self.index2features): self.feature2index[fn] = str(idx)
        self.scale = scale; self.lb, self.ub = lb, ub
        self.diff_flag = diff_flag(self.index2features)

    def set_learnable_coeffs(self, condition):
        self.learn = condition
        if self.learn: print("Grad updates to PDE coeffs.")
        else: print("NO Grad updates to PDE coeffs.")
        self.param0.requires_grad_(self.learn)
        self.param1.requires_grad_(self.learn)
        self.param2.requires_grad_(self.learn)
       
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
            # X_input_noise = X_noise_wiener
            # Work for high noise
            # X_input = self.inp_rpca(X_input, X_input_noise, normalize=True, center=True, is_clamp=True, axis=0, apply_tanh=False)
            X_input = self.inp_rpca(X_input, X_input_noise, normalize=True, center=True, is_clamp=(-1.0, 1.0), axis=0, apply_tanh=True)
            
            # (2) Denoising FFT on y_input
            y_input_noise = y_input-torch.fft.ifft(self.out_fft_nn(y_input_noise[1])*y_input_noise[0]).real.reshape(-1, 1)
            # y_input_noise = u_noise_wiener
            # Work for high noise
            # y_input = self.out_rpca(y_input, y_input_noise, normalize=True, center=True, is_clamp=True, axis=None, apply_tanh=False)
            y_input = self.out_rpca(y_input, y_input_noise, normalize=True, center=True, is_clamp=(-1.0, 1.0), axis=None, apply_tanh=True)
        
        grads_dict, u_t = self.grads_dict(X_input[:, 0:1], X_input[:, 1:2])
        
        total_loss = []
        # MSE Loss
        if update_network_params:
            mse_loss = F.mse_loss(grads_dict["uf"], y_input)
            total_loss.append(mse_loss)
            
        # PDE Loss
        if update_pde_params:
            if not self.learn:
                H = cat(grads_dict["uf"]*grads_dict["u_x"], grads_dict["u_xx"], grads_dict["u_xxxx"])
                self.coeff_buffer = torch.linalg.lstsq(H, u_t).solution.detach()
                l_eq = F.mse_loss(H@(self.coeff_buffer), u_t)
            else:
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

        derivatives = []
        derivatives.append(uf)
        derivatives.append(u_x)
        derivatives.append(u_xx)
        derivatives.append(u_xxx)
        derivatives.append(u_xxxx)
        
        return torch.cat(derivatives, dim=1), u_t
    
    def gradients(self, func, x):
        return grad(func, x, create_graph=True, retain_graph=True, grad_outputs=torch.ones(func.shape).to(device))
    
    def neural_net_scale(self, inp): 
        return -1.0+2.0*(inp-self.lb)/(self.ub-self.lb)

model = TorchMLP(dimensions=[2, 50, 50, 50 ,50, 50, 1], bn=nn.LayerNorm, dropout=None)

# Pretrained model
load_fn = gpu_load
if not next(model.parameters()).is_cuda:
    load_fn = cpu_load

# semisup_model_state_dict = load_fn("./weights/deephpm_KS_chaotic_semisup_model_with_LayerNormDropout_without_physical_reg_trained60000labeledsamples_trained0unlabeledsamples.pth")
semisup_model_state_dict = load_fn("./weights/semisup_model_noisy2_pub.pth")
parameters = OrderedDict()
# Filter only the parts that I care about renaming (to be similar to what defined in TorchMLP).
inner_part = "network.model."
for p in semisup_model_state_dict:
    if inner_part in p:
        parameters[p.replace(inner_part, "")] = semisup_model_state_dict[p]
model.load_state_dict(parameters)

pinn = RobustPINN(model=model, loss_fn=mod, 
                  index2features=feature_names, scale=True, lb=lb, ub=ub, 
                  pretrained=True, noiseless_mode=noiseless_mode)

if state == 0:
    pinn = load_weights(pinn, "./weights/final/cleanall_pinn_pretrained_weights.pth")
elif state == 1:
    pinn = load_weights(pinn, "./weights/rudy_KS_noisy1_chaotic_semisup_model_with_LayerNormDropout_without_physical_reg_trainedfirst30000labeledsamples_trained0unlabeledsamples_work.pth")
elif state == 2:
    # pinn = load_weights(pinn, "./weights/rudy_KS_noisy2_chaotic_semisup_model_with_LayerNormDropout_without_physical_reg_trainedfirst30000labeledsamples_trained0unlabeledsamples_work.pth")
    pinn = load_weights(pinn, "./weights/semisup_model_noisy2_pub.pth")

alpha = 0.1

pinn = RobustPINN(model=pinn.model, loss_fn=mod, 
                  index2features=feature_names, scale=True, lb=lb, ub=ub, 
                  pretrained=True, noiseless_mode=noiseless_mode, 
                  init_cs=(alpha, alpha), init_betas=(1e-2, 1e-2)).to(device)

_, x_fft, x_PSD = fft1d_denoise(X_u_train[:, 0:1], c=-5, return_real=True)
_, t_fft, t_PSD = fft1d_denoise(X_u_train[:, 1:2], c=-5, return_real=True)
_, u_train_fft, u_train_PSD = fft1d_denoise(u_train, c=-5, return_real=True)
u_train_fft = u_train_fft.to(device)
u_train_PSD = u_train_PSD.to(device)

x_fft, x_PSD = x_fft.detach().to(device), x_PSD.detach().to(device)
t_fft, t_PSD = t_fft.detach().to(device), t_PSD.detach().to(device)

X_u_train = X_u_train.to(device)
u_train = u_train.to(device)

WWW = 0.5

def closure1():
    if torch.is_grad_enabled():
        optimizer1.zero_grad()
    losses = pinn.loss(X_u_train, (x_fft, x_PSD, t_fft, t_PSD), u_train, (u_train_fft, u_train_PSD), update_network_params=True, update_pde_params=True)
    # print(losses)
    l = ((1-WWW)*losses[0])+(WWW*losses[1])
    if l.requires_grad:
        l.backward(retain_graph=True)
    return l

def closure2():
    if torch.is_grad_enabled():
        optimizer2.zero_grad()
    losses = pinn.loss(X_u_train, (x_fft, x_PSD, t_fft, t_PSD), u_train, (u_train_fft, u_train_PSD), update_network_params=True, update_pde_params=True)
    l = ((1-WWW)*losses[0])+(WWW*losses[1])
    if l.requires_grad:
        l.backward(retain_graph=True)
    return l

epochs1, epochs2 = 20, 20
MMM = 500
if state == 0: epochs2 = 0
optimizer1 = torch.optim.LBFGS(pinn.parameters(), lr=1e-1, max_iter=MMM, max_eval=int(MMM*1.25), history_size=MMM, line_search_fn='strong_wolfe')
pinn.train(); best_loss = 1e6
saved_last_weights = f"./weights/final/new/more_noise/deephpm_KS_chaotic_{model_name}_learnedcoeffs_last_{name}_double{double}.pth"

pinn.set_learnable_coeffs(True)
print('1st Phase')
for i in range(epochs1):
    optimizer1.step(closure1)
    if (i % 10) == 0 or i == epochs1-1:
        l = closure1()
        print("Epoch {}: ".format(i), l.item())
        pred_params = np.array([pinn.param0.item(), pinn.param1.item(), pinn.param2.item()])
        print(pred_params)

if not pinn.learn: pred_params = pinn.coeff_buffer.cpu().flatten().numpy()
else: pred_params = np.array([pinn.param0.item(), pinn.param1.item(), pinn.param2.item()])
errs = 100*np.abs(pred_params+1)
print(errs.mean(), errs.std())

if epochs2 > 0:
    if not noiseless_mode:
        pinn = RobustPINN(model=pinn.model, loss_fn=mod, 
                          index2features=feature_names, scale=True, lb=lb, ub=ub, 
                          pretrained=True, noiseless_mode=noiseless_mode, 
                          init_cs=(alpha, alpha), init_betas=(1e-2, 1e-2)).to(device)

    pinn.set_learnable_coeffs(False)
    optimizer2 = torch.optim.LBFGS(pinn.parameters(), lr=1e-1, max_iter=MMM, max_eval=int(MMM*1.25), history_size=MMM, line_search_fn='strong_wolfe')
    print('2nd Phase')
    for i in range(epochs2):
        optimizer2.step(closure2)
        if (i % 10) == 0 or i == epochs2-1:
            l = closure2()
            print("Epoch {}: ".format(i), l.item())
            pred_params = pinn.coeff_buffer.cpu().flatten().numpy()
            print(pred_params)

save(pinn, saved_last_weights)
if not pinn.learn: pred_params = pinn.coeff_buffer.cpu().flatten().numpy()
else: pred_params = np.array([pinn.param0.item(), pinn.param1.item(), pinn.param2.item()])
print(pred_params)
errs = 100*np.abs(pred_params+1)
print(errs.mean(), errs.std())
