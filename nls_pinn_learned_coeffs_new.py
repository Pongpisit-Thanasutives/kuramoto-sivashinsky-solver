# coding: utf-8
import torch
from torch.autograd import grad, Variable

import os
from collections import OrderedDict
from scipy import io
from utils import *
from preprocess import *
from models import *

# Let's do facy optimizers
from madgrad import MADGRAD
import lookahead
from lbfgsnew import LBFGSNew

# Tracking
from tqdm import trange

import sympy
import sympytorch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("You're running on", device)

# Adding noise
noise_intensity = 0.01/np.sqrt(2)
noisy_xt = False; noisy_labels = False
DENOISE = True
mode = int(noisy_xt)+int(noisy_labels)

# Doman bounds
lb = np.array([-5.0, 0.0])
ub = np.array([5.0, np.pi/2])

DATA_PATH = 'data/NLS.mat'
data = io.loadmat(DATA_PATH)

t = data['tt'].flatten()[:,None]
x = data['x'].flatten()[:,None]
Exact = data['uu']
Exact_u = np.real(Exact)
Exact_v = np.imag(Exact)

X, T = np.meshgrid(x,t)

X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
u_star = to_column_vector(Exact_u.T)
v_star = to_column_vector(Exact_v.T)

force_save = False
N = 2500
N = min(N, X_star.shape[0])
idx = np.random.choice(X_star.shape[0], N, replace=False)
if force_save: np.save("./nls_weights/new/idx.npy", idx); print("Saving indices and noises...")
else: idx = np.load("./nls_weights/new/idx.npy"); print("Loading indices and noises...")

X_train = X_star[idx, :]
u_train = u_star[idx, :]
v_train = v_star[idx, :]

if noisy_xt:
    print("Noisy (x, t)")
    X_train_noise = perturb2d(X_train, intensity=noise_intensity, overwrite=False)
    if force_save: np.save("./nls_weights/new/X_train_noise.npy", X_train_noise)
    else: X_train_noise = np.load("./nls_weights/new/X_train_noise.npy")
    X_train = X_train + X_train_noise
else: print("Clean (x, t)")

if noisy_labels:
    print("Noisy labels")
    u_noise = perturb(u_train, intensity=noise_intensity, overwrite=False)
    if force_save: np.save("./nls_weights/new/u_noise.npy", u_noise)
    else: u_noise = np.load("./nls_weights/new/u_noise.npy")
    u_train = u_train + u_noise

    v_noise = perturb(v_train, intensity=noise_intensity, overwrite=False)
    if force_save: np.save("./nls_weights/new/v_noise.npy", v_noise)
    else: v_noise = np.load("./nls_weights/new/v_noise.npy")
    v_train = v_train + v_noise

    del v_noise, u_noise
else: print("Clean labels")

X_train = to_tensor(X_train, True).to(device)
u_train = to_tensor(u_train, False).to(device)
v_train = to_tensor(v_train, False).to(device)
h_train = torch.complex(u_train, v_train).to(device)
lb = to_tensor(lb, False).to(device)
ub = to_tensor(ub, False).to(device)

X_star = to_tensor(X_star, True).to(device)

feature_names = ['hf', '|hf|', 'h_xx']

class RobustComplexPINN(nn.Module):
    def __init__(self, model, loss_fn, index2features, scale=False, lb=None, ub=None, init_cs=(0.01, 0.01), init_betas=(0.0, 0.0), learnable_pde_coeffs=True):
        super(RobustComplexPINN, self).__init__()
        # FFTNN
        global N
        self.in_fft_nn = FFTTh(c=init_cs[0])
        self.out_fft_nn = FFTTh(c=init_cs[1])
        
        self.model = model
        
        # Beta-Robust PCA
        self.inp_rpca = RobustPCANN(beta=init_betas[0], is_beta_trainable=False, inp_dims=2, hidden_dims=50)
        self.out_rpca = RobustPCANN(beta=init_betas[1], is_beta_trainable=False, inp_dims=2, hidden_dims=50)

        self.callable_loss_fn = loss_fn
        self.learn = learnable_pde_coeffs; self.coeff_buffer = None
        self.index2features = index2features; self.feature2index = {}
        for idx, fn in enumerate(self.index2features): self.feature2index[fn] = str(idx)
        self.scale = scale; self.lb, self.ub = lb, ub
        if self.scale and (self.lb is None or self.ub is None):
            print("Please provide thw lower and upper bounds of your PDE.")
            print("Otherwise, there will be error(s)")
        self.diff_flag = diff_flag(self.index2features)
        
    def set_learnable_coeffs(self, condition):
        self.learn = condition
        if self.learn: print("Grad updates to PDE coeffs.")
        else: print("NO Grad updates to PDE coeffs, use the unbiased estimation.")
        for param in self.callable_loss_fn.parameters():
            param.requires_grad_(self.learn)
        
    def forward(self, H):
        if self.scale: H = self.neural_net_scale(H)
        return self.model(H)
    
    def loss(self, HL, HS, y_input, y_input_S, update_network_params=True, update_pde_params=True, denoising=True):
        total_loss = []

        if denoising: 
            # Denoising FFT on (x, t)
            HS = cat(torch.fft.ifft(self.in_fft_nn(HS[1])*HS[0]).real.reshape(-1, 1), 
                     torch.fft.ifft(self.in_fft_nn(HS[3])*HS[2]).real.reshape(-1, 1))
            HS = HL-HS
            H = self.inp_rpca(HL, HS, normalize=False, center=False, is_clamp=False, axis=0, apply_tanh=True)
            
            # Denoising FFT on y_input
            y_input_S = y_input-torch.fft.ifft(self.out_fft_nn(y_input_S[1])*y_input_S[0]).reshape(-1, 1)
            y_input = self.out_rpca(cat(y_input.real, y_input.imag), 
                                    cat(y_input_S.real, y_input_S.imag), 
                                    normalize=False, center=False, is_clamp=False, axis=0, apply_tanh=True)
            y_input = torch.complex(y_input[:, 0:1], y_input[:, 1:2])
            
            grads_dict, u_t = self.grads_dict(H[:, 0:1], H[:, 1:2])

        else: grads_dict, u_t = self.grads_dict(HL[:, 0:1], HL[:, 1:2])

        # MSE Loss
        if update_network_params:
            total_loss.append(complex_mse(grads_dict['X'+self.feature2index['hf']], y_input))
        # PDE Loss
        if update_pde_params:
            if self.learn: 
                eq_loss = complex_mse(self.callable_loss_fn(grads_dict), u_t)
                # self.coeff_buffer = self.callable_loss_fn.complex_coeffs().cpu().detach().numpy().ravel()
            else: 
                H = cat(grads_dict['X0']*grads_dict['X1'], grads_dict['X2'])
                # self.coeff_buffer = torch.linalg.lstsq(H, u_t).solution.detach() -> error: autodiff on complex numbers
                self.coeff_buffer = torch.linalg.lstsq(H.detach(), u_t.detach()).solution
                eq_loss = complex_mse(H@self.coeff_buffer, u_t)
            total_loss.append(eq_loss)
            
        return total_loss
    
    def grads_dict(self, x, t):
        uf = self.forward(cat(x, t))
        u_t = complex_diff(uf, t, device)
        u_x = complex_diff(uf, x, device)
        derivatives = {}
        derivatives['X0'] = cplx2tensor(uf)
        derivatives['X1'] = (uf.real**2+uf.imag**2)+0.0j
        derivatives['X2'] = complex_diff(u_x, x, device)
        return derivatives, u_t
    
    def gradients(self, func, x):
        return grad(func, x, create_graph=True, retain_graph=True, grad_outputs=torch.ones(func.shape).to(device))
    
    def neural_net_scale(self, inp): 
        return 2*(inp-self.lb)/(self.ub-self.lb)-1

dft_tag = "nodft"
if DENOISE: dft_tag = "dft"
print(dft_tag)

AAA = 1
BBB = 1

def closure1():
    global X_train, X_train_S, h_train, h_train_S, x_fft, x_PSD, t_fft, t_PSD
    if torch.is_grad_enabled():
        optimizer1.zero_grad(set_to_none=True)
    losses = pinn.loss(X_train, (x_fft, x_PSD, t_fft, t_PSD), h_train, (h_train_fft, h_train_PSD), update_network_params=True, update_pde_params=True, denoising=DENOISE)
    loss = AAA*losses[0] + BBB*losses[1]
    if loss.requires_grad: loss.backward(retain_graph=True)
    return loss

def closure2():
    global X_train, X_train_S, h_train, h_train_S, x_fft, x_PSD, t_fft, t_PSD
    if torch.is_grad_enabled():
        optimizer2.zero_grad(set_to_none=True)
    losses = pinn.loss(X_train, (x_fft, x_PSD, t_fft, t_PSD), h_train, (h_train_fft, h_train_PSD), update_network_params=True, update_pde_params=True, denoising=DENOISE)
    loss = AAA*losses[0] + BBB*losses[1]
    if loss.requires_grad:
        loss.backward(retain_graph=True)
    return loss

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
if mode == 0:
	tag = "cleanall"
	semisup_model_state_dict = cpu_load("./nls_weights/pretrained_weights/cleanall_NLS_complex_model_500labeledsamples_jointtrainwith500unlabeledsamples.pth")
elif mode == 1:
	tag = "noisy1"
	semisup_model_state_dict = cpu_load("./nls_weights/pretrained_weights/noisy_NLS_complex_model_500labeledsamples_jointtrainwith500unlabeledsamples.pth")
elif mode == 2:
	tag = "noisy2"
	semisup_model_state_dict = cpu_load("./nls_weights/pretrained_weights/noisy2_NLS_complex_model_500labeledsamples_jointtrainwith500unlabeledsamples.pth")
parameters = OrderedDict()

# Filter only the parts that I care about renaming (to be similar to what defined in TorchMLP).
inner_part = "network.model."
for p in semisup_model_state_dict:
    if inner_part in p:
        parameters[p.replace(inner_part, "")] = semisup_model_state_dict[p]
complex_model.load_state_dict(parameters)
complex_model = complex_model.to(device)

t_steps = 160 # 1000, 160
t_steps = min(t_steps, t.shape[0])
# should be 1.25
print("Considering up to t = ", t[:, 0][:t_steps].max())
n_test = x.shape[0]*t_steps
idx_test = np.arange(n_test)
X_dis = X_star[:n_test]
xx, tt = dimension_slicing(X_dis)
predictions = complex_model(cat(xx, tt))
h = cplx2tensor(predictions)
h_x = complex_diff(predictions, xx, device)
h_xx = complex_diff(h_x, xx, device)
h_t = complex_diff(predictions, tt, device)
abs_h = (h.real**2+h.imag**2)+0.0j
cns = np.linalg.lstsq(cat(h*abs_h, h_xx).cpu().detach().numpy(), h_t.cpu().detach().numpy(), rcond=-1)[0].flatten().tolist()
program1 = "X0*X1"
pde_expr1, variables1,  = build_exp(program1); print(pde_expr1, variables1)
program2 = "X2"
pde_expr2, variables2,  = build_exp(program2); print(pde_expr2, variables2)
mod = ComplexSymPyModule(expressions=[pde_expr1, pde_expr2], complex_coeffs=cns); mod.train()

noise_x, x_fft, x_PSD = fft1d_denoise(X_train[:, 0:1], c=0, return_real=True)
noise_x = X_train[:, 0:1]-noise_x
noise_t, t_fft, t_PSD = fft1d_denoise(X_train[:, 1:2], c=0, return_real=True)
noise_t = X_train[:, 1:2]-noise_t
X_train_S = cat(noise_x, noise_t)

h_train_S, h_train_fft, h_train_PSD = fft1d_denoise(h_train, c=-1, return_real=False)
h_train_S = h_train-h_train_S

del noise_x, noise_t
del X_star, X_dis, xx, tt 
del predictions, h, h_x, h_xx, abs_h

pinn = RobustComplexPINN(model=complex_model, loss_fn=mod, 
                         index2features=feature_names, scale=False, lb=lb, ub=ub, 
                         init_cs=(1e-1, 1e-1), init_betas=(1e-5, 1e-5)).to(device)

save_weights_at1 = f"./nls_weights/new/nls_{dft_tag}_{tag}_opt1.pth"
save_weights_at2 = f"./nls_weights/new/nls_{dft_tag}_{tag}_opt2.pth"

grounds = np.array([1j, 0+0.5j])

epochs1, epochs2 = 60, 30
optimizer1 = torch.optim.LBFGS(pinn.parameters(), lr=1e-1, max_iter=1000, max_eval=1000, history_size=500, line_search_fn='strong_wolfe')
pinn.train(); pinn.set_learnable_coeffs(True)
print('1st Phase')
for i in range(epochs1):
    optimizer1.step(closure1)
    if (i % 10) == 0 or i == epochs1-1:
        l = closure1()
        print("Epoch {}: ".format(i), l.item())
        pred_params = pinn.callable_loss_fn.complex_coeffs().cpu().detach().numpy().ravel()
        print(pred_params)
        errs = []
        for i in range(len(grounds)):
            # Relative l2 error
            err = pred_params[i]-grounds[i]
            errs.append(100*(abs(err.real+1j*err.imag)/abs(grounds[i])))
        errs = np.array(errs)
        print(errs.mean(), errs.std())

save(pinn, save_weights_at1)

if epochs2 > 0:
    pinn.set_learnable_coeffs(False)
    optimizer2 = torch.optim.LBFGS(pinn.parameters(), lr=1e-1, max_iter=1000, max_eval=1000, history_size=500, line_search_fn='strong_wolfe')
    # optimizer2 = torch.optim.LBFGS(pinn.parameters(), lr=1e-1, line_search_fn='strong_wolfe')
    print('2nd Phase')
    for i in range(epochs2):
        optimizer2.step(closure2)
        if (i % 10) == 0 or i == epochs2-1:
            l = closure2()
            print("Epoch {}: ".format(i), l.item())
            pred_params = pinn.callable_loss_fn.complex_coeffs().cpu().detach().numpy().ravel()
            print(pred_params)
            errs = []
            for i in range(len(grounds)):
                # Relative l2 error
                err = pred_params[i]-grounds[i]
                errs.append(100*(abs(err.real+1j*err.imag)/abs(grounds[i])))
            errs = np.array(errs)
            print(errs.mean(), errs.std())

save(pinn, save_weights_at2)
