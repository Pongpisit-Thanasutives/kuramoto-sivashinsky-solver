import torch
from torch.autograd import grad, Variable
import torch.nn.functional as F

import os
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

from pde_diff import FiniteDiff
import cplxmodule
from cplxmodule import cplx
from cplxmodule.nn import RealToCplx, CplxToReal, CplxSequential, CplxToCplx
from cplxmodule.nn import CplxLinear, CplxModReLU, CplxAdaptiveModReLU, CplxModulus, CplxAngle

# torch device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("You're running on", device)

DATA_PATH = 'data/NLS.mat'
data = io.loadmat(DATA_PATH)

# Doman bounds
lb = np.array([-5.0, 0.0])
ub = np.array([5.0, np.pi/2])

t = data['tt'].flatten()[:,None]
x = data['x'].flatten()[:,None]
Exact = data['uu']
Exact_u = np.real(Exact)
Exact_v = np.imag(Exact)

X, T = np.meshgrid(x,t)

X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
u_star = to_column_vector(Exact_u.T)
v_star = to_column_vector(Exact_v.T)
h_star = u_star + v_star*1j

force_save = False
N = 2500
N = min(N, X_star.shape[0])
idx = np.random.choice(X_star.shape[0], N, replace=False)
if force_save: np.save("./nls_weights/new/idx.npy", idx); print("Saving indices and noises...")
else: idx = np.load("./nls_weights/new/idx.npy"); print(f"Loading {len(idx)} indices and noises...")

X_train = X_star[idx, :]
u_train = u_star[idx, :]
v_train = v_star[idx, :]

noise_intensity = 0.01/np.sqrt(2); noisy_xt = False; noisy_labels = True

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

mode = int(noisy_xt)+int(noisy_labels)
name = "cleanall"
if mode == 1: name = "noise1"
elif mode == 2: name = "noise2"

# Converting to tensor
X_star = to_tensor(X_star, True).to(device)
h_star = to_complex_tensor(h_star, False).to(device)

X_train = to_tensor(X_train, True).to(device)
u_train = to_tensor(u_train, False).to(device)
v_train = to_tensor(v_train, False).to(device)
h_train = torch.complex(u_train, v_train).to(device)

lb = to_tensor(lb, False).to(device)
ub = to_tensor(ub, False).to(device)

# Unsup data
include_N_res = 1
if include_N_res>0:
    N_res = int(N*include_N_res)
    idx_res = np.array(range(X_star.shape[0]))[~idx]
    idx_res = idx_res[:N_res]
    X_res = to_tensor(X_star[idx_res, :], True)
    print(f"Training with {N_res} unsup samples")
    X_train = torch.vstack([X_train, X_res])

feature_names = ['hf', '|hf|', 'h_x', 'h_xx', 'h_xxx']

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

class ComplexNetwork(nn.Module):
    def __init__(self, model, index2features=None, scale=False, lb=None, ub=None):
        super(ComplexNetwork, self).__init__()
        # pls init the self.model before
        self.model = model
        # For tracking, the default tup is for the burgers' equation.
        self.index2features = index2features
        print("Considering", self.index2features)
        self.diff_flag = diff_flag(self.index2features)
        self.uf = None
        self.scale = scale
        self.lb, self.ub = lb, ub
        
    def xavier_init(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x, t):
        if not self.scale: self.uf = self.model(torch.cat([x, t], dim=-1))
        else: self.uf = self.model(self.neural_net_scale(torch.cat([x, t], dim=-1)))
        return self.uf
    
    def get_selector_data(self, x, t):
        uf = self.forward(x, t)
        u_t = complex_diff(uf, t, device=device)
        u_x = complex_diff(uf, x, device=device)
        u_xx = complex_diff(u_x, x, device=device)
        u_xxx = complex_diff(u_xx, x, device=device)
        derivatives = [cplx2tensor(uf), (uf.real**2+uf.imag**2)+0.0j, u_x, u_xx, u_xxx]
        return torch.cat(derivatives, dim=-1), u_t
    
    def neural_net_scale(self, inp):
        return -1+2*(inp-self.lb)/(self.ub-self.lb)

REG_INTENSITY = 0.1; print(REG_INTENSITY)
class ComplexAttentionSelectorNetwork(nn.Module):
    def __init__(self, layers, prob_activation=torch.sigmoid, bn=None, reg_intensity=REG_INTENSITY):
        super(ComplexAttentionSelectorNetwork, self).__init__()
        # Nonlinear model, Training with PDE reg.
        assert len(layers) > 1
        self.linear1 = CplxLinear(layers[0], layers[0], bias=True)
        self.prob_activation = prob_activation
        self.nonlinear_model = ComplexTorchMLP(dimensions=layers, activation_function=CplxToCplx[F.relu](), bn=bn, dropout_rate=0.0)
        self.latest_weighted_features = None
        self.th = (1/layers[0])+(1e-10)
        self.reg_intensity = reg_intensity
        self.al = ApproxL0(sig=1.0)
        self.w = (1e-1)*torch.tensor([1.0, 1.0, 1.0, 2.0, 3.0]).to(device)
        
    def xavier_init(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        
    def forward(self, inn):
        return self.nonlinear_model((inn*(F.relu(self.weighted_features(inn)-self.th))))
    
    def weighted_features(self, inn):
        self.latest_weighted_features = self.prob_activation(self.linear1(inn).real).mean(dim=0)
        return self.latest_weighted_features
    
    def loss(self, X_input, y_input):
        ut_approx = self.forward(X_input)
        l1 = complex_mse(ut_approx, y_input)
        reg_term = F.relu(self.latest_weighted_features-self.th)
        l2 = self.al(reg_term)+torch.dot(self.w, reg_term)
        return l1 + self.reg_intensity*l2

# Only the SemiSupModel has changed to work with the finite difference guidance
class SemiSupModel(nn.Module):
    def __init__(self, network, selector, normalize_derivative_features=False, mini=None, maxi=None):
        super(SemiSupModel, self).__init__()
        self.network = network
        self.selector = selector
        self.normalize_derivative_features = normalize_derivative_features
        self.mini = mini
        self.maxi = maxi
        
    def forward(self, X_h_train, h_train, include_unsup=True):
        X_selector, y_selector = self.network.get_selector_data(*dimension_slicing(X_h_train))
        
        h_row = h_train.shape[0]
        fd_guidance = complex_mse(self.network.uf[:h_row, :], h_train)
        
        if self.normalize_derivative_features:
            X_selector = (X_selector-self.mini)/(self.maxi-self.mini)
        
        if include_unsup: unsup_loss = self.selector.loss(X_selector, y_selector)
        else: unsup_loss = None

        return fd_guidance, unsup_loss

semisup_model = SemiSupModel(
    network=ComplexNetwork(model=complex_model, index2features=feature_names, scale=True, lb=lb, ub=ub),
    selector=ComplexAttentionSelectorNetwork([len(feature_names), 50, 50, 1], prob_activation=TanhProb(), bn=True),
    normalize_derivative_features=True,
    mini=None,
    maxi=None,
).to(device)

untrained_path = "./nls_weights/lambdas/untrained_semisup_model.pth"
use_untrained_path = True
if not use_untrained_path: 
    save(semisup_model, untrained_path); 
    print("Save the untrained weights.")
else: 
    semisup_model = load_weights(semisup_model, untrained_path)
    print("Use the same untrained semisup_model for the starting for alg1.")

X_train = X_train.to(device)
h_train = h_train.to(device)
X_star = X_star.to(device)
h_star = h_star.to(device)

lets_pretrain = True
if lets_pretrain:
    def pretraining_closure():
        global N, X_train, h_train
        if torch.enable_grad(): pretraining_optimizer.zero_grad()
        mse_loss = complex_mse(semisup_model.network(*dimension_slicing(X_train[:N, :])), h_train[:N, :])
        if mse_loss.requires_grad: mse_loss.backward(retain_graph=False)
        return mse_loss
    
    print("Pretraining")
    pretraining_optimizer = LBFGSNew(semisup_model.network.parameters(),
                                     lr=1e-1, max_iter=500,
                                     max_eval=int(500*1.25), history_size=300,
                                     line_search_fn=True, batch_mode=False)

    semisup_model.network.train()    
    for i in range(1):
        pretraining_optimizer.step(pretraining_closure)
            
        if (i%2)==0:
            l = pretraining_closure()
            curr_loss = l.item()
            print("Epoch {}: ".format(i), curr_loss)

            # See how well the model perform on the test set
            semisup_model.network.eval()
            test_performance = complex_mse(semisup_model.network(*dimension_slicing(X_star)), h_star).item()
            string_test_performance = scientific2string(test_performance)
            print('Test MSE:', string_test_performance)

    X_selector, y_selector = semisup_model.network.get_selector_data(*dimension_slicing(X_train))
    semisup_model.mini = torch.tensor(torch.abs(X_selector).min(axis=0).values, dtype=torch.cfloat).to(device)
    semisup_model.maxi = torch.tensor(torch.abs(X_selector).max(axis=0).values, dtype=torch.cfloat).to(device)

WWW = 1e-3
def pcgrad_closure(return_list=False):
    global N, X_train, h_train
    if torch.is_grad_enabled():
        optimizer.zero_grad()
    fd_guidance, unsup_loss = semisup_model(X_train, h_train, include_unsup=True)
    loss = fd_guidance+WWW*unsup_loss
    if loss.requires_grad:
        loss.backward(retain_graph=True)
    if not return_list: return loss
    else: return fd_guidance, unsup_loss

# Joint training
optimizer = MADGRAD([{'params':semisup_model.network.parameters()}, {'params':semisup_model.selector.parameters()}], lr=1e-6)
optimizer.param_groups[0]['lr'] = 1e-7
optimizer.param_groups[1]['lr'] = 1e-1 # 5e-2

# Use ~idx to sample adversarial data points
for i in range(1500):
    semisup_model.train()
    optimizer.step(pcgrad_closure)
    loss = pcgrad_closure(return_list=True)
    if i == 0:
        semisup_model.selector.th = 0.9*semisup_model.selector.latest_weighted_features.min().item()
        print(semisup_model.selector.th)
    if i%25==0:
        print(semisup_model.selector.latest_weighted_features.cpu().detach().numpy())
        print(loss)

# Fine-tuning the solver network
f_opt = torch.optim.LBFGS(semisup_model.network.parameters(), lr=1e-1, max_iter=500, max_eval=int(1.25*500), history_size=300)

def finetuning_closure():
    global N, X_train, h_train
    if torch.is_grad_enabled(): f_opt.zero_grad()
    loss = complex_mse(semisup_model.network(*dimension_slicing(X_train[:N, :])), h_train[:N, :])
    if loss.requires_grad: loss.backward(retain_graph=True)
    return loss

semisup_model.network.train()
semisup_model.selector.eval()

feature_importance = semisup_model.selector.latest_weighted_features.cpu().detach().numpy()
print(semisup_model.selector.th)
print(feature_importance)
print(feature_importance-semisup_model.selector.th+(1/len(feature_importance)))

ep = 0
for i in range(10):
    f_opt.step(finetuning_closure)
    if i%2==0:
        loss = finetuning_closure()
        print(loss.item())
    ep += 1

save(semisup_model, f"./nls_weights/lambdas/semisup_model_{name}_{REG_INTENSITY}_ep{ep}.pth")
