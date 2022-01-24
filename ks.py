# -*- coding: utf-8 -*-
import os
from glob import glob

# DATA_PATH = "data/KS_chaotic.mat"
DATA_PATH = "data/kuramoto_sivishinky.mat"

from utils import *
from lbfgsnew import *
from models import *
from preprocess import *

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from scipy.io import loadmat
from madgrad import MADGRAD

def to_tensor(arr, g=True):
    return torch.tensor(arr).float().requires_grad_(g)

# torch device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using", device)

data = loadmat(DATA_PATH)

t = data['tt'].flatten()[:,None]
x = data['x'].flatten()[:,None]
Exact = np.real(data['uu'])
T_sol, X_sol = np.meshgrid(t, x)

if (X_sol.shape == T_sol.shape == Exact.shape): print("Shapes OK!")
else: print("Shapes NOT OK!")

x_star = X_sol.flatten()[:,None]
t_star = T_sol.flatten()[:,None]

X_star = np.hstack((x_star, t_star))
u_star = Exact.flatten()[:,None]

# Bound
ub = X_star.max(axis=0)
lb = X_star.min(axis=0)

# For identification
# N = 60000
N = 100000

idx = np.random.choice(X_star.shape[0], N, replace=False)
X_train = X_star[idx,:]
u_train = u_star[idx,:]
print("Training with", N, "samples")

noise_intensity_xt = 0.01/np.sqrt(2)
noise_intensity_labels = 0.01
noisy_xt = True; noisy_labels = True
if noisy_xt and noise_intensity_xt > 0.0:
    print("Noisy X_train")
    X_train = perturb2d(X_train, noise_intensity_xt)
else: print("Clean X_train")
if noisy_labels and noise_intensity_labels > 0.0:
    print("Noisy u_train")
    u_train = perturb(u_train, noise_intensity_labels)
else: print("Clean u_train")

# Unsup data
include_N_res = False
if include_N_res:
    N_res = N//2
    idx_res = np.array(range(X_star.shape[0]-1))[~idx]
    idx_res = np.random.choice(idx_res.shape[0], N_res, replace=True)
    X_res = X_star[idx_res, :]
    print(f"Training with {N_res} unsup samples")
    X_u_train = np.vstack([X_train, X_res])
    u_train = np.vstack([u_train, torch.rand(X_res.shape[0], 1) - 1000])
    # del X_res
else: print("Not including N_res")
    
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
feature_names=('uf', 'u_x', 'u_xx', 'u_xxx', 'u_xxxx')

class Network(nn.Module):
    def __init__(self, model, index2features=None, scale=False, lb=None, ub=None):
        super(Network, self).__init__()
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
        if not self.scale: self.uf = self.model(torch.cat([x, t], dim=1))
        else: self.uf = self.model(self.neural_net_scale(torch.cat([x, t], dim=1)))
        return self.uf
    
    def get_selector_data(self, x, t):
        uf = self.forward(x, t)
        u_t = self.gradients(uf, t)[0]
        
        ### PDE Loss calculation ###
        # Without calling grad
        derivatives = []
        for t in self.diff_flag[0]:
            if t=='uf': derivatives.append(uf)
            elif t=='x': derivatives.append(x)
        # With calling grad
        for t in self.diff_flag[1]:
            out = uf
            for c in t:
                if c=='x': out = self.gradients(out, x)[0]
                elif c=='t': out = self.gradients(out, t)[0]
            derivatives.append(out)
        
        return torch.cat(derivatives, dim=1), u_t
    
    def gradients(self, func, x):
        return grad(func, x, create_graph=True, retain_graph=True, grad_outputs=torch.ones(func.shape))
    
    def neural_net_scale(self, inp):
        return 2*(inp-self.lb)/(self.ub-self.lb)-1

class AttentionSelectorNetwork(nn.Module):
    def __init__(self, layers, prob_activation=torch.sigmoid, bn=None, reg_intensity=0.1):
        super(AttentionSelectorNetwork, self).__init__()
        # Nonlinear model, Training with PDE reg.
        assert len(layers) > 1
        self.linear1 = nn.Linear(layers[0], layers[0])
        self.prob_activation = prob_activation
        self.nonlinear_model = TorchMLP(dimensions=layers, activation_function=nn.Tanh(), bn=bn, dropout=nn.Dropout(p=0.1))
        self.latest_weighted_features = None
        self.th = 0.1
        self.reg_intensity = reg_intensity
        
    def xavier_init(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        
    def forward(self, inn):
        return self.nonlinear_model(inn*self.weighted_features(inn))
    
    def weighted_features(self, inn):
        self.latest_weighted_features = self.prob_activation(self.linear1(inn)).mean(axis=0)
        return self.latest_weighted_features
    
    def loss(self, X_input, y_input):
        ut_approx = self.forward(X_input)
        mse_loss = F.mse_loss(ut_approx, y_input, reduction='mean')
        reg_term = F.relu(self.latest_weighted_features-self.th)
        return mse_loss+self.reg_intensity*(torch.norm(reg_term, p=0)+(torch.tensor([1.0, 1.0, 2.0, 3.0, 4.0])*reg_term).sum())

class SemiSupModel(nn.Module):
    def __init__(self, network, selector, normalize_derivative_features=False, mini=None, maxi=None):
        super(SemiSupModel, self).__init__()
        self.network = network
        self.selector = selector
        self.normalize_derivative_features = normalize_derivative_features
        self.mini = mini
        self.maxi = maxi
        
    def forward(self, X_u_train):
        X_selector, y_selector = self.network.get_selector_data(*dimension_slicing(X_u_train))
        if self.normalize_derivative_features:
            X_selector = (X_selector-self.mini)/(self.maxi-self.mini)
        unsup_loss = self.selector.loss(X_selector, y_selector)
        return self.network.uf, unsup_loss

### Version with normalized derivatives ###
use_pretrained_weights = False
lets_pretrain = not use_pretrained_weights

semisup_model = SemiSupModel(network=Network(
                                    model=TorchMLP(dimensions=[2, 50, 50, 50 ,50, 50, 1],
                                                   # activation_function=nn.Tanh(),
                                                   bn=nn.LayerNorm, dropout=None),
                                    index2features=feature_names, scale=True, lb=lb, ub=ub),
                            selector=AttentionSelectorNetwork([len(feature_names), 50, 50, 1], prob_activation=nn.Softmax(dim=1), bn=nn.LayerNorm),
                            normalize_derivative_features=True,
                            mini=None,
                            maxi=None).to(device)

if lets_pretrain:
    print("Pretraining")
    pretraining_optimizer = LBFGSNew(semisup_model.network.parameters(), 
                                     lr=1e-1, max_iter=500, 
                                     max_eval=int(500*1.25), history_size=300, 
                                     line_search_fn=True, batch_mode=False)
    
    best_state_dict = None; curr_loss = 1000
    semisup_model.network.train()
    for i in range(300):
        def pretraining_closure():
            global N, X_u_train, u_train
            if torch.is_grad_enabled():
                pretraining_optimizer.zero_grad()
            # Only focusing on first [:N, :] elements
            mse_loss = F.mse_loss(semisup_model.network(*dimension_slicing(X_train[:N, :])), u_train[:N, :])
            if mse_loss.requires_grad:
                mse_loss.backward(retain_graph=False)
            return mse_loss

        pretraining_optimizer.step(pretraining_closure)

        l = pretraining_closure()
        
        if l.item() < curr_loss:
            curr_loss = l.item()
            best_state_dict = semisup_model.state_dict()
            
        if (i%10)==0:
            print("Epoch {}: ".format(i), curr_loss)

            # Sneak on the test performance...
            semisup_model.network.eval()
            test_performance = F.mse_loss(semisup_model.network(*dimension_slicing(X_star)).detach(), u_star).item()
            string_test_performance = scientific2string(test_performance)
            print('Test MSE:', string_test_performance)
    
    # If there is the best_state_dict
    if best_state_dict is not None: semisup_model.load_state_dict(best_state_dict)
    # print("Computing derivatives features")
    # semisup_model.eval()
    # referenced_derivatives, _ = semisup_model.network.get_selector_data(*dimension_slicing(X_star))
    
    # semisup_model.mini = torch.min(referenced_derivatives, axis=0)[0].detach().requires_grad_(False)
    # semisup_model.maxi = torch.max(referenced_derivatives, axis=0)[0].detach().requires_grad_(False)

    # semisup_model.mini = tmp.min(axis=0)[0].requires_grad_(False)
    # semisup_model.maxi = tmp.max(axis=0)[0].requires_grad_(False)

print("Saving trained weights...")
torch.save(semisup_model.state_dict(), "./weights/rudy_KS_noisy2_chaotic_semisup_model_with_LayerNormDropout_without_physical_reg_trained60000labeledsamples_trained0unlabeledsamples.pth")
