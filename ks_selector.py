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
import pcgrad

def to_tensor(arr, g=True):
    return torch.tensor(arr).float().requires_grad_(g)

# torch device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using", device)

X_sol, T_sol, Exact = space_time_grid(data_path=DATA_PATH, real_solution=True)
X_star, u_star = get_trainable_data(X_sol, T_sol, Exact)

# Bound
ub = X_star.max(axis=0)
lb = X_star.min(axis=0)

# For identification
# N = 30000 # Noisy experiments
N = 1024*21 # Clean experiment

# idx = np.random.choice(X_star.shape[0], N, replace=False)
idx = np.arange(N)
X_train = X_star[idx,:]
u_train = u_star[idx,:]
print("Training with", N, "samples")

# REG = 2e-2 # 0, 1e-2
# REG = 2e-3 # 0, 1e-2
REG = 2e-4 # 0, 1e-2
# REG = 0 # 0, 1e-2
print(REG)

noise_intensity_xt = 0.01/np.sqrt(2)
noise_intensity_labels = 0.01
noisy_xt = False; noisy_labels = False
if noisy_xt and noise_intensity_xt > 0.0:
    print("Noisy X_train")
    X_train = perturb2d(X_train, noise_intensity_xt)
else: print("Clean X_train")
if noisy_labels and noise_intensity_labels > 0.0:
    print("Noisy u_train")
    u_train = perturb(u_train, noise_intensity_labels)
else: print("Clean u_train")

# Unsup data
include_N_res = True
if include_N_res:
    N_res = N//2
    idx_res = np.array(range(X_star.shape[0]-1))[~idx]
    idx_res = np.random.choice(idx_res.shape[0], N_res, replace=True)
    X_res = X_star[idx_res, :]
    print(f"Training with {N_res} unsup samples")
    X_train = np.vstack([X_train, X_res])
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
feature_names=('uf', 'u_x', 'u_xx', 'u_xxx', 'u_xxxx', 'u_xxxxxx')

class ApproxL0(nn.Module):
    def __init__(self, sig=1.0, minmax=(0.0, 10.0), trainable=True):
        super(ApproxL0, self).__init__()
        self.sig = nn.Parameter(data=torch.FloatTensor([float(sig)])).requires_grad_(trainable)
        self.mini = minmax[0]
        self.maxi = minmax[1]

    def forward(self, w):
        sig = torch.clamp(self.sig, min=self.mini, max=self.maxi)
        return approx_l0(w, sig=sig)

def approx_l0(w, sig=1.0):
    sig = sig*torch.var(w)
    return len(w)-torch.exp(torch.div(-torch.square(w), 2*(torch.square(sig)))).sum()

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
        return 2*(inp-self.lb)/(self.ub-self.lb)-1

class AttentionSelectorNetwork(nn.Module):
    def __init__(self, layers, prob_activation=torch.sigmoid, bn=None, reg_intensity=REG):
        super(AttentionSelectorNetwork, self).__init__()
        # Nonlinear model, Training with PDE reg.
        assert len(layers) > 1
        self.linear1 = nn.Linear(layers[0], layers[0])
        self.prob_activation = prob_activation
        torch.nn.init.xavier_uniform_(self.linear1.weight)
        self.linear1.bias.data.fill_(0.01)
        
        self.nonlinear_model = TorchMLP(dimensions=layers, bn=bn, dropout=nn.Dropout(p=0.1))
        self.latest_weighted_features = None
        self.th = (1/layers[0])-(1e-10)
        self.reg_intensity = reg_intensity
        self.w = (1e-1)*torch.tensor([1.0, 1.0, 2.0, 3.0, 4.0, 5.0]).to(device)
        self.al = ApproxL0(sig=1.0)
        self.gamma = nn.Parameter(torch.ones(layers[0]).float()).requires_grad_(True)
        
    def xavier_init(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        
    def forward(self, inn):
        # return self.nonlinear_model((inn*(F.relu(self.weighted_features(inn)-self.th))))
        return self.nonlinear_model((inn*(F.relu(self.weighted_features(inn)-self.th)))*self.gamma)
    
    def weighted_features(self, inn):
        self.latest_weighted_features = self.prob_activation(self.linear1(inn)).mean(axis=0)
        return self.latest_weighted_features
    
    def loss(self, X_input, y_input):
        ut_approx = self.forward(X_input)
        mse_loss = F.mse_loss(ut_approx, y_input, reduction='mean')
        reg_term = F.relu(self.latest_weighted_features-self.th)
        
        l1 = mse_loss
        # l2 = torch.norm(reg_term, p=0)+torch.dot(self.w, reg_term)
        l2 = self.al(reg_term)+torch.dot(self.w, reg_term)
        return l1+self.reg_intensity*(l2)

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

lets_pretrain = True
pretrained_weights = "./weights/semisup_model_with_LayerNormDropout_without_physical_reg_trained30000labeledsamples_trained15000unlabeledsamples.pth"
# pretrained_weights = None
semisup_model = SemiSupModel(network=Network(
                                    model=TorchMLP(dimensions=[2, 50, 50, 50 ,50, 50, 1],
                                                   # activation_function=nn.Tanh(),
                                                   bn=nn.LayerNorm, dropout=None),
                                    index2features=feature_names, scale=True, lb=lb, ub=ub),
                            selector=AttentionSelectorNetwork([len(feature_names), 50, 50, 1], prob_activation=TanhProb(), bn=nn.LayerNorm),
                            normalize_derivative_features=True,
                            mini=None,
                            maxi=None)
if pretrained_weights is not None:
    print("Loaded pretrained_weights")
    semisup_model = load_weights(semisup_model, pretrained_weights)

semisup_model = semisup_model.to(device)

if pretrained_weights is not None:
    referenced_derivatives, _ = semisup_model.network.get_selector_data(*dimension_slicing(X_train))
    semisup_model.mini = torch.min(referenced_derivatives, axis=0)[0].detach().requires_grad_(False)
    semisup_model.maxi = torch.max(referenced_derivatives, axis=0)[0].detach().requires_grad_(False)
    del referenced_derivatives

def pcgrad_closure(return_list=False):
    global N, X_train, u_train
    if torch.is_grad_enabled():
        optimizer.zero_grad()
    predictions, unsup_loss = semisup_model(X_train)
    losses = [F.mse_loss(predictions[:N, :], u_train[:N, :]), 0.1*unsup_loss]
    loss = sum(losses)
    if loss.requires_grad: loss.backward(retain_graph=True)
    if not return_list: return loss
    else: return losses

# Joint training
optimizer = MADGRAD([{'params':semisup_model.network.parameters()}, {'params':semisup_model.selector.parameters()}], lr=1e-6)
optimizer.param_groups[0]['lr'] = 1e-7 # Used to be 1e-11.
optimizer.param_groups[1]['lr'] = 5e-3 # pub: 5e-3

if lets_pretrain:
    print("Pretraining")
    pretraining_optimizer = LBFGSNew(semisup_model.network.parameters(), 
                                     lr=1e-1, max_iter=500, 
                                     max_eval=int(500*1.25), history_size=300, 
                                     line_search_fn=True, batch_mode=False)
    
    best_state_dict = None; curr_loss = 1000
    semisup_model.network.train()
    for i in range(1):
        def pretraining_closure():
            global N, X_train, u_train
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
            
        if (i%1)==0:
            print("Epoch {}: ".format(i), curr_loss)

            # Sneak on the test performance...
            semisup_model.network.eval()
            test_performance = F.mse_loss(semisup_model.network(*dimension_slicing(X_star)).detach(), u_star).item()
            string_test_performance = scientific2string(test_performance)
            print('Test MSE:', string_test_performance)
    
    # If there is the best_state_dict
    if best_state_dict is not None: semisup_model.load_state_dict(best_state_dict)

#    referenced_derivatives, _ = semisup_model.network.get_selector_data(*dimension_slicing(X_train))
#    semisup_model.mini = torch.min(referenced_derivatives, axis=0)[0].detach().requires_grad_(False)
#    semisup_model.maxi = torch.max(referenced_derivatives, axis=0)[0].detach().requires_grad_(False)
#    del referenced_derivatives

semisup_model = load_weights(semisup_model,  "./lambda_study/take2/init.pth")
# semisup_model = load_weights(semisup_model,  "./lambda_study/take2/init_20220412.pth")
# torch.save(semisup_model.state_dict(), "./lambda_study/take2/init_20220412.pth")

# Use ~idx to sample adversarial data points
epochs = 300
for i in range(epochs):
    semisup_model.train()
    optimizer.step(pcgrad_closure)
    loss = pcgrad_closure(return_list=True)
    if i == 0:
        semisup_model.selector.th = 0.8*semisup_model.selector.latest_weighted_features.cpu().min().item() # 0
        # semisup_model.selector.th = 0.1 # 1e-1, 1e-2, 1e-3
        print(semisup_model.selector.th)
    if i%25==0:
        print(semisup_model.selector.latest_weighted_features.cpu().detach().numpy())
        print(loss)

feature_importance = semisup_model.selector.latest_weighted_features.cpu().detach().numpy()
print(feature_importance)
old_th = 1/len(feature_importance); diff = old_th-semisup_model.selector.th
if diff < 0: feature_importance = feature_importance - abs(diff)
else: feature_importance = feature_importance + abs(diff)
print(feature_importance)
# print("Saving trained weights...")
# torch.save(semisup_model.state_dict(), f"./lambda_study/take2/{REG}_nogamma.pth")
