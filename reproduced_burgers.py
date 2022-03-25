#!/usr/bin/env python
# coding: utf-8
import os
import numpy as np
import torch
from torch import tensor
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable, grad

import scipy
import scipy.io as io
from pyDOE import lhs

from sklearn.metrics import mean_squared_error, mean_absolute_error
from utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using", device)

data = io.loadmat('./data/burgers_shock.mat')

N = 3000

t = data['t'].flatten()[:,None]
x = data['x'].flatten()[:,None]
Exact = np.real(data['usol']).T

X, T = np.meshgrid(x,t)

X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
u_star = Exact.flatten()[:,None]              

# Doman bounds
lb = X_star.min(0)
ub = X_star.max(0)

force_save = False
idx = np.random.choice(X_star.shape[0], N, replace=False)
if force_save: 
    np.save("./burgers_weights/idx.npy", idx)
    print("Save indices...")
else: 
    idx = np.load("./burgers_weights/idx2.npy")
    print("Load indices V2...")
X_u_train = X_star[idx, :]
u_train = u_star[idx,:]
print(f"Training with {X_u_train.shape[0]} samples")

noisy_xt = False; noisy_labels = False; state = int(noisy_xt)+int(noisy_labels)
if state == 0:
    name = "cleanall"
elif state == 1:
    name = "noisy1"
elif state == 2:
    name = "noisy2"
print(name)
noise_intensity = 0.01
if noisy_xt:
    X_noise = perturb2d(X_u_train, intensity=noise_intensity/np.sqrt(2), overwrite=False)
    if force_save: np.save("./burgers_weights/X_noise.npy", X_noise)
    else: X_noise = np.load("./burgers_weights/X_noise2.npy")
    X_u_train = X_u_train + X_noise
if noisy_labels:
    u_noise = perturb(u_train, intensity=noise_intensity, overwrite=False)
    if force_save: np.save("./burgers_weights/u_noise.npy", u_noise)
    else: u_noise = np.load("./burgers_weights/u_noise2.npy")
    u_train = u_train + u_noise

class Network(nn.Module):
    def __init__(self, model):
        super(Network, self).__init__()
        self.model = model
        self.model.apply(self.xavier_init)
        self.lambda_1 = torch.nn.Parameter(torch.tensor([0.0]))
        # self.lambda_2 = torch.nn.Parameter(torch.tensor([-6.0]))
        self.lambda_2 = torch.nn.Parameter(torch.tensor([-7.0]))
        
    def xavier_init(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        
    def forward(self, x, t):
        return self.model(torch.cat([x, t], dim=1))
    
    def loss(self, x, t, y_input):
        lambda_1 = self.lambda_1
        lambda_2 = torch.exp(self.lambda_2)
        
        uf = self.forward(x, t)
        
        # PDE Loss calculation
        u_t = self.gradients(uf, t)[0]
        u_x = self.gradients(uf, x)[0]
        u_xx = self.gradients(u_x, x)[0]
        l_eq = (u_t + lambda_1*uf*u_x - lambda_2*u_xx)
        l_eq = (l_eq**2).mean()
        
        # Loss on the boundary condition
        mse = F.mse_loss(uf, y_input, reduction='mean')
        
        return l_eq + mse
    
    def gradients(self, func, x):
        return grad(func, x, create_graph=True, retain_graph=True, grad_outputs=torch.ones(func.shape).to(device))

hidden_nodes = 50

model = nn.Sequential(nn.Linear(2, hidden_nodes), 
                        nn.Tanh(), 
                        nn.Linear(hidden_nodes, hidden_nodes),
                        nn.Tanh(), 
                        nn.Linear(hidden_nodes, hidden_nodes),
                        nn.Tanh(), 
                        nn.Linear(hidden_nodes, hidden_nodes),
                        nn.Tanh(),
                        nn.Linear(hidden_nodes, hidden_nodes),
                        nn.Tanh(),
                        nn.Linear(hidden_nodes, 1))

init_weights = "./burgers_weights/init_reproduced_pinn.pth"
# save(model, init_weights)
model = load_weights(model, init_weights)

network = Network(model=model).to(device)

X_u_train = tensor(X_u_train).float().requires_grad_(True).to(device)
u_train = tensor(u_train).float().requires_grad_(True).to(device)

X_star = tensor(X_star).float().requires_grad_(True)
u_star = tensor(u_star).float().requires_grad_(True)

optimizer = torch.optim.LBFGS(network.parameters(), lr=0.1, max_iter=300, max_eval=300, history_size=150, line_search_fn='strong_wolfe')
if state == 0:
    optimizer = torch.optim.LBFGS(network.parameters(), lr=0.1, max_iter=500, max_eval=500, history_size=300, line_search_fn='strong_wolfe')
epochs = 300
network.train()
# weights_path = None
weights_path = f'./burgers_weights/reproduced_pinn_{state}.pth'

for i in range(epochs):    
    ### Add the closure function to calculate the gradient. For LBFGS.
    def closure():
        if torch.is_grad_enabled():
            optimizer.zero_grad()
        l = network.loss(X_u_train[:, 0:1], X_u_train[:, 1:2], u_train)
        if l.requires_grad:
            l.backward()
        return l

    optimizer.step(closure)

    # calculate the loss again for monitoring
    l = closure()

    if (i % 100) == 0:
        print("Epoch {}: ".format(i), l.item())

if weights_path is not None:
    save(network, weights_path)

network.eval()

nu = 0.01 / np.pi

error_lambda_1 = np.abs(network.lambda_1.cpu().detach().item() - 1.0)*100
error_lambda_2 = np.abs(torch.exp(network.cpu().lambda_2).detach().item() - nu) / nu * 100

error_lambda_1, error_lambda_2

print(1.0, network.lambda_1.cpu().detach().item())

print(nu, torch.exp(network.lambda_2).cpu().detach().item())

errs = 100*np.array([np.abs(1.0-network.lambda_1.cpu().detach().item()), np.abs(nu-torch.exp(network.lambda_2).cpu().detach().item())/nu])

print(errs.mean(), errs.std())
