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

# Adjust the diemnsion of Exact and potential (0.5*x**2)
if Exact.T.shape == X.shape: Exact = Exact.T
if potential.T.shape == X.shape: potential = potential.T
Exact_u = np.real(Exact)
Exact_v = np.imag(Exact)

# Converting in a feature vector for each feature
X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
h_star = to_column_vector(Exact)
u_star = to_column_vector(Exact_u)
v_star = to_column_vector(Exact_v)

# Doman bounds
lb = X_star.min(axis=0)
ub = X_star.max(axis=0)

N = 10000; include_N_res = 1
idx = np.random.choice(X_star.shape[0], N, replace=False)
# idx = np.arange(N) # Just have an easy dataset for experimenting
print(f"Training with {N} labeled samples")

lb = to_tensor(lb, False).to(device)
ub = to_tensor(ub, False).to(device)

X_train = X_star[idx, :]
u_train = u_star[idx, :]
v_train = v_star[idx, :]

# Converting to tensor
X_star = to_tensor(X_star, True)
h_star = to_complex_tensor(h_star, False)

X_train = to_tensor(X_train, True)
u_train = to_tensor(u_train, False)
v_train = to_tensor(v_train, False)
h_train = torch.complex(u_train, v_train)

# Unsup data
if include_N_res>0:
    N_res = int(N*include_N_res)
    idx_res = np.array(range(X_star.shape[0]-1))[~idx]
    idx_res = idx_res[:N_res]
    X_res = to_tensor(X_star[idx_res, :], True)
    print(f"Training with {N_res} unsup samples")
    X_train = torch.vstack([X_train, X_res])

# Potential is calculated from x
# Hence, Quadratic features of x are required.
feature_names = ['hf', 'h_x', 'h_xx', 'h_xxx', 'V']

dt = (t[1]-t[0])
dx = (x[2]-x[1])

fd_h_t = np.zeros((time_dim, spatial_dim), dtype=np.complex64)
fd_h_x = np.zeros((time_dim, spatial_dim), dtype=np.complex64)
fd_h_xx = np.zeros((time_dim, spatial_dim), dtype=np.complex64)
fd_h_xxx = np.zeros((time_dim, spatial_dim), dtype=np.complex64)

for i in range(spatial_dim):
    fd_h_t[:,i] = FiniteDiff(Exact[:,i], dt, 1)
for i in range(time_dim):
    fd_h_x[i,:] = FiniteDiff(Exact[i,:], dx, 1)
    fd_h_xx[i,:] = FiniteDiff(Exact[i,:], dx, 2)
    fd_h_xxx[i,:] = FiniteDiff(Exact[i,:], dx, 3)

fd_h_t = to_column_vector(fd_h_t)
fd_h_x = to_column_vector(fd_h_x)
fd_h_xx = to_column_vector(fd_h_xx)
fd_h_xxx = to_column_vector(fd_h_xxx)
V = to_column_vector(potential)

derivatives = cat_numpy(h_star.detach().numpy(), V, fd_h_x, fd_h_xx, fd_h_xxx)
dictionary = {}
for i in range(len(feature_names)): dictionary[feature_names[i]] = get_feature(derivatives, i)

# PRETRAINED_PATH = "./qho_weights/pretrained_cpinn_2000labeledsamples.pth"
PRETRAINED_PATH = None

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

if PRETRAINED_PATH is not None: complex_model.load_state_dict(cpu_load(PRETRAINED_PATH))

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
        derivatives = [cplx2tensor(uf), u_x, u_xx, u_xxx, 0.5*torch.pow(x,2)]
        return torch.cat(derivatives, dim=-1), u_t
    
    def neural_net_scale(self, inp):
        return -1 + 2*(inp-self.lb)/(self.ub-self.lb)

REG_INTENSITY = 1e-2
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
        # 1e0 = 1
        # self.w = (1e-1)*torch.tensor([1.0, 1.0, 2.0, 3.0, 1.0])
        # 1e-2, 1e-4
        self.w = torch.tensor([1.0, 3.0, 2.0, 3.0, 1.0]).to(device)
        # self.gamma = nn.Parameter(torch.ones(layers[0]).float()).requires_grad_(True)
        
    def xavier_init(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        
    def forward(self, inn):
        return self.nonlinear_model((inn*(F.relu(self.weighted_features(inn)-self.th))))
        # return self.nonlinear_model(inn*F.threshold(self.weighted_features(inn), self.th, 0.0))
    
    def weighted_features(self, inn):
        self.latest_weighted_features = self.prob_activation(self.linear1(inn).real).mean(dim=0)
        return self.latest_weighted_features
    
    def loss(self, X_input, y_input):
        ut_approx = self.forward(X_input)
        l1 = complex_mse(ut_approx, y_input)
        reg_term = F.relu(self.latest_weighted_features-self.th)
        l2 = torch.norm(reg_term, p=0)+torch.dot(self.w, reg_term)
        return l1 + self.reg_intensity*l2

# Only the SemiSupModel has changed to work with the finite difference guidance
class SemiSupModel(nn.Module):
    def __init__(self, network, selector, normalize_derivative_features=False, mini=None, maxi=None, uncert=False):
        super(SemiSupModel, self).__init__()
        self.network = network
        self.selector = selector
        self.normalize_derivative_features = normalize_derivative_features
        self.mini = mini
        self.maxi = maxi
        self.weights = None
        if uncert: 
            self.weights = torch.tensor([0.0, 0.0])
        
    def forward(self, X_h_train, h_train, include_unsup=True):
        X_selector, y_selector = self.network.get_selector_data(*dimension_slicing(X_h_train))
        
        h_row = h_train.shape[0]
        fd_guidance = complex_mse(self.network.uf[:h_row, :], h_train)
        
        # I am not sure a good way to normalize/scale a complex tensor
        if self.normalize_derivative_features:
            X_selector = (X_selector-self.mini)/(self.maxi-self.mini)
        
        if include_unsup: unsup_loss = self.selector.loss(X_selector, y_selector)
        else: unsup_loss = None
            
        if include_unsup and self.weights is not None:
            return (torch.exp(-self.weights[0])*fd_guidance)+self.weights[0], (torch.exp(-self.weights[1])*unsup_loss)+self.weights[1]
        else:
            return fd_guidance, unsup_loss

semisup_model = SemiSupModel(
    network=ComplexNetwork(model=complex_model, index2features=feature_names, scale=True, lb=lb, ub=ub),
    selector=ComplexAttentionSelectorNetwork([len(feature_names), 50, 50, 1], prob_activation=TanhProb(), bn=True),
    normalize_derivative_features=True,
    mini=torch.tensor(np.abs(derivatives).min(axis=0), dtype=torch.cfloat),
    maxi=torch.tensor(np.abs(derivatives).max(axis=0), dtype=torch.cfloat),
    uncert=False,
).to(device)

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
    for i in range(120):
        pretraining_optimizer.step(pretraining_closure)
            
        if (i%10)==0:
            l = pretraining_closure()
            curr_loss = l.item()
            print("Epoch {}: ".format(i), curr_loss)

            # See how well the model perform on the test set
            semisup_model.network.eval()
            test_performance = complex_mse(semisup_model.network(*dimension_slicing(X_star)), h_star).item()
            string_test_performance = scientific2string(test_performance)
            print('Test MSE:', string_test_performance)

def pcgrad_closure(return_list=False):
    global N, X_train, h_train
    fd_guidance, unsup_loss = semisup_model(X_train, h_train, include_unsup=True)
    losses = [fd_guidance, (1e-2)*unsup_loss]
    loss = sum(losses)
    loss.backward(retain_graph=True)
    if not return_list: return loss
    else: return losses

save(semisup_model, f"./qho_weights/pretrained_semisup_model_lambda1_{REG_INTENSITY}.pth")

# Joint training
optimizer = MADGRAD([{'params':semisup_model.network.parameters()}, {'params':semisup_model.selector.parameters()}], lr=1e-6)
optimizer.param_groups[0]['lr'] = 1e-7
optimizer.param_groups[1]['lr'] = 0.1

# Use ~idx to sample adversarial data points
for i in range(500):
    semisup_model.train()
    optimizer.step(pcgrad_closure)
    loss = pcgrad_closure(return_list=True)
    if i == 0:
        semisup_model.selector.th = 0.95*semisup_model.selector.latest_weighted_features.min().item()
        print(semisup_model.selector.th)
    if i%25==0:
        print(semisup_model.selector.latest_weighted_features.cpu().detach().numpy())
        print(loss)

# Fine-tuning the solver network
f_opt = torch.optim.LBFGS(semisup_model.network.parameters(), lr=0.1, max_iter=500, max_eval=int(1.25*500), history_size=300)

def finetuning_closure():
    global N, X_train, h_train
    if torch.is_grad_enabled(): f_opt.zero_grad()
    loss = complex_mse(semisup_model.network(*dimension_slicing(X_train[:N, :])), h_train[:N, :])
    if loss.requires_grad: loss.backward(retain_graph=True)
    return loss

semisup_model.network.train()
semisup_model.selector.eval()

for i in range(50):
    f_opt.step(finetuning_closure)
    if i%10==0:
        loss = finetuning_closure()
        print(loss.item())

feature_importance = semisup_model.selector.latest_weighted_features.cpu().detach().numpy()
old_th = 1/len(feature_importance); diff = abs(old_th-semisup_model.selector.th)
feature_importance = np.where(feature_importance<old_th, feature_importance+diff, feature_importance)
print(feature_importance)

save(semisup_model, f"./qho_weights/jointtrained_semisup_model_lambda1_{REG_INTENSITY}.pth")
