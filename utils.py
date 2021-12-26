import torch
from torch import nn

def scientific2string(x):
    return format(x, '.1e')

def dimension_slicing(a_tensor):
    c = a_tensor.shape[-1]
    out = []
    for i in range(1, c+1): out.append(a_tensor[:, i-1:i])
    return out

def diff_order(dterm):
    return dterm.split("_")[-1][::-1]

def diff_flag(index2feature):
    dd = {0:[], 1:[]}
    for t in index2feature:
        if '_' not in t: dd[0].append(t)
        else: dd[1].append(diff_order(t))
    return dd

class TorchMLP(nn.Module):
    def __init__(self, dimensions, bias=True, activation_function=nn.Tanh(), bn=None, dropout=None):
        super(TorchMLP, self).__init__()
        self.model  = nn.ModuleList()

        for i in range(len(dimensions)-1):
            self.model.append(nn.Linear(dimensions[i], dimensions[i+1], bias=bias))
            if bn is not None and i!=len(dimensions)-2:
                self.model.append(bn(dimensions[i+1]))
                if dropout is not None:
                    self.model.append(dropout)
            if i==len(dimensions)-2: break
            self.model.append(activation_function)

    def xavier_init(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        for i, l in enumerate(self.model): 
            x = l(x)
        return x

# class TorchMLP(nn.Module):
#     def __init__(self, dimensions, bias=True,activation_function=nn.Tanh, bn=None, dropout=None, inp_drop=False, final_activation=None):
#         super(TorchMLP, self).__init__()
#         print("Using old implementation of TorchMLP in utils.py. See models.py for more new model-related source code.")
#         self.model  = nn.ModuleList()
#         # Can I also use the LayerNorm with elementwise_affine=True
#         # This should be a callable module.
#         self.activation_function = activation_function()
#         self.bn = bn
#         if dropout is not None and inp_drop: self.inp_dropout = dropout
#         else: self.inp_dropout = None
#         for i in range(len(dimensions)-1):
#             self.model.append(nn.Linear(dimensions[i], dimensions[i+1], bias=bias))
#             if self.bn is not None and i!=len(dimensions)-2:
#                 self.model.append(self.bn(dimensions[i+1]))
#                 if dropout is not None:
#                     self.model.append(dropout)
#             if i==len(dimensions)-2: break
#             self.model.append(activation_function())
#         if final_activation is not None:
#             self.model.append(final_activation())
#         self.model.apply(self.xavier_init)

#     def xavier_init(self, m):
#         if type(m) == nn.Linear:
#             torch.nn.init.xavier_uniform_(m.weight)
#             m.bias.data.fill_(0.01)

#     def forward(self, x):
#         if hasattr(self, 'inp_dropout'):
#             if self.inp_dropout is not None:
#                 x = self.inp_dropout(x)
#         for i, l in enumerate(self.model): 
#             x = l(x)
#         return x
