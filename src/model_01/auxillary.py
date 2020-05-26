from torch import nn
from torch import LongTensor as LT
from torch import FloatTensor as FT
import numpy as np
import torch

class Indexed_LinearFC_List(nn.Module):
    def __init__(self, num_modules, inp_size, op_size):
        super(Indexed_LinearFC_List, self).__init__()
        self.linears = nn.ModuleList([
            nn.Linear(inp_size, op_size)
            for _ in range(num_modules)
        ])


    def forward(self, x, indices):
        res = []
        for i,j in zip(x,indices):
            y = self.linears[j](i)
            res.append(y)
        res = torch.stack(res,dim=0)
        return res

