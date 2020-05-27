from torch import nn
from torch import LongTensor as LT
from torch import FloatTensor as FT
import numpy as np
import torch
from torch.nn import functional as F
import math

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



class ScaledDotProductAttention(nn.Module):

    def forward(self, query, key, value, mask=None):
        dk = query.size()[-1]
        scores = query.matmul(
            key.transpose(-2, -1)
        ) / math.sqrt(dk)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = F.softmax(scores, dim=-1)
        return attention.matmul(value)


class MultiHeadAttention(nn.Module):

    def __init__(
         self,
         inp_dim,
         num_heads,
         bias=True,
         activation=F.relu):
        """Multi-head attention.
        :param in_features: Size of each input sample.
        :param head_num: Number of heads.
        :param bias: Whether to use the bias term.
        :param activation: The activation after each linear transformation.
        """
        super(MultiHeadAttention, self).__init__()

        op_dim = inp_dim
        self.num_heads = num_heads
        self.activation = activation
        self.bias = bias
        self.attention_wts_k = nn.ParameterList(
            [ nn.Parameter(data= torch.randn(inp_dim, op_dim)) for _ in range(num_heads)]
        )

        self.W = nn.Parameter(torch.zeros(size=(16, 18)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * 18, 1)))


        return

    # input
    def forward(self, inputs):
        # https: // github.com / Diego999 / pyGAT / blob / master / layers.py
        res = []
        # for each head calculate
        print('Input ', inputs.shape)
        h = torch.matmul(inputs,  self.W )
        N = h.size()[0]
        print('h shape ', h.shape)
        print('n ', N)
        a_input = torch.cat(
            [h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)],
            dim=1
        )
        print(a_input.shape)
        a_input = a_input.reshape( N, -1, 2 * 18)
        print(a_input.shape)
        for i in range(self.num_heads):
            op1 = torch.matmul(
                inputs,
                self.attention_wts_k[i]
            )
            # Concat each transformed input

            print('attention_score' , attention_score.shape)
            # attention_score = attention_score.squeeze()
            attention_score = F.softmax(attention_score,dim=-2)
            print('>>', attention_score.shape)
            # reshape(inputs.size(0), inputs.size(1), 1)
            print('>>', attention_score[:,0,0])
            scored_x = inputs * attention_score
            condensed_x = torch.sum(scored_x, dim=1)
            print('condensed_x', condensed_x.size())
            res.append(condensed_x)

        res = torch.cat(res,dim =-1)
        print(res.size())
        return res


net = MultiHeadAttention(
    inp_dim = 16,
    num_heads = 4
)

x = FT(torch.randn(20,30,16))
r = net(x)
print(r.size())

