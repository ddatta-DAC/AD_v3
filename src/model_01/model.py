import torch
import numpy as np
import networkx as nx
import stellargraph
import math
import sklearn
from torch import nn
from torch.nn import functional as F
from torch import LongTensor as LT
from torch import FloatTensor as FT
from stellargraph.data import UniformRandomMetaPathWalk


# ---------------------- #

class gmodel01(torch.nn.Module):
    def __init__(self):
        super(gmodel01, self).__init__()
        return

    def set_hypperparams(
        self,
        graph_obj, # stellargraph object
        num_nodes,
        num_edge_type,          # int
        num_node_type = 1,      # int
        metapath_list = None,   # list of integers values between 0 and nt-1
        embedding_size=128,
        num_layers = 2

    ):
        self.node_emb = nn.Embedding(num_nodes)
        self.edge_matrix = nn.ParameterList()
        for i in range(num_edge_type):
            self.edge_matrix.append(
                nn.Parameter(data = torch.ones(embedding_size,embedding_size))
            )

        self.op_emb = None
        self.num_layers = num_layers
        self.graph_obj = graph_obj
        self.metapath_list = metapath_list
        # For each metapath create RWalker
        self.mprw_obj = []
        for mp in self.metapath_list :
            obj_rw = UniformRandomMetaPathWalk(
                    self.graph_obj,
                    length=len(mp),
                    metapaths=[mp],
            )
            self.mprw_obj.append(obj_rw)
        return

    '''
    input_x :
    target node : x_target
    Input a single type
    
    '''
    def forward(
        self,
        x_input
    ) :


        return


