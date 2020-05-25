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


    # ------------
    # Filter the node types which are valid for a node type
    # ------------
    def get_valid_metapaths(self, x_type):
        # Todo : figure out a better way
        arr = np.array(self.metapath_list)[:0]
        mp = []

        for j in range(len(arr)):
            if x_type == arr[j]: mp.append(j)
        return mp




    '''
    input_x :
    target node : x_target
    Input a single type
    
    '''
    def forward(
        self,
        x_target = None, # shape : [ ?, 1 ]
        x_type = -1
    ) :
        # --------
        # For all meta-paths that are valid
        # --------
        valid_metapaths = self.get_valid_metapaths(x_type)

        # for mp in valid_metapaths:
        #     # Calculate per meta-path value





        return


