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
from torch.autograd import Variable
from . import auxillary
# ---------------------- #

class gmodel01(torch.nn.Module):
    def __init__(self):
        super(gmodel01, self).__init__()

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.__device__ = device
        return

    # =================
    # Assume:
    # 1. input features is None and is initialized as identity vector (one-hot)
    # 2. Single layer "GNN"
    # =================
    def set_hypperparams(
        self,
        graph_obj, # stellargraph object
        num_nodes,
        num_edge_type = 1,
        num_node_type = 1,      # int
        metapath_list = None,   # list of integers values between 0 and nt-1
        embedding_size=128,
    ):
        op_node_emb_size = embedding_size
        # Identity features
        identity_inp_features = np.eye(N=num_nodes,dtype = np.float)
        identity_inp_features = FT(identity_inp_features).to(self.__device__ )

        self.identity_inp_features = nn.Embedding.from_pretrained(
            identity_inp_features
        )

        num_metapaths = len(metapath_list)
        xformed_emb_size = embedding_size
        # ------
        # Transformation for "input feature"
        # ------
        self.fc1 = nn.Linear(
            in_features = num_nodes,
            out_features = xformed_emb_size
        )

        # ----
        # Each edge is represented as vector W_e
        # n1 x w_e x n2
        # ----

        self.W_edge = nn.Embedding( num_edge_type,embedding_size )
        self.W_mp_encoder_wt = auxillary.Indexed_LinearFC_List(
            num_modules=num_metapaths,
            inp_size=embedding_size,
            op_size=embedding_size
        )

        self.op_emb = np.zeros([num_nodes, op_node_emb_size])
        self.graph_obj = graph_obj
        self.metapath_list = metapath_list
        self.num_metapaths = len(metapath_list)
        attn_dropout = 0.1
        num_heads = 8
        self.MP_attention = nn.MultiheadAttention(
            embed_dim = embedding_size * num_heads,
            num_heads = num_heads,
            dropout = attn_dropout
        )
        return

    '''
    input_x :
    target node : x_target
    Input a single type
    
    '''
    def forward(
        self,
        node_idx,     # Shape [?]
        mp_type_idx,  # Shape [?, num_mp_instances ]
        node_mp_walks # Shape [?, num_mp_instances, num_hops, 3]
    ) :
        _split_op = torch.split(node_mp_walks, split_size_or_sections=3, dim=-1)
        n1 = _split_op[0]
        n2 = _split_op[1]
        e_n1n2 = _split_op[2]
        n1_vec = self.fc1(self.self.inp_features(n1))
        n2_vec = self.fc1(self.self.inp_features(n2))
        edge_vec = self.W_edge(LT(e_n1n2))

        # Do element-wise multiplication
        n1n2 = torch.mul(n1_vec,n2_vec)
        n1n2e = torch.mul(n1n2,edge_vec)
        # n1n2e should have shape : [ ?, num_mp_instances, num_hops, 1]
        n1n2e = torch.squeeze(n1n2e, dim =-1)               # op shape : [ ?, num_mp_instances, num_hops]

        # ----------------
        # Aggregate each meta path instance
        # ----------------
        mp_agg_0 = torch.mean(
            n1n2e,
            dim = -1,
            keepdim=False
        ) # op shape : [ ?, num_mp_instances]

        # ------------
        # Linear transform each aggregated meta path instance
        # ------------
        #  op shape [?, num_mp_instances]
        mp_agg_1 = self.W_mp_encoder_wt(mp_agg_0, mp_type_idx)

        attn_output, attn_output_weights =  self.MP_attention(
            query,
            key,
            value
        )



        return


