#!/usr/bin/env python
#-*- coding: utf-8 -*-

# ---------------
# Author : Debanjan Datta
# Email : ddatta@vt.edu
# ---------------

import numpy as np
import networkx as nx
import pandas as pd
import pickle
import stellargraph
import os
from pandarallel import pandarallel
pandarallel.initialize()
import multiprocessing
from multiprocessing import Pool
from stellargraph.data import UniformRandomMetaPathWalk

class graph_container():
    metapaths = None
    node_id2type_dict = None
    sg_obj = None
    node_type_ids = None

    def __init__(
            self,
            node_df, # node_id, type
            edge_df  # format : source, target
    ):
        node_types = list(sorted(set(node_df['type'])))

        # node_type : node_type_id
        # node_types = {
        #     e[1] : e[0] for e in enumerate(self.node_types)
        # }
        print(' node_type_ids ', node_types)

        # Set static variable
        graph_container.node_types = node_types

        # Convert node type to type ids
        # def replace_with_node_type_id(val):
        #     return node_type_ids[val]
        #
        # node_df['type'] = node_df['type'].parallel_apply(replace_with_node_type_id)

        i = node_df['node_id']
        j = node_df['type']
        node_id2type_dict = { _i:_j for _i,_j in zip(i,j)}
        # Set static variable
        graph_container.node_id2type_dict = node_id2type_dict
        node_dict = {}

        for nt in set(node_df['type']):
            tmp =  node_df.loc[node_df['type']==nt].reset_index(drop=True)
            tmp = tmp.set_index('node_id')
            tmp = tmp.rename(columns={'node_id': str(nt)})
            del tmp['type']
            node_dict [nt] = tmp

        def set_e_type(row):
            t1 = node_id2type_dict[row['source']]
            t2 = node_id2type_dict[row['target']]
            et = '_'.join(list(sorted((str(t1),str(t2)))))
            return et

        edge_df['type'] = edge_df.parallel_apply(set_e_type,axis=1)
        # Create numeric ids for each edge type
        self.edge_type2id_dict = { e[1]:e[0] for e in enumerate(set(edge_df['type']))}

        def replace_with_edge_id(val):
            return self.edge_type2id_dict[val]

        edge_df['type'] = edge_df['type'].parallel_apply(replace_with_edge_id)

        graph_container.sg_obj = stellargraph.StellarGraph(
            node_dict,
            edge_df,
            edge_type_column="type"
        )
        return

    @staticmethod
    def _get_nbr_by_type( node_id, node_type=None):
        nodes = graph_container.sg_obj.neighbors(node=node_id)
        nodes = [
            _ for _ in nodes
            if  graph_container.node_id2type_dict[_] == node_type
        ]
        return nodes

    @staticmethod
    def _get_edge_type_count():
        return len(graph_container.sg_obj.edge_types)

    @staticmethod
    def _get_node_types():
        return len(graph_container.node_type_ids)

    # ===========
    # list of list of integers
    # ===========
    @staticmethod
    def _set_metapaths(MP_list):
        graph_container.metapaths = MP_list
        return

    @staticmethod
    def _get_node_type(node_id):
        return str(graph_container.node_id2type_dict[node_id])


    @staticmethod
    def _aux_get_mp_nbrs(
            node_id,
            num_samples=-1
    ):
        node_type = graph_container._get_node_type(node_id)
        metapaths = graph_container.metapaths
        valid_mps = [_ for _ in metapaths if _[0] == node_type]

        all_walks = []
        rw_obj = UniformRandomMetaPathWalk(
            graph_container.sg_obj,
            metapaths=valid_mps
        )

        for mp in valid_mps:
            walk_len = len(mp)

            walks = rw_obj.run(
                [node_id],
                length=walk_len,
                n=num_samples,
                metapaths=[mp]
            )

            all_walks.extend(walks)
        return all_walks

    @staticmethod
    def _get_metapath_nbrs(
            node_id_list,
            num_samples
    ):
        n_jobs = multiprocessing.cpu_count()
        pool = Pool(processes=n_jobs)
        results = [ pool.apply_async(
            graph_container._aux_get_mp_nbrs,
            args=( node_id, num_samples,)) for node_id in node_id_list ]
        output = [p.get() for p in results]
        all_walks = []
        for op in output:
            all_walks.extend(op)
        return all_walks
