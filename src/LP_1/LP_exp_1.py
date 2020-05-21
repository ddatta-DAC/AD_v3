import pandas as pd
import numpy as np
import os
import sys
sys.path.append('./../..')
import pickle
import torch
import stellargraph as sg

# ------------------- #

'''
Procedure : remove 30% of nodes a particular type
DBLP: P-T
'''
model_use_data_DIR = 'model_use_data'

def setup(dataset):
    global model_use_data_DIR
    if os.path.exists(model_use_data_DIR):
        os.mkdir(model_use_data_DIR)
    model_use_data_DIR = os.path.join(model_use_data_DIR, dataset)
    if os.path.exists(model_use_data_DIR):
        os.mkdir(model_use_data_DIR)

def prepare_data(dataset):

    node_dict = None
    train_edges = None
    test_edges = None

    if dataset == 'dblp':
        input_dir = './../../dblp/processed_data/DBLP'
        target_edge_type = 'PT'

        edge_files = {
            'PA' : 'PA_edges.csv',
            'PC' : 'PC_edges.csv',
            'PT' : 'PT_edges.csv'
        }

        node_files = {
            'P': 'nodes_paper.csv',
            'A': 'nodes_author.csv',
            'T': 'nodes_term.csv',
            'C': 'nodes_conf.csv'
        }

        node_dict = {}
        for _type, file in node_files.items() :
            file = os.path.join(input_dir, file)
            _df = pd.read_csv(
                file
            )
            _df = _df.reset_index()
            ids = _df[list(_df.columns)[0]]
            for e in ids:
                node_dict[e] = _type

        edges_df = None
        for k, file in edge_files.items():
            file = os.path.join(input_dir,file)
            if k == target_edge_type: continue
            _df =  _df = pd.read_csv(
                file, index_col=None
            )
            if edges_df is None:
                edges_df = _df
            else :
                edges_df = edges_df.append(_df,ignore_index=True)

        # =====
        # Sample edges such that none of the nodes in edges are not of degree 1
        # i.e. keep the graph connected
        # =====
        valid_nodes = list(edges_df['source'])
        valid_nodes.extend(edges_df['target'])
        valid_nodes = list(set(valid_nodes))

        target_df = pd.read_csv(
            os.path.join(input_dir, edge_files[target_edge_type]),
            index_col=None
        )

        test_size = int(len(target_df) * 0.3)
        print('Test size ::', test_size)
        df1 = target_df.loc[
            (target_df['source'].isin(valid_nodes)) |
            (target_df['target'].isin(valid_nodes))
        ]

        df2 = target_df.loc[
            (~target_df['source'].isin(valid_nodes)) &
            (~target_df['target'].isin(valid_nodes))
        ]

        df1 = df1.sample(frac=1).reset_index(drop=True) # Shuffle
        train_size = len(df1) - test_size
        test_edges = df1.head(test_size)
        train = df1.tail(train_size)
        train = train.append(df2)
        train_edges = train.append(edges_df)
        print('Train edges : ', len(train_edges), 'Test edges : ', len(test_edges))

    print(node_dict)
    '''
    Set up data for each method 
    1. node2vec
    2. metapath2vec
    3. hin2vec
    '''






prepare_data('dblp')