import pandas as pd
import numpy as np
import os
import sys
sys.path.append('./..')
sys.path.append('./../..')
import argparse
import pickle
import torch
import stellargraph as sg
try :
    from src.mp2vec import run_m2pv
    from src.node2vec import run_n2v
    from src.hin2vec import run_hin2vec
except:
    from .src.mp2vec import run_m2pv
    from .src.node2vec import run_n2v
    from .src.hin2vec import run_hin2vec
# ------------------- #

'''
Procedure : remove 30% of nodes a particular type
DBLP: P-T
'''
model_use_data_DIR = 'model_use_data'
model_save_data_DIR = 'model_save_data'

def setup(dataset):
    global model_use_data_DIR
    global model_save_data_DIR
    if not os.path.exists(model_use_data_DIR):
        os.mkdir(model_use_data_DIR)
    model_use_data_DIR = os.path.join(model_use_data_DIR, dataset)
    if not os.path.exists(model_use_data_DIR):
        os.mkdir(model_use_data_DIR)

    if not os.path.exists(model_save_data_DIR):
        os.mkdir(model_save_data_DIR)
    model_save_data_DIR = os.path.join(model_save_data_DIR, dataset)
    if not os.path.exists(model_save_data_DIR):
        os.mkdir(model_save_data_DIR)



def create_data(
        _method = 'node2vec',
        node_dict = None,
        train_edges = None,
        test_edges = None,
        data_save_path = None
):
    if not os.path.exists(data_save_path):
        os.mkdir(data_save_path)
    data_save_path = os.path.join(data_save_path,_method)
    if not os.path.exists(data_save_path):
        os.mkdir(data_save_path)

    if _method == 'hin2vec':
        train_edges = train_edges.rename(
            columns={
                'source':'node1',
                'target':'node2'
            }
        )
        test_edges = test_edges.rename(
            columns={
                'source': 'node1',
                'target': 'node2'
            }
        )
        train_edges['node1'] = train_edges['node1'].astype(int)
        train_edges['node2'] = train_edges['node2'].astype(int)
        train_edges['rel'] = train_edges['rel'].astype(int)

        fpath = os.path.join(data_save_path, 'train_edges.txt')
        train_edges.to_csv(fpath, index=None, sep=',')
        test_edges['node1'] = test_edges['node1'].astype(int)
        test_edges['node2'] = test_edges['node2'].astype(int)
        test_edges['rel'] = test_edges['rel'].astype(int)

        fpath = os.path.join(data_save_path, 'test_edges.txt')
        test_edges.to_csv(fpath, index=None, sep=',')

        # Save the nodes
        df = pd.DataFrame.from_dict({
                'node_id': list(node_dict.keys()),
                'node_type': list(node_dict.values())
            },
            orient='columns'
        )
        fpath = os.path.join(data_save_path, 'nodes.csv')
        df.to_csv(fpath,index=False)

    elif  _method =='mp2vec' or _method == 'node2vec':
        dict_node_df = {}
        for node_type in set(node_dict.values()):
            _list = [ k for k,v in node_dict.items() if v == node_type ]
            _df = pd.DataFrame(data =_list, columns=[node_type])
            _df = _df.set_index(node_type)
            dict_node_df[node_type] = _df
        try:
            del train_edges['rel']
            del test_edges['rel']
        except:
            pass

        fpath = os.path.join(data_save_path, 'train_edges.csv')
        train_edges.to_csv(fpath,index=False)
        fpath = os.path.join(data_save_path, 'test_edges.csv')
        test_edges.to_csv(fpath, index=False)
        # ----------------
        # Save the dict
        # ----------------
        with open(os.path.join(data_save_path,'dict_node_df.pkl'),'wb') as fh:
            pickle.dump(
                dict_node_df, fh, pickle.HIGHEST_PROTOCOL
            )

        return dict_node_df, train_edges, test_edges
    else:
        print('Not supported ::', _method)

    return


def prepare_data(dataset):
    global model_use_data_DIR
    global model_save_data_DIR
    node_dict = None
    train_edges = None
    test_edges = None

    if dataset == 'dblp':
        base_train_edges_file = 'base_train_edges.csv'
        base_test_edges_file = 'base_test_edges.csv'
        input_dir = './../../dblp/processed_data/'

        # =====
        # Check if edges file is saved
        # =====

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
            _df = _df.reset_index(drop=True)
            ids = _df[list(_df.columns)[0]]
            for e in ids:
                node_dict[e] = _type

        print('Number of nodes ::', len(node_dict))
        edges_df = None
        e_type_idx = 0
        for e_type, file in edge_files.items():
            file = os.path.join(input_dir,file)
            if e_type == target_edge_type: continue
            _df = _df = pd.read_csv (file, index_col=None)
            _df['rel'] = e_type_idx
            if edges_df is None: edges_df = _df
            else :  edges_df = edges_df.append(_df,ignore_index=True)
            e_type_idx += 1

        target_df = pd.read_csv(
            os.path.join(input_dir, edge_files[target_edge_type]),
            index_col=None
        )

        target_df['rel'] = e_type_idx
        test_size = int(len(target_df) * 0.3)
        print('Test size ::', test_size)
        # =====
        # Sample edges such that none of the nodes in edges are not of degree 1
        # i.e. keep the graph connected
        # =====

        if os.path.exists(base_test_edges_file) and os.path.exists(base_train_edges_file):
            train_edges = pd.read_csv(base_train_edges_file, index_col=None)
            test_edges =  pd.read_csv(base_test_edges_file, index_col=None)
        else:
            target_nodes = list(set(target_df['target']))
            keep_links_df = pd.DataFrame( columns=list(target_df.columns) )
            test_cand = pd.DataFrame( columns=list(target_df.columns))

            for i,row in target_df.iterrows():
                if row['target'] in target_nodes:
                    keep_links_df = keep_links_df.append(row, ignore_index=True)
                    target_nodes.remove(row['target'])
                else:
                    test_cand = test_cand.append(row, ignore_index=True)

            test_edges = test_cand.head(test_size)
            train_size = len(test_cand) - test_size
            train = test_cand.tail(train_size)
            train_edges = train.append(keep_links_df)
            train_edges = train_edges.append(edges_df)
            train_edges.to_csv(base_train_edges_file,index=False)
            test_edges.to_csv(base_test_edges_file, index=False)

    print('Train edges : ', len(train_edges), 'Test edges : ', len(test_edges))

    '''
    Set up data for each method 
    1. node2vec
    2. metapath2vec
    3. hin2vec
    '''
    create_data(
        _method='hin2vec',
        train_edges = train_edges,
        test_edges = test_edges,
        node_dict = node_dict,
        data_save_path=model_use_data_DIR
    )

    create_data(
        _method='node2vec',
        train_edges=train_edges,
        test_edges=test_edges,
        node_dict=node_dict,
        data_save_path=model_use_data_DIR
    )


    create_data(
        _method='mp2vec',
        train_edges=train_edges,
        test_edges=test_edges,
        node_dict=node_dict,
        data_save_path=model_use_data_DIR
    )
    return

def exec(_dataset,  _method ):
    global model_use_data_DIR
    global model_save_data_DIR

    if not os.path.exists(os.path.join(model_save_data_DIR, _method)):
        os.path.join(model_save_data_DIR, _method)
    data_src_dir = os.path.join(model_use_data_DIR, _method)

    # ==================================== #
    if _method == 'node2vec':
        print('Running Node2vec')
        run_n2v.setup(
            _dataset,
            _model_save_path=model_save_data_DIR,
            _model_use_data_DIR=model_use_data_DIR
        )

        with open (os.path.join(data_src_dir,'dict_node_df.pkl'),'rb') as fh :
            dict_node_df = pickle.load(fh)

        train_edges = pd.read_csv(
            os.path.join(data_src_dir,'train_edges.csv'),
            index_col=None
        )
        node_embeddings = run_n2v.execute_model(
            dict_node_df,
            train_edges
        )
        output_file_name = os.path.join(model_save_data_DIR, _method, 'embeddings.npy')
        np.save(output_file_name, node_embeddings)

    elif _method == 'mp2vec':
        print('Running metapath2vec')
        run_m2pv.setup(
            _dataset,
            _model_save_path=model_save_data_DIR,
            _model_use_data_DIR=model_use_data_DIR
        )
        with open(os.path.join(data_src_dir, 'dict_node_df.pkl'), 'rb') as fh:
            dict_node_df = pickle.load(fh)
        train_edges = pd.read_csv(
            os.path.join(data_src_dir, 'train_edges.csv'),
            index=None
        )
        node_embeddings = run_m2pv.execute_model(
            dict_node_df,
            train_edges
        )
        output_file_name = os.path.join(model_save_data_DIR, _method, 'embeddings.npy')
        np.save(output_file_name, node_embeddings)

    elif _method == 'hin2vec':
        print('Running HN2vec')
        src_dir = os.path.join(model_use_data_DIR, _method )
        input_file_name = os.path.join(src_dir,'train_edges.txt')
        output_file_name = os.path.join(model_save_data_DIR, _method, 'embeddings.npy')
        run_hin2vec.exec(
            _dataset,
            input_file_name=input_file_name,
            output_file_name=output_file_name,
            model_use_data_DIR=None
        )
    else:
        print('Invalid method!!', _method)

parser = argparse.ArgumentParser()
parser.add_argument(
    '--dataset', choices=['dblp'],
    default='dblp'
)
parser.add_argument(
    '--method', choices=['node2vec','hin2vec','mp2vec'],
    default='node2vec'
)
args = parser.parse_args()
_dataset = args.dataset
_method = args.method

setup(_dataset)
prepare_data(_dataset)
exec(_dataset=_dataset, _method=_method )