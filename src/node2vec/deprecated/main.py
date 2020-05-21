import yaml
import pandas as pd
import os
import numpy as np
import sys

sys.path.append('./../..')
from pandarallel import pandarallel

pandarallel.initialize()
import joblib
import multiprocessing
import networkx as nx
from node2vec import Node2Vec
import pickle
import argparse

try:
    import utils
except:
    from . import utils
# ------------------------------------------ #

DIR = None
CONFIG_FILE = 'config.yaml'
CONFIG = None
SOURCE_DATA_DIR = None
SAVE_DATA_DIR = None
id_col = None
model_use_data_DIR = None
serial_mapping_df_file = None
serial_mapping_df = None
SOURCE_DATA_DIR_1 = None
label_col = None


# ------------------------------------------ #
# Set up configuration
# ------------------------------------------ #

def set_up_config(_DIR=None):
    global CONFIG
    global SOURCE_DATA_DIR_1
    global DIR
    global CONFIG_FILE
    global SAVE_DIR
    global id_col
    global SAVE_DATA_DIR
    global model_use_data_DIR
    global serial_mapping_df_file
    global label_col
    global serial_mapping_df
    DIR = _DIR

    with open(CONFIG_FILE) as f:
        CONFIG = yaml.safe_load(f)

    if _DIR is None:
        DIR = CONFIG['DIR']
    else:
        DIR = _DIR

    SOURCE_DATA_DIR_1 = os.path.join(
        CONFIG['SOURCE_DATA_DIR_1'], DIR
    )

    if not os.path.exists(CONFIG['model_use_data_DIR']):
        os.mkdir(CONFIG['model_use_data_DIR'])

    model_use_data_DIR = os.path.join(
        CONFIG['model_use_data_DIR'],
        DIR
    )
    if not os.path.exists(model_use_data_DIR):
        os.mkdir(model_use_data_DIR)

    id_col = CONFIG['id_col']
    label_col = CONFIG['label_col']
    mapping_df_file = 'serial_mapping.csv'

    serial_mapping_df_file = os.path.join(
        SOURCE_DATA_DIR_1,
        mapping_df_file
    )

    serial_mapping_df = pd.read_csv(
        serial_mapping_df_file,
        index_col=None
    )
    return


# ========================================================= #

# noinspection PyTypeChecker
def get_data():
    global DIR
    global SOURCE_DATA_DIR_1
    fpath = os.path.join(
        SOURCE_DATA_DIR_1,
        'train_data.csv'
    )

    train_df = pd.read_csv(
        fpath, index_col=None
    )

    fpath = os.path.join(
        SOURCE_DATA_DIR_1,
        'domain_dims.pkl'
    )

    with open(fpath, 'rb') as fh:
        domain_dims = pickle.load(fh)

    fpath = os.path.join(
        SOURCE_DATA_DIR_1,
        'serialized_train_data.csv'
    )

    serialized_train_df = pd.read_csv(
        fpath, index_col=None
    )

    return train_df, serialized_train_df, domain_dims


def get_edge_types():
    et_list = []
    with open('edge_type_list.txt', 'r') as fh:
        lines = fh.readlines()
        for line in lines:
            line = line.strip(',')
            line = line.strip()
            et_list.append(list(sorted(line.split(','))))

    return et_list


def create_graph(serialized_train_df, train_df, domain_dims):
    global serial_mapping_df
    edge_types = get_edge_types()
    serial_map_dict = {}
    feature_cols = list(domain_dims.keys())
    for dom in domain_dims.keys():
        tmp = serial_mapping_df.loc[serial_mapping_df['Domain'] == dom]
        j = tmp['Serial_ID']
        i = tmp['Entity_ID']
        _dict = {_i: _j for _i, _j in zip(i, j)}
        serial_map_dict[dom] = _dict
    # Get co-occurrence dictionary
    coOccMatrix_dict = utils.get_coOccMatrix_dict(train_df, feature_cols)
    edge_list = []
    for edge_type in edge_types :
        dom1 = edge_type[0]
        dom2 = edge_type[1]
        key = edge_type[0] + '_+_' + edge_type[1]
        matrix = coOccMatrix_dict[key]
        for i in range(matrix.shape[0]):
            serial_i = serial_map_dict[dom1][i]
            for j in range(matrix.shape[1]):
                if matrix[i][j] == 0 : continue
                w = matrix[i][j]
                serial_j = serial_map_dict[dom2][j]
                edge_list.append((serial_i,serial_j,w))

    G = nx.Graph()
    G.add_weighted_edges_from(edge_list)
    print(G.number_of_edges())
    print(G.number_of_nodes())
    return G


def main_exec():
    train_df, serialized_train_df, domain_dims = get_data()
    # Create graph
    create_graph(serialized_train_df, train_df, domain_dims)


parser = argparse.ArgumentParser()
parser.add_argument(
    '--DIR', choices=['us_import1', 'us_import2'],
    default='us_import1'
)

args = parser.parse_args()
DIR = args.DIR
# -------------------------------- #

set_up_config(args.DIR)
main_exec()
