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
    return


# ========================================================= #

def get_data():
    global DIR
    global SOURCE_DATA_DIR_1
    fpath = os.path.join (
        SOURCE_DATA_DIR_1,
        'train_csv.csv'
    )

    train_df = pd.read_csv(
        fpath, index_col=None
    )

    fpath = os.path.join(
        SOURCE_DATA_DIR_1,
        'domain_dims.pkl'
    )

    with open(fpath,'rb') as fh:
        domain_dims = pickle.load(fh)

    fpath = os.path.join(
        SOURCE_DATA_DIR_1,
        'serialized_train_data.csv'
    )

    serialized_train_df = pd.read_csv(
        fpath, index_col=None
    )

    return train_df, serialized_train_df, domain_dims


