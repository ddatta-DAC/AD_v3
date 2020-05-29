import pandas as pd
import numpy as np
import os
import sys
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.linear_model import LogisticRegression

sys.path.append('./..')
sys.path.append('./../..')
import argparse
import pickle
from pandarallel import pandarallel
pandarallel.initialize()
from joblib import Parallel,delayed
import multiprocessing
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
    return


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

def create_neg_test_samples(dataset):

    global model_use_data_DIR
    op_file_path = os.path.join(model_use_data_DIR, 'base_neg_test_edges.csv')
    base_train_edges_file = os.path.join(model_use_data_DIR, 'base_train_edges.csv')
    base_test_edges_file = os.path.join(model_use_data_DIR, 'base_test_edges.csv')

    if dataset == 'dblp':
        input_dir = './../../dblp/processed_data/'
        train_edges = pd.read_csv(base_train_edges_file, index_col=None)
        test_edges = pd.read_csv(base_test_edges_file, index_col=None)

        source_nodes_file = os.path.join(input_dir, 'nodes_paper.csv')
        target_nodes_file = os.path.join(input_dir, 'nodes_term.csv')

        # ======================
        # target_edge_type = PT
        # source : P target : T
        # =====================
        _df = pd.read_csv( source_nodes_file )
        _df = _df.reset_index(drop=True)
        col = list(_df.columns)[0]
        source_set = _df[col]
        _df = pd.read_csv( target_nodes_file )
        _df = _df.reset_index(drop=True)
        col = list(_df.columns)[0]
        target_set = _df[col]
        size = len(test_edges)
        ref_df = train_edges.copy()
        ref_df = ref_df.append(test_edges,ignore_index=True)

        def aux_gen_neg_test( source_set, target_set, ref_df ):
            s = None
            t = None
            trial = 0
            while True:
                trial += 1
                s = np.random.choice(source_set,1)[0]
                t = np.random.choice(target_set,1)[0]
                if len(ref_df.loc[ (ref_df['source']==s) & (ref_df['target']==t) ]) == 0:
                    break
            return [s,t]

        n_jobs =  multiprocessing.cpu_count()
        res = Parallel(n_jobs = n_jobs)(delayed(
            aux_gen_neg_test
        )( source_set, target_set, ref_df) for _ in range(size))

        arr = np.array(res)
        df = pd.DataFrame(data=arr,columns=['source','target'])

        df.to_csv(op_file_path)
        return

def prepare_data(dataset):
    global model_use_data_DIR
    node_dict = None
    train_edges = None
    test_edges = None
    test_ratio = 0.2

    if dataset == 'dblp':
        base_train_edges_file = os.path.join(model_use_data_DIR, 'base_train_edges.csv')
        base_test_edges_file = os.path.join(model_use_data_DIR,'base_test_edges.csv')
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
        test_size = int(len(target_df) * test_ratio)
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

        base_test_neg_edges_file = os.path.join(model_use_data_DIR, 'base_neg_test_edges.csv')
        if not os.path.exists(base_test_neg_edges_file):
            create_neg_test_samples(_dataset)

    elif dataset == 'us_import1':
        base_train_edges_file = os.path.join(model_use_data_DIR, 'base_train_edges.csv')
        base_test_edges_file = os.path.join(model_use_data_DIR, 'base_test_edges.csv')
        input_dir = './../../us_import1/processed/'

        target_edge_types = [
            ['PortOfLading', 'ShipperPanjivaID'],
            ['PortOfUnlading', 'ConsigneePanjivaID'],
            ['ShipmentDestination', 'ConsigneePanjivaID'],
            ['ShipperPanjivaID', 'ShipmentOrigin']
        ]
        target_edge_types = ['_'.join(list(sorted(_))) for _ in  target_edge_types]

        node_types = ['ConsigneePanjivaID', 'HSCode', 'PortOfLading', 'PortOfUnlading', 'ShipmentDestination',
                      'ShipmentOrigin', 'ShipperPanjivaID']
        node_files = {node_type: 'nodes_{}.csv'.format(node_type) for node_type in node_types}
        node_dict = {}

        for _type, file in node_files.items():
            file = os.path.join(input_dir, file)
            _df = pd.read_csv(
                file
            )
            _df = _df.reset_index(drop=True)
            ids = _df[list(_df.columns)[0]]
            for e in ids:
                node_dict[e] = _type


        edge_types = [
            ['ShipperPanjivaID', 'HSCode'],
            ['ConsigneePanjivaID', 'HSCode'],
            ['ShipperPanjivaID', 'ConsigneePanjivaID'],
            ['PortOfLading', 'ShipperPanjivaID'],
            ['PortOfLading', 'PortOfUnlading'],
            ['PortOfUnlading', 'ConsigneePanjivaID'],
            ['ShipmentDestination', 'PortOfUnlading'],
            ['PortOfLading', 'ShipmentOrigin'],
            ['ShipmentDestination', 'ConsigneePanjivaID'],
            ['ShipperPanjivaID','ShipmentOrigin']
        ]
        edge_types = ['_'.join(list(sorted(_))) for _ in  edge_types]
        edge_files = {}

        for edge_type in edge_types:
            # key = '_'.join(sorted(edge_type))
            fname = edge_type + '_edges.csv'
            edge_files[edge_type] =  fname

        node_dict = {}
        for _type, file in node_files.items():
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
            file = os.path.join(input_dir, file)
            # TODO : Check
            if e_type in target_edge_types:
                continue
            _df = _df = pd.read_csv(file, index_col=None)
            _df['rel'] = e_type_idx
            if edges_df is None:
                edges_df = _df
            else:
                edges_df = edges_df.append(_df, ignore_index=True)
            e_type_idx += 1


        target_df =  None
        for target_edge_type in target_edge_types:
            _df = pd.read_csv(
                os.path.join(input_dir, edge_files[target_edge_type]),
                index_col=None
            )
            _df['rel'] = e_type_idx
            e_type_idx += 1
            if target_df is None:
                target_df = _df
            else:
                target_df = target_df.append(_df,ignore_index =True)


        test_size = int(len(target_df) * test_ratio)
        print('Test size ::', test_size)
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

        base_test_neg_edges_file = os.path.join(model_use_data_DIR, 'base_neg_test_edges.csv')
        if not os.path.exists(base_test_neg_edges_file):
            create_neg_test_samples(_dataset)


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

def exec(_dataset, _method):
    global model_use_data_DIR
    global model_save_data_DIR

    if not os.path.exists(os.path.join(model_save_data_DIR, _method)):
        os.mkdir(os.path.join(model_save_data_DIR, _method))

    data_src_dir = os.path.join(model_use_data_DIR, _method)
    output_file_name = os.path.join(model_save_data_DIR, _method, 'embeddings.npy')
    # ==================================== #
    node_emb = None
    if  os.path.exists(output_file_name):
        if _method == 'node2vec' or _method == 'mp2vec':
            node_emb = np.load(output_file_name,allow_pickle=True)
        elif _method == 'hin2vec' :
            emb = np.load(output_file_name, allow_pickle=True)
            node_emb = emb[0]
            rel_emb = emb[1]
    else:
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

            np.save(output_file_name, node_embeddings)
            node_emb = np.load(output_file_name, allow_pickle=True)
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
                index_col=None
            )
            node_embeddings = run_m2pv.execute_model(
                dict_node_df,
                train_edges
            )

            np.save(output_file_name, node_embeddings)
            node_emb = np.load(output_file_name, allow_pickle=True)
        elif _method == 'hin2vec':
            print('Running HN2vec')
            src_dir = os.path.join(model_use_data_DIR, _method )
            input_file_name = os.path.join(src_dir,'train_edges.txt')

            run_hin2vec.exec(
                _dataset,
                input_file_name=input_file_name,
                output_file_name=output_file_name,
                model_use_data_DIR=None
            )
            emb = np.load(output_file_name, allow_pickle=True)
            node_emb = emb[0]
            rel_emb = emb[1]
        else:
            print('Invalid method!!', _method)

    eval_LP ( node_emb, _dataset )
    return node_emb

# ============================================== #
# def calculate_P(node_emb, node1, node2):
#     e1 = node_emb[node1]
#     e2 = node_emb[node2]
#     # p = Sigmoid( e1 . e2 )
#     z = 1 / (1 + np.exp(-np.dot(e1,e2)))
#     return z

# ====================
# Generate 0 label data for classifier that does LP
# ====================
def  generate_LP_clf_neg_samples(
            source_nodes,
            target_nodes,
            exclude_df,
            size
        ):

    def aux_gen_neg_test(source_set, target_set, ref_df):
        s = None
        t = None
        trial = 0
        while True:
            trial += 1
            s = np.random.choice(source_set, 1)[0]
            t = np.random.choice(target_set, 1)[0]
            if len(ref_df.loc[(ref_df['source'] == s) & (ref_df['target'] == t)]) == 0:
                break
        return [s, t]

    n_jobs= multiprocessing.cpu_count()
    res = Parallel(n_jobs=n_jobs)(delayed(
        aux_gen_neg_test
    )(source_nodes, target_nodes, exclude_df) for _ in range(size))

    arr = np.array(res)
    df = pd.DataFrame(data=arr, columns=['source', 'target'])
    return df

def eval_LP(node_emb, dataset):
    global model_use_data_DIR

    def calc_edge_feature(row, node_emb):
        return np.multiply(node_emb[int(row['source'])], node_emb[int(row['target'])])

    test_neg_path = os.path.join(model_use_data_DIR, 'base_neg_test_edges.csv')
    test_pos_path = os.path.join(model_use_data_DIR, 'base_test_edges.csv')
    source_nodes = None
    target_nodes = None

    if dataset == 'dblp':
        input_dir = './../../dblp/processed_data/'
        source_nodes_file = os.path.join(input_dir, 'nodes_paper.csv')
        target_nodes_file = os.path.join(input_dir, 'nodes_term.csv')

        # ======================
        # source : P target : T
        # =====================
        _df = pd.read_csv(source_nodes_file)
        _df = _df.reset_index(drop=True)
        col = list(_df.columns)[0]
        source_nodes = _df[col]
        _df = pd.read_csv(target_nodes_file)
        _df = _df.reset_index(drop=True)
        col = list(_df.columns)[0]
        target_nodes = _df[col]

    # ------------
    # Train a classifier
    # ------------
    lp_clf_train_file = os.path.join(model_use_data_DIR,' lp_clf_train.csv')
    if not os.path.exists(lp_clf_train_file):
        lp_train_pos_df = pd.read_csv(
            os.path.join(model_use_data_DIR,'base_train_edges.csv'),
            index_col=None
        ).sample(frac=1)

        size = len(lp_train_pos_df)

        df_p = pd.read_csv(test_pos_path, index_col=None)
        df_n = pd.read_csv(test_neg_path, index_col=None)
        exclude_df = lp_train_pos_df.append(
            df_p,ignore_index=True).append(df_n,ignore_index=True)
        lp_train_neg_df = generate_LP_clf_neg_samples(
            source_nodes,
            target_nodes,
            exclude_df,
            size
        )
        lp_train_pos_df['y'] =1
        lp_train_neg_df['y'] = 0
        lp_clf_train = lp_train_pos_df.append(lp_train_neg_df,ignore_index=True)
        lp_clf_train.to_csv(lp_clf_train_file,index=False)
    else:
        lp_clf_train = pd.read_csv(lp_clf_train_file,index_col=None)

    x1 = list(lp_clf_train['source'].astype(int))
    x2 = list(lp_clf_train['target'].astype(int))
    x1 = node_emb[x1]
    x2 = node_emb[x2]
    X_train = np.multiply(x1, x2)
    clf_obj = LogisticRegression()
    y = lp_clf_train['y']
    clf_obj.fit(X_train, y)

    # ===== Actual evaluation ==== #

    df_p = pd.read_csv(test_pos_path,index_col=None)
    df_n = pd.read_csv(test_neg_path,index_col=None)
    df_p['y_true'] = 1
    df_n['y_true'] = 0
    df_test = df_p.append(df_n,ignore_index=True)

    x1 = list(df_test['source'].astype(int))
    x2 = list(df_test['target'].astype(int))
    x1 = node_emb[x1]
    x2 = node_emb[x2]
    X_test = np.multiply(x1,x2)
    predict_proba = clf_obj.predict_proba(X_test)
    predict_proba = np.reshape(predict_proba[:,1],-1)

    df_test['y_pred'] = predict_proba
    y_true = list(df_test['y_true'])
    y_scores = list(predict_proba)
    roc = roc_auc_score(y_true, y_scores)
    print(' >>  AUC ROC ', roc)
    fpr, tpr, thresholds = roc_curve( y_true, y_scores)



# ============================================== #
parser = argparse.ArgumentParser()
parser.add_argument(
    '--dataset', choices=['dblp','us_import1'],
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