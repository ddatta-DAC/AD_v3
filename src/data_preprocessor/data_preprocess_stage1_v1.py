import os
import sys
import pandas as pd
import numpy as np
import sklearn
import glob
import pickle
import random
from joblib import Parallel, delayed
import yaml
import math
import re
from pandarallel import pandarallel
from collections import Counter
import argparse

pandarallel.initialize(progress_bar=False)
sys.path.append('.')
sys.path.append('./..')
sys.path.append('./../..')

# ==================================== #
# try:
#     import utils_preprocess
# except:
#     from . import utils_preprocess

# -------------------------- #

DATA_SOURCE = None
DIR_LOC = None
CONFIG = None
CONFIG_FILE = 'config_preprocessor_v01.yaml'
id_col = 'PanjivaRecordID'
use_cols = None
freq_bound = None
column_value_filters = None
num_neg_samples = None
save_dir = None
cleaned_csv_subdir = None
hscode_2_col = 'HSCode_2'
label_col =  'y_label'
# -------------------------- #

# -------------------------- #

def get_regex(_type):
    global DIR
    if DIR == 'us_import1':
        if _type == 'train':
            return '.*0[3-6]_2015.csv'
        if _type == 'test':
            return '.*((07)|(08))_2015.csv'

    if DIR == 'us_import2':
        if _type == 'train':
            return '.*0[3-6]_2016.csv'
        if _type == 'test':
            return '.*((07)|(08))_2016.csv'

    return '*.csv'


def get_files(DIR, _type='all'):
    global DATA_SOURCE
    regex = get_regex(_type)
    c = glob.glob(os.path.join(DATA_SOURCE, '*'))

    def glob_re(pattern, strings):
        return filter(re.compile(pattern).match, strings)

    files = sorted([_ for _ in glob_re(regex, c)])
    print('DIR ::', DIR, ' Type ::', _type, 'Files count::', len(files))
    return files


def set_up_config(_DIR=None):
    global DIR
    global CONFIG
    global CONFIG_FILE
    global use_cols
    global freq_bound
    global num_neg_samples_ape
    global save_dir
    global column_value_filters
    global DATA_SOURCE
    global DIR_LOC

    with open(CONFIG_FILE) as f:
        CONFIG = yaml.safe_load(f)

    if _DIR is not None:
        DIR = _DIR
        CONFIG['DIR'] = _DIR
    else:
        DIR = CONFIG['DIR']

    DIR_LOC = re.sub('[0-9]', '', DIR)
    DATA_SOURCE = os.path.join('./../../Data_Raw', DIR_LOC)
    save_dir = CONFIG['save_dir']
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_dir = os.path.join(
        CONFIG['save_dir'],
        DIR
    )

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    cleaned_csv_subdir = os.path.join(
        save_dir,
        CONFIG['cleaned_csv_subdir']
    )

    if not os.path.exists(cleaned_csv_subdir):
        os.mkdir(cleaned_csv_subdir)

    use_cols = CONFIG[DIR]['use_cols']
    freq_bound = CONFIG[DIR]['low_freq_bound']
    column_value_filters = CONFIG[DIR]['column_value_filters']

'''
Replace attribute with id specific to a domain
'''


def replace_attr_with_id(row, attr, val2id_dict):
    val = row[attr]
    if val not in val2id_dict.keys():
        print(attr, val)
        return None
    else:
        return val2id_dict[val]


'''
Converts the train df to ids 
Returns :
col_val2id_dict  { 'col_name': { 'col1_val1': id1,  ... } , ... }
'''


def convert_to_ids(
        df,
        save_dir
):
    global id_col
    global label_col

    feature_columns = list(df.columns)
    feature_columns.remove(id_col)
    try:
        feature_columns.remove(label_col)
    except:
        pass

    feature_columns = list(sorted(feature_columns))
    dict_DomainDims = {}
    col_val2id_dict = {}

    for col in sorted(feature_columns):
        vals = list(set(df[col]))
        vals = list(sorted(vals))

        id2val_dict = {
            e[0]: e[1]
            for e in enumerate(vals, 0)
        }


        val2id_dict = {
            v: k for k, v in id2val_dict.items()
        }
        col_val2id_dict[col] = val2id_dict

        # Replace
        df[col] = df.parallel_apply(
            replace_attr_with_id,
            axis=1,
            args=(
                col,
                val2id_dict,
            )
        )

        dict_DomainDims[col] = len(id2val_dict)

    print(' Feature columns :: ', feature_columns)
    print(' Dict Domain Dims :: ', dict_DomainDims)

    # -------------
    # Save the domain dimensions
    # -------------

    file = 'domain_dims.pkl'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    f_path = os.path.join(save_dir, file)

    with open(f_path, 'wb') as fh:
        pickle.dump(
            dict_DomainDims,
            fh,
            pickle.HIGHEST_PROTOCOL
        )

    file = 'col_val2id_dict.pkl'
    f_path = os.path.join(save_dir, file)

    with open(f_path, 'wb') as fh:
        pickle.dump(
            col_val2id_dict,
            fh,
            pickle.HIGHEST_PROTOCOL
        )

    return df, col_val2id_dict, dict_DomainDims


'''
Join the csv files into 1 single Dataframe
Removes missing values
Input : file_path_list
'''

'''
Remove the rows with entities that have very low frequency.
'''


def remove_low_frequency_values(df):
    global id_col
    global freq_bound
    print(' DF length : ', len(df))
    freq_column_value_filters = {}
    feature_cols = list(df.columns)
    feature_cols.remove(id_col)

    # calculate the number of entities per column
    counter_df = pd.DataFrame(columns=['domain', 'count'])
    for c in feature_cols:
        count = len(set(df[c]))
        counter_df = counter_df.append({
            'domain': c, 'count': count
        }, ignore_index=True)
        z = np.percentile(
            list(Counter(df[c]).values()), 5)
        print(c, count, z)

    counter_df = counter_df.sort_values(by=['count'], ascending=True)
    print(counter_df)
    # ----
    # figure out which entities are to be removed
    # ----
    for domain in list(counter_df['domain']):
        values = list(df[domain])
        freq_column_value_filters[domain] = []
        obj_counter = Counter(values)

        for _item, _count in obj_counter.items():
            if _count < freq_bound:  freq_column_value_filters[domain].append(_item)

        df = df.loc[~df[domain].isin(freq_column_value_filters[domain])].copy()
    for c in feature_cols:
        count = len(set(df[c]))
        print(c, count)
    print(' DF length : ', len(df))
    return df

# ---------
# This should be a custom clean up function
# Taking hscode of type 4 only
# Taking "normal" class as 44
# ---------

def HSCode_clean_select(
        list_df
):
    global CONFIG
    hscode_col = 'HSCode'
    hscode_2_col = hscode_col + '_2'

    def get_HSCode2(_code):
        return str(_code)[:2]

    def remove_dot(_code):
        return _code.replace('.', '')


    def filter_hscode(val):
        if ';' in val:
            val = val.split(';')
            val = [v.strip() for v in val]
            val = list(set(val))
        else:
            val = [val]
        for v in val:
            if v[:1] == '4':
                return v[:6]
        return None

    list_processed_df = []

    for df in list_df:
        df = df.dropna()

        df[hscode_col] = df[hscode_col].astype(str)
        df[hscode_col] = df[hscode_col].parallel_apply(
            remove_dot
        )
        df[hscode_col] = df[hscode_col].parallel_apply(
            filter_hscode
        )
        df = df.dropna()

        df[hscode_2_col] = df[hscode_col].parallel_apply(
            get_HSCode2
        )

        if df is not None and len(df) > 0:
            df = df.dropna()
            list_processed_df.append(df)

    # --------- #
    print([len(_) for _ in list_processed_df])
    # ---------------
    # join the dataframes
    # ---------------
    merged_df = None
    for _df in list_processed_df:
        if merged_df is None:
            merged_df = pd.DataFrame(_df, copy=True)
        else:
            merged_df = merged_df.append(
                _df,
                ignore_index=True
            )
    print(merged_df.columns)
    return merged_df


'''
Apply :: column_value_filters
Remove values which are garbage & known to us
'''


def apply_value_filters(list_df):
    global column_value_filters

    if type(column_value_filters) != bool:
        list_processed_df = []
        for df in list_df:
            for col, val in column_value_filters.items():
                df = df.loc[~df[col].isin(val)]
            list_processed_df.append(df)
        return list_processed_df
    return list_df

# ---------------------------------------------------- #
# Lexically sorted columns
# ---------------------------------------------------- #
def lexical_sort_cols(df):
    global label_col
    global id_col
    rmv_cols = [id_col, label_col]
    feature_columns = list(df.columns)
    for rc in rmv_cols:
        try:
            feature_columns.remove(rc)
        except:
            pass

    feature_columns = list(sorted(feature_columns))

    ord_cols = rmv_cols + feature_columns
    return df[ord_cols]


# ---------------------------------------------------- #
# Clean up training data
# ---------------------------------------------------- #

def clean_train_data():
    global DIR
    global CONFIG
    global DIR_LOC
    files = get_files(DIR, 'train')
    print(files)
    list_df = [pd.read_csv(_file, usecols=use_cols, low_memory=False) for _file in files]

    merged_df = HSCode_clean_select(list_df)
    merged_df = remove_low_frequency_values(merged_df)
    return merged_df


# ------------------------------------------------- #
# set up testing data
# ------------------------------------------------- #


# noinspection PyTypeChecker
def create_data_sets():
    global use_cols
    global DIR
    global save_dir
    global column_value_filters
    global CONFIG
    global DIR_LOC
    global hscode_2_col
    global label_col

    train_df_file = os.path.join(save_dir, 'train_data.csv')
    column_valuesId_dict_file = 'column_valuesId_dict.pkl'
    column_valuesId_dict_path = os.path.join(
        save_dir,
        column_valuesId_dict_file
    )

    # --- Later on - remove using the saved file ---- #
    if os.path.exists(train_df_file) and False:
        train_df = pd.read_csv(train_df_file)
        with open(column_valuesId_dict_path, 'rb') as fh:
            col_val2id_dict = pickle.load(fh)

        return train_df, col_val2id_dict

    train_df = clean_train_data()

    # ===============
    # Non interesting 44
    # Assumption that interesting : non-interesting entity ratio is 1 : 4 (20%)
    # ===============

    # We are targeting Consignee IDs; exclude overlapping
    exclude_consigneeids = set(train_df.loc[
        train_df[hscode_2_col] == '44']['ConsigneePanjivaID'])

    print('exclude consigneeid', len(exclude_consigneeids))
    non_interesting_data = pd.DataFrame(
        train_df.loc[train_df[hscode_2_col] == '44'],
        copy=True
    )

    # get the 3 2-digit HS codes that are not 44
    k = 3
    top_k = list(train_df.loc[train_df[hscode_2_col] != '44'].groupby(
        by= [hscode_2_col]).size().reset_index(
        name='count').sort_values(
        by='count',ascending=False).head(k)[hscode_2_col])

    print(train_df.groupby(by=[hscode_2_col]).size().reset_index(name='count'))

    interesting_data = None
    factor = 3
    for _hscode2 in top_k:
        tmp_df = train_df.loc[
            (train_df[hscode_2_col] == _hscode2) &
            (~train_df['ConsigneePanjivaID'].isin(exclude_consigneeids))
        ]

        all_con = list(set(tmp_df['ConsigneePanjivaID']))
        consignees = np.random.choice(
            all_con,
            size = len(all_con) // (factor*k),
            replace= False
        )

        tmp_df = tmp_df.loc[tmp_df['ConsigneePanjivaID'].isin(consignees)]
        if interesting_data is None:
            interesting_data = tmp_df.copy()
        else:
            interesting_data = interesting_data.append(
                tmp_df,
                ignore_index=True
            )


    print( len(interesting_data) )
    print(interesting_data.columns)
    print( len(non_interesting_data) )
    print(non_interesting_data.columns)
    print(len(set(non_interesting_data['ConsigneePanjivaID'])), len(set(interesting_data['ConsigneePanjivaID'])))

    train_df = interesting_data.append(non_interesting_data,ignore_index=True)
    print(train_df.columns)

    try:
        del train_df['HSCode']
    except:
        pass


    def label_func(_val):
        if  _val == '44': return 0
        else: return 1

    train_df = train_df.rename(
        columns={hscode_2_col: label_col}
    )

    train_df[label_col] = train_df[label_col].parallel_apply(
        label_func
    )

    train_df, col_val2id_dict, domain_dims = convert_to_ids(
        train_df,
        save_dir
    )
    print(train_df.columns)
    # -------------------------------
    # Create a serialized version
    # -------------------------------

    serial_mapping_df_file = os.path.join(
        save_dir,
        'serial_mapping.csv'
    )

    serialized_train_data_file = os.path.join(
        save_dir,
        'serialized_train_data.csv'
    )

    prev_count = 0
    res = []
    for dn, ds in domain_dims.items():
        for eid in range(ds):
            r = [dn, eid, eid + prev_count]
            res.append(r)
        prev_count += ds

    serial_mapping_df = pd.DataFrame(
        data=res,
        columns=['Domain', 'Entity_ID', 'Serial_ID']
    )

    serial_mapping_df.to_csv(
        serial_mapping_df_file,
        index=False
    )

    def convert_to_SerialID(_row, cols):
        row = _row.copy()
        for c in cols:
            val = row[c]
            res = list(
                serial_mapping_df.loc[
                    (serial_mapping_df['Domain'] == c) &
                    (serial_mapping_df['Entity_ID'] == val)]
                ['Serial_ID']
            )
            row[c] = res[0]
        return row

    cols = list(domain_dims.keys())

    serialized_train_data = train_df.parallel_apply(
        convert_to_SerialID,
        axis=1,
        args= (cols,)
    )

    print('Length of train data ', len(train_df))
    train_df = lexical_sort_cols(train_df)
    serialized_train_data = lexical_sort_cols(serialized_train_data)
    train_df.to_csv(train_df_file, index=False)
    serialized_train_data.to_csv(serialized_train_data_file,index=False)

    # -----------------------
    # Save col_val2id_dict
    # -----------------------
    with open(column_valuesId_dict_path, 'wb') as fh:
        pickle.dump(col_val2id_dict, fh, pickle.HIGHEST_PROTOCOL)

    return train_df,  col_val2id_dict


# -------------------------------- #

parser = argparse.ArgumentParser()
parser.add_argument(
    '--DIR', choices=['us_import1', 'us_import2'],
    default='us_import2'
)

args = parser.parse_args()
DIR = args.DIR
# -------------------------------- #

set_up_config(args.DIR)
create_data_sets()
