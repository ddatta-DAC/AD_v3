import pandas as pd
import numpy as np
import joblib
from joblib import delayed, Parallel

def create_coocc_matrix(*arg_list):
    df, col_1, col_2 = arg_list
    set_elements_1 = set(list(df[col_1]))
    set_elements_2 = set(list(df[col_2]))
    count_1 = len(set_elements_1)
    count_2 = len(set_elements_2)
    coocc = np.zeros([count_1, count_2])
    df = df[[col_1, col_2]]
    new_df = df.groupby([col_1, col_2]).size().reset_index(name='count')

    for _, row in new_df.iterrows():
        i = row[col_1]
        j = row[col_2]
        coocc[i][j] = row['count']

    print('Col 1 & 2', col_1, col_2, coocc.shape, '>>', (count_1, count_2))
    key = col_1 + '_+_' + col_2
    return (key,coocc)


'''
Create co-occurrence between entities using training data. 
Returns a dict { Domain1_+_Domain2 : __matrix__ }
Domain1 and Domain2 are sorted lexicographically
'''


def get_coOccMatrix_dict(df, feature_cols):

    feature_cols = list(sorted(feature_cols))
    columnWise_coOccMatrix_dict = {}
    arg_list = []
    for i in range(len(feature_cols)):
        for j in range(i + 1, len(feature_cols)):
            col_1 = feature_cols[i]
            col_2 = feature_cols[j]

            arg_list.append((df, col_1, col_2))

    n_jobs = len(arg_list)
    results = Parallel(n_jobs=n_jobs)(
        delayed(create_coocc_matrix)(*_args)
        for _args in arg_list
    )

    for res in results:
        key = res[0]
        matrix = res[1]
        columnWise_coOccMatrix_dict[key] = matrix

    return columnWise_coOccMatrix_dict



