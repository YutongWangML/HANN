# NOTE: this script does not need to be run.
# The output ("metadata/datasets_summary.csv") of this script has be precomputed.
import os
from datasets import load_trn, load_tst, get_data_list
import pandas as pd
from io import StringIO

L2_dist_est = pd.read_csv("metadata/L2_dist_est.csv",index_col=0)

dataset_names = get_data_list('all')

L2_dist_est.loc['abalone','L2_dist_est']

rows = ["dataset_name,n_samples,n_train,n_test,n_features,n_classes,L2_dist_est"]
for dataset_name in dataset_names:
    D_train = load_trn(dataset_name)
    D_test = load_tst(dataset_name)
    x_train = D_train.data
    y_train = D_train.target
    x_test = D_test.data
    n_test = x_test.shape[0]
    n_train = x_train.shape[0]
    n_samples = n_test + n_train
    n_features = x_train.shape[1]
    n_classes = max(y_train)+1
    dist_est = L2_dist_est.loc[dataset_name, 'L2_dist_est']
    row = [dataset_name,n_samples,n_train,n_test,n_features,n_classes,dist_est]
    row = ",".join(map(str,row))
    rows.append(row)
    
md_str = "\n".join(rows)

df = pd.read_csv(StringIO(md_str),index_col=0)

df.to_csv("metadata/datasets_summary.csv")