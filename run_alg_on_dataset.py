# Usage example
# python run_algorithm_on_dataset.py --algorithm WWSVMGaussianKernel --dataset iris

import importlib
import argparse
import UCI_classification.datasets as datasets
import numpy as np
import os
import timeit



parser = argparse.ArgumentParser()

parser.add_argument('--dataset',
                    type = str,
                    help = 'Dataset directory name, e.g., "abalone". See the "data/" directory.')

parser.add_argument('--algorithm',
                    type = str,
                    help = 'Algorithm name, e.g., "SVMGaussianKernel". See "algorithms/" directory.')

FLAGS = parser.parse_args()


if __name__ == '__main__':
    print(FLAGS.dataset)
    dataset_name = FLAGS.dataset
    algorithm_name = FLAGS.algorithm
    DS_trn = datasets.load_trn(dataset_name)
    DS_tst = datasets.load_tst(dataset_name)

    cv_folds = DS_trn.cv_folds


    X_train = DS_trn.data
    y_train = DS_trn.target
    X_test = DS_tst.data
    y_test = DS_tst.target
    
    # only one dataset has a bad single bad CV fold, namely 'low-res-spect'
    # out of the 4 cv folds, the second fold has only 8 classes while all others have 9 classes
    # the following code eliminates only this bad example
    n_classes = len(np.unique(np.array(y_train)))
    good_cv_folds = [(train,val) for train,val in cv_folds if len(np.unique(np.array(y_train[train]))) == n_classes]
    
    
    mod = importlib.import_module("algorithms."+algorithm_name)
    mod.clf.fit(X_train, y_train, X_test, y_test, cv_folds = good_cv_folds)
