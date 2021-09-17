from sklearn.utils import Bunch
import pandas as pd
import numpy as np

def get_data_list(name):
    return open(ROOTDIR+"/data_lists/"+ name + ".txt", "r").read().splitlines()


# Load the UCI datasets in the sklearn format
# For example, see https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html
def load_trn(dataset_name,return_X_y=False):
    """ Load the training datasets """
    data_dir = ROOTDIR+"/data/" + dataset_name + "/"
    data_file = data_dir + dataset_name + "_py.dat"
    label_file = data_dir + "labels_py.dat"
    validation_folds_file = data_dir + "validation_folds_py.dat"
    test_folds_file = data_dir + "folds_py.dat"

    data = pd.read_csv(data_file,header=None).values
    label = pd.read_csv(label_file, header=None).values.flatten()
    validation_folds = pd.read_csv(validation_folds_file, header=None).values
    test_folds = pd.read_csv(test_folds_file, header=None).values
    
    train_index = (test_folds[:, 0] == 0)
    test_index = (test_folds[:, 0] == 1)
    
    cv_folds = []
    for i in range(validation_folds.shape[1]):
        val = (validation_folds[:, i] == 1)
        validation_index = (train_index & (val == 1))
        validation_index = validation_index[train_index]
        cv_folds += [(~validation_index,validation_index)]
#     train_index = (train_index & (val == 0))

    train_x = data[train_index, :]
    train_y = label[train_index]
    train_y[np.isnan(train_y)] = -1

    if return_X_y:
        return (train_x,train_y)
    else:
        return Bunch(data = train_x, target = train_y, cv_folds = cv_folds)
    

# Load the UCI datasets in the sklearn format
# For example, see https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html
def load_tst(dataset_name,return_X_y=False):
    """ Load the testing datasets """
    data_dir = ROOTDIR+"/data/" + dataset_name + "/"
    data_file = data_dir + dataset_name + "_py.dat"
    label_file = data_dir + "labels_py.dat"
    test_folds_file = data_dir + "folds_py.dat"

    data = pd.read_csv(data_file,header=None).values
    label = pd.read_csv(label_file, header=None).values.flatten()
    test_folds = pd.read_csv(test_folds_file, header=None).values
    
    train_index = (test_folds[:, 0] == 0)
    test_index = (test_folds[:, 0] == 1)

    test_x = data[test_index, :]
    test_y = label[test_index]
    test_y[np.isnan(test_y)] = -1

    if return_X_y:
        return (test_x, test_y)
    else:
        return Bunch(data = test_x, target = test_y)
