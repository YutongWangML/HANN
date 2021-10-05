from HANN_model import HANN_model_factory

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from tf_multiclass.utils import get_label_manipulators

from tensorflow import keras

import numpy as np
import time



from multiprocessing import Lock, Pool
import itertools



def dict_product(dicts):
    # This function is written by Arash Rouhani
    # taken from https://stackoverflow.com/a/40623158/636276
    # for creating the hyperparameter grids Python iterator.
    # Syntatically, this looks nicer than tons of nested for loops.
    """
    >>> list(dict_product(dict(number=[1,2], character='ab')))
    [{'character': 'a', 'number': 1},
     {'character': 'a', 'number': 2},
     {'character': 'b', 'number': 1},
     {'character': 'b', 'number': 2}]
    """
    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))



s_print_lock = Lock()
def s_print(*a, **b):
    """Thread safe print function"""
    with s_print_lock:
        print(*a, **b)

hparams = list(dict_product(dict(
    n_hyperplanes = [15],
    n_hidden = [1000],
    learning_rate = [0.01],
    loss_func = ["OCE"],
    dropout_rate = [0.1,0.25,0.5],
    quantizer = ["SwishSign"],
    quantizer_param = [None],
    hidden_layer_act = ["relu"]
)))


class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)
        
def run(hparam_idx,hparam):
    global global_vars
    gv = Bunch(global_vars)
    opt = keras.optimizers.SGD(learning_rate = hparam['learning_rate'])
    opt_init = keras.optimizers.SGD()
    loss_func = hparam['loss_func']

    n_features_in = gv.X_train.shape[1]
    n_classes = gv.n_classes
    n_hyperplanes = hparam['n_hyperplanes']
    n_hidden = hparam['n_hidden']
    dropout_rate = hparam['dropout_rate']
    quantizer = hparam['quantizer']
    hidden_layer_act = hparam['hidden_layer_act']
    quantizer_param = hparam['quantizer_param']
    loss_func = hparam['loss_func']
    
    HMF = HANN_model_factory(n_features_in,
                             n_classes,
                             n_hyperplanes,
                             n_hidden,
                             use_skip_conn = True,
                             dropout_rate = dropout_rate,
                             quantizer = quantizer,
                             hidden_layer_act = hidden_layer_act,
                             quantizer_param = quantizer_param)
    
    label_encoder, label_decoder, label_predictor, acc_name = get_label_manipulators(loss_func, n_classes)
    

    X_train = gv.X_train
    X_valid = gv.X_valid
    X_test = gv.X_test
    Y_train = label_encoder(gv.y_train)
    Y_valid = label_encoder(gv.y_valid)
    y_test = gv.y_test
    
    model = HMF.get_model(X_train, opt, loss_func=loss_func)
    
    val_acc_sm = 0 # smoothed validation accuracy
    sm_param = 0.1 # smoothing parameter for the validation accuracy
    
    acc2perc = lambda acc: round(100*acc,2) # Converts accuracy to percentage

    
    if X_train.shape[0] == 77904: # if the dataset is miniboone
        n_outer_epochs = 50
    else:
        n_outer_epochs = 500
    for i in range(n_outer_epochs):

        history = model.fit(X_train,
                            Y_train,
                            epochs=10,
                            validation_data=(X_valid, Y_valid),
                            shuffle= True,
                            batch_size = 128,
                            validation_batch_size = X_valid.shape[0],
                            verbose=0)
        
        val_acc = np.mean(history.history['val_' + acc_name])
        trn_acc = np.mean(history.history[acc_name])
        val_acc_sm = (1-sm_param)*val_acc_sm + sm_param*val_acc
        tst_acc = np.mean(y_test == label_predictor(model(X_test,training = False)))
        s_print("RES "+",".join(map(str,
                                    [hparam_idx,
                                     i,
                                     acc2perc(trn_acc),
                                     acc2perc(val_acc),
                                     acc2perc(val_acc_sm),
                                     acc2perc(tst_acc)])))


global_vars = {}
class HANN():
    def __init__(self):
        self.best_model = None
        self.best_accuracy = 0.0
        self.best_params = None
        self.best_y_pred = None
        

    def fit(self, X_train_valid, y_train_valid, X_test, y_test, **kwargs):
        
        print("HYP "+str(hparams))
        print("RES hparam_id,epoch,trn_acc,val_acc,val_acc_sm,tst_acc")
        
        train, valid = kwargs['cv_folds'][0]

        scaler = StandardScaler().fit(X_train_valid)
        X_train_valid = scaler.transform(X_train_valid)
        X_test = scaler.transform(X_test)
        
        if min(X_train_valid.shape) > 50:
            pca = PCA(n_components = 50).fit(X_train_valid)
            X_train_valid = pca.transform(X_train_valid)
            X_test = pca.transform(X_test)
        
        global global_vars
        
        global_vars['n_features_in'] = X_train_valid.shape[1]
        global_vars['n_classes'] = len(np.unique(y_train_valid))
        global_vars['y_train_valid'] = y_train_valid
        global_vars['X_train'] = X_train_valid[train,:]
        global_vars['y_train'] = y_train_valid[train]
        global_vars['X_valid'] = X_train_valid[valid,:]
        global_vars['y_valid'] = y_train_valid[valid]
        global_vars['X_test'] = X_test
        global_vars['y_test'] = y_test

        start = time.perf_counter()
        pool = Pool()
        pool.starmap(run, enumerate(hparams))

        pool.close()
        pool.join()
        time_elapsed = time.perf_counter() - start
        s_print("TME " + str(time_elapsed))
                
        return None

clf = HANN()