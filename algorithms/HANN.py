from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import tf_multiclass.losses as losses
import tf_multiclass.metrics as metrics
from tf_multiclass.utils import to_t_categorical, from_t_categorical, predict_classes_from_r_margin

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers

from sklearn import datasets
import numpy as np
import larq as lq


global_vars = {}

loss_func_handles = {"OCE": tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                     "CE" : losses.CrossEntropy(),
                     "WW" : losses.WWHinge(),
                     "CS" : losses.CSHinge(),
                     "DKR": losses.DKRHinge()}

quantizer_handles = {"SteSign" : lq.quantizers.SteSign,
                     "SwishSign" : lq.quantizers.SwishSign}

def create_HANN_model(X_train,
                      y_train,
                      n_classes,
                      n_hyperplanes,
                      n_hidden,
                      opt,
                      loss_func = "CE",
                      first_layer_trainable = True,
                      use_skip_conn = True,
                      dropout_rate = 0.1,
                      quantizer = "SteSign",
                      hidden_layer_act = "relu",
                      quantizer_param = None):
    
    n_features_in = X_train.shape[1]
    inputs = keras.Input(shape=(n_features_in,), name="features_in")
    
    if quantizer_param is not None:
        qtz = quantizer_handles[quantizer](quantizer_param)
    else:
        qtz = quantizer_handles[quantizer]()

    hyperplane_enc = layers.Dense(n_hyperplanes, activation = qtz)(inputs)

    hyperplane_enc = layers.Dropout(dropout_rate)(hyperplane_enc)

    if loss_func == "OCE":
        output_dim = n_classes
        accuracy = keras.metrics.CategoricalAccuracy()
    else:
        output_dim = n_classes-1
        accuracy = metrics.RMarginAccuracy()
        
    if use_skip_conn:
        hidden = layers.Dense(n_hidden, activation=hidden_layer_act)(hyperplane_enc)
        out_hidden = layers.Dense(output_dim, activation = "linear")(hidden)
        out_skip = layers.Dense(output_dim, activation = "linear")(hyperplane_enc)
        outputs = tf.keras.layers.Add()([out_skip,out_hidden])
    else:
        hidden = layers.Dense(n_hidden, activation="relu")(quantized_inputs)
        outputs = layers.Dense(output_dim, activation = "linear")(hidden)

    model = keras.Model(inputs=inputs, outputs=outputs)
    
    model.layers[1].trainable = first_layer_trainable
    
    # initializing the bias
    W = model.layers[1].weights[0].numpy()
    b = np.matmul(W.T,X_train.T)
    initial_bias = -b[range(b.shape[0]), np.random.choice(b.shape[1],b.shape[0])]
    model.layers[1].weights[1].assign(initial_bias)
    
    
    loss_func_handle = loss_func_handles[loss_func]
    model.compile(loss=loss_func_handle, optimizer=opt, metrics=[accuracy])
    
    return model

def get_model_creation_functions(X_train, 
                                 y_train,
                                n_classes,
                                n_hyperplanes,
                                n_hidden,
                                opt,
                                opt_init,
                                loss_func,
                                use_skip_conn,
                                dropout_rate,
                                quantizer,
                                hidden_layer_act,
                                quantizer_param):
    def get_model():
        return create_HANN_model(X_train,
                                 y_train,
                                 n_classes=n_classes,
                                 n_hyperplanes=n_hyperplanes,
                                 n_hidden=n_hidden,
                                 opt = opt,
                                 loss_func = loss_func,
                                 first_layer_trainable = True,
                                 use_skip_conn = use_skip_conn,
                                 dropout_rate = dropout_rate,
                                 quantizer = quantizer,
                                 hidden_layer_act = hidden_layer_act,
                                 quantizer_param = quantizer_param)
    def get_init_model():
        return create_HANN_model(X_train,
                                 y_train,
                                 n_classes=n_classes,
                                 n_hyperplanes=n_hyperplanes,
                                 n_hidden=n_hidden,
                                 opt = opt_init,
                                 loss_func = loss_func,
                                 first_layer_trainable = False,
                                 use_skip_conn = use_skip_conn,
                                 dropout_rate = 0.0,
                                 quantizer = quantizer,
                                 hidden_layer_act = hidden_layer_act,
                                 quantizer_param = quantizer_param)
    return get_model, get_init_model



def get_label_manipulators(loss_func, n_classes):
    if loss_func == "OCE":
        label_encoder = lambda x: to_categorical(x, num_classes = n_classes)
        label_decoder = lambda x: tf.math.argmax(x,axis=-1)
        class_predictor = label_decoder
        acc_name = "categorical_accuracy"
    else:
        label_encoder = lambda x : to_t_categorical(x, num_classes = n_classes)
        label_decoder = from_t_categorical
        class_predictor = predict_classes_from_r_margin
        acc_name = "accuracy"
    return label_encoder, label_decoder, class_predictor, acc_name

def run_model_fitting(model,epochs):
    gv = Bunch(global_vars)
    return model.fit(gv.X_train,
                     gv.Y_train,
                     epochs=epochs,
                     validation_data=(gv.X_valid, gv.Y_valid),
#                      shuffle= True,
                     batch_size=gv.X_train.shape[0],
                     validation_batch_size = gv.X_valid.shape[0],
                     verbose=0)

def get_best_initial_weights_by_sampling(get_init_model, acc_name, n_models = 10, epochs = 10):
    best_acc = 0
    all_accs = []
    for i in range(n_models):
        init_model = get_init_model()
        history = run_model_fitting(init_model,epochs)
        hist = history.history
        final_acc = hist['val_'+acc_name][-1]
        all_accs.append(final_acc)
        if final_acc > best_acc:
            best_init = init_model.get_weights()
            best_acc = final_acc
    return best_init, all_accs






from multiprocessing import Lock, Pool


sm_param = 0.1 # smoothing parameter for the validation accuracy

import itertools


def acc2perc(acc):
    # Converts accuracy to percentage
    return round(100*acc,2)

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


class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)

s_print_lock = Lock()

def s_print(*a, **b):
    """Thread safe print function"""
    with s_print_lock:
        print(*a, **b)

        
        

hparams = list(dict_product(dict(
    n_hyperplanes = [100],
    n_hidden = [1000],
    learning_rate = [0.01],
    loss_func = ["DKR", "WW", "CS"],
    dropout_rate = [0.5],
    quantizer = ["SwishSign"],
    quantizer_param = [None],
    hidden_layer_act = ["relu"]
)))
    

def run(hparam_idx,hparam):
    global global_vars
    gv = Bunch(global_vars)
    opt = keras.optimizers.SGD(learning_rate = hparam['learning_rate'])
    opt_init = keras.optimizers.SGD()
    loss_func = hparam['loss_func']
    get_model, get_init_model = get_model_creation_functions(X_train = gv.X_train, 
                                                             y_train = gv.y_train,
                                                             n_classes = gv.n_classes,
                                                             n_hyperplanes = hparam['n_hyperplanes'],
                                                             n_hidden = hparam['n_hidden'],
                                                             opt = opt,
                                                             opt_init = opt_init,
                                                             loss_func = loss_func,
                                                             use_skip_conn = True,
                                                             dropout_rate = hparam['dropout_rate'],
                                                             quantizer = hparam['quantizer'],
                                                             hidden_layer_act = hparam['hidden_layer_act'],
                                                             quantizer_param = hparam['quantizer_param'])

    label_encoder, label_decoder, class_predictor, acc_name = get_label_manipulators(loss_func, gv.n_classes)
    

    global_vars['Y_train'] = label_encoder(gv.y_train)
    global_vars['Y_valid'] = label_encoder(gv.y_valid)
    
    best_init, _ = get_best_initial_weights_by_sampling(get_init_model, acc_name, n_models = 30, epochs=10)

    tf.keras.backend.clear_session()
    
    model = get_model()
    model.set_weights(best_init)

    val_acc_sm = 0

#     for i in range(500):
    for i in range(5):
        history = run_model_fitting(model,epochs=10)

        val_acc = np.mean(history.history['val_' + acc_name])
        trn_acc = np.mean(history.history[acc_name])

        val_acc_sm = (1-sm_param)*val_acc_sm + sm_param*val_acc

        tst_acc = np.mean(gv.y_test == class_predictor(model(gv.X_test,training = False)))



        s_print("RES "+",".join(map(str,[hparam_idx,
                                i,
                                acc2perc(trn_acc),
                                acc2perc(val_acc),
                                acc2perc(val_acc_sm),
                                acc2perc(tst_acc)])))


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

        pool = Pool()
        pool.starmap(run, enumerate(hparams))

        pool.close()
        pool.join()

                
        return None

clf = HANN()