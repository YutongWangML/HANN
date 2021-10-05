import tf_multiclass.losses as losses
import tf_multiclass.metrics as metrics
from tf_multiclass.utils import get_label_manipulators

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers

import numpy as np
import larq as lq

loss_func_handles = {"OCE": tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                     "CE" : losses.CrossEntropy(),
                     "WW" : losses.WWHinge(),
                     "CS" : losses.CSHinge(),
                     "DKR": losses.DKRHinge()}

quantizer_handles = {"SteSign" : lq.quantizers.SteSign,
                     "SwishSign" : lq.quantizers.SwishSign}

class HANN_model_factory:
    
    def __init__(
        self, 
        n_features_in,
        n_classes,
        n_hyperplanes,
        n_hidden,
        use_skip_conn = True,
        dropout_rate = 0.1,
        quantizer = "SwishSign",
        hidden_layer_act = "relu",
        quantizer_param = None
    ):
        self.n_features_in = n_features_in
        self.n_classes = n_classes
        self.n_hyperplanes = n_hyperplanes
        self.n_hidden = n_hidden
        self.use_skip_conn = use_skip_conn
        self.dropout_rate = dropout_rate
        self.quantizer = quantizer
        self.hidden_layer_act = hidden_layer_act
        self.quantizer_param = quantizer_param
        
    def get_model(
        self,
        X_train,
        opt,
        first_layer_trainable = True,
        loss_func = "CE"):
        
        
        n_features_in = self.n_features_in
        n_classes = self.n_classes
        n_hyperplanes = self.n_hyperplanes
        n_hidden = self.n_hidden
        use_skip_conn = self.use_skip_conn
        dropout_rate = self.dropout_rate
        quantizer = self.quantizer
        hidden_layer_act = self.hidden_layer_act
        quantizer_param = self.quantizer_param
        
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
            hidden = layers.Dense(n_hidden, activation="relu")(hyperplane_enc)
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