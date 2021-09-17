from .utils import to_t_categorical, from_t_categorical, predict_classes_from_r_margin
import tensorflow.keras.backend as K


# Note, this is a hacky to get the class keras.metrics.MeanMetricWrapper
# In TF version 2.6, MeanMetricWrapper is exported and one can directly use
# from keras.metrics import MeanMetricWrapper
from tensorflow.keras.metrics import Accuracy
MeanMetricWrapper = Accuracy.__base__ 
del Accuracy
# from tf.keras.metrics import MeanMetricWrapper



def r_margin_accuracy(y_true, y_pred):
    # relative margin accuracy
    return K.cast(K.equal(from_t_categorical(y_true),
                          predict_classes_from_r_margin(y_pred)),
                  K.floatx())


class RMarginAccuracy(MeanMetricWrapper):
    """ 
    Analogous to the ordinary accuracy as defined in
    tf.keras.metrics.Accuracy
    https://www.tensorflow.org/api_docs/python/tf/keras/metrics/Accuracy
    """

    def __init__(self, name='accuracy', dtype=None):
        super(RMarginAccuracy, self).__init__(r_margin_accuracy, name, dtype=dtype)
