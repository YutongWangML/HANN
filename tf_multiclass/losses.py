from tensorflow.python.keras.utils import losses_utils
import tensorflow.keras.backend as K
from tensorflow.keras.losses import Loss
import tensorflow as tf

def get_relative_margins(y, z):
    # Output: 
    #     a tensor with shape (_, n_class - 1) whose i-th row is
    #     z[i,:]*rho[y[i]]^T 
    #         where rho[y] is the y-th involution code
    # Inputs: 
    #     y is a (_, n_classes - 1) tensor.
    #           must be a trimmed categorical vector, 
    #           i.e., y should be the output of tf_multiclass.utils.to_t_categorical(y_raw)
    #           y_raw here refers to the labels that takes values in {0, 1,..., n_classes}
    #           the raw labels are then transformed to one-hot representation and the last entry is dropped
    #
    #     z is a (_, n_classes - 1) tensor
    #           should be the real-valued output of a model whose final layer has n_classes units
    A = z
    C = z*y
    B = K.expand_dims(K.sum(C,axis=1),axis=1)
    return A - B - C


class PERMLoss(Loss):
    def __init__(self,fn,reduction=losses_utils.ReductionV2.AUTO, name=None):
        super().__init__(reduction=reduction, name=name)
        self.fn = fn

    def call(self, y_true, y_pred):
        """
        Args:
          y_true: Ground truth values.
          y_pred: The predicted values.

        Returns:
          Loss values per sample.
        """
        z = get_relative_margins(y_true, y_pred)
        return self.fn(z)



def cs_hinge(z):
    # Crammer-Singer hinge function
    return K.max(K.maximum(1-z,0.),axis=1)


class CSHinge(PERMLoss):
    def __init__(self,
                 reduction=losses_utils.ReductionV2.AUTO,
                 name='cs_hinge'):
        super(CSHinge, self).__init__(cs_hinge, name=name, reduction=reduction)



def ww_hinge(z):
    # Weston-Watkins hinge function
    return K.sum(K.maximum(1-z,0.),axis=1)


class WWHinge(PERMLoss):
    def __init__(self,
                 reduction=losses_utils.ReductionV2.AUTO,
                 name='ww_hinge'):
        super(WWHinge, self).__init__(ww_hinge, name=name, reduction=reduction)


        
def dkr_hinge(z):    
    # Duchi-Khosravi-Ruan hinge function
    z0 = tf.pad( z, [[0,0],[0,1]] ) # add a column of zeros to z
    z0sorted = -tf.nn.top_k(-z0,k = z0.shape[1], sorted=True).values
    z0sorted_csummed = K.cumsum(z0sorted,axis=1)+1
    
    ell = K.arange(1,z0.shape[1]+1)
    ell_inv = K.cast_to_floatx(K.reshape(1/ell,[1,z0.shape[1]]))
    
    z0sorted_csummed_scaled = z0sorted_csummed * ell_inv

    return 1-K.min(z0sorted_csummed_scaled,axis=1)


class DKRHinge(PERMLoss):
    def __init__(self,
                 reduction=losses_utils.ReductionV2.AUTO,
                 name='dkr_hinge'):
        super(DKRHinge, self).__init__(dkr_hinge, name=name, reduction=reduction)

        

def cross_entropy(z):
    # Cross Entropy implemented in the PERM loss framework
    return K.log(1+K.sum(K.exp(-z),axis=1))


class CrossEntropy(PERMLoss):
    def __init__(self,
                 reduction=losses_utils.ReductionV2.AUTO,
                 name='crossentropy'):
        super(CrossEntropy, self).__init__(cross_entropy, name=name, reduction=reduction)