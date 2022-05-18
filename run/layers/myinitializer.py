from __future__ import division
import tensorflow as tf
import numpy as np

import tensorflow.keras.backend as K
from tensorflow.keras.layers import InputSpec,Layer

def myreg(l2_reg):
    def reg(weight_matrix):
        temp = np.zeros(weight_matrix.shape)
        _,_,m,n = temp.shape
        for i in range(n):
            temp[1,1,int(i*m/n),i]=1
        return l2_reg*K.sum((weight_matrix-temp)**2)
    return reg

class IdentityConvInitializer(tf.keras.initializers.Initializer):
    """ Apply a prior probability to the weights.
    """

    def __init__(self, value=1):
        self.value = value

    def get_config(self):
        return {
            'value': self.value
        }

    def __call__(self, shape, dtype=None):
        # set bias to -log((1 - p)/p) for foreground

        result = np.zeros(shape,dtype = np.float32)
        print("IdentityConvInitializer-shape:",shape)
        a,b,c,d = shape
        for i in range(d):
            result[1,1,int(i*c/d),i]=1
        return result


class BiasInitializer(tf.keras.initializers.Initializer):
    """ Apply a prior probability to the weights.
    """

    def __init__(self, value=1):
        self.value = value

    def get_config(self):
        return {
            'value': self.value
        }

    def __call__(self, shape, dtype=None):
        # set bias to -log((1 - p)/p) for foreground

        result = np.zeros(shape,dtype = np.float32)
        print("BiasIntializer-shape:",shape)
        result[0]=self.value
        print(list(result))
        return result


class ResNetInitializer(tf.keras.initializers.Initializer):
    """ Apply a prior probability to the weights.
    """

    def __init__(self, value=1):
        self.value = value

    def get_config(self):
        return {
            'value': self.value
        }

    def __call__(self, shape, dtype=None):
        # set bias to -log((1 - p)/p) for foreground

        #result = np.zeros(shape,dtype = np.float32)
        result = np.random.normal(0,2/(shape[-1]+shape[-2]),shape)
        print("ResNetInitializer-shape:",shape)
        a,b,c,d = shape
        for i in range(d):
            result[1,1,int(i*c/d),i]=1
        return result

