import tensorflow.keras.backend as K
from tensorflow.keras.layers import InputSpec,Layer
import numpy as np


class My_Normalization(Layer):
    

    def __init__(self, gamma = 1.0,axis=[1,2],**kwargs):
        self.axis = axis
        self.gamma = gamma
        super(My_Normalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        super(My_Normalization, self).build(input_shape)

    def call(self, x, mask=None):
        output = K.l2_normalize(x, self.axis)
        output = self.gamma*output
        return output

    def get_config(self):       
        return {
            'axis': self.axis,
            'gamma': self.gamma,
            "name": self.name
        }
