import tensorflow.keras.backend as K
from tensorflow.keras.layers import InputSpec,Layer
import numpy as np


class LayerNormalization(Layer):

    def __init__(self, axis=[1,2,3],gamma_init=1.0, **kwargs):
        super(LayerNormalization, self).__init__(**kwargs)
        self.axis = axis
        self.eps = 1e-6
        self.gamma_init = gamma_init

    def build(self, input_shape):

        gamma = self.gamma_init * np.ones((input_shape[-1],))
        self.gamma = K.variable(gamma, name='{}_gamma'.format(self.name))
        self._trainable_weights = [self.gamma]
     #   self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:],
     #                                initializer=initializers.Ones(), trainable=True)
      #  self.beta = self.add_weight(name='beta', shape=input_shape[-1:],
      #                              initializer=initializers.Zeros(), trainable=True)
        super(LayerNormalization, self).build(input_shape)

    def call(self, x):
       # mean = K.mean(x, axis=-1, keepdims=True)
        std = K.std(x, axis=self.axis, keepdims=True)
        return self.gamma * (x) / (std + self.eps)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(LayerNormalization, self).get_config()
        config.update({
            'axis': self.axis,
            'gamma_init':self.gamma_init

        })
        return config
