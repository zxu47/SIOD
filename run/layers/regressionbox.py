import tensorflow as tf
import numpy as np
from layers.func import regress

class RegressBoxes(tf.keras.layers.Layer):
    """ Keras layer for applying regression values to boxes.
    """

    def __init__(self, anchors,config,*args, **kwargs):
      
        self.anchors = tf.Variable(np.copy(anchors))
        self.config = config
        info = config["info"]
        
        length = len(info)
        layer_index = []
        for i in range(length):
            layer_index+=[i+1]*(info[i]["shape"][0]*info[i]["shape"][1])
        layer_index = np.array(layer_index,dtype = np.float32)
        layer_index = np.reshape(layer_index,(-1,1))
        self.layer_index = tf.Variable(np.array(layer_index,dtype = np.float32))

        super(RegressBoxes, self).__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):

        out = regress(inputs,self.config,self.anchors,self.layer_index)
        return out

    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[1],input_shape[2]+1)

    def get_config(self):
        config = super(RegressBoxes, self).get_config()
        return config


