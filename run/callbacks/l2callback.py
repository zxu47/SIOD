import tensorflow as tf
from tensorflow.keras.layers import Conv2D
import numpy as np



class L2LossCallback(tf.keras.callbacks.Callback):

    def __init__(self,config,worker):
       
        self.config = config
        self.worker = worker
        super(L2LossCallback, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        l2loss = 0
        for layer in self.worker.layers:
            if isinstance(layer,Conv2D):
                kernel = layer.weights[0].numpy()
                l2loss+=np.sum(kernel**2)
            if "model" in layer.name:
                for item in layer.layers:
                    if isinstance(item,Conv2D):
                        kernel = item.weights[0].numpy()
                        l2loss+=np.sum(kernel**2)

        logs['model_weights'] = l2loss
        logs["l2loss"] = l2loss*self.config["l2_reg"]
        logs["real_loss"]=logs["loss"]-l2loss*self.config["l2_reg"]
