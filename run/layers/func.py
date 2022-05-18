from __future__ import division
import tensorflow as tf
import numpy as np

import tensorflow.keras.backend as K
from tensorflow.keras.layers import InputSpec
from tensorflow.keras.layers import Layer


def wnorm(scale):
    def norm(w):
        temp = tf.reduce_sum(w*w,axis = [0,1,2])
        temp = temp-1
        return scale*tf.reduce_sum(temp*temp)
    return norm

class Normalization1(Layer):
    

    def __init__(self, **kwargs):
        self.axis = 3
        super(Normalization1, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
       # gamma = self.gamma_init * np.ones((input_shape[self.axis],))
       # self.gamma = K.variable(gamma, name='{}_gamma'.format(self.name))
       # self.trainable_weights = [self.gamma]
        super(Normalization1, self).build(input_shape)

    def call(self, x, mask=None):
        output = K.l2_normalize(x, self.axis)
        return output

    def get_config(self):       
        base_config = super(Normalization1, self).get_config()
        return dict(list(base_config.items()))

def preprocess_inputs(inputs,config):
    start = config["nclasses"]
    e1 = inputs[...,start]>=-3.5
    e2 = inputs[...,start]<=3.5
    e3 = inputs[...,start+1]>=-3.5
    e4 = inputs[...,start+1]<=3.5
    length = tf.math.maximum(inputs[...,start+2],inputs[...,start+3])
    e5 = length<=np.log(2)/0.2
    e6 = length>=0

    e = e1&e2&e3&e4&e5&e6
    e_indicator = tf.where(e,tf.ones(tf.shape(e)),tf.zeros(tf.shape(e)))

    a = 1-tf.abs(inputs[...,start])/10.0
    b = 1-tf.abs(inputs[...,start+1])/10.0
    indicator = e_indicator*a*b
    indicator = tf.tile(tf.expand_dims(indicator,axis=-1),(1,1,start))
    left = inputs[...,:start]*indicator
    right = inputs[...,start:]
    return tf.concat(values = [left,right],axis = -1)
   # return inputs

def standard_regress(inputs, config,anchors,layer_index):

  #  inputs = preprocess_inputs(inputs,config)

    inputs_shape = tf.keras.backend.shape(inputs)
    batch_anchor = tf.tile(tf.expand_dims(anchors,axis=0),(inputs_shape[0],1,1))
    batch_index = tf.tile(tf.expand_dims(layer_index,axis=0),(inputs_shape[0],1,1))

    start = config["nclasses"]
   # cx = inputs[...,start]*config["variances"][0]*(batch_anchor[...,-2]-batch_anchor[...,-4])+(batch_anchor[...,-4]+batch_anchor[...,-2])/2
   # cy = inputs[...,-3]*config["variances"][1]*(batch_anchor[...,-1]-batch_anchor[...,-3])+(batch_anchor[...,-3]+batch_anchor[...,-1])/2
   # w = tf.exp(inputs[...,-2]*config["variances"][2])*(batch_anchor[...,-2]-batch_anchor[...,-4])
   # h = tf.exp(inputs[...,-1]*config["variances"][3])*(batch_anchor[...,-1]-batch_anchor[...,-3])
    
    cx = inputs[...,start]*config["variances"][0]*batch_anchor[...,4]+batch_anchor[...,0]
    cy = inputs[...,start+1]*config["variances"][1]*batch_anchor[...,5]+batch_anchor[...,1]
    w = tf.exp(inputs[...,start+2]*config["variances"][2])*(batch_anchor[...,4])
    h = tf.exp(inputs[...,start+3]*config["variances"][3])*(batch_anchor[...,5])

    xmin = cx-w/2
    ymin = cy-h/2
    xmax = xmin+w
    ymax = ymin+h

    xmin = tf.clip_by_value(tf.expand_dims(xmin,axis = -1),0,config["image_size"][1])
    xmax = tf.clip_by_value(tf.expand_dims(xmax,axis = -1),0,config["image_size"][0])
    ymin = tf.clip_by_value(tf.expand_dims(ymin,axis = -1),0,config["image_size"][1])
    ymax = tf.clip_by_value(tf.expand_dims(ymax,axis = -1),0,config["image_size"][0])

    out = tf.concat(values=[inputs[...,:-4], xmin, ymin, xmax, ymax,batch_index], axis=-1)
    return out


def standard_regress_80(inputs, config,anchors,layer_index):

    inputs_shape = tf.keras.backend.shape(inputs)
    batch_anchor = tf.tile(tf.expand_dims(anchors,axis=0),(inputs_shape[0],1,1))
    batch_index = tf.tile(tf.expand_dims(layer_index,axis=0),(inputs_shape[0],1,1))

    items = []
    start = config["nclasses"]
    for i in range(0,start-1):    
        cx = inputs[...,start+i*4]*config["variances"][0]*batch_anchor[...,4]+batch_anchor[...,0]
        cy = inputs[...,start+1+i*4]*config["variances"][1]*batch_anchor[...,5]+batch_anchor[...,1]
        w = tf.exp(inputs[...,start+2+i*4]*config["variances"][2])*(batch_anchor[...,4])
        h = tf.exp(inputs[...,start+3+i*4]*config["variances"][3])*(batch_anchor[...,5])

        xmin = cx-w/2
        ymin = cy-h/2
        xmax = xmin+w
        ymax = ymin+h

        xmin = tf.clip_by_value(tf.expand_dims(xmin,axis = -1),0,config["image_size"][1])
        xmax = tf.clip_by_value(tf.expand_dims(xmax,axis = -1),0,config["image_size"][0])
        ymin = tf.clip_by_value(tf.expand_dims(ymin,axis = -1),0,config["image_size"][1])
        ymax = tf.clip_by_value(tf.expand_dims(ymax,axis = -1),0,config["image_size"][0])
        items+=[xmin,ymin,xmax,ymax]

    out = tf.concat(values=[inputs[...,:start]]+items+[batch_index], axis=-1)
    return out


def label_center_regress_80(inputs, config,anchors,layer_index):

    inputs_shape = keras.backend.shape(inputs)
    batch_anchor = tf.tile(tf.expand_dims(anchors,axis=0),(inputs_shape[0],1,1))
    batch_index = tf.tile(tf.expand_dims(layer_index,axis=0),(inputs_shape[0],1,1))

    items = []
    for i in range(0,20):     
        w = tf.exp(inputs[...,23+i*4]*config["variances"][2])*(batch_anchor[...,4])
        h = tf.exp(inputs[...,24+i*4]*config["variances"][3])*(batch_anchor[...,5])
        cx = inputs[...,21+i*4]*config["variances"][0]*w+batch_anchor[...,0]
        cy = inputs[...,22+i*4]*config["variances"][1]*h+batch_anchor[...,1]

        xmin = cx-w/2
        ymin = cy-h/2
        xmax = xmin+w
        ymax = ymin+h

        xmin = tf.clip_by_value(tf.expand_dims(xmin,axis = -1),0,config["image_size"][1])
        xmax = tf.clip_by_value(tf.expand_dims(xmax,axis = -1),0,config["image_size"][0])
        ymin = tf.clip_by_value(tf.expand_dims(ymin,axis = -1),0,config["image_size"][1])
        ymax = tf.clip_by_value(tf.expand_dims(ymax,axis = -1),0,config["image_size"][0])
        items+=[xmin,ymin,xmax,ymax]

    out = tf.concat(values=[inputs[...,:21]]+items+[batch_index], axis=-1)
    return out


def corner_regress(inputs, config,anchors,layer_index):

    inputs_shape = keras.backend.shape(inputs)
    batch_anchor = tf.tile(tf.expand_dims(anchors,axis=0),(inputs_shape[0],1,1))
    batch_index = tf.tile(tf.expand_dims(layer_index,axis=0),(inputs_shape[0],1,1))

    xmin = inputs[...,-4]*config["variances"][0]*batch_anchor[...,4]+batch_anchor[...,0]
    ymin = inputs[...,-3]*config["variances"][1]*batch_anchor[...,5]+batch_anchor[...,1]
    xmax = inputs[...,-2]*config["variances"][2]*batch_anchor[...,4]+batch_anchor[...,0]
    ymax = inputs[...,-1]*config["variances"][3]*batch_anchor[...,5]+batch_anchor[...,1]

    xmin = tf.clip_by_value(tf.expand_dims(xmin,axis = -1),0,config["image_size"][1])
    xmax = tf.clip_by_value(tf.expand_dims(xmax,axis = -1),0,config["image_size"][0])
    ymin = tf.clip_by_value(tf.expand_dims(ymin,axis = -1),0,config["image_size"][1])
    ymax = tf.clip_by_value(tf.expand_dims(ymax,axis = -1),0,config["image_size"][0])

    out = tf.concat(values=[inputs[...,:-4], xmin, ymin, xmax, ymax,batch_index], axis=-1)
    return out


def corner_regress_80(inputs, config,anchors,layer_index):

    inputs_shape = keras.backend.shape(inputs)
    batch_anchor = tf.tile(tf.expand_dims(anchors,axis=0),(inputs_shape[0],1,1))
    batch_index = tf.tile(tf.expand_dims(layer_index,axis=0),(inputs_shape[0],1,1))
    items = []

    for i in range(20):


        xmin = -tf.math.exp(inputs[...,4*i+21]*config["variances"][0])*batch_anchor[...,4]+batch_anchor[...,0]
        ymin = -tf.math.exp(inputs[...,4*i+22]*config["variances"][1])*batch_anchor[...,5]+batch_anchor[...,1]
        xmax = tf.math.exp(inputs[...,4*i+23]*config["variances"][2])*batch_anchor[...,4]+batch_anchor[...,0]
        ymax = tf.math.exp(inputs[...,4*i+24]*config["variances"][3])*batch_anchor[...,5]+batch_anchor[...,1]

        xmin = tf.clip_by_value(tf.expand_dims(xmin,axis = -1),0,config["image_size"][1])
        xmax = tf.clip_by_value(tf.expand_dims(xmax,axis = -1),0,config["image_size"][0])
        ymin = tf.clip_by_value(tf.expand_dims(ymin,axis = -1),0,config["image_size"][1])
        ymax = tf.clip_by_value(tf.expand_dims(ymax,axis = -1),0,config["image_size"][0])
        items += [xmin,ymin,xmax,ymax]

    out = tf.concat(values=[inputs[...,:21]]+items+[batch_index], axis=-1)
    return out

def iou_regress_80(inputs, config,anchors,layer_index):

    inputs_shape = keras.backend.shape(inputs)
    batch_anchor = tf.tile(tf.expand_dims(anchors,axis=0),(inputs_shape[0],1,1))
    batch_index = tf.tile(tf.expand_dims(layer_index,axis=0),(inputs_shape[0],1,1))

    items = []
    for i in range(0,20):    
        cx = 0.1*inputs[...,21+i*4]*batch_anchor[...,4]+batch_anchor[...,0]
        cy = 0.1*inputs[...,22+i*4]*batch_anchor[...,5]+batch_anchor[...,1]
       # dia = tf.exp(inputs[...,23+i*4])*batch_anchor[...,4]
      #  ratio = tf.exp(inputs[...,24+i*4])
       # h = dia/tf.math.sqrt(1+ratio*ratio)
       # w = ratio*h
        w = tf.exp(inputs[...,23+i*4]*0.4)*batch_anchor[...,4]
        h = tf.exp(inputs[...,24+i*4]*0.4)*batch_anchor[...,5]

        xmin = cx-w/2
        ymin = cy-h/2
        xmax = xmin+w
        ymax = ymin+h

        xmin = tf.clip_by_value(tf.expand_dims(xmin,axis = -1),0,config["image_size"][1])
        xmax = tf.clip_by_value(tf.expand_dims(xmax,axis = -1),0,config["image_size"][0])
        ymin = tf.clip_by_value(tf.expand_dims(ymin,axis = -1),0,config["image_size"][1])
        ymax = tf.clip_by_value(tf.expand_dims(ymax,axis = -1),0,config["image_size"][0])
        items+=[xmin,ymin,xmax,ymax]

    out = tf.concat(values=[inputs[...,:21]]+items+[batch_index], axis=-1)
    return out

def regress(inputs, config,anchors,layer_index):
    if config["regression_type"]=="fine":
        return standard_regress_80(inputs, config,anchors,layer_index)
    else:
        return standard_regress(inputs, config,anchors,layer_index)
