from __future__ import division
import sys
sys.path.append("../")
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D,Dense,Multiply,Reshape
from tensorflow.keras.layers import Input, Lambda, Activation, Conv2D, MaxPooling2D, ZeroPadding2D, Reshape, Concatenate
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import BatchNormalization,LayerNormalization
from layers.regressionbox import RegressBoxes

#from layers.mynorm import My_Normalization
#from layers.layernorm import LayerNormalization
from tensorflow.keras.models import load_model

from layers.suppression import SuppressionFine
from layers.suppression import SuppressionCoarse
from layers.myinitializer import BiasInitializer
from anchor import get_anchors

from loss import SSDLoss80,SSDLoss


from layers.myinitializer import IdentityConvInitializer,BiasInitializer

head_length=768

class CosineDecayWithWarmUp(tf.keras.experimental.CosineDecay):
    def __init__(self,lr,ds,alpha=0.0,warms=0,name=None):
        self.warms = warms
        super(CosineDecayWithWarmUp,self).__init__(initial_learning_rate = lr,
                                                  decay_steps = ds,
                                                  alpha = alpha,
                                                  name=name)
        
    @tf.function
    def __call__(self,step):
        if step<=self.warms:
            return float(step/self.warms*self.initial_learning_rate)
        else:
            return super(CosineDecayWithWarmUp,self).__call__(step-self.warms)

class MyResNetInitializer(tf.keras.initializers.Initializer):
    """ Apply a prior probability to the weights.
    """

    def __init__(self, value = 0.5,var = 1):
        self.value = value
        self.var = var

    def get_config(self):
        return {
            'value': self.value,
            'var': self.var
        }

    def __call__(self, shape, dtype=None):

        print("--------------------------")
        a,b,c,d = shape
        std = np.sqrt(6/(a*b*c+a*b*d))
        print("std:",std)
       # result = K.truncated_normal(shape,0.0,std).numpy()
        result = np.random.uniform(-std,std,shape)
        if a%2==1:
            for i in range(d):
                result[a//2,b//2,int(i*c/d),i]=self.value
        else:
            for i in range(d):
                result[:,:,int(i*c/d),i]=self.value/(a*b)
        return result


def normal_layer(x,
                 filters_in,
                 filters_out,
                 name,
                 kernel=(3,3),
                 strides=(1,1),
                 kernel_initializer=None,
                 kernel_regularizer=None,
                 with_bn = False):
    l2_reg=0.0
    if kernel_initializer==None:
        kernel_initializer = "he_normal"
    if kernel_regularizer==None:
        kernel_regularizer = l2(l2_reg)
    if with_bn==False:
        x = Conv2D(filters_out,kernel,strides = strides,padding="same",kernel_initializer=kernel_initializer,kernel_regularizer=kernel_regularizer,name=name,activation="relu")(x)
    else:
        x = Conv2D(filters_out,kernel,strides = strides,padding="same",kernel_initializer=kernel_initializer,kernel_regularizer=kernel_regularizer,name=name,use_bias=False)(x)
        x = BatchNormalization(name=name+"_bn")(x)
        x = Activation("relu")(x)
    return x


def my_resnet_layer(x,
                 filters_in,
                 filters_out,
                 name,
                 kernel=(3,3),
                 strides=(1,1),
                 kernel_initializer=None,
                 kernel_regularizer=None,
                 activation = "relu",
                 with_bn = False):
    scale = 0.75
    if kernel_initializer==None:
        kernel_initializer = MyResNetInitializer(0.85*scale,1.7*scale)
    if kernel_regularizer==None:
        kernel_regularizer = l2(0.0)
    #    kernel_regularizer = nocenterreg(0.0)
    if with_bn==False:
        x = Conv2D(filters_out,kernel,strides = strides,padding="same",kernel_initializer=kernel_initializer,kernel_regularizer=kernel_regularizer,name=name,activation=activation)(x)
    else:
        x = Conv2D(filters_out,kernel,strides = strides,padding="same",kernel_initializer=kernel_initializer,kernel_regularizer=kernel_regularizer,name=name,use_bias=False)(x)
        x = BatchNormalization(name=name+"_bn")(x)
        x = Activation("relu")(x)
    return x

def nms(x):
    hmax = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    keep = K.cast(K.equal(hmax,x),K.floatx())
    x=  x*keep
    return x

def get_classification_model(config,name,length):

    padding = "same"

    nclasses = config["nclasses"]
    l2_reg = config["l2_reg"]
  #  kernel_initializer = config["kernel_initializer"]
    kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0,stddev=0.01,seed=None)
    inputs = Input(shape = (None,None,head_length))
    outputs = inputs
    bi = BiasInitializer(5)
    channels =384
   # outputs = ZeroPadding2D(padding=((4, 4), (4, 4)), name='padding')(inputs)
    
    outputs = Conv2D(channels, (3, 3), activation='relu', padding=padding, kernel_initializer=kernel_initializer, kernel_regularizer=l2(l2_reg), name='get_cfeature')(outputs)
   # outputs = se(outputs,256,"c_feature1",config)
 #   outputs = BatchNormalization(name="c_feature1_bn")(outputs)
    outputs = Conv2D(channels, (3, 3), activation='relu', padding=padding, kernel_initializer=kernel_initializer, kernel_regularizer=l2(l2_reg), name='get_cfeature1')(outputs)
   # outputs = BatchNormalization(name="c_feature2_bn")(outputs)
  #  outputs = Conv2D(256, (3, 3), activation='relu', padding=padding, kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='get_feature2')(outputs)
  #  outputs = Conv2D(256, (3, 3), activation='relu', padding=padding, kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='get_feature3')(outputs)
  
    outputs1 = Conv2D(filters =nclasses,
                kernel_size=(3,3),
                strides=1,
                padding=padding,
                kernel_initializer = config["kernel_initializer"],
             #   bias_initializer = bi,
                kernel_regularizer=l2(l2_reg),
                name = "pyramid_classification1")(outputs)
  #  outputs1 = nms(outputs1)
    
    outputs1 = Activation('softmax',name = "pyramid_classification_softmax1")(outputs1)
 #   outputs1 = Lambda(nms)(outputs1)
    outputs1 = Reshape((-1,nclasses),name = "pyramid_classification_reshape1")(outputs1)
   # return Model(inputs = inputs,outputs = outputs1)


    
    outputs2 = Conv2D(filters =nclasses,
                kernel_size=(3,3),
                strides=1,
                padding=padding,
                kernel_initializer = config["kernel_initializer"],
              #  bias_initializer = bi,
                kernel_regularizer=l2(l2_reg),
                name = "pyramid_classification2")(outputs)
    
    outputs2 = Activation('softmax',name = "pyramid_classification_softmax2")(outputs2)
  #  outputs2 = Lambda(nms)(outputs2)
    outputs2 = Reshape((-1,nclasses),name = "pyramid_classification_reshape2")(outputs2)

    outputs3 = Conv2D(filters =nclasses,
                kernel_size=(3,3),
                strides=1,
                padding=padding,
                kernel_initializer = config["kernel_initializer"],
               # bias_initializer = bi,
                kernel_regularizer=l2(l2_reg),
                name = "pyramid_classification3")(outputs)
    
    outputs3 = Activation('softmax',name = "pyramid_classification_softmax3")(outputs3)
   # outputs3 = Lambda(nms)(outputs3)
    outputs3 = Reshape((-1,nclasses),name = "pyramid_classification_reshape3")(outputs3)
    
    outputs = [outputs1,outputs2,outputs3]
    if length!=1:
        return Model(inputs = inputs,outputs = outputs[0:length],name=name)
    else:
        return Model(inputs = inputs,outputs = outputs1,name=name)



def get_regression_model(config,name,length):
    norm = 1

    if config["regression_type"]=="fine":
        number_values = 4*(config["nclasses"]-1)
    else:
        number_values = 4
    l2_reg = config["l2_reg"]
   # kernel_initializer = config["kernel_initializer"]
    kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0,stddev=0.01,seed=None)
    inputs = Input(shape = (None,None,head_length))

    padding = "same"
    
    outputs = inputs
    channels=384
   # outputs = ZeroPadding2D(padding=((4, 4), (4, 4)), name='padding')(inputs)
    outputs = Conv2D(channels, (3, 3), activation='relu', padding=padding, kernel_initializer=kernel_initializer, kernel_regularizer=l2(l2_reg), name='get_rfeature')(outputs)
   # outputs = se(outputs,256,"r_feature1",config)
  #  outputs = BatchNormalization(name="r_feature1_bn")(outputs)
    outputs = Conv2D(channels, (3, 3), activation='relu', padding=padding, kernel_initializer=kernel_initializer, kernel_regularizer=l2(l2_reg), name='get_rfeature1')(outputs)
  #  outputs = Conv2D(256, (3, 3), activation='relu', padding=padding, kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='get_feature2')(outputs)
  #  outputs = Conv2D(256, (3, 3), activation='relu', padding=padding, kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='get_feature3')(outputs)
    

    #outputs1 = Normalization(gamma=norm, name='norm')(outputs)

  #  outputs1 = Conv2D(256, (1, 1), strides = (1,1),activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='pre1')(outputs1)
    
    
    outputs1 = Conv2D(filters =number_values,
                kernel_size=(3,3),
                strides=1,
                padding=padding,
                kernel_initializer = config["kernel_initializer"],
                kernel_regularizer=l2(l2_reg),
             #   dtype="float32",
            #    bias_initializer = PriorProbability(0),
                name = "pyramid_regression1")(outputs)
    outputs1 = Reshape((-1,number_values),name = "pyramid_regression_reshape1")(outputs1)
  #  return Model(inputs = inputs,outputs = outputs1)

    

   # outputs2 = Normalization(gamma=norm, name='norm2')(outputs)
   # outputs2 = Conv2D(256, (1, 1), strides = (1,1),activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='pre2')(outputs2)
    outputs2 = Conv2D(filters =number_values,
                kernel_size=(3,3),
                strides=1,
                padding=padding,
                kernel_initializer = config["kernel_initializer"],
                kernel_regularizer=l2(l2_reg),
             #   dtype="float32",
            #    bias_initializer = PriorProbability(0),
                name = "pyramid_regression2")(outputs)
    outputs2 = Reshape((-1,number_values),name = "pyramid_regression_reshape2")(outputs2)


    outputs3 = Conv2D(filters =number_values,
                kernel_size=(3,3),
                strides=1,
                padding=padding,
                kernel_initializer = config["kernel_initializer"],
                kernel_regularizer=l2(l2_reg),
             #   dtype="float32",
            #    bias_initializer = PriorProbability(0),
                name = "pyramid_regression3")(outputs)
    outputs3 = Reshape((-1,number_values),name = "pyramid_regression_reshape3")(outputs3)

   

  #  return Model(inputs = inputs,outputs = [outputs1,outputs2,outputs3])
    outputs = [outputs1,outputs2,outputs3]
    if length!=1:
        return Model(inputs = inputs,outputs = outputs[0:length],name=name)
    else:
        return Model(inputs = inputs,outputs = outputs1,name=name)

def layer(x,
          depth=head_length,
          kernel=(3,3),
          strides=(1,1),
          padding="same",
          kernel_initializer = "glorot_uniform",
          kernel_regularizer = l2(0.00005),
          with_bn=True,
          name=None):

    if with_bn==False:
        x = Conv2D(depth,kernel,strides = strides,padding = padding, kernel_initializer = kernel_initializer, kernel_regularizer = kernel_regularizer,name=name,activation="relu")(x)
    else:
        x = Conv2D(depth,kernel,strides = strides,padding = padding, kernel_initializer = kernel_initializer, kernel_regularizer = kernel_regularizer,name=name, use_bias = False)(x)
        x = BatchNormalization(name=name+'_bn',epsilon=1e-5)(x)
        x = Activation("relu")(x)
    return x

def layer2(x,
          depth=head_length,
          kernel=(3,3),
          strides=(1,1),
          padding="same",
          kernel_initializer = "he_normal",
          kernel_regularizer = l2(0.0001),
          with_bn=True,
          name=None):

    if with_bn==False:
        x = Conv2D(depth,kernel,strides = strides,padding = padding, kernel_initializer = kernel_initializer, kernel_regularizer = kernel_regularizer,name=name,activation="relu")(x)
    else:
        x = Conv2D(depth,kernel,strides = strides,padding = padding, kernel_initializer = kernel_initializer, kernel_regularizer = kernel_regularizer,name=name, use_bias = False)(x)
        x = BatchNormalization(name=name+'_bn',epsilon=1e-5)(x)
        x = Activation("relu")(x)
    return x

def nocenterreg(l2_reg):
    def reg(weight_matrix):
        temp = np.ones(weight_matrix.shape)
        a,b,m,n = temp.shape
        if a%2==1:
            for i in range(n):
                temp[a//2,b//2,int(i*m/n),i] = 0
        else:
            for i in range(n):
                temp[:,:,int(i*m/n),i] = 0
        return l2_reg*K.sum((weight_matrix*temp)**2)
    return reg

def get_name(a,b,index):
    if index==0:
        return "conv"+str(a)+"_"+str(b)
    else:
        return "block"+str(a)+"_conv"+str(b)

def my_pool(x,depth,name,with_bn=False):
    if with_bn:
        x = Conv2D(depth,kernel_size=(2,2),strides = (2,2),name=name,use_bias=False)(x)
        x  = BatchNormalization(name=name+"_bn")(x)
        x = Activation("relu")(x)
    else:
        x = Conv2D(depth,kernel_size=(2,2),strides = (2,2),name=name)(x)
    return x

def se(x,channel,name,config):
    l2_reg = config["l2_reg"]
    input = x
    x = GlobalAveragePooling2D()(x)
    x = Dense(channel//8,activation="relu",name=name+"_dense1",kernel_regularizer=l2(l2_reg))(x)
    x = Dense(channel,activation="sigmoid",name=name+"_dense2",kernel_regularizer=l2(l2_reg))(x)
    x = Reshape((1,1,channel))(x)
    x = Multiply()([x,input])
    return x



def get_model(config):
    image_size = config["image_size"]
    n_classes = len(config["classes"])
    l2_reg = config["l2_reg"]
    number_values = 4*20
    kernel_initializer = "he_normal"
    ki = IdentityConvInitializer(1)
    img_height, img_width = image_size[0], image_size[1]
    img_channels = 3

    
    input = Input(shape=(img_height, img_width, img_channels))
    he_normal="he_normal"
    resnet_ini = MyResNetInitializer(value = 1, var = 1)

    kr = l2(l2_reg)
    nc = kr
  #  nc = nocenterreg(0.0001)
    
    x = ZeroPadding2D(padding=((2,2),(2,2)),name="padding_1")(input)
    conv1_1 = layer(x,depth=64,kernel=(6,6),strides = (2,2),padding="valid",kernel_initializer = he_normal,kernel_regularizer=kr,name=get_name(1,1,0),with_bn=True)
    
    conv2_1 = layer(conv1_1,depth=128,kernel_initializer = resnet_ini,kernel_regularizer=nc,name=get_name(2,1,0),with_bn=True)
    conv2_2 = layer(conv2_1,depth=128,kernel_initializer = resnet_ini,kernel_regularizer=nc,name=get_name(2,2,0),with_bn=True)
    x = ZeroPadding2D(padding=((1,1),(1,1)),name="padding_2")(conv2_2)
    conv2_3 = layer(x,depth=128,kernel=(4,4),strides = (2,2),padding="valid",kernel_initializer = resnet_ini,kernel_regularizer=nc,name=get_name(2,3,0),with_bn=True)

    conv3_1 = layer(conv2_3,depth=256,kernel_initializer = resnet_ini,kernel_regularizer=nc,name=get_name(3,1,0),with_bn=True)
    conv3_2 = layer(conv3_1,depth=256,kernel_initializer = resnet_ini,kernel_regularizer=nc,name=get_name(3,2,0),with_bn=True)
    conv3_3 = layer(conv3_2,depth=256,kernel_initializer = resnet_ini,kernel_regularizer=nc,name=get_name(3,3,0),with_bn=True)
    x = ZeroPadding2D(padding=((1,1),(1,1)),name="padding_3")(conv3_3)
    conv3_4 = layer(x,depth=256,kernel=(4,4),strides = (2,2),padding="valid",kernel_initializer = resnet_ini,kernel_regularizer=nc,name=get_name(3,4,0),with_bn=True)

    conv4_1 = layer(conv3_4,kernel_initializer = resnet_ini,kernel_regularizer=nc,name=get_name(4,1,0),with_bn=True)
    conv4_2 = layer(conv4_1,kernel_initializer = resnet_ini,kernel_regularizer=nc,name=get_name(4,2,0),with_bn=True)
    conv4_3 = layer(conv4_2,kernel_initializer = resnet_ini,kernel_regularizer=nc,name=get_name(4,3,0),with_bn=True)
    conv4_4 = layer(conv4_3,kernel=(2,2),strides = (2,2),padding="valid",kernel_initializer = resnet_ini,kernel_regularizer = nc,name=get_name(4,4,0))

    conv5_1 = layer(conv4_4,kernel_initializer = resnet_ini,kernel_regularizer=nc,name=get_name(5,1,0),with_bn=True)
    conv5_2 = layer(conv5_1,kernel_initializer = resnet_ini,kernel_regularizer=nc,name=get_name(5,2,0),with_bn=True)
    conv5_3 = layer(conv5_2,kernel_initializer = resnet_ini,kernel_regularizer=nc,name=get_name(5,3,0),with_bn=True)
    conv5_4 = layer(conv5_3,kernel=(2,2),strides = (2,2),padding ="valid",kernel_initializer = resnet_ini,kernel_regularizer = nc,name=get_name(5,4,0))

    conv6_1 = layer(conv5_4,kernel_initializer = resnet_ini,kernel_regularizer=nc,name=get_name(6,1,0),with_bn=True)
    conv6_2 = layer(conv6_1,kernel_initializer = resnet_ini,kernel_regularizer=nc,name=get_name(6,2,0),with_bn=True)
    conv6_3 = layer(conv6_2,kernel_initializer = resnet_ini,kernel_regularizer=nc,name=get_name(6,3,0),with_bn=True)
    conv6_4 = layer(conv6_3,kernel=(2,2),strides = (2,2),padding="valid",kernel_initializer = resnet_ini,kernel_regularizer = kr,name=get_name(6,4,0))

    conv7_1 = layer(conv6_4,kernel_initializer = resnet_ini,kernel_regularizer=kr,name=get_name(7,1,0),with_bn=True)
    conv7_2 = layer(conv7_1,kernel_initializer = resnet_ini,kernel_regularizer=kr,name=get_name(7,2,0),with_bn=True)
    conv7_3 = layer(conv7_2,kernel_initializer = resnet_ini,kernel_regularizer=kr,name=get_name(7,3,0),with_bn=True) 
    
    conv7_4 = layer(conv7_3,kernel=(2,2),strides = (2,2),padding="valid",kernel_initializer = resnet_ini,kernel_regularizer = kr,name=get_name(7,4,0))

    conv8_1 = layer(conv7_4,kernel_initializer = resnet_ini,kernel_regularizer=kr,name=get_name(8,1,0),with_bn=True)
    conv8_2 = layer(conv8_1,kernel_initializer = resnet_ini,kernel_regularizer=kr,name=get_name(8,2,0),with_bn=True)
    conv8_3 = layer(conv8_2,kernel_initializer = resnet_ini,kernel_regularizer=kr,name=get_name(8,3,0),with_bn=True) 
    
    layer1 = conv4_3
    layer2 = conv5_3
    layer3 = conv6_3
    layer4 = conv7_3
    layer5 = conv8_3
    #layer1 = BatchNormalization(name="bn_1")(layer1)
    #layer2 = BatchNormalization(name="bn_2")(layer2)
    #layer3 = BatchNormalization(name="bn_3")(layer3)
    #layer4 = BatchNormalization(name="bn_4")(layer4)
    clayers = [layer1,layer2,layer3,layer4]
    rlayers = [layer1,layer2,layer3,layer4]

    #return Model(inputs = input,outputs=[layer1,layer2,layer3,layer4])
    
    blength = config["blength"]
    classification_model = get_classification_model(config,"classification_model",blength)
    regression_model = get_regression_model(config,"regression_model",blength)

    classification_tensor_list=[]
    regression_tensor_list=[]
   
    for i in range(len(clayers)):
        clayer = clayers[i]
        rlayer = rlayers[i]
        if i<5:
            classification_tensor_list+=classification_model(clayer)
            regression_tensor_list+=regression_model(rlayer)
        else:
            classification_tensor_list+=[classification_model(clayer)]
            regression_tensor_list+=[regression_model(rlayer)]
  #  classification_tensor_list = [classification_model(feature_layer) for feature_layer in layers]
   # regression_tensor_list = [regression_model(featuer_layer) for featuer_layer in layers]

    length = len(config["info"])
    classification_tensor_list = classification_tensor_list[:length]
    regression_tensor_list = regression_tensor_list[:length]
  #  print(classification_tensor_list)

   
    classification_tensor = Concatenate(axis = 1,dtype="float32",name="classification_concatenate")(classification_tensor_list)
    regression_tensor = Concatenate(axis = 1,dtype="float32",name="regression_concatenate")(regression_tensor_list)

    combine = Concatenate(axis = -1,name = "combine")([classification_tensor,regression_tensor])
    train_model = Model(inputs = input,outputs = [classification_tensor,regression_tensor])
    return train_model

    out = RegressBoxes(anchors = get_anchors(config),config = config)(combine)
   # train_model = Model(inputs = input,outputs = out)
   # return train_model
    if config["regression_type"]=="fine":
        print("regression type:",config["regression_type"],",use SuppressionFine")
        infer_out = SuppressionFine(config,0.3,0.45,200,400)(out)
        eval_out = SuppressionFine(config,0.01,0.45,200,400)(out)
    else:
        print("regression type:",config["regression_type"],",use SuppressionCoarse")
        infer_out = SuppressionCoarse(config,0.3,0.50,200,400)(out)
        eval_out = SuppressionCoarse(config,0.01,0.50,200,400)(out)
    infer_model = Model(inputs =input,outputs = infer_out)
    eval_model = Model(inputs =input,outputs = eval_out)
    return train_model,infer_model,eval_model

def load(config):

    path = "weights/320_cos/epoch-212_loss-3.9396.h5"
    print("load")
    print("loading model from:"+path)
    print(path)
    if config["regression_type"]=="fine":
        ssdLoss = SSDLoss80(config)
    else:
        ssdLoss = SSDLoss(config)
    train_model = load_model(path,
                         custom_objects = {
                             #"Normalization":Normalization,
                                        "MyResNetInitializer":MyResNetInitializer,
                                      #  "nocenterreg":nocenterreg,
                                        "reg":nocenterreg(0),
                                           "CosineDecayWithWarmUp":CosineDecayWithWarmUp,
                                           "IdentityConvInitializer":IdentityConvInitializer,
                                           "BiasInitializer":BiasInitializer,
                                           'class_loss':ssdLoss.class_loss,
                                           "regress_loss":ssdLoss.regress_loss})
    x = train_model.inputs
  #  combine = train_model.outputs[0]
    combine = Concatenate(axis = -1,name = "combine")(train_model.outputs)
  #  return train_model

    out = RegressBoxes(anchors = get_anchors(config),config = config)(combine)
    if config["regression_type"]=="fine":
        print("regression type:",config["regression_type"],",use SuppressionFine")
        infer_out = SuppressionFine(config,0.3,0.45,200,400)(out)
        eval_out = SuppressionFine(config,0.01,0.45,200,400)(out)
    else:
        print("regression type:",config["regression_type"],",use SuppressionCoarse")
        infer_out = SuppressionCoarse(config,0.3,0.45,200,400)(out)
        eval_out = SuppressionCoarse(config,0.01,0.45,200,400)(out)
    infer_model = Model(inputs =x,outputs = infer_out)
    eval_model = Model(inputs =x,outputs = eval_out)
    return train_model,infer_model,eval_model
