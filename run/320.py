import sys
sys.path.append("../")
from dataset.coco import DataSet
from generator import Generator
from model import get_model,load
from loss import SSDLoss80,SSDLoss,SSDLossTest,SSDLossSplit

from callbacks.l2callback import L2LossCallback
#from util.image import show_image_with_label,show_result

import numpy as np
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TerminateOnNaN, CSVLogger,ReduceLROnPlateau
import cv2
import time
import tensorflow as tf
from layers.newsuppression import my_nms
import os
from tensorflow.keras.mixed_precision import experimental as mixed_precision


gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.5)

config = tf.compat.v1.ConfigProto(gpu_options = gpu_options)
config.gpu_options.allow_growth=True

sess = tf.compat.v1.Session(config = config)

policy = mixed_precision.Policy("mixed_float16")
mixed_precision.set_policy(policy)

class CosineDecayWithWarmUp(tf.keras.experimental.CosineDecay):
    def __init__(self,lr,ds,alpha=0.0,warms=0,name=None):
        self.warms = warms
        super(CosineDecayWithWarmUp,self).__init__(initial_learning_rate = lr,
                                                  decay_steps = ds,
                                                  alpha = alpha,
                                                  name=name)

    @tf.function
    def __call__(self,step):
 #       step = step+1000*40
        if step<=self.warms:
            return float(step/self.warms*self.initial_learning_rate)
        else:
            return super(CosineDecayWithWarmUp,self).__call__(step-self.warms)


ratio,ratio1,ratio2 = [1/6.55,6.55],[-1,1],[-1,10]
s1,s2,s3 = 20,32,40
size = 512
t = 1
a1 = int(size/8)
a2 = int(size/16)
a3 = int(size/32)
a4 = int(size/64)
a5 = int(size/128)
config = {
    "image_size":(size,size),
    "blength":2,
    "info" : [
    {"shape":[a1,a1],"starts":3.5, "steps":8,  "scale":[s1,2*s1],  "boxes":[t*s1,t*s1],  "ratio":ratio}, 
    {"shape":[a1,a1],"starts":3.5, "steps":8,  "scale":[s2,2*s2],  "boxes":[t*s2,t*s2],  "ratio":ratio},
   # {"shape":[a1,a1],"starts":3.5, "steps":8,  "scale":[s3,2*s3],  "boxes":[t*s3,t*s3],  "ratio":ratio},
       
    {"shape":[a2,a2],"starts":7.5, "steps":16, "scale":[2*s1,4*s1], "boxes":[t*2*s1,t*2*s1],  "ratio":ratio},
    {"shape":[a2,a2],"starts":7.5, "steps":16, "scale":[2*s2,4*s2], "boxes":[t*2*s2,t*2*s2],  "ratio":ratio},
   # {"shape":[a2,a2],"starts":7.5, "steps":16, "scale":[2*s3,4*s3], "boxes":[t*2*s3,t*2*s3],  "ratio":ratio},
        
    {"shape":[a3,a3],"starts":15.5,"steps":32, "scale":[4*s1,8*s1],"boxes":[t*4*s1,t*4*s1],  "ratio":ratio},
    {"shape":[a3,a3],"starts":15.5,"steps":32, "scale":[4*s2,8*s2],"boxes":[t*4*s2,t*4*s2],  "ratio":ratio}, 
   # {"shape":[a3,a3],"starts":15.5,"steps":32, "scale":[4*s3,8*s3],"boxes":[t*4*s3,t*4*s3],  "ratio":ratio}, 
        
    {"shape":[a4,a4],  "starts":31.5,"steps":64, "scale":[8*s1,16*s1],"boxes":[t*8*s1,t*8*s1],"ratio":ratio},
    {"shape":[a4,a4],  "starts":31.5,"steps":64, "scale":[8*s2,16*s2],"boxes":[8*s2,8*s2],"ratio":ratio},
   # {"shape":[a4,a4],  "starts":31.5,"steps":64, "scale":[8*s3,16*s3],"boxes":[8*s3,8*s3],"ratio":ratio},
  
   # {"shape":[a5,a5],  "starts":63.5,"steps":128, "scale":[16*s1,32*s1],"boxes":[t*16*s1,t*16*s1],"ratio":ratio},
   # {"shape":[a5,a5],  "starts":63.5,"steps":128, "scale":[16*s2,32*s2],"boxes":[16*s2,16*s2],"ratio":ratio},
    ],
    "variances":np.array([0.1,0.1,0.2,0.2],dtype = np.float32),
    "l2_reg" :0.0001,
    "regression_type":"coarse",
    "kernel_initializer":"he_normal",#keras.initializers.normal(mean = 0.0,stddev= 0.01,seed= None),
    "load_images_into_memory":False,
    "verbose":True,
    "nclasses":81,
    "classes":['background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 
               'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 
               'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 
               'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 
               'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 
               'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 
               'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 
               'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 
               'teddy bear', 'hair drier', 'toothbrush'],

    "image_mean":np.array([123, 117, 104]),
    "include_classes":"all",
    "batch":32,
    "shuffle":True,
    "map_mode":"sample"
}
print(config["info"])



train_model = get_model(config)
#train_model,infer_model,eval_model = get_model(config)
train_model.summary()

#train_model.load_weights("/u40/xuz131/program/coco/mynet/weights/temp/epoch-92_loss-4.7036.h5",by_name=True,skip_mismatch=True)
print("load_weights")
train_model.load_weights("/u40/xuz131/pretrain/768_wd_0.0001.h5", by_name = True,skip_mismatch=True)
#train_model.load_weights("weights/temp/epoch-1200_loss-2.8496.h5",by_name=True,skip_mismatch=True)
#train_model.load_weights("weights/512_800/epoch-800_loss-3.8667.h5",by_name=True,skip_mismatch=True)
# Define model callbacks.
bs =32

coco_train = DataSet(config,
              ["/u40/xuz131/dataset/coco/train2017"],
                    ["/u40/xuz131/dataset/coco/annotations/instances_train2017.json"],True)
train_generator = Generator(config,coco_train,"train",bs)
#for i in range(30):
 #   train_model.layers[i].trainable=False
for layer in train_model.layers:
    if "conv1_" in layer.name:# or "classification" not in layer.name or "regression" not in layer.name:
        layer.trainable=False
    print(layer.name,layer.trainable)


base = "weights/temp/"

def lr_schedule(epoch):
   # if epoch==0:
    #    return 0.0001
 #   if epoch<20:
 #       return 0.00005*(epoch+1)
    if epoch < 300:
        return 0.001
    elif epoch < 350:
        return 0.0001
    else:
        return 0.00001

l2loss = L2LossCallback(config,train_model)

model_checkpoint = ModelCheckpoint(filepath=base+'epoch-{epoch:02d}_loss-{loss:.4f}.h5',
                                   monitor='loss',
                                   verbose=1,
                                   save_best_only=False,
                                   save_weights_only=True,
                                   mode='min',
                                   period=10)
csv_logger = CSVLogger(filename=base+'training_log.csv',
                       separator=',',
                       append=True)

learning_rate_scheduler = LearningRateScheduler(schedule=lr_schedule,
                                                verbose=1)

terminate_on_nan = TerminateOnNaN()

tensorboard = TensorBoard(log_dir=base+'logs',
                          histogram_freq=1,
                          batch_size=32,
                          write_graph=False,
                          write_grads=True)

callbacks = [
            l2loss,
#            tensorboard,
            model_checkpoint,
             csv_logger,
          #   learning_rate_scheduler,
            # lr_reduce,
             terminate_on_nan]

schedual = CosineDecayWithWarmUp(lr=0.01,
                                 ds=1200*32000//bs,
                                 alpha=0.0001,
                                warms=4*32000//bs)

sgd = SGD(learning_rate=schedual, momentum=0.9, decay=0, nesterov=False)
#sgd = Adam(learning_rate = schedual)
neg_pos_ratio = 4
if config["regression_type"]=="fine":
    myloss = SSDLoss80(config)
else:
    myloss = SSDLoss(config)
#ssdLoss = SSDLoss80()
#ssdLoss = SSDLoss(config)

train_model.compile(
    loss=[myloss.class_loss,myloss.regress_loss],
    optimizer=sgd
)


train_model.fit_generator(
        generator=train_generator,
        steps_per_epoch=32000//bs,
        epochs=1200,
        verbose=1,
        callbacks=callbacks,
        use_multiprocessing=True,
        workers = 15,
        initial_epoch=0
        )
