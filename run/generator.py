import sys
sys.path.append("../")
import numpy as np
#import random
import cv2
from tensorflow.keras.utils import Sequence
#from augmentation.scale import Scale,RandomPad
from augmentation.resize import Resize
from augmentation.visiual import VisiualAugmentation
from augmentation.geometric import GeometricAugmentation
from augmentation.randomcrop import RandomCrop
from augmentation.removebackground import RemoveBackground
from augmentation.flip import Flip
from augmentation.aug import CutAug,RandomPad
from data_generator.data_augmentation_chain_original_ssd import SSDDataAugmentation

from anchor import get_anchors,iou

from copy import deepcopy
import time

class Generator(Sequence):

    def __init__(self,config,dataset = None,mode = "train",batch = 32,remove_diff_sample=False):

        self.remove_diff_sample=remove_diff_sample
        self.image_size = config["image_size"]
        self.classes = len(config["classes"])
        self.regression_type = "center"
        self.config = config
        self.dataset = dataset
        self.mode = mode
        self.batch = batch

        self.augmentations = []

     
        self.ssdaug = SSDDataAugmentation(config["image_size"][0],config["image_size"][1],np.array(config["image_mean"]))
        
        if mode!="val" and mode!="eval":
            self.augmentations.append(self.ssdaug)
           # self.augmentations.append(VisiualAugmentation())
            
           # self.augmentations.append(Flip())
           # self.augmentations.append(CutAug(image_shape=config["image_size"]))
           # self.augmentations.append(RandomPad(image_shape=config["image_size"]))
      #      self.augmentations.append(Resize(image_shape = config["image_size"],method = 1))
        else:
           
           # self.augmentations.append(VisiualAugmentation())
            
           # self.augmentations.append(Flip())
           # self.augmentations.append(CutAug(image_shape=config["image_size"]))
            
            #self.augmentations.append(Scale(image_shape=config["image_size"]))
            #self.augmentations.append(RandomPad(image_shape=config["image_size"]))
           # self.augmentations.append(self.ssdaug)
            self.augmentations.append(Resize(image_shape = config["image_size"],method = 1))
        
        self.size = len(self.dataset.image_ids)
        self.order = list(range(self.size))
       # if config["shuffle"]:
        #    random.shuffle(self.order)

        self.anchors = get_anchors(config)
        self.n=0
        self.max = self.__len__()
        print(self.anchors)

    def on_epoch_end(self):

        if self.config["shuffle"]:
            np.random.shuffle(self.order)

    def __len__(self):
        return int(self.size/self.batch)

    def __next__(self):
        if self.n>=self.max:
            self.n=0
        result = self.__getitem__(self.n)
        self.n+=1
        return result

    def __getitem__(self,idx):

        order = self.order
        image_batch,label_batch,image_id_batch= [],[],[]
        for  i in range(self.batch):
            image_id_batch.append(self.dataset.image_ids[order[(i+idx*self.batch)%self.size]])
            
            image = self.dataset.get_images(order[(i+idx*self.batch)%self.size])
            if len(image.shape)!=3:
                image = np.expand_dims(image,axis = -1)
                image = np.tile(image,(1,1,3))
            image_batch.append(image)


            temp_label = np.array(self.dataset.get_labels(order[(i+idx*self.batch)%self.size],self.remove_diff_sample),dtype = np.float32)
        #    print("-----------")
        #    print(image.shape)
        #    print(temp_label)
         #   print(len(temp_label))
            if temp_label.shape[0]==0:   
 #
                height,width,depth = image_batch[-1].shape
                temp_label = np.array([[-1,0,0,width,height]],dtype = np.float32)

            label_batch.append(temp_label)
        original_label_batch = deepcopy(label_batch)
        ratio = []
        for j in range(len(image_batch)):
            ratio.append([image_batch[j].shape[1]/self.config["image_size"][0],image_batch[j].shape[0]/self.config["image_size"][1]])
            

        image_batch,label_batch,neural_output = self.process(image_batch,label_batch,image_id_batch)

        image_batch = self.preprocess_image(image_batch)
        if self.mode == "train":
            return (image_batch,(neural_output,np.copy(neural_output)))
              #  yield [image_batch,neural_output,label_batch,image_id_batch,np.array(ratio)]
        elif self.mode =="val":
            return (image_batch,neural_output)
        else:
            return (image_batch,neural_output,label_batch,image_id_batch,np.array(ratio))



    def bbox_regression(self,anchor_match,label):


        anchors = self.anchors[anchor_match,:]
        if self.regression_type == "corner":
            number = anchors.shape[0]
            output = np.zeros((number,4))
            output[:,0]=(label[:,0]-anchors[:,0])/anchors[:,4]
            output[:,1]=(label[:,1]-anchors[:,1])/anchors[:,5]
            output[:,2]=(label[:,2]-anchors[:,0])/anchors[:,4]
            output[:,3]=(label[:,3]-anchors[:,1])/anchors[:,5]
            output = output/self.config["variances"]
            return output
        elif self.regression_type == "center":
            number = anchors.shape[0]
          #  print(anchors.shape)
            output = np.zeros((number,4))
            output[:,0]=((label[:,0]+label[:,2])/2-anchors[:,0])/(anchors[:,4])
            output[:,1]=((label[:,1]+label[:,3])/2-anchors[:,1])/(anchors[:,5])
            output[:,2]=np.log((label[:,2]-label[:,0])/(anchors[:,4]))
            output[:,3]=np.log((label[:,3]-label[:,1])/(anchors[:,5]))
            output = output/self.config["variances"]
            return output
        elif self.regression_type =="label_center":
            number = anchors.shape[0]
          #  print(anchors.shape)
            output = np.zeros((number,4))
            output[:,0]=((label[:,0]+label[:,2])/2-anchors[:,0])/(label[:,2]-label[:,0])
            output[:,1]=((label[:,1]+label[:,3])/2-anchors[:,1])/(label[:,3]-label[:,1])
            output[:,2]=np.log((label[:,2]-label[:,0])/(anchors[:,4]))
            output[:,3]=np.log((label[:,3]-label[:,1])/(anchors[:,5]))
            output = output/self.config["variances"]
         #   print(output)
            return output


    def compute_output(self,labels,ids):

        batch = len(labels)
        output = np.zeros((batch,self.anchors.shape[0],self.classes+5),dtype = np.float32)
        output[:,:,0]=1

        positive_threshold = 0.35
        negative_threshold = 0.40

      #  positive_threshold = 0.55
       # negative_threshold = 0.60

        for i in range(batch):
            anchor_match,gt_match=[],[]

            label = labels[i]
            if label.shape[0]==0:
                continue

            overlaps = iou(self.anchors,label[:,1:])
         
            min_inds = np.argmin(overlaps,axis = 1)

            min_overlaps = overlaps[np.arange(overlaps.shape[0]),min_inds]
            positive_indices = min_overlaps<=positive_threshold
          #  print("positive:",np.sum(positive_indices))
            """
            if label[0][0]==-1:
                print("label:",label)
                print(min_inds)
                print(np.sum(positive_indices))
            """
            ignore_indices = (min_overlaps<=negative_threshold) & ~positive_indices
           # print("ignore:",np.sum(ignore_indices))

            output[i,positive_indices,-1]=1
            output[i,ignore_indices,-1]=-1

            label_match = min_inds[positive_indices]
            output[i,positive_indices,0]=0
            output[i,positive_indices,np.array(label[label_match,0],dtype = np.int32)]=1
            

            output[i,positive_indices,-5:-1]=self.bbox_regression(positive_indices,label[min_inds[positive_indices],1:])
        return output

    def compute_output1(self,labels,ids):

        batch = len(labels)
        output = np.zeros((batch,self.anchors.shape[0],self.classes+81),dtype = np.float32)
        output[:,:,0]=1
        positive_threshold = 0.35
     #   negative_threshold = 0.40

        for i in range(batch):
            anchor_match,gt_match=[],[]

            alllabel = labels[i]
            if alllabel.shape[0]==0:
                continue

            for j in range(1,21):
                label = alllabel[alllabel[:,0]==j]
                if label.shape[0]==0:
                    continue

                overlaps = iou(self.anchors,label[:,1:])
         
                min_inds = np.argmin(overlaps,axis = 1)
                min_overlaps = overlaps[np.arange(overlaps.shape[0]),min_inds]
                positive_indices = min_overlaps<=positive_threshold
               # ignore_indices = (min_overlaps<=negative_threshold) & ~positive_indices

                output[i,positive_indices,-1]=1
               # output[i,ignore_indices,-1]=-1

                label_match = min_inds[positive_indices]
                output[i,positive_indices,0]=0
                output[i,positive_indices,np.array(label[label_match,0],dtype = np.int32)]=1
            

                output[i,positive_indices,21+4*(j-1):25+4*(j-1)]=self.bbox_regression(positive_indices,label[min_inds[positive_indices],1:])
       
        return output

    def cutmix(self,images,labels,number):
        h,w,_ = images[0].shape
        total = len(images)
        new_images,new_labels=[],[]
        for i in range(number):
            index = np.random.randint(0,total)
            image1 = cv2.resize(images[index],(h//2,w//2))
            label1 = np.copy(labels[index])
            label1[:,[1,2,3,4]]/=2

            index = np.random.randint(0,total)
            image2 = cv2.resize(images[index],(h//2,w//2))
            label2 = np.copy(labels[index])
            label2[:,[1,2,3,4]]/=2
            label2[:,[1,3]]+=w//2
            
            index = np.random.randint(0,total)
            image3 = cv2.resize(images[index],(h//2,w//2))
            label3 = np.copy(labels[index])
            label3[:,[1,2,3,4]]/=2
            label3[:,[2,4]]+=h//2


            index = np.random.randint(0,total)
            image4 = cv2.resize(images[index],(h//2,w//2))
            label4 = np.copy(labels[index])
            label4[:,[1,2,3,4]]/=2
            label4[:,[1,2,3,4]]+=h//2
            

            image = np.zeros((h,w,3))
            image[:h//2,:w//2,:]=image1
            image[:h//2,w//2:,:]=image2
            image[h//2:,:w//2,:]=image3
            image[h//2:,w//2:,:]=image4
            label1 = list(label1)
            label2 = list(label2)
            label3 = list(label3)
            label4 = list(label4)
            new_images.append(image)
            new_labels.append(np.array(label1+label2+label3+label4))
        return new_images,new_labels

    def process(self,images,labels,ids):
        for augmentation in self.augmentations:
            images,labels = augmentation(images,labels)
       # for i in range(len(images)):
           # std = abs(np.random.normal(0,20))
        #    std = 2*i
         #   images[i]=np.array(images[0],dtype=np.float32)+np.random.normal(0,std,(320,320,3))
          #  images[i]=np.clip(images[i],0,255)
       # new_images,new_labels = self.cutmix(images,labels,16)
       # for i in range(16):
       #     images[i]=new_images[i]
       #     labels[i]=new_labels[i]
        for i in range(len(labels)):
            if len(labels[i])>0:
                labels[i]=labels[i][labels[i][:,0]!=-1]
        '''
        for i in range(16):
            a,b = np.random.randint(0,16),np.random.randint(0,16)
            if a==b:
                b=(a+1)%16
            image =(images[a]+images[b])/2
            label = list(labels[a])+list(labels[b])
            label = np.array(label)
            images.append(image)
            labels.append(label)

        images = images[16:]
        labels = labels[16:]
        '''
        output = self.compute_output(labels,ids)
        #print(time.time()-start)
        return images,labels,output

    def preprocess_image(self,image_batch):
        
        image_batch = np.array(image_batch,dtype = np.float32) 
#        return image_batch/255.0
         
        image_batch[:,:,:,[0,2]]=image_batch[:,:,:,[2,0]]
        cmean=[103.939, 116.779, 123.68]
        sstd=[58.393, 57.12, 57.375]
        for i in range(3):
            image_batch[..., i] = (image_batch[..., i] - cmean[i]) / sstd[i]
        return image_batch
        
        image_batch = np.array(image_batch,dtype=np.float32)-self.config["image_mean"]
#        image_batch[:,:,:,[0,2]]=image_batch[:,:,:,[2,0]]
        return image_batch

    



