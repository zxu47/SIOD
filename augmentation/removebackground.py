import sys
sys.path.append("../")
import numpy as np
from util.box import iou
import cv2

class RemoveBackground:

    def __init__(self,config,prob=0.5,ratio = 0.2):
        self.prob = prob
        self.ratio = ratio
        self.config = config
        
    def transform(self,image,label):


        p = np.random.uniform(0,1)
        if p>self.prob:
            return image,label

        config = self.config

        height,width,depth = image.shape
        cancas = np.zeros(image.shape,dtype = np.uint8)
        cancas[:,:] = config["image_mean"]
        length = label.shape[0]
        ratio = self.ratio
        for i in range(length):
            classindex,xmin,ymin,xmax,ymax = label[i]
            t1 = int(xmin-np.random.rand()*ratio*(xmax-xmin))
            t2 = int(ymin-np.random.rand()*ratio*(ymax-ymin))
            t3 = int(xmax+np.random.rand()*ratio*(xmax-xmin))
            t4 = int(ymax+np.random.rand()*ratio*(ymax-ymin))
            t1 = max(t1,0)
            t2 = max(t2,0)
            t3 = min(t3,width)
            t4 = min(t4,height)
            cancas[t2:t4,t1:t3,:]=image[t2:t4,t1:t3,:]

        return cancas,label

    


    def __call__(self, images,labels):

        batch = len(images)
        for i in range(batch):
            image,label = images[i],labels[i]

            image, label = self.transform(image, label)
            images[i]=image
            labels[i]=label
        return images,labels



   