import numpy as np
import cv2

class Resize:

    def __init__(self,image_shape=None,method = None):
        self.image_shape = image_shape
        self.interpolation_modes=[cv2.INTER_NEAREST,
                                      cv2.INTER_LINEAR,
                                      cv2.INTER_CUBIC,
                                      cv2.INTER_AREA,
                                      cv2.INTER_LANCZOS4]
        self.method = method

    def __call__(self,images,labels):

     #   print(labels)

        batch = len(images)
        for i in range(batch):
            ratio = self.image_shape[0]/images[i].shape[0],self.image_shape[1]/images[i].shape[1]
            if self.method:
                interpolation = self.interpolation_modes[self.method]
            else:
                interpolation = np.random.choice(self.interpolation_modes)
            images[i] = cv2.resize(images[i],
                           dsize=self.image_shape,
                           interpolation=interpolation)
            if labels is None:
                continue
            if len(labels[i])==0:
                continue
            labels[i][:,1]*=ratio[1]
            labels[i][:,2]*=ratio[0]
            labels[i][:,3]*=ratio[1]
            labels[i][:,4]*=ratio[0]
        return images,labels

class CenterResize:

    def __init__(self,image_shape=None,method = None,padding = 15):
        self.image_shape = image_shape
        self.interpolation_modes=[cv2.INTER_NEAREST,
                                      cv2.INTER_LINEAR,
                                      cv2.INTER_CUBIC,
                                      cv2.INTER_AREA,
                                      cv2.INTER_LANCZOS4]
        self.method = method
        self.padding = padding

    def __call__(self,images,labels):

        batch = len(images)
        for i in range(batch):
            if images[i].shape[0]!=self.image_shape[0] or images[i].shape[1]!=self.image_shape[1]:
                ratio = self.image_shape[0]/images[i].shape[0],self.image_shape[1]/images[i].shape[1]
                if self.method:
                    interpolation = self.interpolation_modes[self.method]
                else:
                    interpolation = np.random.choice(self.interpolation_modes)
                images[i] = cv2.resize(images[i],
                           dsize=self.image_shape,
                           interpolation=interpolation)

                labels[i][:,1]*=ratio[1]
                labels[i][:,2]*=ratio[0]
                labels[i][:,3]*=ratio[1]
                labels[i][:,4]*=ratio[0]
            temp = np.zeros((self.image_shape[0]+self.padding*2,self.image_shape[1]+self.padding*2,3),dtype = np.uint8)
            temp[:,:]=np.array([123, 117, 104])
            temp[self.padding:self.padding+self.image_shape[0],self.padding:self.padding+self.image_shape[1]]=images[i]
            images[i]=temp
            labels[i][:,[1,2,3,4]]+=self.padding
        return images,labels
