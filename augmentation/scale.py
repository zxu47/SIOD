import numpy as np
import cv2

class Scale:

    def __init__(self,image_shape=None,method = None,scale_range=(0.5,2.0)):
        self.image_shape = image_shape
        self.interpolation_modes=[cv2.INTER_NEAREST,
                                      cv2.INTER_LINEAR,
                                      cv2.INTER_CUBIC,
                                      cv2.INTER_AREA,
                                      cv2.INTER_LANCZOS4]
        self.method = method
        self.scale_range = scale_range

    def __call__(self,images,labels):


        batch = len(images)
        for i in range(batch):
            h,w,_ = images[i].shape
                
            scale = np.random.uniform(self.scale_range[0],self.scale_range[1])
            hn,wn=h,w*scale
            target = self.image_shape[0]
            ratio = target/max(hn,wn)
            hn,wn=int(hn*ratio),int(wn*ratio)

            if self.method:
                interpolation = self.interpolation_modes[self.method]
            else:
                interpolation = np.random.choice(self.interpolation_modes)

            images[i] = cv2.resize(images[i],
                           dsize=(wn,hn),
                           interpolation=interpolation)

            if labels is None:
                continue
            h_ratio,w_ratio = hn/h,wn/w


            labels[i][:,1]*=w_ratio
            labels[i][:,2]*=h_ratio
            labels[i][:,3]*=w_ratio
            labels[i][:,4]*=h_ratio

        return images,labels

class RandomPad:

    def __init__(self,image_shape=None):
        self.image_shape = image_shape

    def __call__(self,images,labels):

        batch = len(images)
        for i in range(batch):
            h,w,_ = images[i].shape
          #  print(images[i])
            h_range=self.image_shape[0]-h
            w_range=self.image_shape[1]-w
            h_start = np.random.randint(0,h_range+1)
            w_start = np.random.randint(0,w_range+1)
            h_start=0
            w_start=0
            canvs = np.zeros((self.image_shape[0],self.image_shape[1],3),dtype = np.uint8)
            canvs[:,:]=[123, 117, 104]
            canvs[h_start:h_start+h,w_start:w_start+w,:]=images[i]
            images[i]=canvs
          #  print("-------------")
          #  print(images[i][h_start,w_start,:])
          #  print(h_start,w_start)
           # print(images[i])
            labels[i][:,1]+=w_start
            labels[i][:,2]+=h_start
            labels[i][:,3]+=w_start
            labels[i][:,4]+=h_start
        return images,labels
