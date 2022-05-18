import cv2
import numpy as np

class Crop:

    def __init__(self,background_size=(600,600),crop_size=(500,500),background_color=[0,0,0]):

        self.background_size = background_size
        self.crop_size = crop_size
        self.background_color=background_color


    def __call__(self,image,label):

        b_w,b_h = self.background_size[0],self.background_size[1]
        height,width,channel = image.shape


        # make sure background is bigger than image
        if height>b_h:
            b_h=height
        if width>b_w:
            b_w=width


        # create new image and label 
        out = np.zeros((b_h,b_w,3),dtype = np.uint8)
        out[:,:]=self.background_color
        
        h_limit = b_h-height
        w_limit = b_w-width
        h_start = np.random.randint(0,h_limit+1)
        w_start = np.random.randint(0,w_limit+1)

        out[h_start:h_start+image.shape[0],w_start:w_start+image.shape[1],:]=image
        label[:,[1,3]]+=w_start
        label[:,[2,4]]+=h_start

        for i in range(10):
            x = np.random.randint(0,b_w-self.crop_size[0]+1)
            y = np.random.randint(0,b_h-self.crop_size[1]+1)
            if not self.test(label,x,y):
                true_label_index = (label[:,1]>=x)*(label[:,2]>=y)*(label[:,3]<=x+self.crop_size[0]-1)*(label[:,4]<=y+self.crop_size[1]-1)
                if np.sum(true_label_index)==0:
                    continue

                label = label[true_label_index]
                label[:,[1,3]]-=x
                label[:,[2,4]]-=y
                label[:,[1,3]] = np.clip(label[:,[1,3]], a_min=0, a_max=self.crop_size[0]-1)
                label[:,[2,4]] = np.clip(label[:,[2,4]], a_min=0, a_max=self.crop_size[1]-1)
                return out[y:y+self.crop_size[1],x:x+self.crop_size[0],:],label


        return out,label

    def test(self,label,x,y):
        x1 = x+self.crop_size[0]-1
        y1 = y+self.crop_size[1]-1
        t1 = (label[:,1]<x)*(label[:,3]>x)+(label[:,1]<x1)*(label[:,3]>x1)
        t2 = (label[:,2]<y)*(label[:,4]>y)+(label[:,2]<y1)*(label[:,4]>y1)
        s1 = (label[:,4]<y)+(label[:,2]>=y1)
        s2 = (label[:,3]<x)+(label[:,1]>=x1)
        index = t1*(1-s1)+t2*(1-s2)
        return np.sum(index)

    def test1(self,label,x,y):
       # label = np.copy(label)
        x1 = x+self.crop_size[0]-1
        y1 = y+self.crop_size[1]-1
        lt = (label[:,1]>=x)*(label[:,1]<=x1)*(label[:,2]>=y)*(label[:,2]<=y1)
        lb = (label[:,1]>=x)*(label[:,1]<=x1)*(label[:,4]>=y)*(label[:,4]<=y1)
        rt = (label[:,3]>=x)*(label[:,3]<=x1)*(label[:,2]>=y)*(label[:,2]<=y1)
        rb = (label[:,3]>=x)*(label[:,3]<=x1)*(label[:,4]>=y)*(label[:,4]<=y1)
        length = label.shape[0]
        for i in range(length):
            if lt[i] and not rb[i]:
                return True
            if not lt[i] and rb[i]:
                return True
            if lb[i] and not rt[i]:
                return True
            if not lb[i] and rt[i]:
                return True
        return False


class SquareImage:

    def __init__(self,background_color=[0,0,0]):

        self.background_color = background_color

    def __call__(self,image,labels):

        height,width = image.shape[0],image.shape[1]
        output_size = max(height,width)
        out = np.zeros((output_size,output_size,3),dtype = np.uint8)
        out[:,:]=self.background_color
        h_start = int((output_size-height)/2)
        w_start = int((output_size-width)/2)
        out[h_start:h_start+height,w_start:w_start+width,:]=image
        labels[:,[1,3]]+=w_start
        labels[:,[2,4]]+=h_start

        return out,labels
