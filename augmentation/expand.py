import numpy as np


class Expand:

    def __init__(self,config,min_scale=1,max_scale=4,prob=0.5):
        self.name = "Expand"
        self.config = config
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.prob = prob


    def __call__(self,image,label):


        p = np.random.uniform(0,1)
        if p <self.prob:
            return image,label
        factor = np.random.uniform(self.min_scale, self.max_scale)
        height = int(factor*image.shape[0])
        width = int(factor*image.shape[1])
        h_limit = height - image.shape[0]
        w_limit = width - image.shape[1]
        h_start = np.random.randint(0,h_limit+1)
        w_start = np.random.randint(0,w_limit+1)
       # print("height,width,h_start,w_start",height,width,h_start,w_start)
        out = np.zeros((height,width,3),dtype = np.uint8)
        out[:,:]=self.config["image_mean"]
        out[h_start:h_start+image.shape[0],w_start:w_start+image.shape[1],:]=image
        label[:,[1,3]]+=w_start
        label[:,[2,4]]+=h_start
       # print(label)
        return out,label
