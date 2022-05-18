import numpy as np
import cv2

class CutAug:

    def __init__(self,image_shape=None,method = None,scale_range=(0.5,2.0)):
        self.image_shape = image_shape
        self.interpolation_modes=[cv2.INTER_NEAREST,
                                      cv2.INTER_LINEAR,
                                      cv2.INTER_CUBIC,
                                      cv2.INTER_AREA,
                                      cv2.INTER_LANCZOS4]
        self.method = method
        self.scale_range = scale_range
        self.trails = 40
    def get_scale(self):
        low=0.5
        high=1.0
        scale_range=(0.5,2)
        while True:
            x = np.random.uniform(low,high)
            y = np.random.uniform(low,high)
            if x/y>=scale_range[0] and x/y<=scale_range[1]:
                return x/y

    def cut_y(self,image,label):
        height,width,_ = image.shape
        if len(label)==0:
            h_length = np.random.randint(height//3,height+1)
            start = np.random.randint(0,height-h_length+1)
            image = image[start:start+h_length,:,:]
            return image,label

        start,end = 0,height
        for i in range(self.trails):
            temp = np.random.randint(0,height-1)+0.5
            valid=True
            for item in label:
                if item[2]<temp and item[4]>temp:
                    valid = False
                    break
            if not valid:
                continue
            left = np.sum(label[:,4]<temp)
            right = np.sum(label[:,2]>temp)
            if left==0:
                start=int(temp+0.5)
                break
            if right==0:
                end = int(temp)
                break
            if np.random.randint(0,2)==0:
                end = int(temp)
            else:
                start = int(temp+0.5)
            break
     #   print(height,width,start,end)
        if start==0 and end==height:
            return image,label
        image = image[start:end,:,:]
        index =(label[:,2]>=start)*(label[:,4]<=end)
        label = label[index]
        label[:,2]-=start
        label[:,4]-=start
        return image,label

    def cut_x(self,image,label):
        height,width,_ = image.shape
        if len(label)==0:
            w_length = np.random.randint(width//3,width+1)
            start = np.random.randint(0,width-w_length+1)
            image = image[:,start:start+w_length,:]
            return image,label

        start,end = 0,width
        for i in range(self.trails):
            temp = np.random.randint(0,width-1)+0.5
            valid=True
            for item in label:
                if item[1]<temp and item[3]>temp:
                    valid = False
                    break
            if not valid:
                continue
            left = np.sum(label[:,3]<temp)
            right = np.sum(label[:,1]>temp)
            if left==0:
                start=int(temp+0.5)
                break
            if right==0:
                end = int(temp)
                break
            if np.random.randint(0,2)==0:
                end = int(temp)
            else:
                start = int(temp+0.5)
            break
     #   print(height,width,start,end)
        if start==0 and end==width:
            return image,label
        image = image[:,start:end,:]
        index =(label[:,1]>=start)*(label[:,3]<=end)
        label = label[index]
        label[:,1]-=start
        label[:,3]-=start
        return image,label


    def __call__(self,images,labels):

        batch = len(images)
        for i in range(batch):
            image,label = images[i],labels[i]
      #      print("---------")
      #      print(label)
            if np.random.randint(0,2)==0:
                image,label = self.cut_x(image,label)
                #image,label = self.cut_y(image,label)
            else:
                #image,label = self.cut_y(image,label)
                image,label = self.cut_x(image,label)
       #     print(label)
            image,label = self.ratio_aug(image,label)
            images[i]=image
            labels[i]=label

        return images,labels

    def ratio_aug(self,image,label):

        h,w,_ = image.shape
        scale = self.get_scale()
        hn,wn=h,w*scale
        target = self.image_shape[0]
        if np.random.randint(0,3)>0:
            target = np.random.randint(target//2,target)
        ratio = target/max(hn,wn)
        hn,wn=int(hn*ratio),int(wn*ratio)
        hn = max(hn,1)
        wn = max(wn,1)

        if self.method:
            interpolation = self.interpolation_modes[self.method]
        else:
            interpolation = np.random.choice(self.interpolation_modes)

        image = cv2.resize(image,dsize=(wn,hn),interpolation=interpolation)

        h_ratio,w_ratio = hn/h,wn/w

        if len(label)>0:
            label[:,1]*=w_ratio
            label[:,2]*=h_ratio
            label[:,3]*=w_ratio
            label[:,4]*=h_ratio

        return image,label

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
        #    h_start=0
        #    w_start=0
            canvs = np.zeros((self.image_shape[0],self.image_shape[1],3),dtype = np.uint8)
            canvs[:,:]=[123, 117, 104]
            canvs[h_start:h_start+h,w_start:w_start+w,:]=images[i]
            images[i]=canvs
          #  print("-------------")
          #  print(images[i][h_start,w_start,:])
          #  print(h_start,w_start)
           # print(images[i])
            if len(labels[i])==0:
                continue
            labels[i][:,1]+=w_start
            labels[i][:,2]+=h_start
            labels[i][:,3]+=w_start
            labels[i][:,4]+=h_start
        return images,labels
