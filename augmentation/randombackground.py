import sys
sys.path.append("../")
import numpy as np
from util.box import iou
import cv2

class RandomCrop:

    def __init__(self,config,prob=0.143):
        self.prob = prob
        self.config = config
        
    def expand(self,image,label):
        config = self.config
        min_scale = 1.0
        max_scale = 4.0
        height = image.shape[0]
        width = image.shape[1]
        new_height = int(height*max_scale)
        new_width  = int(width*max_scale)
        cancas = np.zeros((new_height,new_width,3),dtype = np.uint8)
        cancas[:,:] = config["image_mean"]
        factor = np.random.uniform(min_scale, max_scale)
        w = int(factor*width)
        h = int(factor*height)
        image = cv2.resize(image,(w,h))
        label[:,[1,2,3,4]]*=factor
        w_range = new_width-w
        h_range = new_height-h
        w_start = np.random.randint(0,w_range+1)
        h_start = np.random.randint(0,h_range+1)

        cancas[h_start:h_start+h,w_start:w_start+w,:]=image
        label[:,[1,3]]+=w_start
        label[:,[2,4]]+=h_start
        return cancas,label,w_start,w_start+w,h_start,h_start+h

    def test(self,label,xmin,xmax,ymin,ymax):
        has_label = False
        for item in label:
            inside = False
            if item[1]>=xmin and item[2]>=ymin and item[3]<xmax and item[4]<ymax:
                has_label=True
                inside = True
            t1 = max(xmin,item[1])
            t2 = max(ymin,item[2])
            t3 = min(xmax,item[3])
            t4 = min(ymax,item[4])
         #   print(t3-t1,t4-t2,inside)
            if t1<t3 and t2<t4 and not inside:
                return False
        return has_label

    def crop(self,image,label,w_start,w_end,h_start,h_end):
        min_scale = 0.25
        max_scale = 1.0
        height = image.shape[0]
        width = image.shape[1]
        trail = 100
        for i in range(trail):
            h_middle = np.random.randint(h_start,h_end)
            w_middle = np.random.randint(w_start,w_end)
          #  print(i,h_middle,w_middle)
            index = np.random.randint(0,8)
            if index==0:
                xmin,xmax,ymin,ymax = 0,w_middle,0,h_middle
            elif index==1:
                xmin,xmax,ymin,ymax = w_middle,width,0,h_middle
            elif index==2:
                xmin,xmax,ymin,ymax = 0,w_middle,h_middle,height
            elif index ==3:
                xmin,xmax,ymin,ymax = w_middle,width,h_middle,height
            elif index == 4:
                xmin,xmax,ymin,ymax = 0,width,0,h_middle
            elif index == 5:
                xmin,xmax,ymin,ymax = 0,width,h_middle,height
            elif index == 6:
                xmin,xmax,ymin,ymax = 0,w_middle,0,height
            elif index == 7:
                xmin,xmax,ymin,ymax = w_middle,width,0,height
            if not self.test(label,xmin,xmax,ymin,ymax):
                continue
            print("success",index,w_start,w_end,h_start,h_end,w_middle,h_middle)
            
            image = image[ymin:ymax,xmin:xmax,:]
            indicator = []
            for item in label:
                if item[1]>=xmin and item[2]>=ymin and item[3]<xmax and item[4]<ymax:
                    indicator.append(True)
                else:
                    indicator.append(False)
            label = label[indicator]
            label[:,[1,3]]-=xmin
            label[:,[2,4]]-=ymin
            return image,label


        return image,label


    def __call__(self, images,labels):

        batch = len(images)
        for i in range(batch):
            image,label = images[i],labels[i]

            image, label = self.transform(image, label)
            images[i]=image
            labels[i]=label
        return images,labels



    def transform(self,image,label):
        print(label)

        p = np.random.uniform(0,1)
        if p>self.prob:
            return image,label

        cv2.imshow("show",image[:,:,[2,1,0]])
        k=cv2.waitKey(0)
        cv2.destroyAllWindows()

        image,label,w_start,w_end,h_start,h_end = self.expand(image,label)
        cv2.imshow("show",image[:,:,[2,1,0]])
        k=cv2.waitKey(0)
        cv2.destroyAllWindows()
        image,label = self.crop(image,label,w_start,w_end,h_start,h_end)
        return image,label
