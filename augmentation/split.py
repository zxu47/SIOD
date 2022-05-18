import cv2
import numpy as np

class Split:

    def __init__(self,config,index = 0):
        self.background = config["image_mean"]
        self.index = index
        self.config = config


    def __call__(self,images,labels):
        new_images = []
        new_labels = []
        config = self.config
        batch = len(images)
        padding = 10
        scale = 2
        for i in range(batch):
            image,label = images[i],labels[i]
            area = (label[:,3]-label[:,1])*(label[:,4]-label[:,2])
            print(area)
            if self.index:
                label = label[label[:,0]==self.index]
            target_number = label.shape[0]
          #  target_number = min(target_number,5)


            new_images.append(image)
            new_labels.append(label)
         #   if image.shape[0]>=300 and image.shape[1]>=300:
         #       w_start = int(image.shape[1]/2)-150
          #      h_start = int(image.shape[0]/2)-150
         #       temp = image[h_start:h_start+300,w_start:w_start+300,:]                  
         #       new_images.append(temp)
         #       new_labels.append(np.array([[1,0.0,0.0,1.0,1.0]]))

         #   new_image = cv2.resize(image,(int(image.shape[0]*1.5),int(image.shape[1]*1.5)))
         #   height,width,_ = new_image.shape
         #   for j in range(10):
         #       w_start = np.random.randint(0,width-300)
         #       h_start = np.random.randint(0,height-300)
          #      temp =new_image[h_start:h_start+300,w_start:w_start+300,:] 
          #      new_images.append(temp)
           #     new_labels.append(np.array([[1,0.0,0.0,1.0,1.0]]))
         #   continue
            for j in range(target_number):
              #  temp = np.zeros(image.shape,dtype = np.uint8)
              #  temp[:,:]=self.background
              #  temp1 = np.random.randint(0,256,temp.shape,dtype = np.uint8)
                x1,y1,x2,y2 = label[j,1:]
                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)
              #  temp[y1:y2,x1:x2,:]=image[y1:y2,x1:x2,:]
              #  temp1[y1:y2,x1:x2,:]=image[y1:y2,x1:x2,:]
              #  new_images.append(temp)
              #  new_labels.append(np.array([np.copy(label[j])]))
              #  new_images.append(temp1)
              #  new_labels.append(np.array([np.copy(label[j])]))


                width = x2-x1
                height = y2-y1

               # temp = np.zeros(image.shape,dtype = np.uint8)
              #  temp[:,:]=self.background
              #  temp1 = np.random.randint(0,256,temp.shape,dtype = np.uint8)
             #   w_start = int((image.shape[1]-(x2-x1))/2)
             #   h_start = int((image.shape[0]-(y2-y1))/2)
             #   temp[h_start:h_start+y2-y1,w_start:w_start+x2-x1,:]=image[y1:y2,x1:x2,:]
              #  temp1[h_start:h_start+y2-y1,w_start:w_start+x2-x1,:]=image[y1:y2,x1:x2,:]
              #  new_images.append(temp)
              #  new_images.append(temp1)
              #  new_labels.append(np.array([[self.index,w_start,h_start,w_start+width-1,h_start+height-1]],dtype = np.float64))
              #  new_labels.append(np.array([[self.index,w_start,h_start,w_start+width-1,h_start+height-1]],dtype = np.float64))
                


                move = (scale-1)/2
               
              #  t1 = x1-15 if x1>=15 else x1
              #  t2 = y1-15 if y1>=15 else y1
              #  t3 = x2+15 if x2+15<image.shape[1] else x2
               # t4 = y2+15 if y2+15<image.shape[0] else y2
                t1 = int(max(0,x1-width*move))
                t2 = int(max(0,y1-height*move))
                t3 = int(min(image.shape[1],x2+move*width))
                t4 = int(min(image.shape[0],y2+move*height))
                temp = np.zeros(image.shape,dtype = np.uint8)
                temp[:,:]=self.background
                temp1 = np.random.randint(0,256,temp.shape,dtype = np.uint8)
              #  print(x1,y1,x2,y2,t1,t2,t3,t4)

                w_start = int((image.shape[1]-(t3-t1))/2)
                h_start = int((image.shape[0]-(t4-t2))/2)
              #  print(image.shape,t1,t2,t3,t4,w_start,h_start)
                temp[h_start:h_start+t4-t2,w_start:w_start+t3-t1,:]=image[t2:t4,t1:t3,:]
                temp1[h_start:h_start+t4-t2,w_start:w_start+t3-t1,:]=image[t2:t4,t1:t3,:]
                new_images.append(temp)
                new_images.append(temp1)

                new_labels.append(np.array([[self.index,w_start,h_start,w_start+width-1,h_start+height-1]],dtype = np.float64))
                new_labels.append(np.array([[self.index,w_start,h_start,w_start+width-1,h_start+height-1]],dtype = np.float64))
                

                temp = np.zeros(image.shape,dtype = np.uint8)
                temp[:,:]=self.background
                temp1 = np.copy(temp)
                w_start = 0
                h_start = 0
                temp[h_start:h_start+y2-y1,w_start:w_start+x2-x1,:]=image[y1:y2,x1:x2,:]
             #   temp1[h_start:h_start+y2-y1,w_start:w_start+x2-x1,:]=image[y1:y2,x1:x2,:]
                new_images.append(temp)
              #  new_images.append(temp1)
                new_labels.append(np.array([[self.index,w_start,h_start,w_start+width-1,h_start+height-1]],dtype = np.float64))
              #  new_labels.append(np.array([[self.index,w_start,h_start,w_start+width-1,h_start+height-1]],dtype = np.float64))

                if width+10<temp.shape[1] and height+10<temp.shape[0]:
                    temp1[10:10+y2-y1,10:10+x2-x1,:]=image[y1:y2,x1:x2,:]
                    new_images.append(temp1)
                    new_labels.append(np.array([[self.index,w_start,h_start,w_start+width-1,h_start+height-1]],dtype = np.float64))

            
        return new_images,new_labels

