import numpy as np
class Flip:
  
    def transform(self,image,label):
        p = np.random.uniform(0,1)
        if p>0.5:
            return image,label
        img_height, img_width = image.shape[:2]
       # print(image.shape)
        image = image[:,::-1]
        label = np.copy(label)
       # print(label)
        if len(label)>0:
            label[:, [1, 3]] = img_width - label[:, [3, 1]]
       # print(label)
        return image,label

    def __call__(self, images, labels=None, return_inverter=False):

        batch = len(images)
        for i in range(batch):
            image,label = images[i],labels[i]

            image, label = self.transform(image, label)
            images[i]=image
            labels[i]=label
        return images,labels

