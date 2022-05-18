from random import uniform
import cv2
import numpy as np

class VisiualAugmentation:


    def __init__(self,
            contrast_range=(0.9, 1.1),
            brightness_range=(-.1, .1),
            hue_range=(-0.05, 0.05),
            saturation_range=(0.95, 1.05)
                 ):
        self.contrast_range = contrast_range
        self.brightness_range = brightness_range
        self.hue_range = hue_range
        self.saturation_range = saturation_range


    def __call__(self, images,labels):

        batch = len(images)
        for i in range(batch):
            image,label = images[i],labels[i]
            image, label = self.transform(image, label)
            images[i]=image
            labels[i]=label
        return images,labels

    def transform(self, image,labels):
        """ Apply a visual effect on the image.

        Args
            image: Image to adjust
        """
        if self.contrast_range:
            factor = uniform(self.contrast_range[0],self.contrast_range[1])
            mean = image.mean(axis=0).mean(axis=0)
            image = np.clip((image - mean) * factor + mean,0,255).astype(np.uint8)

        if self.brightness_range:
            factor = uniform(self.brightness_range[0],self.brightness_range[1])
            image = np.clip(image+factor*255,0,255).astype(np.uint8)

        if self.hue_range or self.saturation_range:

            image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

            if self.hue_range:
                factor = uniform(self.hue_range[0],self.hue_range[1])
                image[..., 0] = np.mod(image[..., 0] + factor * 180, 180)
            if self.saturation_range:
                factor = uniform(self.saturation_range[0],self.saturation_range[1])
                image[..., 1] = np.clip(image[..., 1] * factor, 0 , 255)

            image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

        return image,labels

if __name__=="__main__":
    path = '/home/zhiwei/program/dp/dataset/VOCdevkit/VOC2007/JPEGImages/009963.jpg'
    image = cv2.imread(path)
    aug = VisiualAugmentation()
    for i in range(5):
        image,labels= aug(image,None)
        cv2.imshow('test',image)
        k=cv2.waitKey(0)
    cv2.destroyAllWindows()
