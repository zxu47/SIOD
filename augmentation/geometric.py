from random import uniform
import random
import numpy as np
import cv2


class GeometricAugmentation:


    def __init__(self,
            rotation=None,#[-0.1,0.1],
            translation=None,#[[-0.1,0.1],[-0.1,0.1]],
            shear=None,#[-0.1,0.1],
            scale =None,#[[0.9,1,1],[0.9,1,1]],
            flip = 0.5
                ):

        self.rotation = rotation
        self.translation = translation
        self.shear = shear
        self.scale = scale
        self.flip = flip
        self.background_color = [123, 117, 104]


    def __call__(self, images,labels):

        batch = len(images)
        for i in range(batch):
            image,label = images[i],labels[i]

            image, label = self.transform(image, label)
            images[i]=image
            labels[i]=label
        return images,labels


    def transform(self,image,bbox):
        random.seed()
        matrix = np.identity(3)
        if self.rotation:
            angle = uniform(self.rotation[0],self.rotation[1])
            temp = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle),  np.cos(angle), 0],
            [0, 0, 1]
            ])
            matrix = np.matmul(matrix,temp)
        if self.translation:
            x,y = uniform(self.translation[0][0],self.translation[0][1]),uniform(self.translation[1][0],self.translation[1][1])
            shape = image.shape
            x = x*shape[0]
            y = y*shape[1]
            temp = np.array([
                    [1, 0, x],
                    [0, 1, y],
                    [0, 0, 1]])
            matrix = np.matmul(matrix,temp)
        if self.shear:
            angle = uniform(self.shear[0],self.shear[1])
            temp = np.array([
                    [1, -np.sin(angle), 0],
                    [0,  np.cos(angle), 0],
                    [0, 0, 1]])
            matrix = np.matmul(matrix,temp)
        if self.scale:
            x,y = uniform(self.scale[0][0],self.scale[0][1]),uniform(self.scale[1][0],self.scale[1][1])
            temp = np.array([
                [x, 0, 0],
                [0, y, 0],
                [0, 0, 1]])
            matrix = np.matmul(matrix,temp)
        if self.flip:
            factor = uniform(0,1)
            if factor<=self.flip:
                temp = np.array([
                    [-1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]])
                matrix = np.matmul(matrix,temp)
        height,width,channel = image.shape
        move = np.array([
                    [1, 0, -width/2],
                    [0, 1, -height/2],
                    [0, 0, 1]])
        anti_move=np.array([
                    [1, 0, width/2],
                    [0, 1, height/2],
                    [0, 0, 1]])
        matrix = np.linalg.multi_dot([anti_move,matrix,move])
        image = cv2.warpAffine(
            image,
            matrix[:2, :],
            dsize       = (image.shape[1], image.shape[0]),
            borderValue = self.background_color
            )

        for i in range(len(bbox)):
            index,x1,y1,x2,y2 = bbox[i]
            points = matrix.dot([
                [x1, x2, x1, x2],
                [y1, y2, y2, y1],
                [1,  1,  1,  1 ],
            ])
            min_corner = points.min(axis=1)
            max_corner = points.max(axis=1)
            bbox[i][1]=min_corner[0]
            bbox[i][2]=min_corner[1]
            bbox[i][3]=max_corner[0]
            bbox[i][4]=max_corner[1]
        return image,bbox

if __name__=="__main__":
    path = '/home/zhiwei/program/dp/dataset/VOCdevkit/VOC2007/JPEGImages/000005.jpg'
    image = cv2.imread(path)
    labels = [[1,263,211,324,339],
              [1,165,264,253,372],
              [1,5,244,67,374],
              [1,241,194,295,299],
              [1,277,186,312,220]]
    aug = GeometricAugmentation()
    height,width,channel = image.shape
    for i in range(1):
        image,labels = aug(image,labels)
        temp = image
        for label in labels:
            index,x1,y1,x2,y2 = map(int,label)
            if x1>=0 and x2<width and x1<x2 and y1>=0 and y2<height and y1<y2:
                temp = cv2.rectangle(temp, (x1,y1), (x2,y2),(0,255,0), 2)

        cv2.imshow('test',temp)
        k=cv2.waitKey(0)
    cv2.destroyAllWindows()
   