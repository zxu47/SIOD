from __future__ import division
import numpy as np
import inspect
from collections import defaultdict
import warnings
import sklearn.utils
from copy import deepcopy
from PIL import Image
import cv2
import csv
import os
import sys
import random
from tqdm import tqdm, trange
import h5py
from bs4 import BeautifulSoup

class Pascal:

    def __init__(self,config,images_dirs,images_set_filenames,annotations_dirs):

        self.labels_format={'class_id': 0,
                            'xmin': 1,
                            'ymin': 2,
                            'xmax': 3,
                            'ymax': 4} 
        self.labels_output_format=('class_id','xmin','ymin','xmax','ymax')
        self.load_images_into_memory = config["load_images_into_memory"]
        self.images_dirs = images_dirs
        self.images_set_filenames=images_set_filenames
        self.annotations_dirs = annotations_dirs
        self.classes = config["classes"]
        self.include_classes=config["include_classes"]
        self.config = config
        self.verbose = config["verbose"]


        self.dataset_size = 0 
        self.images = None 
        self.image_shape={}
        self.parse()

    def get_images(self,index):
        if self.images:
            return deepcopy(self.images[index])
        else:
            filename = self.filenames[index]
            image = Image.open(filename)
            return np.array(image,dtype = np.uint8)

    def get_labels(self,index,remove_diff_sample = False):
        if not remove_diff_sample:
            return deepcopy(self.labels[index])
        else:
            return self.labels[index][self.indicator[index]]

    

    def parse(self):        

        self.filenames = []
        self.image_ids = []
        self.labels = []
        self.image_infos = []
        self.indicator = []
        number=0
        total = 0

        for images_dir, image_set_filename, annotations_dir in zip(self.images_dirs, self.images_set_filenames, self.annotations_dirs):
            with open(image_set_filename) as f:
                image_ids = [line.strip() for line in f] # Note: These are strings, not integers.
                self.image_ids += image_ids

            if self.verbose: it = tqdm(image_ids, desc="Processing image set '{}'".format(os.path.basename(image_set_filename)), file=sys.stdout)
            else: it = image_ids

            for image_id in it:

                filename = '{}'.format(image_id) + '.jpg'
                self.filenames.append(os.path.join(images_dir, filename))

                if not annotations_dir is None:
                    with open(os.path.join(annotations_dir, image_id + '.xml')) as f:
                        soup = BeautifulSoup(f, 'xml')

                    boxes = [] 
                    difficult_indicator = []
                    objects = soup.find_all('object')

                    for obj in objects:
                        class_name = obj.find('name', recursive=False).text
                        class_id = self.classes.index(class_name)
                        if (not self.include_classes == "all") and class_name not in self.include_classes:
                            continue

                      #  difficult = int(obj.find("difficult",recursive = False).txt)
                        difficult = 1-int(obj.find('difficult', recursive=False).text)
                        bndbox = obj.find('bndbox', recursive=False)
                        xmin = int(bndbox.xmin.text)
                        ymin = int(bndbox.ymin.text)
                        xmax = int(bndbox.xmax.text)
                        ymax = int(bndbox.ymax.text)
                     
                        item_dict = {
                                     'image_name': filename,
                                     'image_id': image_id,
                                     'class_name': class_name,
                                     'class_id': class_id,
                                     'xmin': xmin,
                                     'ymin': ymin,
                                     'xmax': xmax,
                                     'ymax': ymax}
                        box = []
                        for item in self.labels_output_format:
                            box.append(item_dict[item])
                        boxes.append(box)
                        if difficult:
                            difficult_indicator.append(True)
                        else:
                            difficult_indicator.append(False)

                 #   if len(boxes)==0:
                  #      print("no label")
                  #  else:
                  #      print(len(boxes))

                    self.labels.append(np.array(boxes,dtype = np.float32))
                    self.indicator.append(difficult_indicator)
       # print("small object:",number)
       # print("total object:",total)


        self.dataset_size = len(self.filenames)
        self.dataset_indices = np.arange(self.dataset_size, dtype=np.int32)

        if self.load_images_into_memory:
            self.images = []
            if self.verbose: it = tqdm(range(len(self.filenames)), desc='Loading images into memory', file=sys.stdout)
            else: it = range(len(self.filenames))
            for i in it:
                with Image.open(self.filenames[i]) as image:
                    self.images.append(np.array(image, dtype=np.uint8))
                    self.image_shape[self.image_ids[i]]=[self.images[-1].shape[0],self.images[-1].shape[1]]



class Pascal_single:

    def __init__(self,config,images_dirs,images_set_filenames,annotations_dirs):

        self.labels_format={'class_id': 0,
                            'xmin': 1,
                            'ymin': 2,
                            'xmax': 3,
                            'ymax': 4} 
        self.labels_output_format=('class_id','xmin','ymin','xmax','ymax')
        self.load_images_into_memory = config["load_images_into_memory"]
        self.images_dirs = images_dirs
        self.images_set_filenames=images_set_filenames
        self.annotations_dirs = annotations_dirs
        self.classes = config["classes"]
        self.include_classes=config["include_classes"]
        self.config = config
        self.verbose = config["verbose"]


        self.dataset_size = 0 
        self.images = None 
        self.image_shape={}
        self.parse()

    def get_images(self,index):
        if self.images:
            return deepcopy(self.images[index])
        else:
            filename = self.filenames[index]
            image = Image.open(filename)
            return np.array(image,dtype = np.uint8)

    def get_labels(self,index,remove_diff_sample = False):
        if not remove_diff_sample:
            return deepcopy(self.labels[index])
        else:
            return self.labels[index][self.indicator[index]]

    

    def parse(self):        

        self.filenames = []
        self.image_ids = []
        self.labels = []
        self.image_infos = []
        self.indicator = []
        number=0
        total = 0

        for images_dir, image_set_filename, annotations_dir in zip(self.images_dirs, self.images_set_filenames, self.annotations_dirs):
            with open(image_set_filename) as f:
                image_ids = []
                for line in f:
                   # print(line)
                    line = line.strip().split(" ")
                    if len(line)==1 or line[-1]=="1":
                        image_ids.append(line[0])
                self.image_ids += image_ids
               # print()

            if self.verbose: it = tqdm(image_ids, desc="Processing image set '{}'".format(os.path.basename(image_set_filename)), file=sys.stdout)
            else: it = image_ids

            for image_id in it:

                filename = '{}'.format(image_id) + '.jpg'
                self.filenames.append(os.path.join(images_dir, filename))

                if not annotations_dir is None:
                    with open(os.path.join(annotations_dir, image_id + '.xml')) as f:
                        soup = BeautifulSoup(f, 'xml')

                    boxes = [] 
                    difficult_indicator = []
                    objects = soup.find_all('object')

                    for obj in objects:
                        class_name = obj.find('name', recursive=False).text
                        class_id = self.classes.index(class_name)
                        if (not self.include_classes == "all") and class_name not in self.include_classes:
                            continue

                      #  difficult = int(obj.find("difficult",recursive = False).txt)
                        difficult = 1-int(obj.find('difficult', recursive=False).text)
                        bndbox = obj.find('bndbox', recursive=False)
                        xmin = int(bndbox.xmin.text)
                        ymin = int(bndbox.ymin.text)
                        xmax = int(bndbox.xmax.text)
                        ymax = int(bndbox.ymax.text)
                     
                        item_dict = {
                                     'image_name': filename,
                                     'image_id': image_id,
                                     'class_name': class_name,
                                     'class_id': class_id,
                                     'xmin': xmin,
                                     'ymin': ymin,
                                     'xmax': xmax,
                                     'ymax': ymax}
                        box = []
                        for item in self.labels_output_format:
                            box.append(item_dict[item])
                        boxes.append(box)
                        if difficult:
                            difficult_indicator.append(True)
                        else:
                            difficult_indicator.append(False)

                    self.labels.append(np.array(boxes,dtype = np.float32))
                    self.indicator.append(difficult_indicator)


        self.dataset_size = len(self.filenames)
        self.dataset_indices = np.arange(self.dataset_size, dtype=np.int32)

        if self.load_images_into_memory:
            self.images = []
            if self.verbose: it = tqdm(range(len(self.filenames)), desc='Loading images into memory', file=sys.stdout)
            else: it = range(len(self.filenames))
            for i in it:
                with Image.open(self.filenames[i]) as image:
                    self.images.append(np.array(image, dtype=np.uint8))
                    self.image_shape[self.image_ids[i]]=[self.images[-1].shape[0],self.images[-1].shape[1]]

