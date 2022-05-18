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
import json

class DataSet:

    def __init__(self,config,images_dirs,annotations_filenames,ground_truth_available):

        self.labels_format={'class_id': 0,
                            'xmin': 1,
                            'ymin': 2,
                            'xmax': 3,
                            'ymax': 4} 
        self.labels_output_format=('class_id','xmin','ymin','xmax','ymax')
        self.load_images_into_memory = config["load_images_into_memory"]

        self.images_dirs = images_dirs
        self.annotations_filenames = annotations_filenames
        self.ground_truth_available = ground_truth_available

        self.classes = config["classes"]
        self.include_classes=config["include_classes"]

        self.config = config
        self.verbose = config["verbose"]


        self.dataset_size = 0 
        self.images = None 

        self.parse()

    def get_images(self,index):
        if self.images:
            return deepcopy(self.images[index])
        else:
            filename = self.filenames[index]
            image = Image.open(filename)
            return np.array(image,dtype = np.uint8)

    def get_labels(self,index,remove_diff_sample):

        if self.labels:
            return deepcopy(self.labels[index])
        else:
            return np.array([[0,0,0,0,0]],dtype = np.float32)


    def parse(self):

      #  self.include_classes = include_classes
        # Erase data that might have been parsed before.
        self.filenames = []
        self.image_ids = []
        self.labels = []
        ground_truth_available = self.ground_truth_available
        verbose = self.verbose
        if not ground_truth_available:
            self.labels = None

        # Build the dictionaries that map between class names and class IDs.
        with open(self.annotations_filenames[0], 'r') as f:
            annotations = json.load(f)

        self.cats_to_names = {} # The map between class names (values) and their original IDs (keys)
        self.classes_to_names = [] # A list of the class names with their indices representing the transformed IDs
        self.classes_to_names.append('background') # Need to add the background class first so that the indexing is right.
        self.cats_to_classes = {} # A dictionary that maps between the original (keys) and the transformed IDs (values)
        self.classes_to_cats = {} # A dictionary that maps between the transformed (keys) and the original IDs (values)
        for i, cat in enumerate(annotations['categories']):
            self.cats_to_names[cat['id']] = cat['name']
            self.classes_to_names.append(cat['name'])
            self.cats_to_classes[cat['id']] = i + 1
            self.classes_to_cats[i + 1] = cat['id']

        print("-------------------------")

        print(self.cats_to_names)
        print(self.classes_to_names)
        print(self.cats_to_classes)
        print(self.classes_to_cats)
        print("-------------------------")


        print("parse")
        total=0
        # Iterate over all datasets.
        for images_dir, annotations_filename in zip(self.images_dirs, self.annotations_filenames):
            # Load the JSON file.
            with open(annotations_filename, 'r') as f:
                annotations = json.load(f)

            if ground_truth_available:
                # Create the annotations map, a dictionary whose keys are the image IDs
                # and whose values are the annotations for the respective image ID.
                image_ids_to_annotations = defaultdict(list)
                for annotation in annotations['annotations']:
                    image_ids_to_annotations[annotation['image_id']].append(annotation)

            if verbose: it = tqdm(annotations['images'], desc="Processing '{}'".format(os.path.basename(annotations_filename)), file=sys.stdout)
            else: it = annotations['images']

            # Loop over all images in this dataset.
            for img in it:

                self.filenames.append(os.path.join(images_dir, img['file_name']))
                self.image_ids.append(img['id'])

                if ground_truth_available:
                    # Get all annotations for this image.
                    annotations = image_ids_to_annotations[img['id']]
                    boxes = []
                    for annotation in annotations:
                        cat_id = annotation['category_id']
                        # Check if this class is supposed to be included in the dataset.
                        if (not self.include_classes == 'all') and (not cat_id in self.include_classes): continue
                        # Transform the original class ID to fit in the sequence of consecutive IDs.
                        class_id = self.cats_to_classes[cat_id]
                        xmin = annotation['bbox'][0]
                        ymin = annotation['bbox'][1]
                        width = annotation['bbox'][2]
                        height = annotation['bbox'][3]
                        # Compute `xmax` and `ymax`.
                        xmax = xmin + width
                        ymax = ymin + height
                        item_dict = {'image_name': img['file_name'],
                                     'image_id': img['id'],
                                     'class_id': class_id,
                                     'xmin': xmin,
                                     'ymin': ymin,
                                     'xmax': xmax,
                                     'ymax': ymax}
                        box = []
                        for item in self.labels_output_format:
                            box.append(item_dict[item])
                        boxes.append(box)
                    if len(boxes)==0:
                        total+=1
                  #  else:
                   #     print(len(boxes))
                    self.labels.append(boxes)
        print("total:",total)
        self.dataset_size = len(self.filenames)
        self.dataset_indices = np.arange(self.dataset_size, dtype=np.int32)
        if self.load_images_into_memory:
            self.images = []
            if verbose: it = tqdm(self.filenames, desc='Loading images into memory', file=sys.stdout)
            else: it = self.filenames
            for filename in it:
                with Image.open(filename) as image:
                    self.images.append(np.array(image, dtype=np.uint8))
