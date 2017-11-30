
# coding: utf-8

# In[1]:

# import warnings
# warnings.filterwarnings('ignore')


# In[2]:

# from keras import backend as K
# from importlib import reload
# import os

# def set_keras_backend(backend):

#     if K.backend() != backend:
#         os.environ['KERAS_BACKEND'] = backend
#         reload(K)
#         assert K.backend() == backend

# set_keras_backend("tensorflow")


# # Setup

# In[3]:

import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

from config import Config
import utils
import model as modellib
import visualize
from model import log
# import IPython
# get_ipython().magic(u'matplotlib inline')

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Path to COCO trained weights
# COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")


# In[4]:

from FRBDataset import FRBDataset
from FRBConfig import FRBConfig
config = FRBConfig()


# ### Creating Training and Validation Datasets

# In[5]:

dataset_train = FRBDataset(mode="train")
dataset_train.load_frbs(config.STEPS_PER_EPOCH, config.IMAGE_MAX_DIM, config.IMAGE_MIN_DIM)
dataset_train.prepare()

dataset_val = FRBDataset(mode="val")
dataset_val.load_frbs(config.VALIDATION_STEPS, config.IMAGE_MAX_DIM, config.IMAGE_MIN_DIM) # make sure these aren't in training set
dataset_val.prepare()


# In[6]:

# average_image = np.zeros((384, 256, 3))
# for i in dataset_train.image_ids:
#     image = dataset_train.load_image(i)
#     average_image = average_image + image
# average_image = average_image / 500
# r = np.mean(average_image[:, :, 0])
# r.shape


# In[7]:

# r = np.mean(average_image[:, :, 0])
# g = np.mean(average_image[:, :, 1])
# b = np.mean(average_image[:, :, 2])
# print("{}".format((r, g, b)))


# In[8]:

# image_ids = np.random.choice(dataset_train.image_ids, 4)
# for image_id in image_ids:
#     image = dataset_train.load_image(image_id)
#     mask, class_ids = dataset_train.load_mask(image_id)
#     print(mask.shape)


# ### Creating the Model

# In[9]:

model = modellib.MaskRCNN(mode="training", config=FRBConfig(),
                          model_dir=MODEL_DIR)


# # Training

# In[10]:

# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE, 
            epochs=10, 
            layers='heads')


# In[ ]:



