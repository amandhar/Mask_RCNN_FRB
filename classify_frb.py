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

#%matplotlib inline 

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))





# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

from FRBDataset import FRBDataset
from FRBConfig import FRBConfig
config = FRBConfig()

dataset_train = FRBDataset(mode="train")
dataset_train.load_frbs(10, config.IMAGE_MAX_DIM, config.IMAGE_MIN_DIM)
dataset_train.prepare()

dataset_val = FRBDataset(mode="val")
dataset_val.load_frbs(10, config.IMAGE_MAX_DIM, config.IMAGE_MIN_DIM) # make sure these aren't in training set
dataset_val.prepare()

'''
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE, 
            epochs=1, 
            layers='heads')
'''

model = modellib.MaskRCNN(mode="inference", 
                          config=config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
model_path = model.find_last()[1]

# Load trained weights (fill in path to trained weights here)
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)


for i in range(5):

    # Test on a random image
    image_id = random.choice(dataset_val.image_ids)
    original_image, image_meta, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset_val, config, 
                               image_id, use_mini_mask=False)
        
    log("original_image", original_image)
    log("image_meta", image_meta)
    log("gt_bbox", gt_bbox)
    log("gt_mask", gt_mask)

    visualize.display_instances(original_image, gt_bbox[:,:4], gt_mask, gt_bbox[:,4], 
                                dataset_train.class_names, figsize=(8, 8))
    plt.savefig("frb_training_images/gt_" + str(i))

    results = model.detect([original_image], verbose=1)

    r = results[0]
    visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], 
                                dataset_val.class_names, r['scores'], ax=get_ax())
    plt.savefig("frb_training_images/test_" + str(i))



