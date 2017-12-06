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
import tensorflow as tf

from config import Config
import utils
import model as modellib
import visualize
from model import log

#%matplotlib inline 

data_path = "sample-files/"
num_imgs_to_load = 10 # Number of validation set images
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.3
# config.gpu_options.allow_growth = True
# set_session(tf.Session(config=config))

# Root directory of the project
ROOT_DIR = os.getcwd()
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

from frb import FRBConfig, FRBDataset
config = FRBConfig()

dataset_val = FRBDataset(mode="val", data_path=data_path)
dataset_val.load_frbs(num_imgs_to_load, config.IMAGE_MIN_DIM, config.IMAGE_MAX_DIM)
dataset_val.prepare()

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


# Ground Truth Image with Mask
# original_image, image_meta, gt_bbox, gt_mask =\
#     modellib.load_image_gt(dataset_val, config, 
#                            image_id, use_mini_mask=False)
    
# log("original_image", original_image)
# log("image_meta", image_meta)
# log("gt_bbox", gt_bbox)
# log("gt_mask", gt_mask)

# visualize.display_instances(original_image, gt_bbox[:,:4], gt_mask, gt_bbox[:,4], 
#                             dataset_train.class_names, figsize=(8, 8))
# plt.savefig("frb_training_images/gt_" + str(i))

i = 0
batch_size = config.BATCH_SIZE
while i < len(dataset_val.image_ids):
    batch_ids = dataset_val.image_ids[i : (i+batch_size)]
    # Load images and remove padding (42 pixels)
    batch = [dataset_val.load_image(image_id)[:, 42:, :] for image_id in batch_ids]
    results = model.detect(batch, verbose=1)
    for j in range(len(results)):
        r = results[j]
        visualize.display_instances(batch[j], r['rois'], r['masks'], r['class_ids'], 
                                    dataset_val.class_names, r['scores'])
        # Save r['masks'][0]
        # Save r['scores'][0]
        image_name = dataset_val.image_reference(batch_ids[j])['path'].split('/')[-1]
        print(image_name)
        plt.savefig("sample-results/" + image_name[:-3] + "png")
        plt.close()
    i += batch_size


