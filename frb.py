from config import Config
import utils
import numpy as np

class FRBConfig(Config):
    """Configuration for training on the frb dataset.
    Derives from the base Config class and overrides values specific
    to the frb dataset.
    """
    # Give the configuration a recognizable name
    NAME = "frb"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2 # was 8 for 128x128 images

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 1 frb

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 342+42 # 42 zeros for padding

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels
    # RPN_ANCHOR_RATIOS

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 8

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 5 #4000

    # use small validation steps since the epoch is small
    VALIDATION_STEPS =1  #1000

    # Image mean (RGB)
    MEAN_PIXEL = np.zeros([3,]) #np.array([123.7, 116.8, 103.9])

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = False

'''
    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 2000
    POST_NMS_ROIS_INFERENCE = 1000

    # Pooled ROIs
    POOL_SIZE = 7
    MASK_POOL_SIZE = 14
'''
#    MASK_SHAPE = [384, 256]


"""
Extend the Dataset class and add a method to load the shapes dataset, load_shapes(), and override the following methods:
load_image()
load_mask()
image_reference()
"""
import glob
import pdb
import skimage
# import imageio
class FRBDataset(utils.Dataset):
    """Generates the frb dataset.
    """
    def __init__(self, mode="train", data_path="../data/", class_map=None):
        super().__init__()
        self.mode = mode
        self.data_path = data_path


    # Load our images the same way we did with VGG, except use self.add_image here with proper params
    def load_frbs(self, count, height, width):
        """Load the requested number of images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        self.add_class("frbs", 1, "frb")

        # Add images
        image_id = 0
        if self.mode == "train":
            paths = glob.glob(self.data_path + '*.npy')[:count]
        else:
            paths = glob.glob(self.data_path + '*.npy')[-count:]
        for img_path in paths:
            self.image_ids.append(image_id)
            self.add_image("frbs", image_id=image_id, path=img_path)
            image_id += 1


    # Load the image with corresponding image_id
    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but
        in this case it generates the image on the fly from the
        specs in image_info.
        """
        info = self.image_info[image_id]
        image = np.load(info['path'])
        # print(image.shape)
        image = image[:, :, 0]

        # print((np.min(image), np.max(image)))
        # image = (image - np.min(image)) / (np.max(image) - np.min(image))
        # image = image * 2 - 1
        # print((np.min(image), np.max(image)))
        # skimage.io.imsave('raw.png', image)
        # imageio.imwrite('rawIO.png', image)
        # print(image.shape)

        # image = np.swapaxes(image, 0, 1)[:, :, 0] # makes shape 342 * 256
        # print(image.shape)
        # image = image.reshape((342, 256, 1))
        image = np.concatenate([np.zeros((256, 42)), image], axis=1)
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
            # print((np.min(image), np.max(image)))
            # print(image.shape)
        # print(image.shape)
        # skimage.io.imsave('gray2rgb.png', image)
        # imageio.imwrite('gray2rgbIO.png', image)
        return image

    # Could use this function to return additional image info if needed
    def image_reference(self, image_id):
        """Return the data of the image."""
        return self.image_info[image_id]


    # Returns shape (384, 384, 1) mask, along with class number 1 in an np array
    def load_mask(self, image_id):
        """Generate instance masks for frbs of the given image ID."""
        info = self.image_info[image_id]
        mask = np.load(info['path'])['mask']
        mask = np.concatenate([np.zeros((42, 256)).astype(bool), mask])
        return mask[:,:,np.newaxis], np.array([1])




