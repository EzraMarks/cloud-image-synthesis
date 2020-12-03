import glob
from PIL import Image
import numpy as np


def get_data(clouds_file_path, masks_file_path, dimension=16):
    """
    :param clouds_file_path: the filepath to a directory containing .png cloud images to read in
    :param masks_file_path: the filepath to a directory containing the same number of .png cloud masks to read in
    :param dimension: an integer representing the input width and height
    :return: A normalized NumPy array of cloud images of type np.float32 with shape (num_inputs, 286, 286, 3)
    representing (num_inputs, width, height, num_channels) and a normalized Numpy array of masks of type np.float32 with
    shape (num_inputs, 256, 256) representing (num_inputs, width, height)
    """

    # Save image paths
    cloud_image_paths = []
    for image_path in glob.glob(clouds_file_path + "/*.png"):
        cloud_image_paths.append(image_path)

    mask_image_paths = []
    for image_path in glob.glob(masks_file_path + "/*.png"):
        mask_image_paths.append(image_path)

    # Count inputs and assert that there are the same number of clouds as masks
    num_inputs = len(cloud_image_paths)
    assert(num_inputs == len(mask_image_paths))

    # Read in images, resize them, and them save as NumPy arrays
    clouds = np.empty((num_inputs, dimension, dimension, 3))
    for i in range(num_inputs):
        image = Image.open(cloud_image_paths[i])
        image = image.resize((dimension, dimension))
        clouds[i] = np.asarray(image)

    masks = np.empty((num_inputs, dimension, dimension))
    for i in range(num_inputs):
        image = Image.open(mask_image_paths[i])
        image = image.resize((dimension, dimension))
        masks[i] = np.asarray(image)

    # Normalize inputs
    clouds = clouds / np.float32(255.0)
    masks = masks / np.float32(255.0)

    return clouds, masks
