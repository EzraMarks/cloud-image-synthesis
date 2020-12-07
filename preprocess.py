import glob
from PIL import Image
import numpy as np


class Preprocess:
    def __init__(self, clouds_file_path, masks_file_path, batch_size, dimension=16):
        """
        The preprocess class preprocesses data by batch
        :param clouds_file_path: the filepath to a directory containing .png cloud images to read in
        :param masks_file_path: the filepath to a directory containing the same number of .png cloud masks to read in
        :param batch_size: the number of inputs to be returned by each call to get_data
        :param dimension: an integer representing the input width and height
        """
        self.batch_size = batch_size
        self.dimension = dimension
        self.inputs_processed = 0

        # Save image paths
        self.cloud_image_paths = []
        for image_path in glob.glob(clouds_file_path + "/*.png"):
            self.cloud_image_paths.append(image_path)

        self.mask_image_paths = []
        for image_path in glob.glob(masks_file_path + "/*.png"):
            self.mask_image_paths.append(image_path)

        # Count inputs and assert that there are the same number of clouds as masks
        self.num_inputs = len(self.cloud_image_paths)
        assert(self.num_inputs == len(self.mask_image_paths))

    def get_data(self):
        """
        :return: A normalized NumPy array of the next batch_size cloud images of type np.float32 with shape
        (batch_size, width, height, num_channels) and a normalized Numpy array of the next batch_size masks of type
        np.float32 with shape (batch_size, width, height) or None, None if there are not batch_size more inputs
        """

        if self.inputs_processed + self.batch_size > self.num_inputs:
            return None, None

        # Read in images, resize them, and them save as NumPy arrays
        clouds = np.empty((self.batch_size, self.dimension, self.dimension, 3))
        for i in range(self.batch_size):
            print(self.cloud_image_paths[self.inputs_processed + i])
            image = Image.open(self.cloud_image_paths[self.inputs_processed + i])
            image = image.resize((self.dimension, self.dimension))
            clouds[i] = np.asarray(image)

        masks = np.empty((self.batch_size, self.dimension, self.dimension))
        for i in range(self.batch_size):
            image = Image.open(self.mask_image_paths[self.inputs_processed + i])
            image = image.resize((self.dimension, self.dimension))
            masks[i] = np.asarray(image)

        self.inputs_processed += self.batch_size

        # Normalize inputs
        clouds = clouds / np.float32(255.0)
        masks = masks / np.float32(255.0)

        return clouds, masks
