from PIL import Image
import numpy as np


def save_images(generator_output, output_directory):
    """
    :param generator_output: a Numpy array of shape (num_inputs, 16, 16, 3) representing
    (num_inputs, width, height, num_channels)
    :param output_directory: the filepath to the directory where outputs will be saved
    """

    # For each cloud array convert to png and save
    for cloud_num in len(generator_output):
        image = Image.fromarray(generator_output[cloud_num])
        file_name = "{}/cloud{}".format(output_directory, cloud_num)
        image.save(file_name, "PNG")
