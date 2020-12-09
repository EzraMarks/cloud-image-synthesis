from PIL import Image
import numpy as np
import os


def save_images(images, output_directory, file_name):
    """
    :param images: a Tensor of shape (num_inputs, 16, 16, 3) representing
    (num_inputs, width, height, num_channels)
    :param output_directory: the filepath to the directory where outputs will be saved
    :param file_name: the name of the output file (without file extensions)
    """

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # For each cloud array convert to png and save
    for i in range(len(images)):
        image = images[i].numpy() * 255
        image = image.astype(np.uint8)
        image = Image.fromarray(image)
        image = image.convert("RGB")
        path = "{}/{}-{}.png".format(output_directory, file_name, i)
        image.save(path, "PNG")
