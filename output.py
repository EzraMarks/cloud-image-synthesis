from PIL import Image
import numpy as np
import time


def save_images(generator_output, output_directory):
    """
    :param generator_output: a Numpy array of shape (num_inputs, 16, 16, 3) representing
    (num_inputs, width, height, num_channels)
    :param output_directory: the filepath to the directory where outputs will be saved
    """

    # For each cloud array convert to png and save
    for cloud_num in range(len(generator_output)):
        image = generator_output[cloud_num].numpy() * 255
        image = image.astype(np.uint8)
        image = Image.fromarray(image)
        image = image.convert("RGB")
        timestamp = time.strftime("%H-%M-%S", time.localtime())
        file_name = "{}/cloud-{}-{}.png".format(output_directory, timestamp, cloud_num)
        image.save(file_name, "PNG")
