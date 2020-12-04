from output import save_images
import tensorflow as tf
from generator import Generator
from discriminator import Discriminator
from preprocess import get_data
import math

def train(real_images, masks, generator, discriminator, optimizer):
    """
    Executes one training step on a batch of inputs
    :param real_images: a tensor of shape (batch_size, width, height, channels)
    :param masks: a tensor of shape (batch_size, width, height)
    :param generator: a generator model
    :param discriminator: a discriminator model
    :return: None
    """

    with tf.GradientTape(persistent=True) as tape:
        masks = tf.expand_dims(masks, axis=-1)
        fake_images = generator(masks)

        real_inputs = tf.concat([real_images, masks], axis=-1)
        fake_inputs = tf.concat([fake_images, masks], axis=-1)

        logits_real = discriminator(real_inputs)
        logits_fake = discriminator(fake_inputs)

        d_loss = discriminator.loss(logits_fake, logits_real) / 2
        g_loss = generator.loss(logits_fake)

    d_gradients = tape.gradient(d_loss, discriminator.trainable_variables)
    optimizer.apply_gradients(zip(d_gradients, discriminator.trainable_variables))

    g_gradients = tape.gradient(g_loss, generator.trainable_variables)
    optimizer.apply_gradients(zip(g_gradients, generator.trainable_variables))

    save_images(fake_images[0:2], "../results")


def main():
    # Define constants
    output_width_and_height = 256
    batch_size = 2
    num_epochs = 10

    # Read in training data
    ground_truth_images, masks = get_data("../swimseg/images", "../swimseg/GTmaps", dimension=output_width_and_height)
    num_inputs = len(ground_truth_images)
    assert num_inputs == len(masks), "There must be the same number of input masks and ground truth images"

    # Initialize models
    generator = Generator()
    discriminator = Discriminator(dimension=output_width_and_height)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.999)

    # For each epoch batch inputs and train models
    for epoch in range(num_epochs):
        num_batches = math.floor(num_inputs / batch_size)
        for batch in range(num_batches):
            start_index = batch * batch_size
            end_index = (batch + 1) * batch_size
            train(tf.convert_to_tensor(ground_truth_images[start_index:end_index], dtype=tf.float32),
                  tf.convert_to_tensor(masks[start_index:end_index], dtype=tf.float32), generator, discriminator, optimizer)


if __name__ == '__main__':
    main()
