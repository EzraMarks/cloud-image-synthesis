from output import save_images
import tensorflow as tf
from generator import Generator
from discriminator import Discriminator
from preprocess import Preprocess
import math


def train(real_images, masks, generator, discriminator):
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
    discriminator.optimizer.apply_gradients(zip(d_gradients, discriminator.trainable_variables))

    g_gradients = tape.gradient(g_loss, generator.trainable_variables)
    generator.optimizer.apply_gradients(zip(g_gradients, generator.trainable_variables))

    save_images(fake_images[0:1], "../results")


def main():
    # Define constants
    output_width_and_height = 256
    batch_size = 4
    num_epochs = 1

    # Initialize preprocess and the models
    preprocess = Preprocess("../swimseg/images", "../swimseg/GTmaps", batch_size, dimension=output_width_and_height)
    generator = Generator()
    discriminator = Discriminator(dimension=output_width_and_height)

    # For each epoch train the models on each batch of inputs
    for epoch in range(num_epochs):
        while True:
            clouds, masks = preprocess.get_data()
            if clouds is None or masks is None:
                break

            train(tf.convert_to_tensor(clouds, dtype=tf.float32), tf.convert_to_tensor(masks, dtype=tf.float32),
                  generator, discriminator)


if __name__ == '__main__':
    main()
