from output import save_images
import tensorflow as tf
import numpy as np
from generator import Generator
from discriminator import Discriminator
from preprocess import Preprocess
import os


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

    return real_images, fake_images, d_loss, g_loss

def test(masks, generator):
    """
    Generates images based on a batch of input masks
    :param masks: a tensor of shape (batch_size, width, height)
    :param generator: a generator model
    :return generated_images: a tensor of shape (batch_size, width, height)
    """

    masks = tf.expand_dims(masks, axis=-1)
    generated_images = generator(masks)

    return generated_images

def training_loop(generator, discriminator, image_size):
    # NOTE: Define constants
    batch_size = 10
    num_epochs = 25

    preprocess = Preprocess("../swimseg/images", "../swimseg/GTmaps", batch_size, dimension=output_width_and_height)

    # For each epoch train the models on each batch of inputs
    for epoch in range(num_epochs):
        print("Epoch ", epoch)
        preprocess.inputs_processed = 0
        real_images = None
        fake_images = None
        discriminator_losses = []
        generator_losses = []

        while True:
            clouds, masks = preprocess.get_data()
            if clouds is None or masks is None:
                break

            real_images, fake_images, d_loss, g_loss = train(
                tf.convert_to_tensor(clouds, dtype=tf.float32),
                tf.convert_to_tensor(masks, dtype=tf.float32),
                generator, discriminator)

            discriminator_losses.append(str(np.average(d_loss)))
            generator_losses.append(str(np.average(g_loss)))
        
        # Save the model after every epoch
        generator.save_weights("../checkpoints-{}/generator".format(epoch))
        discriminator.save_weights("../checkpoints-{}/discriminator".format(epoch))
        save_images(real_images, "../results/real", "real-0")
        save_images(fake_images, "../results/fake", "fake-{}".format(epoch))
        if not os.path.exists("../losses"):
            os.makedirs("../losses")
        discriminator_losses_file = open("../losses/discriminator-losses.csv", "a")
        discriminator_losses_file.write("Epoch {},{}\n".format(epoch, ",".join(discriminator_losses)))
        discriminator_losses_file.close()
        generator_losses_file = open("../losses/generator-losses.csv", "a")
        generator_losses_file.write("Epoch {},{}\n".format(epoch, ",".join(generator_losses)))
        generator_losses_file.close()

def testing_loop(generator, image_size):
    # NOTE: Define constants
    batch_size = 10

    preprocess = Preprocess("../swimseg/images", "../swimseg/GTmaps", batch_size, dimension=image_size)
    preprocess.inputs_processed = 0
    
    batch_num = 0
    while True:
        clouds, masks = preprocess.get_data()
        if clouds is None or masks is None:
            break

        real_images =  tf.convert_to_tensor(clouds, dtype=tf.float32)
        fake_images = test(tf.convert_to_tensor(masks, dtype=tf.float32), generator)
        save_images(real_images, "../results/real", "real-{}".format(batch_num))
        save_images(fake_images, "../results/fake", "fake-{}".format(batch_num))
        batch_num += 1

def main():
    # NOTE: Define constants
    image_size = 256   
    is_training = False

    # Initialize the models
    generator = Generator()
    discriminator = Discriminator(dimension=image_size)
    # Load model weights from saved checkpoint
    try:
        generator.load_weights("../checkpoints/generator")
        discriminator.load_weights("../checkpoints/discriminator")
    except:
        print("WARNING: Failed to load model weights from checkpoint")
    
    if (is_training):
        training_loop(generator, discriminator, image_size)
    else:
        testing_loop(generator, image_size)


if __name__ == '__main__':
    main()
