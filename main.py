import tensorflow as tf
from generator import Generator
from discriminator import Discriminator
from preprocess import get_data


# TODO: This is a somewhat generic train function (adapted from GANs lab) which should be altered.
#       It's not currently taking a different number of steps for the generator and
#       the discriminator.

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
        fake_images = generator(masks)

        # TODO: concatenate masks and ground_truth_images on the channels axis and modify discriminator call to pass one
        # tensor with 4 channels (you will need to reshape masks to add the extra dimension first)
        logits_real = discriminator(real_images, masks)
        logits_fake = discriminator(fake_images, masks)

        d_loss = discriminator.loss(logits_fake, logits_real)
        g_loss = generator.loss(logits_fake)

    d_gradients = tape.gradient(d_loss, discriminator.trainable_variables)
    discriminator.optimizer.apply_gradients(zip(d_gradients, discriminator.trainable_variables))

    g_gradients = tape.gradient(g_loss, generator.trainable_variables)
    generator.optimizer.apply_gradients(zip(g_gradients, generator.trainable_variables))

    return


def main():
    # Define constants
    output_width_and_height = 16
    batch_size = 10
    num_epochs = 1

    # Read in training data
    ground_truth_images, masks = get_data("../swimseg/images", "../swimseg/GTmaps", dimension=output_width_and_height)
    num_inputs = len(ground_truth_images)
    assert(num_inputs == len(masks), "There must be the same number of input masks and ground truth images")

    # Initialize models
    generator = Generator()
    discriminator = Discriminator(dimension=output_width_and_height)

    # For each epoch batch inputs and train models
    for epoch in range(num_epochs):
        num_batches = math.floor(num_inputs / batch_size)
        for batch in range(num_batches):
            start_index = batch * batch_size
            end_index = (batch + 1) * batch_size
            train(tf.convert_to_tensor(ground_truth_images[start_index:end_index]),
                  tf.convert_to_tensor(masks[start_index:end_index]), generator, discriminator)


if __name__ == '__main__':
    main()
