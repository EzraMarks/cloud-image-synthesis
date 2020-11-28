import tensorflow as tf
from generator import Generator

generator = Generator()
discriminator = Discriminator()

# TODO: This is a somewhat generic train function (adapted from GANs lab) which should be altered.
#       It's not currently taking a different number of steps for the generator and
#       the discriminator.
def train(real_images, masks):
    with tf.GradientTape(persistent=True) as tape:
        fake_images = generator(masks)

        logits_real = discriminator(real_images, masks)
        logits_fake = discriminator(fake_images, masks)

        d_loss = discriminator.loss(logits_fake, logits_real)
        g_loss = generator.loss(logits_fake)

    d_gradients = tape.gradient(d_loss, discriminator.trainable_variables)
    discriminator.optimizer.apply_gradients(zip(d_gradients, discriminator.trainable_variables))

    g_gradients = tape.gradient(g_loss, generator.trainable_variables)
    generator.optimizer.apply_gradients(zip(g_gradients, generator.trainable_variables))