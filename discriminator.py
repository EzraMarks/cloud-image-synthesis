import tensorflow as tf


class Discriminator(tf.keras.Model):
    def __init__(self, dimension=16):
        """
        The Discriminator class contains the model architecture for the classification network which determines whether
        mask, image pairs are real or fabricated
        :param size: the width and height of the input images and masks. Must be 16 or 286.
        """
        assert(dimension == 16 or dimension == 286, "Discriminator initializer param size must be 16 or 286")

        # Arguments to be used for most layers
        kernel_initializer = tf.keras.initializers.RandomNormal(stddev=0.02)
        conv_args = dict(kernel_size=4, strides=2, padding='same', use_bias=False,
                         kernel_initializer=kernel_initializer)

        # This layer has no trainable params; it can be reused multiple times
        self.leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.2)

        # Batch norm is not applied to the first layer so we use bias
        self.conv_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, padding='same', use_bias=True,
                                             kernel_initializer=kernel_initializer)

        self.conv_2 = tf.keras.layers.Conv2D(filters=128, **conv_args)
        self.batch_norm_2 = tf.keras.layers.BatchNormalization()

        # Layers for 286x286 images only
        if self.dimension == 286:
            self.conv_3 = tf.keras.layers.Conv2D(filters=256, **conv_args)
            self.batch_norm_3 = tf.keras.layers.BatchNormalization()
            self.conv_4 = tf.keras.layers.Conv2D(filters=512, **conv_args)
            self.batch_norm_4 = tf.keras.layers.BatchNormalization()
            self.conv_5 = tf.keras.layers.Conv2D(filters=512, **conv_args)
            self.batch_norm_5 = tf.keras.layers.BatchNormalization()
            self.conv_6 = tf.keras.layers.Conv2D(filters=512, **conv_args)
            self.batch_norm_6 = tf.keras.layers.BatchNormalization()

        # Maps output to one dimension and applies a sigmoid function
        self.conv_final = tf.keras.layers.Conv2D(filters=1, kernel_size=4, strides=1, padding='same',
                                                 activation=tf.keras.activations.sigmoid(),
                                                 kernel_initializer=kernel_initializer)

    def call(self, inputs):
        """
        Runs a forward pass on a batch of mask, image pairs
        :param inputs: a batch of inputs represented by a Tensor of size (num_inputs, width, height, num_channels) where
        num_channels is 4: one channel for the mask and 3 for the image
        :return: A tensor of the shape (num_inputs, 1) with a scalar score for each mask, image pair
        """

        output = self.conv_1(inputs)
        output = self.leaky_relu(output)

        output = self.conv_2(output)
        output = self.batch_norm_2(output)
        output = self.leaky_relu(output)

        # Layers for 286x286 images only
        if self.dimension == 286:
            output = self.conv_3(output)
            output = self.batch_norm_3(output)
            output = self.leaky_relu(output)

            output = self.conv_4(output)
            output = self.batch_norm_4(output)
            output = self.leaky_relu(output)

            output = self.conv_5(output)
            output = self.batch_norm_5(output)
            output = self.leaky_relu(output)

            output = self.conv_6(output)
            output = self.batch_norm_6(output)
            output = self.leaky_relu(output)

        output = self.conv_final(output)
        return output

    def loss(self, logits_fake, logits_real):
        """
        Calculates discriminator loss for a batch of inputs
        :param logits_fake: a tensor of size (num_inputs, 1) containing discriminator scores for each fake input
        :param logits_real: a tensor of size (num_inputs, 1) containing discriminator scores for each real input
        """
        # NOTE: Borrowed from GANs lab
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(logits_fake),
                                                                      logits=logits_fake))
        loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits_real),
                                                                       logits=logits_real))
        return loss
