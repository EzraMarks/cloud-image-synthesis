import tensorflow as tf

class Discriminator(tf.keras.Model):
    def __init__(self, size="16x16"):
        """
        The Discriminator class contains the model architecture for the classification network which determines whether
        mask, image pairs are real or fabricated
        :param size: the size of the input images and masks. Must be "16x16" or "286x286"
        """

        # Validate and record the input size
        assert(size == "16x16" or size == "286x286",
               "Discriminator initializer param size must be \"16x16\" or \"1286x286\"")
        if self.size == "16x16":
            self.dimension = 16
        else:
            self.dimension = 286

        kernel_initializer = tf.keras.initializers.RandomNormal(stddev=0.02)
        conv_args = dict(kernel_size=4, strides=2, padding='same', use_bias=False,
                         kernel_initializer=kernel_initializer)

        # Batch norm is not applied to the first layer so we use bias
        self.conv_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, padding='same', use_bias=False,
                                             kernel_initializer=kernel_initializer)

        self.conv_2 = tf.keras.layers.Conv2D(filters=128, **conv_args)
        self.batch_norm_2 = tf.keras.layers.BatchNormalization()

        # Layers for 286x286 images
        if self.size == 286:
            self.conv_3 = tf.keras.layers.Conv2D(filters=256, **conv_args)
            self.batch_norm_3 = tf.keras.layers.BatchNormalization()
            self.conv_4 = tf.keras.layers.Conv2D(filters=512, **conv_args)
            self.batch_norm_4 = tf.keras.layers.BatchNormalization()
            self.conv_5 = tf.keras.layers.Conv2D(filters=512, **conv_args)
            self.batch_norm_5 = tf.keras.layers.BatchNormalization()
            self.conv_6 = tf.keras.layers.Conv2D(filters=512, **conv_args)
            self.batch_norm_6 = tf.keras.layers.BatchNormalization()

        # This layer has no trainable params; it can be reused multiple times
        self.relu = tf.keras.layers.LeakyReLU(alpha=0.2)

        # Maps to one dimension
        self.conv_final = tf.keras.layers.Conv2D(
            filters=1, kernel_size=1, strides=1, padding='same', kernel_initializer=kernel_initializer)
        self.batch_norm_final = tf.keras.layers.BatchNormalization()

    def call(self, inputs):
        """
        Runs a forward pass on a batch of mask, image pairs
        :param inputs: a batch of inputs represented by a Tensor of size (num_inputs, width, height, num_channels) where
        num_channels is 4: one channel for the mask and 3 for the image
        :return: A tensor with a scalar score for each pair of the shape (num_inputs, 1)
        """
       
