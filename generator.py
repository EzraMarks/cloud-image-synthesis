import tensorflow as tf

class Generator(tf.keras.Model):
    def __init__(self):
        """
        The Generator class contains the model architecture for the encoder-decoder
        network which generates output images from input images.
        """
        super(Generator, self).__init__()
    
        kernel_initializer = tf.keras.initializers.RandomNormal(stddev=0.02)
        conv_args = dict(kernel_size=4, strides=2, padding='same', kernel_initializer=kernel_initializer)

        # Encoder Layers:

        self.encoder_conv_1 = tf.keras.layers.Conv2D(filters=64, **conv_args)
        # Batch norm is not applied to the first layer
        self.encoder_conv_2 = tf.keras.layers.Conv2D(filters=128, **conv_args)
        self.encoder_batch_norm_2 = tf.keras.layers.BatchNormalization()
        self.encoder_conv_3 = tf.keras.layers.Conv2D(filters=256, **conv_args)
        self.encoder_batch_norm_3 = tf.keras.layers.BatchNormalization()
        self.encoder_conv_4 = tf.keras.layers.Conv2D(filters=512, **conv_args)
        self.encoder_batch_norm_4 = tf.keras.layers.BatchNormalization()
        self.encoder_conv_5 = tf.keras.layers.Conv2D(filters=512, **conv_args)
        self.encoder_batch_norm_5 = tf.keras.layers.BatchNormalization()
        self.encoder_conv_6 = tf.keras.layers.Conv2D(filters=512, **conv_args)
        self.encoder_batch_norm_6 = tf.keras.layers.BatchNormalization()
        self.encoder_conv_7 = tf.keras.layers.Conv2D(filters=512, **conv_args)
        self.encoder_batch_norm_7 = tf.keras.layers.BatchNormalization()
        self.encoder_conv_8 = tf.keras.layers.Conv2D(filters=512, **conv_args)
        # Batch norm is skipped for the bottleneck layer (paper revision)

        # This layer has no trainable params; it can be reused multiple times
        self.encoder_relu = tf.keras.layers.LeakyReLU(alpha=0.2)

        # Decoder Layers:

        self.decoder_deconv_1 = tf.keras.layers.Conv2DTranspose(filters=512, **conv_args)
        self.decoder_batch_norm_1 = tf.keras.layers.BatchNormalization()
        self.decoder_deconv_2 = tf.keras.layers.Conv2DTranspose(filters=1024, **conv_args)
        self.decoder_batch_norm_2 = tf.keras.layers.BatchNormalization()
        self.decoder_deconv_3 = tf.keras.layers.Conv2DTranspose(filters=1024, **conv_args)
        self.decoder_batch_norm_3 = tf.keras.layers.BatchNormalization()
        self.decoder_deconv_4 = tf.keras.layers.Conv2DTranspose(filters=1024, **conv_args)
        self.decoder_batch_norm_4 = tf.keras.layers.BatchNormalization()
        self.decoder_deconv_5 = tf.keras.layers.Conv2DTranspose(filters=1024, **conv_args)
        self.decoder_batch_norm_5 = tf.keras.layers.BatchNormalization()
        self.decoder_deconv_6 = tf.keras.layers.Conv2DTranspose(filters=512, **conv_args)
        self.decoder_batch_norm_6 = tf.keras.layers.BatchNormalization()
        self.decoder_deconv_7 = tf.keras.layers.Conv2DTranspose(filters=256, **conv_args)
        self.decoder_batch_norm_7 = tf.keras.layers.BatchNormalization()
        self.decoder_deconv_8 = tf.keras.layers.Conv2DTranspose(filters=128, **conv_args)
        self.decoder_batch_norm_8 = tf.keras.layers.BatchNormalization()

        # These layers have no trainable params; they can be reused multiple times
        self.decoder_relu = tf.keras.layers.ReLU()
        # Call with training=True (even when testing)
        self.decoder_dropout = tf.keras.layers.Dropout(rate=0.5)

        # Maps to RGB output
        self.conv_rgb = tf.keras.layers.Conv2D(
            filters=3, kernel_size=1, strides=1, padding='same', kernel_initializer=kernel_initializer)

    def call(self, inputs):
        """
        Runs a forward pass on an input batch of images.
        :param inputs: image masks, shape of (num_inputs, img_height, img_width, 1);
                       during training, the shape is (batch_size, img_height, img_width, 1)
        :return: output images, shape of (num_inputs, img_height, img_width, 3);
                 each output image is a 3-channel (rgb) matrix of floats from 0-1
        """
        # Encoder Forward Pass:

        # C64
        encoder_1 = self.encoder_conv_1(inputs)
        encoder_1 = self.encoder_relu(encoder_1)
        # C128
        encoder_2 = self.encoder_conv_2(encoder_1)
        # Batch norm should be called with training=True (even when testing)
        encoder_2 = self.encoder_batch_norm_2(encoder_2, training=True)
        encoder_2 = self.encoder_relu(encoder_2)
        # C256
        encoder_3 = self.encoder_conv_3(encoder_2)
        encoder_3 = self.encoder_batch_norm_3(encoder_3, training=True)
        encoder_3 = self.encoder_relu(encoder_3)
        # C512 (1)
        encoder_4 = self.encoder_conv_4(encoder_3)
        encoder_4 = self.encoder_batch_norm_4(encoder_4, training=True)
        encoder_4 = self.encoder_relu(encoder_4)
        # C512 (2)
        encoder_5 = self.encoder_conv_5(encoder_4)
        encoder_5 = self.encoder_batch_norm_5(encoder_5, training=True)
        encoder_5 = self.encoder_relu(encoder_5)
        # C512 (3)
        encoder_6 = self.encoder_conv_6(encoder_5)
        encoder_6 = self.encoder_batch_norm_6(encoder_6, training=True)
        encoder_6 = self.encoder_relu(encoder_6)
        # C512 (4)
        encoder_7 = self.encoder_conv_7(encoder_6)
        encoder_7 = self.encoder_batch_norm_7(encoder_7, training=True)
        encoder_7 = self.encoder_relu(encoder_7)
        # C512 (5)
        encoder_8 = self.encoder_conv_8(encoder_7)
        encoder_8 = self.encoder_relu(encoder_8)

        # Decoder Forward Pass:

        # CD512
        decoder_1 = self.decoder_deconv_1(encoder_8)
        decoder_1 = self.decoder_batch_norm_1(decoder_1, training=True)
        decoder_1 = self.decoder_dropout(decoder_1, training=True)
        decoder_1 = self.decoder_relu(decoder_1)
        # Skip connection
        decoder_2 = tf.concat([decoder_1, encoder_7], axis=-1)
        # CD1024
        decoder_2 = self.decoder_deconv_2(decoder_2)
        decoder_2 = self.decoder_batch_norm_2(decoder_2, training=True)
        decoder_2 = self.decoder_dropout(decoder_2, training=True)
        decoder_2 = self.decoder_relu(decoder_2)
        # Skip connection
        decoder_3 = tf.concat([decoder_2, encoder_6], axis=-1)
        # CD1024
        decoder_3 = self.decoder_deconv_3(decoder_3)
        decoder_3 = self.decoder_batch_norm_3(decoder_3, training=True)
        decoder_3 = self.decoder_dropout(decoder_3, training=True)
        decoder_3 = self.decoder_relu(decoder_3)
        # Skip connection
        decoder_4 = tf.concat([decoder_3, encoder_5], axis=-1)
        # C1024
        decoder_4 = self.decoder_deconv_4(decoder_4)
        decoder_4 = self.decoder_batch_norm_4(decoder_4, training=True)
        decoder_4 = self.decoder_relu(decoder_4)
        # Skip connection
        decoder_5 = tf.concat([decoder_4, encoder_4], axis=-1)
        # C1024
        decoder_5 = self.decoder_deconv_5(decoder_5)
        decoder_5 = self.decoder_batch_norm_5(decoder_5, training=True)
        decoder_5 = self.decoder_relu(decoder_5)
        # Skip connection
        decoder_6 = tf.concat([decoder_5, encoder_3], axis=-1)
        # C512
        decoder_6 = self.decoder_deconv_6(decoder_6)
        decoder_6 = self.decoder_batch_norm_6(decoder_6, training=True)
        decoder_6 = self.decoder_relu(decoder_6)
        # Skip connection
        decoder_7 = tf.concat([decoder_6, encoder_2], axis=-1)
        # C256
        decoder_7 = self.decoder_deconv_7(decoder_7)
        decoder_7 = self.decoder_batch_norm_7(decoder_7, training=True)
        decoder_7 = self.decoder_relu(decoder_7)
        # Skip connection
        decoder_8 = tf.concat([decoder_7, encoder_1], axis=-1)
        # C128
        decoder_8 = self.decoder_deconv_8(decoder_8)
        decoder_8 = self.decoder_batch_norm_8(decoder_8, training=True)
        decoder_8 = self.decoder_relu(decoder_8)

        # Maps to RGB output
        output = self.conv_rgb(decoder_8)
        output = tf.math.tanh(output)

        return output

    def loss(self, logits_fake):
        # NOTE: Borrowed from GANs lab
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits_fake), logits=logits_fake))
