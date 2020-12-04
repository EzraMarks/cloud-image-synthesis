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
        self.decoder_deconv_2 = tf.keras.layers.Conv2DTranspose(filters=512, **conv_args)
        self.decoder_batch_norm_2 = tf.keras.layers.BatchNormalization()
        self.decoder_deconv_3 = tf.keras.layers.Conv2DTranspose(filters=512, **conv_args)
        self.decoder_batch_norm_3 = tf.keras.layers.BatchNormalization()
        self.decoder_deconv_4 = tf.keras.layers.Conv2DTranspose(filters=512, **conv_args)
        self.decoder_batch_norm_4 = tf.keras.layers.BatchNormalization()
        self.decoder_deconv_5 = tf.keras.layers.Conv2DTranspose(filters=256, **conv_args)
        self.decoder_batch_norm_5 = tf.keras.layers.BatchNormalization()
        self.decoder_deconv_6 = tf.keras.layers.Conv2DTranspose(filters=128, **conv_args)
        self.decoder_batch_norm_6 = tf.keras.layers.BatchNormalization()
        self.decoder_deconv_7 = tf.keras.layers.Conv2DTranspose(filters=64, **conv_args)
        self.decoder_batch_norm_7 = tf.keras.layers.BatchNormalization()
        self.decoder_deconv_8 = tf.keras.layers.Conv2DTranspose(filters=64, **conv_args)
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
        output = self.encoder_conv_1(inputs)
        output = self.encoder_relu(output)
        # C128
        output = self.encoder_conv_2(output)
        # Batch norm should be called with training=True (even when testing)
        output = self.encoder_batch_norm_2(output, training=True)
        output = self.encoder_relu(output)
        # C256
        output = self.encoder_conv_3(output)
        output = self.encoder_batch_norm_3(output, training=True)
        output = self.encoder_relu(output)
        # C512 (1)
        output = self.encoder_conv_4(output)
        output = self.encoder_batch_norm_4(output, training=True)
        output = self.encoder_relu(output)
        # C512  (2)
        output = self.encoder_conv_5(output)
        output = self.encoder_batch_norm_5(output, training=True)
        output = self.encoder_relu(output)
        # C512  (3)
        output = self.encoder_conv_6(output)
        output = self.encoder_batch_norm_6(output, training=True)
        output = self.encoder_relu(output)
        # C512  (4)
        output = self.encoder_conv_7(output)
        output = self.encoder_batch_norm_7(output, training=True)
        output = self.encoder_relu(output)
        # C512  (5)
        output = self.encoder_conv_8(output)
        output = self.encoder_relu(output)

        # Decoder Forward Pass:

        # CD512
        output = self.decoder_deconv_1(output)
        output = self.decoder_batch_norm_1(output, training=True)
        output = self.decoder_dropout(output, training=True)
        output = self.decoder_relu(output)
        # CD512
        output = self.decoder_deconv_2(output)
        output = self.decoder_batch_norm_2(output, training=True)
        output = self.decoder_dropout(output, training=True)
        output = self.decoder_relu(output)
        # CD512
        output = self.decoder_deconv_3(output)
        output = self.decoder_batch_norm_3(output, training=True)
        output = self.decoder_dropout(output, training=True)
        output = self.decoder_relu(output)
        # C512
        output = self.decoder_deconv_4(output)
        output = self.decoder_batch_norm_4(output, training=True)
        output = self.decoder_relu(output)
        # C256
        output = self.decoder_deconv_5(output)
        output = self.decoder_batch_norm_5(output, training=True)
        output = self.decoder_relu(output)
        # C128
        output = self.decoder_deconv_6(output)
        output = self.decoder_batch_norm_6(output, training=True)
        output = self.decoder_relu(output)
        # C64
        output = self.decoder_deconv_7(output)
        output = self.decoder_batch_norm_7(output, training=True)
        output = self.decoder_relu(output)
        
        output = self.decoder_deconv_8(output)
        output = self.decoder_batch_norm_8(output, training=True)
        output = self.decoder_relu(output)

        # Maps to RGB output
        output = self.conv_rgb(output)
        output = tf.math.tanh(output)

        return output

    def loss(self, logits_fake):
        # NOTE: Borrowed from GANs lab
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits_fake), logits=logits_fake))
