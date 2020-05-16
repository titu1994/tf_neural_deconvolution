import tensorflow as tf
from tf_deconv_mixed_prec import FastDeconv1D, FastDeconv2D, ChannelDeconv1D, ChannelDeconv2D


class SimpleNet1D(tf.keras.Model):

    def __init__(self, num_classes, num_channels=64, groups=1, channel_deconv_loc="pre", blocks=64, **kwargs):
        super().__init__(**kwargs)

        self.num_channels = num_channels
        self.num_classes = num_channels
        self.channel_deconv_loc = channel_deconv_loc

        self.conv1 = FastDeconv1D(3, num_channels, kernel_size=3, stride=2,
                                  padding='same', activation='relu', groups=1,
                                  n_iter=5, momentum=0.1, block=blocks, **kwargs)

        self.conv2 = FastDeconv1D(num_channels, num_channels, kernel_size=3, stride=2,
                                  padding='same', activation='relu', groups=groups,
                                  n_iter=5, momentum=0.1, block=blocks, **kwargs)

        self.final_conv = ChannelDeconv1D(block=num_channels, momentum=0.1, **kwargs)

        self.gap = tf.keras.layers.GlobalAveragePooling1D(**kwargs)
        self.clf = tf.keras.layers.Dense(num_classes, activation=None, **kwargs)
        self.softmax = tf.keras.layers.Activation('softmax', dtype='float32')

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs, training=training)
        x = self.conv2(x, training=training)

        if self.channel_deconv_loc == 'pre':
            x = self.final_conv(x, training=training)
            x = self.gap(x)
        else:
            x = self.gap(x)
            x = self.final_conv(x, training=training)

        x = self.clf(x)
        x = self.softmax(x)

        return x


class SimpleNet2D(tf.keras.Model):

    def __init__(self, num_classes, num_channels=64, groups=1, channel_deconv_loc="pre", blocks=64, **kwargs):
        super().__init__(**kwargs)

        self.num_channels = num_channels
        self.num_classes = num_channels
        self.channel_deconv_loc = channel_deconv_loc

        self.conv1 = FastDeconv2D(3, num_channels, kernel_size=(3, 3), stride=(2, 2),
                                  padding='same', activation='relu', groups=1,
                                  n_iter=5, momentum=0.9, block=blocks, **kwargs)

        self.conv2 = FastDeconv2D(num_channels, num_channels, kernel_size=(3, 3), stride=(2, 2),
                                  padding='same', activation='relu', groups=groups,
                                  n_iter=5, momentum=0.9, block=blocks, **kwargs)

        self.final_conv = ChannelDeconv2D(block=num_channels, momentum=0.1, **kwargs)

        self.gap = tf.keras.layers.GlobalAveragePooling2D(**kwargs)
        self.clf = tf.keras.layers.Dense(num_classes, activation=None, **kwargs)
        self.softmax = tf.keras.layers.Activation('softmax', dtype='float32')

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs, training=training)
        x = self.conv2(x, training=training)

        if self.channel_deconv_loc == 'pre':
            x = self.final_conv(x, training=training)
            x = self.gap(x)
        else:
            x = self.gap(x)
            x = self.final_conv(x, training=training)

        x = self.clf(x)
        x = self.softmax(x)

        return x
