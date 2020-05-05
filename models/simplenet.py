import tensorflow as tf
from tf_deconv import FastDeconv2D, ChannelDeconv2D


class SimpleNet(tf.keras.Model):

    def __init__(self, num_classes, num_channels=64):
        super().__init__()

        self.num_channels = num_channels
        self.num_classes = num_channels

        self.conv1 = FastDeconv2D(3, num_channels, kernel_size=(3, 3), stride=(2, 2),
                                  padding='same', activation='relu',
                                  n_iter=5, momentum=0.1, block=64)

        self.conv2 = FastDeconv2D(num_channels, num_channels, kernel_size=(3, 3), stride=(2, 2),
                                  padding='same', activation='relu',
                                  n_iter=5, momentum=0.1, block=64)

        self.final_conv = ChannelDeconv2D(block=64, momentum=0.1)

        self.gap = tf.keras.layers.GlobalAveragePooling2D()
        self.clf = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs, training=training)
        x = self.conv2(x, training=training)
        x = self.final_conv(x, training=training)
        x = self.gap(x)
        x = self.clf(x)

        return x


if __name__ == '__main__':
    x = tf.zeros([16, 32, 32, 3])
    model = SimpleNet(num_classes=10, num_channels=64)

    # trace the model
    model_traced = tf.function(model)

    out = model_traced(x, training=True)
    print(out.shape)

    model.summary()
