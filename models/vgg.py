"""VGG11/13/16/19 in Pytorch."""
import tensorflow as tf
from tf_deconv import FastDeconv2D, ChannelDeconv2D, BiasHeUniform

kernel_size = 3

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(tf.keras.Model):
    def __init__(self, vgg_name, num_classes=10):
        super(VGG, self).__init__()
        assert vgg_name in cfg.keys(), "Choose VGG model from {}".format(cfg.keys())

        self.features = self._make_layers(cfg[vgg_name])
        self.channel_deconv = ChannelDeconv2D(block=512)
        self.classifier = tf.keras.layers.Dense(num_classes, activation='softmax',
                                                kernel_initializer='he_uniform',
                                                bias_initializer=BiasHeUniform(),
                                                )

    def call(self, x, training=None, mask=None):
        out = self.features(x, training=training)
        out = self.channel_deconv(out, training=training)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3

        for x in cfg:
            if x == 'M':
                layers.append(tf.keras.layers.MaxPool2D())
            else:
                if in_channels == 3:
                    deconv = FastDeconv2D(in_channels, x, kernel_size=(kernel_size, kernel_size), padding='same',
                                          freeze=True, n_iter=15, block=64, activation='relu')
                else:
                    deconv = FastDeconv2D(in_channels, x, kernel_size=(kernel_size, kernel_size), padding='same',
                                          block=64, activation='relu')

                layers.append(deconv)
                in_channels = x

        layers.append(tf.keras.layers.GlobalAveragePooling2D())
        return tf.keras.Sequential(layers)


if __name__ == '__main__':
    x = tf.zeros([16, 32, 32, 3])
    model = VGG(vgg_name='VGG16', num_classes=10)

    # trace the model
    model_traced = tf.function(model)

    out = model_traced(x, training=True)
    print(out.shape)

    # 14.71 M trainable params, 18.97 total params; matches paper
    model.summary()
