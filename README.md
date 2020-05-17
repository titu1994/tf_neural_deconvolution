# Neural Deconvolutions (Tensorflow)

Tensorflow implementation of the `FastDeconv2D` and `ChannelDeconv` layers from the paper [Network Deconvolution](https://openreview.net/forum?id=rkeu30EtvS) by Ye et al. Code ported from the repository - https://github.com/yechengxi/deconvolution/.

Tensorflow implementation also support mixed precision training, allowing larger training sizes with no reduction in accuracy (found in `tf_deconv_mixed_prec.py`).

# Usage

Simply download the `tf_deconv.py` script and import `ChannelDeconv2D` and `FastDeconv2D` layers. Mixed precision support can be found in equivalent classes inside `tf_deconv_mixed_prec.py`.

A baseline model has been provided in `models/vgg.py` to try out the architecture. `FastDeconv2D` can replace most Conv2D layer operations.

## Important Note
-----------------

It is crucial to initialize your models properly to obtain correct performance. 

1) All `FastDeconv2D` kernels are initialized by default using `he_uniform`, and their bias by `BiasHeUniform`. 

2) Final `Dense` layer `kernel_initializer` should be `he_uniform` and `bias_initializer` should be `BiasHeUniform`.

--------

```python
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
```

# Dependencies
- Tensorflow 2.1+
