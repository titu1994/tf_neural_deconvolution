# Tensorflow Implementation of Neural Deconvolutions

Tensorflow implementation of the `FastDeconv2D` and `ChannelDeconv` layers from the paper [Network Deconvolution](https://openreview.net/forum?id=rkeu30EtvS) by Ye et al. Code ported from the repository - https://github.com/yechengxi/deconvolution/.

# Usage

Simply download the `tf_deconv.py` script and import `ChannelDeconv2D` and `FastDeconv2D` layers. Support for most parameters other than `groups` is available.

A simple baseline model has been provided in `models/simplenet.py` to try out the architecture. `FastDeconv2D` can replace most Conv2D layer operations.

```python
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
```

# Dependencies
- Tensorflow 2.1+
