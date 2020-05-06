import os
import math
import datetime as dt
import tensorflow as tf

from utils.optim import AdamW, SGDW
from utils.schedule import CosineDecay

from models.simplenet2d import SimpleNet2D
from models.vgg import VGG


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

CIFAR_10_MEAN = tf.convert_to_tensor([0.4914, 0.4822, 0.4465])
CIFAR_10_STD = tf.convert_to_tensor([0.2023, 0.1994, 0.2010])
CIFAR_MEAN = tf.reshape(CIFAR_10_MEAN, [1, 1, 3])
CIFAR_STD = tf.reshape(CIFAR_10_STD, [1, 1, 3])


def augment(x, y):
    x = tf.image.resize_with_crop_or_pad(x, 36, 36)
    x = tf.image.random_crop(x, [32, 32, 3])
    x = tf.image.random_flip_left_right(x)
    return x, y


# Dataset pipelines
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000)
train_dataset = train_dataset.map(lambda x, y: (tf.cast(x, tf.float32) / 255., y))
train_dataset = train_dataset.map(augment, num_parallel_calls=os.cpu_count())
train_dataset = train_dataset.map(lambda x, y: ((x - CIFAR_MEAN) / CIFAR_STD, y))
train_dataset = train_dataset.batch(128)
train_dataset = train_dataset.prefetch(4)

test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = test_dataset.map(lambda x, y: (tf.cast(x, tf.float32) / 255., y))
test_dataset = test_dataset.map(lambda x, y: ((x - CIFAR_MEAN) / CIFAR_STD, y))
test_dataset = test_dataset.batch(100)

logdir = './log_cifar10/{}'.format(dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
if not os.path.exists(logdir):
    os.makedirs(logdir)

if not os.path.exists('checkpoints/cifar10/'):
    os.makedirs('checkpoints/cifar10/')

callbacks = [
    # Write TensorBoard logs to `./logs` directory
    tf.keras.callbacks.TensorBoard(log_dir=logdir, update_freq='batch', profile_batch=0),
    tf.keras.callbacks.ModelCheckpoint('checkpoints/cifar10/', monitor='loss',
                                       verbose=2, save_best_only=True,
                                       save_weights_only=True, mode='min')
]

# model = SimpleNet(num_classes=10, num_channels=64)
model = VGG(vgg_name='VGG16', num_classes=10)

epochs = 1  # should be 1, 20 or 100

# SGDW Optimizer
total_steps = math.ceil(len(x_train) / float(128)) * max(1, epochs)
lr = CosineDecay(0.1, decay_steps=total_steps, alpha=1e-6)
optimizer = SGDW(lr, momentum=0.9, nesterov=True, weight_decay=0.001)

model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['acc'])

model.fit(train_dataset, epochs=epochs,
          validation_data=test_dataset,
          callbacks=callbacks)
