import os
import datetime as dt
import tensorflow as tf
from models.simplenet import SimpleNet

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000)
train_dataset = train_dataset.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))
train_dataset = train_dataset.map(lambda x, y: (tf.image.resize_with_crop_or_pad(x, 40, 40), y))
train_dataset = train_dataset.map(lambda x, y: (tf.image.random_crop(x, [32, 32, 3]), y))
train_dataset = train_dataset.map(lambda x, y: (tf.image.random_flip_left_right(x), y))
train_dataset = train_dataset.batch(128)

test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(100)
test_dataset = test_dataset.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))


logdir = './log_cifar10/{}'.format(dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
if not os.path.exists(logdir):
    os.makedirs(logdir)

if not os.path.exists('checkpoints/cifar10/'):
    os.makedirs('checkpoints/cifar10/')

callbacks = [
    # Write TensorBoard logs to `./logs` directory
    tf.keras.callbacks.TensorBoard(log_dir=logdir, update_freq='batch', profile_batch=0),
    tf.keras.callbacks.ModelCheckpoint('checkpoints/cifar10/', monitor='loss', verbose=2, save_best_only=True,
                                       save_weights_only=True, mode='min')
]

model = SimpleNet(num_classes=10, num_channels=64)

lr = tf.keras.optimizers.schedules.ExponentialDecay(0.1, decay_steps=5000, decay_rate=0.9)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
              loss='sparse_categorical_crossentropy',
              metrics=['acc'])

model.fit(train_dataset, epochs=50,
          validation_data=test_dataset,
          callbacks=callbacks)

