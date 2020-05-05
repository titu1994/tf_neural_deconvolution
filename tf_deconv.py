import math
import tensorflow as tf

from tensorflow.python.keras.layers.convolutional import Conv


# iteratively solve for inverse sqrt of a matrix
def isqrt_newton_schulz_autograd(A: tf.Tensor, numIters: int):
    dim = tf.shape(A)[0]
    normA = tf.norm(A, ord='fro', axis=[0, 1])
    Y = A / normA

    with tf.device(A.device):
        I = tf.eye(dim, dtype=A.dtype)
        Z = tf.eye(dim, dtype=A.dtype)

    for i in range(numIters):
        T = 0.5 * (3.0 * I - tf.matmul(Z, Y))
        Y = tf.matmul(Y, T)
        Z = tf.matmul(T, Z)

    A_isqrt = Z / tf.sqrt(normA)
    return A_isqrt


def isqrt_newton_schulz_autograd_batch(A: tf.Tensor, numIters: int):
    Ashape = tf.shape(A)  # [batch, _, C]
    batchSize, dim = Ashape[0], Ashape[-1]

    normA = tf.reshape(A, (batchSize, -1))
    normA = tf.norm(normA, ord=2, axis=1)
    normA = tf.reshape(normA, [batchSize, 1, 1])

    Y = A / normA

    with tf.device(A.device):
        I = tf.expand_dims(tf.eye(dim, dtype=A.dtype), 0)
        Z = tf.expand_dims(tf.eye(dim, dtype=A.dtype), 0)

        I = tf.broadcast_to(I, Ashape)
        Z = tf.broadcast_to(Z, Ashape)

    for i in range(numIters):
        T = 0.5 * (3.0 * I - tf.matmul(Z, Y))  # Z.bmm(Y)
        Y = tf.matmul(Y, T)  # Y.bmm(T)
        Z = tf.matmul(T, Z)  # T.bmm(Z)

    A_isqrt = Z / tf.sqrt(normA)

    return A_isqrt


class ChannelDeconv2D(tf.keras.layers.Layer):
    def __init__(self, block, eps=1e-2, n_iter=5, momentum=0.1, sampling_stride=3):
        super(ChannelDeconv2D, self).__init__()

        self.eps = eps
        self.n_iter = n_iter
        self.momentum = momentum
        self.block = block
        self.sampling_stride = sampling_stride

        self.running_mean1 = tf.Variable(tf.zeros([block, 1]), trainable=False, dtype=self.dtype)
        self.running_mean2 = tf.Variable(tf.zeros([]), trainable=False, dtype=self.dtype)
        self.running_var = tf.Variable(tf.ones([]), trainable=False, dtype=self.dtype)
        self.running_deconv = tf.Variable(tf.eye(block), trainable=False, dtype=self.dtype)
        self.num_batches_tracked = tf.Variable(tf.convert_to_tensor(0, dtype=tf.int64), trainable=False)

        self.block_eye = tf.eye(block)

    def build(self, input_shape):
        in_channels = input_shape[-1]
        self.in_channels = in_channels

        if int(in_channels / self.block) * self.block == 0:
            raise ValueError("`block` must be smaller than in_channels.")

    @tf.function
    def call(self, x, training=None):
        x_shape = tf.shape(x)
        x_original_shape = x_shape

        if len(x.shape) == 2:
            x = tf.reshape(x, [x_shape[0], 1, 1, x_shape[1]])

        x_shape = tf.shape(x)

        N, H, W, C = x_shape[0], x_shape[1], x_shape[2], x_shape[3]
        block = self.block

        # take the first c channels out for deconv
        c = tf.cast(C / block, tf.int32) * block

        # step 1. remove mean
        if tf.not_equal(c, C):
            x1 = tf.reshape(tf.transpose(x[:, :, :, :c], [3, 0, 1, 2]), [block, -1])
        else:
            x1 = tf.reshape(tf.transpose(x, [3, 0, 1, 2]), [block, -1])

        if self.sampling_stride > 1 and H >= self.sampling_stride and W >= self.sampling_stride:
            x1_s = x1[:, ::self.sampling_stride ** 2]
        else:
            x1_s = x1

        mean1 = tf.reduce_mean(x1_s, axis=-1, keepdims=True)  # [blocks, 1]

        if self.num_batches_tracked == 0:
            self.running_mean1.assign(mean1)

        if training:
            running_mean1 = self.momentum * mean1 + (1. - self.momentum) * self.running_mean1
            self.running_mean1.assign(running_mean1)
        else:
            mean1 = self.running_mean1

        x1 = x1 - mean1

        # step 2. calculate deconv@x1 = cov^(-0.5)@x1
        if training:
            scale_ = tf.cast(tf.shape(x1_s)[1], x1_s.dtype)
            cov = (tf.matmul(x1_s, tf.transpose(x1_s)) / scale_) + self.eps * self.block_eye
            deconv = isqrt_newton_schulz_autograd(cov, self.n_iter)
        else:
            deconv = self.running_deconv

        if self.num_batches_tracked == 0:
            self.running_deconv.assign(deconv)

        if training:
            running_deconv = self.momentum * deconv + (1. - self.momentum) * self.running_deconv
            self.running_deconv.assign(running_deconv)
        else:
            deconv = self.running_deconv

        x1 = tf.matmul(deconv, x1)

        # reshape to N,c,J,W
        x1 = tf.reshape(x1, [c, N, H, W])
        x1 = tf.transpose(x1, [1, 2, 3, 0])  # [N, H, W, C]

        # normalize the remaining channels
        if c != C:
            x_tmp = tf.reshape(x[:, c:], [N, -1])

            if self.sampling_stride > 1 and H >= self.sampling_stride and W >= self.sampling_stride:
                x_s = x_tmp[:, ::self.sampling_stride ** 2]
            else:
                x_s = x_tmp

            mean2, var = tf.nn.moments(x_s, axes=[0, 1])

            if self.num_batches_tracked == 0:
                self.running_mean2.assign(mean2)
                self.running_var.assign(var)

            if training:
                running_mean2 = self.momentum * mean2 + (1. - self.momentum) * self.running_mean2
                running_var = self.momentum * var + (1. - self.momentum) * self.running_var
                self.running_mean2.assign(running_mean2)
                self.running_var.assign(running_var)
            else:
                mean2 = self.running_mean2
                var = self.running_var

            x_tmp = tf.sqrt((x[:, c:] - mean2) / (var + self.eps))
            x1 = tf.concat([x1, x_tmp], axis=1)

        if training:
            self.num_batches_tracked.assign_add(1)

        if len(x_original_shape) == 2:
            x1 = tf.reshape(x1, x_original_shape)

        x_intshape = x.shape
        x1 = tf.ensure_shape(x1, x_intshape)

        return x1


class FastDeconv2D(Conv):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='valid', dilation_rate=1,
                 activation=None, use_bias=True, eps=1e-5, n_iter=5, momentum=0.1, block=64, sampling_stride=3,
                 freeze=False, freeze_iter=100):
        self.in_channels = in_channels
        self.momentum = momentum
        self.n_iter = n_iter
        self.eps = eps
        self.counter = 0
        self.track_running_stats = True

        super(FastDeconv2D, self).__init__(
            2, out_channels, kernel_size, stride, padding, dilation_rate=dilation_rate,
            activation=activation, use_bias=use_bias)

        if block > in_channels:
            block = in_channels
        else:
            if in_channels % block != 0:
                block = math.gcd(block, in_channels)
                print("`in_channels` not divisible by `block`, computing new `block` value as %d" % (block))

        self.block = block

        kernel_size_int = kernel_size[0] if type(kernel_size) in (list, tuple) else kernel_size
        self.num_features = kernel_size_int ** 2 * block
        self.running_mean = tf.Variable(tf.zeros(self.num_features), trainable=False, dtype=self.dtype)
        self.running_deconv = tf.Variable(tf.eye(self.num_features), trainable=False, dtype=self.dtype)

        stride_int = stride[0] if type(stride) in (list, tuple) else stride
        self.sampling_stride = sampling_stride * stride_int
        self.counter = tf.Variable(tf.convert_to_tensor(0, dtype=tf.int64), trainable=False)
        self.freeze_iter = freeze_iter
        self.freeze = freeze

    @tf.function
    def call(self, x, training=None):
        x_shape = tf.shape(x)
        N, H, W, C = x_shape[0], x_shape[1], x_shape[2], x_shape[3]

        block = self.block
        frozen = self.freeze and (self.counter > self.freeze_iter)

        if training and self.track_running_stats:
            counter = self.counter + 1
            counter = counter % (self.freeze_iter * 10)
            self.counter.assign(counter)

        if training and (not frozen):

            # 1. im2col: N x cols x pixels -> N*pixles x cols
            if self.kernel_size[0] > 1:
                # [N, X, Y, C]
                X = tf.image.extract_patches(x,
                                             sizes=[1] + list(self.kernel_size) + [1],
                                             strides=[1, self.sampling_stride, self.sampling_stride, 1],
                                             rates=[1, self.dilation_rate[0], self.dilation_rate[1], 1],
                                             padding=str(self.padding).upper())

                X = tf.reshape(X, [N, -1, C])
            else:
                # channel wise
                X = tf.reshape(x, [-1, C])[::self.sampling_stride ** 2, :]

            X = tf.reshape(X, [-1, self.num_features, C // block])
            X = tf.transpose(X, [0, 2, 1])
            X = tf.reshape(X, [-1, self.num_features])

            # 2. subtract mean
            X_mean = tf.reduce_mean(X, axis=0)
            X = X - tf.expand_dims(X_mean, axis=0)

            # 3. calculate COV, COV^(-0.5), then deconv
            scale = tf.cast(tf.shape(X)[0], X.dtype)
            Id = tf.eye(X.shape[1], dtype=X.dtype)
            # addmm op
            Cov = self.eps * Id + (1. / scale) * tf.matmul(tf.transpose(X), X)
            deconv = isqrt_newton_schulz_autograd(Cov, self.n_iter)

            if self.track_running_stats:
                running_mean = self.momentum * X_mean + (1. - self.momentum) * self.running_mean
                running_deconv = self.momentum * deconv + (1. - self.momentum) * self.running_deconv

                # track stats for evaluation
                self.running_mean.assign(running_mean)
                self.running_deconv.assign(running_deconv)

        else:
            X_mean = self.running_mean
            deconv = self.running_deconv

        # 4. X * deconv * conv = X * (deconv * conv)
        w = tf.reshape(self.kernel, [C // block, self.num_features, -1])
        w = tf.transpose(w, [0, 2, 1])
        w = tf.reshape(w, [-1, self.num_features])
        w = tf.matmul(w, deconv)

        b_dash = tf.matmul(w, (tf.expand_dims(X_mean, axis=-1)))
        b_dash = tf.reshape(b_dash, [self.filters, -1])
        b_dash = tf.reduce_sum(b_dash, axis=1)
        b = self.bias - b_dash

        w = tf.reshape(w, [C // block, -1, self.num_features])
        w = tf.transpose(w, [0, 2, 1])
        w = tf.reshape(w, self.kernel.shape)

        x = tf.nn.conv2d(x, w, self.strides, str(self.padding).upper(), dilations=self.dilation_rate)
        x = tf.nn.bias_add(x, b, data_format="NHWC")

        if self.activation is not None:
            return self.activation(x)
        else:
            return x
