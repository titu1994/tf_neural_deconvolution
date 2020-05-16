import pytest
import six
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision

from models import simplenet_tf_mixed


def tf_context(func):
    @six.wraps(func)
    def wrapped(*args, **kwargs):
        # Run tests only on the gpu as grouped convs are not supported on cpu
        with tf.device('gpu:0'):
            out = func(*args, **kwargs)
        return out
    return wrapped


def tf_mixed_precision(func):
    @six.wraps(func)
    def wrapped(*args, **kwargs):
        # Run in mixed precision mode, then return to float32 mode
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_policy(policy)

        # call method in mixed precision mode
        try:
            out = func(*args, **kwargs)
        finally:
            # Return to float32 precision mode
            policy = mixed_precision.Policy('float32')
            mixed_precision.set_policy(policy)

        return out
    return wrapped


@pytest.mark.parametrize("groups", [1, 32, 64])
@pytest.mark.parametrize("channel_deconv_loc", ['pre', 'post'])
@pytest.mark.parametrize("blocks", [1, 32, 64])
@tf_context
@tf_mixed_precision
def test_fastdeconv_1d_mixed_precision(groups, channel_deconv_loc, blocks):
    """ Test 1D variant """
    policy = mixed_precision.global_policy()

    x = tf.zeros([16, 24, 3], dtype=tf.float16)
    model2 = simplenet_tf_mixed.SimpleNet1D(num_classes=10, num_channels=64, groups=groups,
                                            channel_deconv_loc=channel_deconv_loc, blocks=blocks,
                                            dtype=policy)

    # trace the model
    model_traced2 = tf.function(model2)

    out = model_traced2(x, training=True)
    assert out.shape == [16, 10]

    out = model_traced2(x, training=False)
    assert out.shape == [16, 10]


@pytest.mark.parametrize("groups", [1, 32, 64])
@pytest.mark.parametrize("channe_deconv_loc", ['pre', 'post'])
@pytest.mark.parametrize("blocks", [1, 32, 64])
@tf_context
@tf_mixed_precision
def test_fastdeconv_2d_mixed_precision(groups, channe_deconv_loc, blocks):
    """ Test 1D variant """
    policy = mixed_precision.global_policy()

    x = tf.zeros([16, 32, 32, 3], dtype=tf.float16)
    model2 = simplenet_tf_mixed.SimpleNet2D(num_classes=10, num_channels=64, groups=groups,
                                            channel_deconv_loc=channe_deconv_loc, blocks=blocks,
                                            dtype=policy)

    # trace the model
    model_traced2 = tf.function(model2)

    out = model_traced2(x, training=True)
    assert out.shape == [16, 10]

    out = model_traced2(x, training=False)
    assert out.shape == [16, 10]


if __name__ == '__main__':
    pytest.main([__file__])
