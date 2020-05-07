import pytest
import six
import tensorflow as tf

from models import simplenet_tf


def tf_context(func):
    @six.wraps(func)
    def wrapped(*args, **kwargs):
        # Run tests only on the gpu as grouped convs are not supported on cpu
        with tf.device('gpu:0'):
            out = func(*args, **kwargs)
        return out
    return wrapped


@pytest.mark.parametrize("groups", [1, 32, 64])
@pytest.mark.parametrize("channel_deconv_loc", ['pre', 'post'])
@pytest.mark.parametrize("blocks", [1, 32, 64])
@tf_context
def test_fastdeconv_1d(groups, channel_deconv_loc, blocks):
    """ Test 1D variant """
    x = tf.zeros([16, 24, 3])
    model2 = simplenet_tf.SimpleNet1D(num_classes=10, num_channels=64, groups=groups,
                                      channel_deconv_loc=channel_deconv_loc, blocks=blocks)

    # trace the model
    model_traced2 = tf.function(model2)

    out = model_traced2(x, training=True)
    assert out.shape == [16, 10]


@pytest.mark.parametrize("groups", [1, 32, 64])
@pytest.mark.parametrize("channe_deconv_loc", ['pre', 'post'])
@pytest.mark.parametrize("blocks", [1, 32, 64])
@tf_context
def test_fastdeconv_2d_no_groups(groups, channe_deconv_loc, blocks):
    """ Test 1D variant """
    x = tf.zeros([16, 32, 32, 3])
    model2 = simplenet_tf.SimpleNet2D(num_classes=10, num_channels=64, groups=groups,
                                      channel_deconv_loc=channe_deconv_loc, blocks=blocks)

    # trace the model
    model_traced2 = tf.function(model2)

    out = model_traced2(x, training=True)
    assert out.shape == [16, 10]


if __name__ == '__main__':
    pytest.main([__file__])
