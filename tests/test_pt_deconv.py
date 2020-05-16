import pytest

import torch
from models import simplenet_pt


@pytest.mark.parametrize("groups", [1, 32, 64])
@pytest.mark.parametrize("channel_deconv_loc", ['pre', 'post'])
@pytest.mark.parametrize("blocks", [1, 32, 64])
def test_fastdeconv_1d(groups, channel_deconv_loc, blocks):
    """ Test 1D variant """
    x = torch.zeros([16, 3, 24])
    model2 = simplenet_pt.SimpleNet1D(num_classes=10, num_channels=64, groups=groups,
                                      channel_deconv_loc=channel_deconv_loc, blocks=blocks)
    model2.train()
    out = model2(x)
    assert list(out.shape) == [16, 10]

    model2.eval()
    out = model2(x)
    assert list(out.shape) == [16, 10]


@pytest.mark.parametrize("groups", [1, 32, 64])
@pytest.mark.parametrize("channe_deconv_loc", ['pre', 'post'])
@pytest.mark.parametrize("blocks", [1, 32, 64])
def test_fastdeconv_2d_no_groups(groups, channe_deconv_loc, blocks):
    """ Test 1D variant """
    x = torch.zeros([16, 3, 32, 32])
    model2 = simplenet_pt.SimpleNet2D(num_classes=10, num_channels=64, groups=groups,
                                      channel_deconv_loc=channe_deconv_loc, blocks=blocks)
    model2.train()
    out = model2(x)
    assert list(out.shape) == [16, 10]

    model2.eval()
    out = model2(x)
    assert list(out.shape) == [16, 10]


if __name__ == '__main__':
    pytest.main([__file__])
