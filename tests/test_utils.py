import tensorflow as tf

import docclean


def test_get_png_data(png_file='../data/train/2.png'):
    png_data = docclean.utils.get_png_data(png_file)
    assert png_data.shape == tf.TensorShape([258, 540, 1])
    assert png_data.dtype == tf.uint8
