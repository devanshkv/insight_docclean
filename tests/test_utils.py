from urllib.request import urlretrieve

import pytest
import tensorflow as tf

import docclean


@pytest.fixture(scope="session", autouse=True)
def png_file(tmpdir_factory):
    temp_dirty_path = str(tmpdir_factory.mktemp("train")) + '/23.png'
    temp_clean_path = str(tmpdir_factory.mktemp("train_cleaned")) + '/23.png'
    url = "https://storage.googleapis.com/kaggle-competitions/kaggle/4406/media/23.png"
    urlretrieve(url, temp_dirty_path)
    urlretrieve(url, temp_clean_path)
    return temp_dirty_path


def test_get_png_data(png_file):
    png_data = docclean.utils.get_png_data(png_file)
    assert png_data.shape == tf.TensorShape([258, 540, 1])
    assert png_data.dtype == tf.uint8
