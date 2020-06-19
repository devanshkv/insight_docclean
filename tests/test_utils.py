from urllib.request import urlretrieve

import numpy as np
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


def test_get_kaggle_paired_data(png_file):
    a, b = docclean.utils.get_kaggle_paired_data(png_file)
    assert a.shape == tf.TensorShape([512, 512, 3])
    assert a.dtype == tf.float32
    assert b.shape == tf.TensorShape([512, 512, 3])
    assert b.dtype == tf.float32


def test_get_kaggle_data(png_file):
    png_data = docclean.utils.get_kaggle_data(png_file)
    assert png_data.shape == tf.TensorShape([258, 540, 3])
    assert png_data.dtype == tf.uint8


def test_normalize():
    data = tf.cast(tf.random.uniform(shape=(256, 256, 3), minval=0, maxval=255, dtype=tf.int32), tf.uint8)
    data = docclean.utils.normalize(data)
    assert tf.reduce_max(data) < 1.1
    assert tf.reduce_min(data) > -1.1


def test_books_crop_and_augment():
    data = tf.random.uniform(shape=(512, 512, 3), minval=-1, maxval=1, dtype=tf.float32)
    augmented_data = docclean.utils.books_crop_and_augment(data)
    assert augmented_data.shape == tf.TensorShape([256, 256, 3])


def test_kaggle_crop_and_augment():
    data = tf.random.uniform(shape=(512, 512, 3), minval=-1, maxval=1, dtype=tf.float32)
    augmented_data = docclean.utils.kaggle_crop_and_augment(data)
    assert augmented_data.shape == tf.TensorShape([256, 256, 3])


def test_kaggle_paired_augment():
    data_clean = tf.random.uniform(shape=(256, 256, 3), minval=-1, maxval=1, dtype=tf.float32)
    data_dirty = tf.random.uniform(shape=(256, 256, 3), minval=-1, maxval=1, dtype=tf.float32)
    aug_clean, aug_dirty = docclean.utils.kaggle_paired_augment(data_clean, data_dirty)
    assert not tf.reduce_all(tf.math.equal(data_dirty, aug_dirty))
    assert not tf.reduce_all(tf.math.equal(data_clean, aug_clean))


def test_normed_to_uint8():
    data = np.random.uniform(size=(256, 256, 3), low=0, high=1)
    normed_data = docclean.utils.normed_to_uint8(data)
    assert normed_data.max() == 255
    assert normed_data.min() == 0


def test_ImageMosaic_even():
    data = np.random.uniform(size=(256, 256, 3), low=0, high=1)
    im = docclean.utils.ImageMosaic(data)
    patches = im.make_patches()
    assert patches.shape == (1, 256, 256, 3)
    unpatch = im.combine_patches(patches)
    assert data.shape == unpatch.shape


def test_ImageMosaic_odd():
    data = np.random.uniform(size=(1023, 256, 3), low=0, high=1)
    im = docclean.utils.ImageMosaic(data)
    patches = im.make_patches()
    assert patches.shape == (7, 256, 256, 3)
    unpatch = im.combine_patches(patches)
    assert data.shape == unpatch.shape
