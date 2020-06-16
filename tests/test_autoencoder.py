import tensorflow as tf

import docclean


def test_autoencoder():
    ae = docclean.autoencoder.Autoencoder()
    clean_data = tf.data.Dataset.from_tensors(tf.random.normal(shape=(32, 256, 256, 3), dtype=tf.float32))
    dirty_data = tf.data.Dataset.from_tensors(tf.random.uniform(shape=(32, 256, 256, 3), dtype=tf.float32))
    old_weights = ae.autoencoder_model.get_weights()
    ae.train_model(tf.data.Dataset.zip(dirty_data, clean_data), epochs=3)
    new_weights = ae.autoencoder_model.get_weights()
    for x, y in zip(old_weights, new_weights):
        assert (x - y).any() != 0
