import tensorflow as tf

import docclean


def test_autoencoder():
    ae = docclean.autoencoder.Autoencoder()
    clean_data = tf.data.Dataset.from_tensors(tf.random.normal(shape=(128, 256, 256, 3))).batch(32)
    dirty_data = tf.data.Dataset.from_tensors(tf.random.uniform(shape=(128, 256, 256, 3))).batch(32)
    old_weights = ae.autoencoder_model.get_weights()
    ae.train_model(tf.data.Dataset.zip(dirty_data, clean_data), epochs=3)
    new_weights = ae.autoencoder_model.get_weights()
    assert old_weights.any() != new_weights.any()
