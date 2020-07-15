import numpy as np
import tensorflow as tf

import docclean

clean_data = tf.data.Dataset.from_tensors(tf.random.normal(shape=(8, 256, 256, 3), dtype=tf.float32))
dirty_data = tf.data.Dataset.from_tensors(tf.random.uniform(shape=(8, 256, 256, 3), dtype=tf.float32))


def test_cyclegan():
    gan = docclean.cycle_gan.CycleGan()
    old_weights = [x.get_weights() for x in
                   [gan.generator_g, gan.generator_f, gan.discriminator_x, gan.discriminator_y]]

    for image_x, image_y in tf.data.Dataset.zip((dirty_data, clean_data)):
        gan.train_step(image_x, image_y)
    new_weights = [x.get_weights() for x in
                   [gan.generator_g, gan.generator_f, gan.discriminator_x, gan.discriminator_y]]
    for old_models, new_models in zip(old_weights, new_weights):
        _old_weights = np.concatenate([x.flatten() for x in old_models])
        _new_weights = np.concatenate([x.flatten() for x in new_models])
        assert not np.array_equal(_old_weights, _new_weights)

    gan.train(dirty_data, clean_data, epochs=6)
    new_weights = [x.get_weights() for x in
                   [gan.generator_g, gan.generator_f, gan.discriminator_x, gan.discriminator_y]]
    for old_models, new_models in zip(old_weights, new_weights):
        _old_weights = np.concatenate([x.flatten() for x in old_models])
        _new_weights = np.concatenate([x.flatten() for x in new_models])
        assert not np.array_equal(_old_weights, _new_weights)


def test_autoencoder():
    ae = docclean.autoencoder.Autoencoder()
    old_weights = ae.autoencoder_model.get_weights()
    ae.train_model(tf.data.Dataset.zip((dirty_data, clean_data)), epochs=6)
    new_weights = ae.autoencoder_model.get_weights()
    old_weights = np.concatenate([x.flatten() for x in old_weights])
    new_weights = np.concatenate([x.flatten() for x in new_weights])
    assert not np.array_equal(old_weights, new_weights)


def test_autoencoder_mp():
    ae = docclean.autoencoder.Autoencoder(mixed_precision_training=True)
    old_weights = ae.autoencoder_model.get_weights()
    ae.train_model(tf.data.Dataset.zip((dirty_data, clean_data)), epochs=3)
    new_weights = ae.autoencoder_model.get_weights()
    old_weights = np.concatenate([x.flatten() for x in old_weights])
    new_weights = np.concatenate([x.flatten() for x in new_weights])
    assert not np.array_equal(old_weights, new_weights)
