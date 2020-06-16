import tensorflow as tf

import docclean


def test_cyclegan():
    gan = docclean.cycle_gan.CycleGan()
    old_weights = [x.get_weights() for x in
                   [gan.generator_g, gan.generator_f, gan.discriminator_x, gan.discriminator_y]]
    clean_data = tf.data.Dataset.from_tensors(tf.random.normal(shape=(128, 256, 256, 3))).batch(32)
    dirty_data = tf.data.Dataset.from_tensors(tf.random.uniform(shape=(128, 256, 256, 3))).batch(32)
    gan.train(dirty_data, clean_data, epochs=3)
    new_weights = [x.get_weights() for x in
                   [gan.generator_g, gan.generator_f, gan.discriminator_x, gan.discriminator_y]]
    for x, y in zip(old_weights, new_weights):
        assert x.all() != y.all()
