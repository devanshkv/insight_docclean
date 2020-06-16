import tensorflow as tf

import docclean


def test_cyclegan():
    gan = docclean.cycle_gan.CycleGan()
    old_weights = [x.get_weights() for x in
                   [gan.generator_g, gan.generator_f, gan.discriminator_x, gan.discriminator_y]]
    clean_data = tf.data.Dataset.from_tensors(tf.random.normal(shape=(32, 256, 256, 3), dtype=tf.float32))
    dirty_data = tf.data.Dataset.from_tensors(tf.random.uniform(shape=(32, 256, 256, 3), dtype=tf.float32))
    gan.train(dirty_data, clean_data, epochs=3)
    new_weights = [x.get_weights() for x in
                   [gan.generator_g, gan.generator_f, gan.discriminator_x, gan.discriminator_y]]
    for x, y in zip(old_weights, new_weights):
        assert (x - y).any() != 0
