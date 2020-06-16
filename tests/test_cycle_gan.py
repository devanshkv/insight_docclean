import tensorflow as tf

import docclean


def test_cyclegan():
    gan = docclean.cycle_gan.CycleGan()
    old_weights = [x.get_weights() for x in
                   [gan.generator_g, gan.generator_f, gan.discriminator_x, gan.discriminator_y]]
    clean_data = tf.data.Dataset.from_tensors(tf.random.normal(shape=(8, 256, 256, 3), dtype=tf.float32))
    dirty_data = tf.data.Dataset.from_tensors(tf.random.uniform(shape=(8, 256, 256, 3), dtype=tf.float32))
    gan.train(dirty_data, clean_data, epochs=3)
    new_weights = [x.get_weights() for x in
                   [gan.generator_g, gan.generator_f, gan.discriminator_x, gan.discriminator_y]]
    for old_models, new_models in zip(old_weights, new_weights):
        _old_weights = np.concatenate([x.flatten() for x in old_models])
        _new_weights = np.concatenate([x.flatten() for x in new_models])
        assert not np.array_equal(_old_weights, _new_weights)
