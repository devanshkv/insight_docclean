import logging

import tensorflow as tf
import tqdm
from tensorflow_examples.models.pix2pix import pix2pix

logger = logging.getLogger(__name__)


class CycleGan:
    """
    Cycle GAN in tensorflow based on the pix2pix.

    Args:

        checkpoint_path: Where to put the checkpoints

        restore_checkpoint: Restore old checkpoints or not


    """

    def __init__(self, checkpoint_path: str = None, restore_checkpoint: bool = True):

        output_channels = 3
        logger.info("Creating Generators and Discriminators")
        self.generator_g = pix2pix.unet_generator(output_channels, norm_type='instancenorm')
        self.generator_f = pix2pix.unet_generator(output_channels, norm_type='instancenorm')

        self.discriminator_x = pix2pix.discriminator(norm_type='instancenorm', target=False)
        self.discriminator_y = pix2pix.discriminator(norm_type='instancenorm', target=False)

        logger.info("Setting up the optimizers")

        self.generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        self.discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        self.loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.LAMBDA = 10

        if checkpoint_path is None:
            self.checkpoint_path = '..'
        else:
            self.checkpoint_path = checkpoint_path

        self.ckpt = tf.train.Checkpoint(generator_g=self.generator_g,
                                        generator_f=self.generator_f,
                                        discriminator_x=self.discriminator_x,
                                        discriminator_y=self.discriminator_y,
                                        generator_g_optimizer=self.generator_g_optimizer,
                                        generator_f_optimizer=self.generator_f_optimizer,
                                        discriminator_x_optimizer=self.discriminator_x_optimizer,
                                        discriminator_y_optimizer=self.discriminator_y_optimizer)

        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_path, max_to_keep=5)

        self.restore_checkpoint = restore_checkpoint

        if self.restore_checkpoint:
            # if a checkpoint exists, restore the latest checkpoint.
            if self.ckpt_manager.latest_checkpoint:
                self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
                print('Latest checkpoint restored!!')

    def discriminator_loss(self, real, generated):
        real_loss = self.loss_obj(tf.ones_like(real), real)
        generated_loss = self.loss_obj(tf.zeros_like(generated), generated)
        total_disc_loss = real_loss + generated_loss
        return total_disc_loss * 0.5

    def generator_loss(self, generated):
        return self.loss_obj(tf.ones_like(generated), generated)

    def calc_cycle_loss(self, real_image, cycled_image):
        loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))
        return self.LAMBDA * loss1

    def identity_loss(self, real_image, same_image):
        loss = tf.reduce_mean(tf.abs(real_image - same_image))
        return self.LAMBDA * 0.5 * loss

    @tf.function
    def train_step(self, real_x, real_y):
        """
        A single training step

        Args:

            real_x (tf.Tensor): real training image X

            real_y (tf.Tensor): real training image y

        """
        # persistent is set to True because the tape is used more than
        # once to calculate the gradients.
        with tf.GradientTape(persistent=True) as tape:
            # Generator G translates X -> Y
            # Generator F translates Y -> X.

            fake_y = self.generator_g(real_x, training=True)
            cycled_x = self.generator_f(fake_y, training=True)

            fake_x = self.generator_f(real_y, training=True)
            cycled_y = self.generator_g(fake_x, training=True)

            # same_x and same_y are used for identity loss.
            same_x = self.generator_f(real_x, training=True)
            same_y = self.generator_g(real_y, training=True)

            disc_real_x = self.discriminator_x(real_x, training=True)
            disc_real_y = self.discriminator_y(real_y, training=True)

            disc_fake_x = self.discriminator_x(fake_x, training=True)
            disc_fake_y = self.discriminator_y(fake_y, training=True)

            # calculate the loss
            gen_g_loss = self.generator_loss(disc_fake_y)
            gen_f_loss = self.generator_loss(disc_fake_x)

            total_cycle_loss = self.calc_cycle_loss(real_x, cycled_x) + self.calc_cycle_loss(real_y, cycled_y)

            # Total generator loss = adversarial loss + cycle loss
            total_gen_g_loss = gen_g_loss + total_cycle_loss + self.identity_loss(real_y, same_y)
            total_gen_f_loss = gen_f_loss + total_cycle_loss + self.identity_loss(real_x, same_x)

            disc_x_loss = self.discriminator_loss(disc_real_x, disc_fake_x)
            disc_y_loss = self.discriminator_loss(disc_real_y, disc_fake_y)

        # Calculate the gradients for generator and discriminator
        self.generator_g_gradients = tape.gradient(total_gen_g_loss,
                                                   self.generator_g.trainable_variables)
        self.generator_f_gradients = tape.gradient(total_gen_f_loss,
                                                   self.generator_f.trainable_variables)

        self.discriminator_x_gradients = tape.gradient(disc_x_loss,
                                                       self.discriminator_x.trainable_variables)
        self.discriminator_y_gradients = tape.gradient(disc_y_loss,
                                                       self.discriminator_y.trainable_variables)

        # Apply the gradients to the optimizer
        self.generator_g_optimizer.apply_gradients(zip(self.generator_g_gradients,
                                                       self.generator_g.trainable_variables))

        self.generator_f_optimizer.apply_gradients(zip(self.generator_f_gradients,
                                                       self.generator_f.trainable_variables))

        self.discriminator_x_optimizer.apply_gradients(zip(self.discriminator_x_gradients,
                                                           self.discriminator_x.trainable_variables))

        self.discriminator_y_optimizer.apply_gradients(zip(self.discriminator_y_gradients,
                                                           self.discriminator_y.trainable_variables))

    def train(self, dirty_images: tf.data.Dataset, clean_images: tf.data.Dataset, epochs: int = 50):
        """
        Training function.

        Args:

            dirty_images (tf.data.Dataset): dirty images dataset

            clean_images (tf.data.Dataset): clean images dataset

            epochs (int): Number of epochs to run

        """
        for epoch in tqdm.tqdm(range(epochs), leave=False):
            for image_x, image_y in tqdm.tqdm(tf.data.Dataset.zip((dirty_images, clean_images)), leave=False):
                self.train_step(image_x, image_y)

            if (epoch + 1) % 5 == 0:
                ckpt_save_path = self.ckpt_manager.save()
                print('Saving checkpoint for epoch {} at {}'.format(epoch + 1, ckpt_save_path))
