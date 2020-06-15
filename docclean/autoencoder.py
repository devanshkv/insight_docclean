import logging

import tensorflow as tf

logger = logging.getLogger(__name__)


class Autoencoder():
    """
    Autoencoder class.

    Args:

        img_rows (int): Number of pixel rows

        img_cols (int): Number of pixel columns

        nchans (int): Number of color channels

        mixed_precision (bool): Train with fp16 mixed precision

        early_stopping (int): Patience for early stopping


    """

    def __init__(self, img_rows: int = 256, img_cols: int = 256, nchans: int = 3,
                 mixed_precision_training: bool = False,
                 early_stopping: int = 25):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channels = nchans
        self.early_stopping = early_stopping
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        if mixed_precision_training:
            from tensorflow.keras.mixed_precision import experimental as mixed_precision

            self.policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_policy(self.policy)

            self.autoencoder_model = self.build_model()
            optimizer = tf.optimizers.Adam()
            self.autoencoder_model.compile(loss='mse', optimizer=optimizer,
                                           metrics=[tf.keras.metrics.RootMeanSquaredError()])
        else:
            self.autoencoder_model = self.build_model()
            optimizer = tf.optimizers.Adam()
            self.autoencoder_model.compile(loss='mse', optimizer=optimizer,
                                           metrics=[tf.keras.metrics.RootMeanSquaredError()])

        self.autoencoder_model.summary()

    def build_model(self):
        """

        Build the autoencoder model

        Returns:

            tf.keras.Model: The autoencoder model

        """
        # ENCODER
        input_layer = tf.keras.Input(shape=self.img_shape)
        e = tf.keras.layers.Conv2D(32, (4, 4), strides=2, activation='relu')(input_layer)
        e = tf.keras.layers.Conv2D(64, (3, 3), strides=2, activation='relu')(e)
        e = tf.keras.layers.Conv2D(128, (3, 3), strides=2, activation='relu')(e)

        # DECODER
        d = tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=2, activation='relu')(e)
        d = tf.keras.layers.BatchNormalization()(d)
        d = tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=2, activation='relu')(d)
        d = tf.keras.layers.BatchNormalization()(d)
        d = tf.keras.layers.Conv2DTranspose(164, (4, 4), strides=2, activation='relu')(d)
        output_layer = tf.keras.layers.Conv2D(self.channels, (3, 3), activation='sigmoid', padding='same')(d)
        return tf.keras.Model(input_layer, output_layer)

    def train_model(self, train_dataset: tf.data.Dataset, epochs: int = 100, validation_data: tf.data.Dataset = None):
        """

        Train the autoencoder

        Args:

            train_dataset (tf.data.Dataset) : Training dataset

            epochs (int): Number of epochs to train for

            validation_data (tf.data.Dataset) : Validation dataset

        Returns:

            dict: Training history

        """
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                          min_delta=1e-2,
                                                          patience=self.early_stopping,
                                                          verbose=1,
                                                          mode='auto')
        if validation_data is None:
            self.history = self.autoencoder_model.fit(train_dataset, epochs=epochs,
                                                      callbacks=[early_stopping],
                                                      verbose=1)
        else:
            self.history = self.autoencoder_model.fit(train_dataset, epochs=epochs,
                                                      validation_data=validation_data,
                                                      callbacks=[early_stopping],
                                                      verbose=1)
        return self.history
