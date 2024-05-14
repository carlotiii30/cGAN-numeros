import os
import keras
import numpy as np
import tensorflow as tf

from cGAN import conditionalGAN

BATCH_SIZE = 64
NUM_CHANNELS = 1
NUM_CLASSES = 10
IMAGE_SIZE = 28
LATENT_DIM = 128


def load_dataset():
    """
    Loads the MNIST dataset, preprocesses it, and returns it as a TensorFlow dataset.

    Returns:
        tf.data.Dataset: Dataset containing the preprocessed MNIST images and labels.
    """
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    all_digits = np.concatenate([x_train, x_test])
    all_labels = np.concatenate([y_train, y_test])

    all_digits = all_digits.astype("float32") / 255.0
    all_digits = np.reshape(all_digits, (-1, 28, 28, 1))
    all_labels = keras.utils.to_categorical(all_labels, 10)

    dataset = tf.data.Dataset.from_tensor_slices((all_digits, all_labels))
    dataset = dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE)

    return dataset


def build_models():
    """
    Builds the generator and discriminator models for the cGAN.

    Returns:
        keras.Model: Generator model.
        keras.Model: Discriminator model.
    """
    # - - - - - - - Calculate the number of input channels - - - - - - -
    gen_channels = LATENT_DIM + NUM_CLASSES
    dis_channels = NUM_CHANNELS + NUM_CLASSES

    # - - - - - - - Generator - - - - - - -
    generator = keras.Sequential(
        [
            keras.layers.InputLayer((gen_channels,)),
            keras.layers.Dense(7 * 7 * gen_channels),
            keras.layers.LeakyReLU(negative_slope=0.2),
            keras.layers.Reshape((7, 7, gen_channels)),
            keras.layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"),
            keras.layers.LeakyReLU(negative_slope=0.2),
            keras.layers.Conv2DTranspose(
                BATCH_SIZE, kernel_size=4, strides=2, padding="same"
            ),
            keras.layers.LeakyReLU(negative_slope=0.2),
            keras.layers.Conv2DTranspose(
                1, kernel_size=7, strides=1, padding="same", activation="sigmoid"
            ),
        ],
        name="generator",
    )

    # - - - - - - - Discriminator - - - - - - -
    discriminator = keras.Sequential(
        [
            keras.layers.InputLayer((28, 28, dis_channels)),
            keras.layers.Conv2D(BATCH_SIZE, kernel_size=3, strides=2, padding="same"),
            keras.layers.LeakyReLU(negative_slope=0.2),
            keras.layers.Conv2D(128, kernel_size=3, strides=2, padding="same"),
            keras.layers.LeakyReLU(negative_slope=0.2),
            keras.layers.GlobalMaxPool2D(),
            keras.layers.Dense(1),
        ],
        name="discriminator",
    )

    return generator, discriminator


def build_conditional_gan(generator, discriminator):
    """
    Builds the conditional GAN (cGAN) model.

    Args:
        generator (keras.Model): Generator model.
        discriminator (keras.Model): Discriminator model.

    Returns:
        conditionalGAN: Compiled cGAN model.
    """
    cond_gan = conditionalGAN(
        discriminator=discriminator,
        generator=generator,
        latent_dim=LATENT_DIM,
        image_size=IMAGE_SIZE,
        num_classes=NUM_CLASSES,
    )
    cond_gan.compile(
        d_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
        g_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
        loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
    )

    return cond_gan


def train_model(dataset, cond_gan):
    """
    Trains the conditional GAN (cGAN) model.

    Args:
        dataset (tf.data.Dataset): Dataset containing the training images and labels.
        cond_gan (conditionalGAN): Compiled cGAN model.
    """
    cond_gan.fit(dataset, epochs=30)


def save_model_weights(cond_gan, filename):
    """
    Saves the weights of the conditional GAN (cGAN) model.

    Args:
        cond_gan (conditionalGAN): Compiled cGAN model.
        filename (str): Filepath to save the model weights.
    """
    if os.path.exists(filename):
        os.remove(filename)
    cond_gan.save_weights(filename)


def load_model_with_weights(filename):
    """
    Loads the conditional GAN (cGAN) model with saved weights.

    Args:
        filename (str): Filepath to the saved model weights.

    Returns:
        conditionalGAN: cGAN model with loaded weights.
    """
    generator, discriminator = build_models()

    new_cond_gan = build_conditional_gan(generator, discriminator)

    new_cond_gan.load_weights(filename)
    return new_cond_gan
