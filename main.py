import keras
import numpy as np
import tensorflow as tf
from cGAN import conditionalGAN


# - - - - - - - Constants - - - - - - -
batch_size = 64
num_channels = 1
num_classes = 10
image_size = 28
latent_dim = 128

# - - - - - - - Load the dataset - - - - - - -
# We'll use all the available examples from both the training and test
# sets.
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
all_digits = np.concatenate([x_train, x_test])
all_labels = np.concatenate([y_train, y_test])

# Scale the pixel values to [0, 1] range, add a channel dimension to
# the images, and one-hot encode the labels.
all_digits = all_digits.astype("float32") / 255.0
all_digits = np.reshape(all_digits, (-1, 28, 28, 1))
all_labels = keras.utils.to_categorical(all_labels, 10)

# Create tf.data.Dataset.
dataset = tf.data.Dataset.from_tensor_slices((all_digits, all_labels))
dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)

print(f"Shape of training images: {all_digits.shape}")
print(f"Shape of training labels: {all_labels.shape}")

# - - - - - - - Calculate the number of input channels - - - - - - -
gen_channels = latent_dim + num_classes
dis_channels = num_channels + num_classes

# - - - - - - - Generator - - - - - - -
generator = keras.Sequential(
    [
        keras.layers.InputLayer((gen_channels,)),
        # We want to generate 128 + num_classes coefficients to reshape
        # into a 7x7x(128 + num_classes) map
        keras.layers.Dense(7 * 7 * gen_channels),
        keras.layers.LeakyReLU(negative_slope=0.2),
        # Layer
        keras.layers.Reshape((7, 7, gen_channels)),
        keras.layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"),
        # Layer
        keras.layers.LeakyReLU(negative_slope=0.2),
        keras.layers.Conv2DTranspose(
            batch_size, kernel_size=4, strides=2, padding="same"
        ),
        # Output layer
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
        keras.layers.Conv2D(batch_size, kernel_size=3, strides=2, padding="same"),
        keras.layers.LeakyReLU(negative_slope=0.2),
        keras.layers.Conv2D(128, kernel_size=3, strides=2, padding="same"),
        keras.layers.LeakyReLU(negative_slope=0.2),
        keras.layers.GlobalMaxPool2D(),
        keras.layers.Dense(1),
    ],
    name="discriminator",
)

# - - - - - - - Conditional GAN - - - - - - -
cond_gan = conditionalGAN(
    discriminator=discriminator,
    generator=generator,
    latent_dim=latent_dim,
    image_size=image_size,
    num_classes=num_classes,
)
cond_gan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    g_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
)

cond_gan.fit(dataset, epochs=20)
