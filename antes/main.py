import os
import keras
import imageio
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
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
all_digits = np.concatenate([x_train, x_test])
all_labels = np.concatenate([y_train, y_test])

all_digits = all_digits.astype("float32") / 255.0
all_digits = np.reshape(all_digits, (-1, 28, 28, 1))
all_labels = keras.utils.to_categorical(all_labels, 10)

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
        keras.layers.Dense(7 * 7 * gen_channels),
        keras.layers.LeakyReLU(negative_slope=0.2),

        keras.layers.Reshape((7, 7, gen_channels)),
        keras.layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"),

        keras.layers.LeakyReLU(negative_slope=0.2),
        keras.layers.Conv2DTranspose(
            batch_size, kernel_size=4, strides=2, padding="same"
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

#cond_gan.fit(dataset, epochs=1)

# Guardar los pesos del modelo
if os.path.exists("cond_gan_weights.weights.h5"):
    os.remove("cond_gan_weights.weights.h5")

cond_gan.save_weights("cond_gan_weights.weights.h5")

# Cargar los pesos en un nuevo modelo
new_cond_gan = conditionalGAN(
    discriminator=discriminator,
    generator=generator,
    latent_dim=latent_dim,
    image_size=image_size,
    num_classes=num_classes,
)

new_cond_gan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    g_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
)

new_cond_gan.load_weights("cond_gan_weights.weights.h5")

# - - - - - - - Interpolation - - - - - - -

trained_gen = new_cond_gan.generator

num_interpolation = 9

interpolation_noise = keras.random.normal(shape=(1, latent_dim))
interpolation_noise = keras.ops.repeat(interpolation_noise, repeats=num_interpolation)
interpolation_noise = keras.ops.reshape(
    interpolation_noise, (num_interpolation, latent_dim)
)


def interpolate_class(first_number, second_number):
    first_label = keras.utils.to_categorical([first_number], num_classes)
    second_label = keras.utils.to_categorical([second_number], num_classes)
    first_label = keras.ops.cast(first_label, "float32")
    second_label = keras.ops.cast(second_label, "float32")

    percent_second_label = keras.ops.linspace(0, 1, num_interpolation)[:, None]
    percent_second_label = keras.ops.cast(percent_second_label, "float32")
    interpolation_labels = (
        first_label * (1 - percent_second_label) + second_label * percent_second_label
    )

    noise_and_labels = keras.ops.concatenate(
        [interpolation_noise, interpolation_labels], 1
    )
    fake_images = trained_gen.predict(noise_and_labels)
    return fake_images


start_class = 2
end_class = 6
fake_images = interpolate_class(start_class, end_class)

fake_images_rgb = np.tile(fake_images, (1, 1, 1, 3))

for i, image in enumerate(fake_images_rgb):
    image *= 255.0
    converted_image = image.astype(np.uint8)
    filename = f"./images/interpolated_image_{i}.png"
    imageio.imwrite(filename, converted_image)


# - - - - - - Generate a number - - - - - -
def draw_number(number):
    noise = keras.random.normal(shape=(1, latent_dim))
    label = keras.utils.to_categorical([number], num_classes)
    label = keras.ops.cast(label, "float32")

    noise_and_label = keras.ops.concatenate([noise, label], 1)
    generated_image = cond_gan.generator.predict(noise_and_label)

    generated_image *= 255.0
    converted_image = generated_image.astype(np.uint8)
    filename = f"./images/drawn_number_{number}.png"
    imageio.imwrite(filename, converted_image)