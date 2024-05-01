import keras
import imageio
import numpy as np

from utils import latent_dim, num_classes


def interpolate_images(cond_gan):
    trained_gen = cond_gan.generator

    num_interpolation = 9
    interpolation_noise = keras.random.normal(shape=(1, latent_dim))
    interpolation_noise = keras.ops.repeat(
        interpolation_noise, repeats=num_interpolation
    )
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
            first_label * (1 - percent_second_label)
            + second_label * percent_second_label
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
        filename = f"./images/interpolated/interpolated_image_{i}.png"
        imageio.imwrite(filename, converted_image)
