import keras
import imageio
import numpy as np

from utils import num_classes, latent_dim, load_model_with_weights

def draw_number(number, cond_gan):
    number = int(number)
    noise = np.random.normal(size=(1, latent_dim))
    label = keras.utils.to_categorical([number], num_classes)
    label = label.astype("float32")

    noise_and_label = np.concatenate([noise, label], 1)
    generated_image = cond_gan.generator.predict(noise_and_label)

    generated_image = np.squeeze(generated_image)

    if len(generated_image.shape) > 2:
        generated_image = np.mean(generated_image, axis=-1)

    generated_image = np.clip(generated_image * 255, 0, 255)
    generated_image = generated_image.astype(np.uint8)

    filename = f"./images/drawn_number_{number}.png"
    imageio.imwrite(filename, generated_image)
