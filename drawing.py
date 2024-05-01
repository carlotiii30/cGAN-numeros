import keras
import imageio
import numpy as np

from utils import num_classes, latent_dim

def draw_number(number, cond_gan):
    # Genera ruido aleatorio y la etiqueta correspondiente al número
    noise = np.random.normal(size=(1, latent_dim))
    label = keras.utils.to_categorical([number], num_classes)
    label = label.astype("float32")

    # Concatena el ruido y la etiqueta para generar la imagen
    noise_and_label = np.concatenate([noise, label], 1)
    generated_image = cond_gan.generator.predict(noise_and_label)

    # Elimina cualquier dimensión extra
    generated_image = np.squeeze(generated_image)

    # Verifica la forma de la imagen generada
    if len(generated_image.shape) > 2:
        # Si la imagen tiene más de dos dimensiones, conviértela a escala de grises
        generated_image = np.mean(generated_image, axis=-1)

    # Asegúrate de que los valores estén en el rango [0, 255]
    generated_image = np.clip(generated_image * 255, 0, 255)

    # Convierte la imagen a tipo de datos uint8
    generated_image = generated_image.astype(np.uint8)

    # Guarda la imagen en el archivo
    filename = f"./images/drawn_number_{number}.png"
    imageio.imwrite(filename, generated_image)