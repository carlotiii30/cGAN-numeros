import numpy as np
import keras
import imageio

latent_dim = 128
num_classes = 10

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