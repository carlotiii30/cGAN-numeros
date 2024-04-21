import keras
import numpy as np
import tensorflow as tf

class Dataset:
    def __init__(self):
        self.dataset = None

    def load(self):
        # Usamos el dataset MNIST, con 60,000 imágenes de entrenamiento y 10,000 de test
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        all_digits = np.concatenate([x_train, x_test])
        all_labels = np.concatenate([y_train, y_test])

        # Escalamos los píxeles a valores entre 0 y 1, añadimos una dimensión extra
        # para el canal y utilizamos one-hot encoding para las etiquetas.
        all_digits = all_digits.astype("float32") / 255.0
        all_digits = np.reshape(all_digits, (-1, 28, 28, 1))
        all_labels = keras.utils.to_categorical(all_labels, 10)

        # Creamos un dataset de TensorFlow a partir de los datos
        dataset = tf.data.Dataset.from_tensor_slices((all_digits, all_labels))
        dataset = dataset.shuffle(buffer_size=1024).batch(64)

        print(f"Shape of training images: {all_digits.shape}")
        print(f"Shape of training labels: {all_labels.shape}")

        self.dataset = dataset