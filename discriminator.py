import keras

class Discriminator:
    def __init__(self, channels):
        self.channels = channels
        self.model = self.build_model(channels)

    def build_model(self, channels):
        keras.Sequential(
            [
               keras.layers.InputLayer((28, 28, channels)),
               keras.layers.Conv2D(64, kernel_size=3, strides=2, padding="same"),
                keras.layers.LeakyReLU(negative_slope=0.2),
                keras.layers.Conv2D(128, kernel_size=3, strides=2, padding="same"),
                keras.layers.LeakyReLU(negative_slope=0.2),
                keras.layers.GlobalMaxPool2D(),
                keras.layers.Dense(1),
            ]
        )