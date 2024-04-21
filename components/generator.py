import keras

class Generator:
    def __init__(self, channels):
        self.channels = channels
        self.model = self.build_model(channels)

    def build_model(self, channels):
        keras.Sequential(
            [
                keras.layers.InputLayer((channels, )),
                # We want to generate 128 + num_classes coefficients to reshape
                # into a 7x7x(128 + num_classes) map
                keras.layers.Dense(7 * 7 * channels),
                keras.layers.LeakyReLU(negative_slope=0.2),

                # Layer
                keras.layers.Reshape((7, 7, channels)),
                keras.layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"),

                # Layer
                keras.layers.LeakyReLU(negative_slope=0.2),
                keras.layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding="same"),

                # Output layer
                keras.layers.LeakyReLU(negative_slope=0.2),
                keras.layers.Conv2DTranspose(1, kernel_size=7, strides=1, padding="same", activation="sigmoid"),
            ]
        )