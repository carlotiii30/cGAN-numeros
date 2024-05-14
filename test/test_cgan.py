import tensorflow as tf
from src.cGAN import conditionalGAN

class TestConditionalGAN:
    @classmethod
    def setup_class(cls):
        discriminator = tf.keras.models.Sequential()
        generator = tf.keras.models.Sequential()
        latent_dim = 100
        image_size = 28
        num_classes = 10
        cls.cgan = conditionalGAN(discriminator, generator, latent_dim, image_size, num_classes)
        cls.cgan.compile(
            d_optimizer=tf.keras.optimizers.Adam(),
            g_optimizer=tf.keras.optimizers.Adam(),
            loss_fn=tf.keras.losses.BinaryCrossentropy(),
        )

    def test_train_step(self):
        # Create dummy data for testing
        real_images = tf.random.normal((32, 28, 28, 1))
        one_hot_labels = tf.one_hot([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], depth=10)

        # Perform a single train step
        train_logs = self.cgan.train_step((real_images, one_hot_labels))

        # Check if the loss values are updated
        assert "g_loss" in train_logs
        assert "d_loss" in train_logs
        assert train_logs["g_loss"] != 0.0
        assert train_logs["d_loss"] != 0.0

    def test_metrics(self):
        # Check if the metrics are correctly initialized
        metrics = self.cgan.metrics
        assert len(metrics) == 2
        assert isinstance(metrics[0], tf.keras.metrics.Mean)
        assert isinstance(metrics[1], tf.keras.metrics.Mean)