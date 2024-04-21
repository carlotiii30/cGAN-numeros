import keras
import tensorflow as tf


class conditionalGAN(keras.Model):
    def __init__(self, generator, discriminator, latent_dim):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim
        self.seed_generator = keras.random.SeedGenerator(1337)
        self.gen_loss_tracker = keras.metrics.Mean(name="generator_loss")
        self.dis_loss_tracker = keras.metrics.Mean(name="discriminator_loss")

    @property
    def metrics(self):
        return [self.gen_loss_tracker, self.dis_loss_tracker]

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, data):
        # Unpack the data
        real_images, labels = data

        # Discriminator
        # Add dummy dimensions to the labels so they can be concatenated with
        # the images.
        image_labels = labels[:, :, None, None]
        image_labels = keras.ops.repeat(image_labels, repeat=[28 * 28])
        image_labels = keras.ops.reshape(image_labels, (-1, 28, 28, 10))

        # Generator
        # Sample random points in the latent space.
        # Concatenate them with the labels.
        batch_size = keras.ops.shape(real_images)[0]
        random_latent_vectors = self.seed_generator.normal(
            shape=(batch_size, self.latent_dim)
        )
        random_vectors_labels = keras.ops.concatenate(
            [random_latent_vectors, labels], axis=1
        )

        # Decode them to fake images
        generated_images = self.generator(random_vectors_labels)

        # Combine them with real images
        fake_images_and_labels = keras.ops.concatenate(
            [generated_images, image_labels], -1
        )
        real_images_and_labels = keras.ops.concatenate([real_images, image_labels], -1)
        combined_images = keras.ops.concatenate(
            [fake_images_and_labels, real_images_and_labels], axis=0
        )

        # Assemble labels discriminating real from fake images
        labels = keras.ops.concatenate(
            [keras.ops.ones((batch_size, 1)), keras.ops.zeros((batch_size, 1))], axis=0
        )

        # Add random noise to the labels - important trick!
        labels += 0.05 * keras.random.uniform(keras.ops.shape(labels))

        # Train the discriminator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions)

        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        # Sample random points in the latent space
        random_latent_vectors = self.seed_generator.normal(
            shape=(batch_size, self.latent_dim)
        )
        random_vectors_labels = keras.ops.concatenate(
            [random_latent_vectors, labels], axis=1
        )

        # Assemble labels that say "all real images"
        misleading_labels = keras.ops.zeros((batch_size, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            fake_images = self.generator(random_vectors_labels)
            fake_images_and_labels = keras.ops.concatenate(
                [fake_images, image_labels], -1
            )
            predictions = self.discriminator(fake_images_and_labels)
            g_loss = self.loss_fn(misleading_labels, predictions)

        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # Update metrics
        self.gen_loss_tracker.update_state(g_loss)
        self.dis_loss_tracker.update_state(d_loss)

        return {
            "generator_loss": self.gen_loss_tracker.result(),
            "discriminator_loss": self.dis_loss_tracker.result(),
        }
