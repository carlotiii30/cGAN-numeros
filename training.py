from utils import (
    load_dataset,
    build_models,
    build_conditional_gan,
    train_model,
    save_model_weights,
)

dataset = load_dataset()
generator, discriminator = build_models()
cond_gan = build_conditional_gan(generator, discriminator)
train_model(dataset, cond_gan)
save_model_weights(cond_gan, "cond_gan_weights.weights.h5")
