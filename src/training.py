import src.utils as utils

dataset = utils.load_dataset()
generator, discriminator = utils.build_models()
cond_gan = utils.build_conditional_gan(generator, discriminator)
utils.train_model(dataset, cond_gan)
utils.save_model_weights(cond_gan, "cond_gan_weights_alto.weights.h5")
