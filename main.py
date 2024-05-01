from utils import (
    load_dataset,
    build_models,
    build_conditional_gan,
    train_model,
    save_model_weights,
    load_model_with_weights,
)
from drawing import draw_number
from interpolation import interpolate_images

#dataset = load_dataset()
#generator, discriminator = build_models()
#cond_gan = build_conditional_gan(generator, discriminator)
#train_model(dataset, cond_gan)
#save_model_weights(cond_gan, "cond_gan_weights.weights.h5")
cond_gan = load_model_with_weights("cond_gan_weights.weights.h5")
interpolate_images(cond_gan)
draw_number(7, cond_gan)
