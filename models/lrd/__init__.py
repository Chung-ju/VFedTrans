from .base import *
from .beta_vae import *
from .ae import *
from .gan import *

lrd_models = {
    'AE': AutoEncoder,
    'BetaVAE': BetaVAE,
    'GAN': GAN
}