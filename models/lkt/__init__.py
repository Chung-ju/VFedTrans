from .base import *
from .beta_vae import *
from .ae import *
from .vae import *
from .sae import *
from .wae import *
from .mine import *
from .cross_domain import *

lrd_models = {
    'AE': AE,
    'VAE': VAE,
    'WAE': WAE,
    'SAE': SparseAE,
    'BetaVAE': BetaVAE,
}