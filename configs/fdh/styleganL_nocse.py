from tops.config import LazyCall as L

from ..datasets.fdh import data
from ..defaults import EMA, common, train
from ..discriminators.sg2_discriminator import D_optim, G_optim, discriminator, loss_fnc
from ..generators.stylegan_unet import generator

train.max_images_to_train = int(50e6)
G_optim.lr = 0.002
D_optim.lr = 0.002
generator.input_cse = False
data.load_embeddings = False
common.model_url = (
    "https://folk.ntnu.no/haakohu/checkpoints/deep_privacy2/fdh_styleganL_nocse.ckpt"
)
common.model_md5sum = "fda0d809741bc67487abada793975c37"
generator.fix_errors = False
