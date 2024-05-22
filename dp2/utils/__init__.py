import pathlib
from pathlib import Path
from typing import Union

from omegaconf import DictConfig
from tops.config import LazyConfig

from .cse import from_E_to_vertex
from .ema import EMA
from .torch_utils import (
    binary_dilation,
    crop_box,
    denormalize_img,
    forward_D_fake,
    im2numpy,
    im2torch,
    remove_pad,
    set_requires_grad,
    torch_wasserstein_loss,
)
from .utils import config_to_str, init_tops, print_config, tqdm_, trange_


def load_config(config_path: Union[str, Path]) -> DictConfig:
    if isinstance(config_path, str):
        config_path = pathlib.Path(config_path)
    assert config_path.is_file()
    cfg = LazyConfig.load(str(config_path))
    cfg.output_dir = pathlib.Path(
        str(config_path)
        .replace("configs", str(cfg.common.output_dir))
        .replace(".py", "")
    )
    if cfg.common.experiment_name is None:
        cfg.experiment_name = str(config_path)
    else:
        cfg.experiment_name = cfg.common.experiment_name
    cfg.checkpoint_dir = cfg.output_dir.joinpath("checkpoints")
    print("Saving outputs to:", cfg.output_dir)
    return cfg
