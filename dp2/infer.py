from typing import Dict, Optional

import tops
import torch
from omegaconf import DictConfig
from tops import checkpointer
from tops.config import instantiate
from tops.logger import warn

from dp2.generator.base import BaseGenerator
from dp2.generator.deep_privacy1 import MSGGenerator


def load_generator_state(ckpt: Dict, generator: BaseGenerator, ckpt_mapper=None) -> None:
    state = ckpt["EMA_generator"] if "EMA_generator" in ckpt else ckpt["running_average_generator"]
    if ckpt_mapper is not None:
        state = ckpt_mapper(state)
    if isinstance(generator, MSGGenerator):
        generator.load_state_dict(state)
    else:
        load_state_dict(generator, state)
    tops.logger.log(f"Generator loaded, num parameters: {tops.num_parameters(generator)/1e6}M")
    if "w_centers" in ckpt:
        generator.style_net.register_buffer("w_centers", ckpt["w_centers"])
        tops.logger.log(f"W cluster centers loaded. Number of centers: {len(generator.style_net.w_centers)}")
    if "style_net.w_centers" in state:
        generator.style_net.register_buffer("w_centers", state["style_net.w_centers"])
        tops.logger.log(f"W cluster centers loaded. Number of centers: {len(generator.style_net.w_centers)}")


def build_trained_generator(cfg: DictConfig, map_location: Optional[str] = None) -> BaseGenerator:
    map_location = map_location if map_location is not None else tops.get_device()
    generator = instantiate(cfg.generator)
    generator.eval()
    generator.imsize = tuple(cfg.data.imsize) if hasattr(cfg, "data") else None
    if hasattr(cfg, "ckpt_mapper"):
        ckpt_mapper = instantiate(cfg.ckpt_mapper)
    else:
        ckpt_mapper = None
    if "model_url" in cfg.common:
        ckpt = tops.load_file_or_url(cfg.common.model_url, md5sum=cfg.common.model_md5sum)
        load_generator_state(ckpt, generator, ckpt_mapper=ckpt_mapper)
        return generator.to(map_location)
    try:
        ckpt = checkpointer.load_checkpoint(cfg.checkpoint_dir, map_location="cpu")
        load_generator_state(ckpt, generator, ckpt_mapper=ckpt_mapper)
    except FileNotFoundError:
        tops.logger.warn(f"Did not find generator checkpoint in: {cfg.checkpoint_dir}")
    return generator.to(map_location)


def build_trained_discriminator(cfg: DictConfig, map_location: Optional[str] = None):
    map_location = map_location if map_location is not None else tops.get_device()
    D = instantiate(cfg.discriminator).to(map_location)
    D.eval()
    try:
        ckpt = checkpointer.load_checkpoint(cfg.checkpoint_dir, map_location="cpu")
        if hasattr(cfg, "ckpt_mapper_D"):
            ckpt["discriminator"] = instantiate(cfg.ckpt_mapper_D)(ckpt["discriminator"])
        D.load_state_dict(ckpt["discriminator"])
    except FileNotFoundError:
        tops.logger.warn(f"Did not find discriminator checkpoint in: {cfg.checkpoint_dir}")
    return D


def load_state_dict(module: torch.nn.Module, state_dict: dict):
    ignore_key = "style_net.w_centers"  # Loaded by buyild_trained_generator
    module_sd = module.state_dict()
    to_remove = []
    for key, item in state_dict.items():
        if key not in module_sd:
            continue
        if item.shape != module_sd[key].shape:
            to_remove.append(key)
            warn(f"Incorrect shape. Current model: {module_sd[key].shape}, in state dict: {item.shape} for key: {key}")
    for key in to_remove:
        state_dict.pop(key)
    for key, item in state_dict.items():
        if key == ignore_key:
            continue
        if key not in module_sd:
            warn(f"Did not find key in model state dict: {key}")
    for key, item in module_sd.items():
        if key not in state_dict:
            warn(f"Did not find key in state dict: {key}")
    module.load_state_dict(state_dict, strict=False)
