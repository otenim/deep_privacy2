import lzma
import pickle
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Any, Union

import torch
from tops import logger


class BaseDetector(metaclass=ABCMeta):
    def __init__(self, cache_directory: str) -> None:
        if cache_directory is not None:
            self.cache_directory = Path(cache_directory, str(self.__class__.__name__))
            self.cache_directory.mkdir(exist_ok=True, parents=True)

    def save_to_cache(self, detection, cache_path: Path, after_preprocess=True):
        logger.log(f"Caching detection to: {cache_path}")
        with lzma.open(cache_path, "wb") as fp:
            torch.save(
                [det.state_dict(after_preprocess=after_preprocess) for det in detection],
                fp,
                pickle_protocol=pickle.HIGHEST_PROTOCOL,
            )

    def load_from_cache(self, cache_path: Path):
        logger.log(f"Loading detection from cache path: {cache_path}")
        with lzma.open(cache_path, "rb") as fp:
            state_dict = torch.load(fp)
        return [state["cls"].from_state_dict(state_dict=state) for state in state_dict]

    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        pass

    def forward_and_cache(self, im: torch.Tensor, cache_id: Union[str, None], load_cache: bool):
        if cache_id is None:
            return self.forward(im)
        cache_path = self.cache_directory.joinpath(cache_id + ".torch")
        if cache_path.is_file() and load_cache:
            try:
                return self.load_from_cache(cache_path)
            except Exception:
                logger.warn(f"The cache file was corrupted: {cache_path}")
                exit()
        detections = self.forward(im)
        self.save_to_cache(detections, cache_path)
        return detections
