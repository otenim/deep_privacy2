import pathlib

import click
import torch
from tops.config import instantiate

from dp2.utils import load_config, print_config


@click.command()
@click.argument("config-path", type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=pathlib.Path))
@click.option("--batch-size", "-b", type=click.IntRange(min=1), default=1)
def main(config_path: pathlib.Path, batch_size: int) -> None:
    # Load anonymizer config
    anonymizer_cfg = load_config(config_path)
    print_config(anonymizer_cfg)

    # Load generator config
    generator_cfg = load_config(anonymizer_cfg.anonymizer.face_G_cfg)
    print_config(generator_cfg)

    # Instantiate generator
    generator = instantiate(generator_cfg.generator)
    print(generator)

    # To onnx graph
    dummy_patch = torch.randn(batch_size, 3, 256, 256)
    dummy_mask = torch.randn(batch_size, 1, 256, 256)
    torch.onnx.export(generator, (dummy_patch, dummy_mask), "hogehoge.onnx", verbose=False)


if __name__ == "__main__":
    main()
