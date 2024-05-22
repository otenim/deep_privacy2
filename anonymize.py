import hashlib
from pathlib import Path
from typing import Optional, Union

import click
import cv2
import moviepy.editor as mp
import numpy as np
import tops
import tqdm
from detectron2.data.detection_utils import _apply_exif_orientation
from PIL import Image
from tops import logger
from tops.config import instantiate

from dp2 import utils
from dp2.utils.bufferless_video_capture import BufferlessVideoCapture


def show_video(video_path):
    video_cap = cv2.VideoCapture(str(video_path))
    while video_cap.isOpened():
        _, frame = video_cap.read()
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(25)
        if key == ord("q"):
            break
    video_cap.release()
    cv2.destroyAllWindows()


class ImageIndexTracker:
    def __init__(self, fn) -> None:
        self.fn = fn
        self.idx = 0

    def fl_image(self, frame):
        self.idx += 1
        return self.fn(frame, self.idx - 1)


def anonymize_video(
    video_path,
    output_path: Path,
    anonymizer,
    max_res: int,
    fps: Union[int, None],
    start_time: int,
    end_time: Union[int, None],
    visualize_detection: bool,
    track: bool,
    synthesis_kwargs,
    **kwargs,
):
    video: mp.VideoFileClip
    video = mp.VideoFileClip(str(video_path))
    if track:
        anonymizer.initialize_tracker(video.fps)

    def process_frame(frame, idx):
        frame = np.array(resize(Image.fromarray(frame), max_res))
        cache_id = hashlib.md5(frame).hexdigest()
        frame = utils.im2torch(frame, to_float=False, normalize=False)[0]
        cache_id_ = cache_id + str(idx)
        synthesis_kwargs["cache_id"] = cache_id_
        if visualize_detection:
            anonymized = anonymizer.visualize_detection(frame, cache_id=cache_id_)
        else:
            anonymized = anonymizer(frame, **synthesis_kwargs)
        anonymized = utils.im2numpy(anonymized)
        return anonymized

    video = video.subclip(start_time, end_time)

    if fps is not None:
        video = video.set_fps(fps)

    video = video.fl_image(ImageIndexTracker(process_frame).fl_image)
    if output_path.suffix == ".avi":
        output_path = output_path.with_suffix(".mp4")
    video.write_videofile(output_path)


def resize(frame: Image.Image, max_res: Optional[int] = None) -> Image.Image:
    if max_res is None:
        return frame
    f = max(*[x / max_res for x in frame.size], 1)
    if f == 1:
        return frame
    new_shape = [int(x / f) for x in frame.size]
    return frame.resize(new_shape, resample=Image.BILINEAR)


def anonymize_image(
    image_path: Path,
    output_path: Path,
    anonymizer,
    max_res: int,
    visualize_detection: bool,
    synthesis_kwargs: dict,
    **kwargs,
) -> None:
    with Image.open(image_path) as im:
        im = _apply_exif_orientation(im)
        orig_im_mode = im.mode
        im = im.convert("RGB")
        im = resize(im, max_res)
    im = np.array(im)
    md5_ = hashlib.md5(im).hexdigest()
    im = utils.im2torch(np.array(im), to_float=False, normalize=False)[0]
    synthesis_kwargs["cache_id"] = md5_
    if visualize_detection:
        im_ = anonymizer.visualize_detection(tops.to_cuda(im), cache_id=md5_)
    else:
        im_ = anonymizer(im, **synthesis_kwargs)
    im_ = utils.im2numpy(im_)
    im = Image.fromarray(im_).convert(orig_im_mode)
    if output_path is not None:
        output_path.parent.mkdir(exist_ok=True, parents=True)
        im.save(output_path, optimize=False, quality=100)
        print(f"Saved to: {output_path}")


def anonymize_file(input_path: Path, output_path: Path, **kwargs) -> None:
    if output_path is not None and output_path.is_file():
        logger.warn(f"Overwriting previous file: {output_path}")
    if tops.is_image(input_path):
        anonymize_image(input_path, output_path, **kwargs)
    elif tops.is_video(input_path):
        anonymize_video(input_path, output_path, **kwargs)
    else:
        logger.log(f"Filepath not a video or image file: {input_path}")


def anonymize_directory(input_dir: Path, output_dir: Path, **kwargs) -> None:
    for childname in tqdm.tqdm(input_dir.iterdir()):
        childpath = input_dir.joinpath(childname.name)
        output_path = output_dir.joinpath(childname.name)
        if not childpath.is_file():
            anonymize_directory(childpath, output_path, **kwargs)
        else:
            anonymize_file(childpath, output_path, **kwargs)


def anonymize_webcam(
    anonymizer,
    max_res: int,
    synthesis_kwargs,
    visualize_detection,
    track: bool,
    **kwargs,
):
    import time

    cap = BufferlessVideoCapture(0, width=1920, height=1080)
    t = time.time()
    frames = 0
    if track:
        anonymizer.initialize_tracker(fps=5)  # FPS used for tracking objects
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame = Image.fromarray(frame[:, :, ::-1])
        frame = resize(frame, max_res)
        frame = np.array(frame)
        im = utils.im2torch(np.array(frame), to_float=False, normalize=False)[0]
        if visualize_detection:
            im_ = anonymizer.visualize_detection(tops.to_cuda(im))
        else:
            im_ = anonymizer(im, **synthesis_kwargs)
        im_ = utils.im2numpy(im_)

        frames += 1
        delta = time.time() - t
        fps = "?"
        if delta > 1e-6:
            fps = frames / delta
        print(f"FPS: {fps:.3f}", end="\r")
        cv2.imshow("frame", im_[:, :, ::-1])
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


@click.command()
@click.argument(
    "config-path",
    type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path),
)
@click.argument(
    "input-path",
    type=click.Path(exists=True, dir_okay=True, file_okay=True, path_type=Path),
)
@click.argument(
    "output-path",
    type=click.Path(exists=False, dir_okay=True, file_okay=True, path_type=Path),
)
@click.option(
    "--max-res", default=None, type=int, help="Maximum resolution of height/width."
)
@click.option(
    "--start-time",
    "--st",
    default=0,
    type=int,
    help="Start time (second) for vide anonymization",
)
@click.option(
    "--end-time",
    "--et",
    default=None,
    type=int,
    help="End time (second) for vide anonymization",
)
@click.option("--fps", default=None, type=int, help="FPS for anonymization")
@click.option(
    "--detection-score-threshold",
    "--dst",
    default=0.3,
    type=click.FloatRange(0, 1),
    help="Detection threshold, threshold applied for all detection models.",
)
@click.option(
    "--visualize-detection",
    "--vd",
    default=False,
    is_flag=True,
    help="Visualize only detections without running anonymization.",
)
@click.option(
    "--multi-modal-truncation",
    "--mt",
    default=False,
    is_flag=True,
    help="Enable multi-modal truncation proposed by: https://arxiv.org/pdf/2202.12211.pdf",
)
@click.option(
    "--cache",
    default=False,
    is_flag=True,
    help="Enable detection caching. Will save and load detections from cache.",
)
@click.option(
    "--amp",
    default=True,
    is_flag=True,
    help="Use automatic mixed precision for generator forward pass",
)
@click.option(
    "-t",
    "--truncation-value",
    default=0,
    type=click.FloatRange(0, 1),
    help="Latent interpolation truncation value.",
)
@click.option(
    "--track",
    default=False,
    is_flag=True,
    help="Track detections over frames. Will use the same latent variable (z) for tracked identities.",
)
@click.option(
    "--seed", default=0, type=int, help="Set random seed for generating images."
)
@click.option(
    "--person-generator",
    default=None,
    help="Config path to unconditional person generator",
    type=click.Path(),
)
@click.option(
    "--cse-person-generator",
    default=None,
    help="Config path to CSE-guided person generator",
    type=click.Path(),
)
@click.option(
    "--webcam", default=False, is_flag=True, help="Read image from webcam feed."
)
@click.option(
    "--text-prompt",
    default=None,
    type=str,
    help="Text prompt for attribute guided anonymization. Requires the validation dataset downloaded.",
)
@click.option(
    "--text-prompt-strength",
    default=0.5,
    type=float,
    help="Strength for attribute-guided anonymization",
)
def anonymize_path(
    config_path: Path,
    input_path: Path,
    output_path: Path,
    detection_score_threshold: float,
    visualize_detection: bool,
    cache: bool,
    seed: int,
    person_generator: Union[str, None],
    cse_person_generator: Union[str, None],
    webcam: bool,
    **kwargs,
) -> None:
    tops.set_seed(seed)
    cfg = utils.load_config(config_path)
    if person_generator is not None:
        cfg.anonymizer.person_G_cfg = person_generator
    if cse_person_generator is not None:
        cfg.anonymizer.cse_person_G_cfg = cse_person_generator
    cfg.detector.score_threshold = detection_score_threshold
    utils.print_config(cfg)

    anonymizer = instantiate(cfg.anonymizer, load_cache=cache)
    synthesis_kwargs = {
        k: kwargs.pop(k)
        for k in [
            "amp",
            "multi_modal_truncation",
            "truncation_value",
            "text_prompt",
            "text_prompt_strength",
        ]
    }

    kwargs["anonymizer"] = anonymizer
    kwargs["visualize_detection"] = visualize_detection
    kwargs["synthesis_kwargs"] = synthesis_kwargs

    if webcam:
        anonymize_webcam(**kwargs)
        return
    if input_path.is_dir():
        assert output_path.is_dir()
        anonymize_directory(input_path, output_path, **kwargs)
    elif input_path.is_file():
        assert output_path.is_file()
        anonymize_file(input_path, output_path, **kwargs)


if __name__ == "__main__":
    anonymize_path()
