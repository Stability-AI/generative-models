import math
import os
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import imageio
import numpy as np
import torch
import torchvision.transforms as TT
from einops import rearrange, repeat
from omegaconf import ListConfig, OmegaConf
from PIL import Image, ImageSequence
from rembg import remove
from scripts.util.detection.nsfw_and_watermark_dectection import DeepFloydDataFiltering
from sgm.modules.autoencoding.temporal_ae import VideoDecoder
from sgm.modules.diffusionmodules.guiders import (
    LinearPredictionGuider,
    SpatiotemporalPredictionGuider,
    TrapezoidPredictionGuider,
    TrianglePredictionGuider,
    VanillaCFG,
)
from sgm.modules.diffusionmodules.sampling import (
    DPMPP2MSampler,
    DPMPP2SAncestralSampler,
    EulerAncestralSampler,
    EulerEDMSampler,
    HeunEDMSampler,
    LinearMultistepSampler,
)
from sgm.util import default, instantiate_from_config
from torch import autocast
from torchvision.transforms import ToTensor


def load_module_gpu(model):
    model.cuda()


def unload_module_gpu(model):
    model.cpu()
    torch.cuda.empty_cache()


def initial_model_load(model):
    model.model.half()
    return model


def get_resizing_factor(
    desired_shape: Tuple[int, int], current_shape: Tuple[int, int]
) -> float:
    r_bound = desired_shape[1] / desired_shape[0]
    aspect_r = current_shape[1] / current_shape[0]
    if r_bound >= 1.0:
        if aspect_r >= r_bound:
            factor = min(desired_shape) / min(current_shape)
        else:
            if aspect_r < 1.0:
                factor = max(desired_shape) / min(current_shape)
            else:
                factor = max(desired_shape) / max(current_shape)
    else:
        if aspect_r <= r_bound:
            factor = min(desired_shape) / min(current_shape)
        else:
            if aspect_r > 1:
                factor = max(desired_shape) / min(current_shape)
            else:
                factor = max(desired_shape) / max(current_shape)
    return factor


def read_gif(input_path, n_frames):
    frames = []
    video = Image.open(input_path)
    for img in ImageSequence.Iterator(video):
        frames.append(img.convert("RGBA"))
        if len(frames) == n_frames:
            break
    return frames


def read_mp4(input_path, n_frames):
    frames = []
    vidcap = cv2.VideoCapture(input_path)
    success, image = vidcap.read()
    while success:
        frames.append(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
        success, image = vidcap.read()
        if len(frames) == n_frames:
            break
    return frames


def save_img(file_name, img):
    output_dir = os.path.dirname(file_name)
    os.makedirs(output_dir, exist_ok=True)
    imageio.imwrite(
        file_name,
        (((img[0].permute(1, 2, 0) + 1) / 2).cpu().numpy() * 255.0).astype(np.uint8),
    )


def save_video(file_name, imgs, fps=10):
    output_dir = os.path.dirname(file_name)
    os.makedirs(output_dir, exist_ok=True)
    img_grid = [
        (((img[0].permute(1, 2, 0) + 1) / 2).cpu().numpy() * 255.0).astype(np.uint8)
        for img in imgs
    ]
    if file_name.endswith(".gif"):
        imageio.mimwrite(file_name, img_grid, fps=fps, loop=0)
    else:
        imageio.mimwrite(file_name, img_grid, fps=fps)


def read_video(
    input_path: str,
    n_frames: int,
    device: str = "cuda",
):
    path = Path(input_path)
    is_video_file = False
    all_img_paths = []
    if path.is_file():
        if any([input_path.endswith(x) for x in [".gif", ".mp4"]]):
            is_video_file = True
        else:
            raise ValueError("Path is not a valid video file.")
    elif path.is_dir():
        all_img_paths = sorted(
            [
                f
                for f in path.iterdir()
                if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]
            ]
        )[:n_frames]
    elif "*" in input_path:
        all_img_paths = sorted(glob(input_path))[:n_frames]
    else:
        raise ValueError

    if is_video_file and input_path.endswith(".gif"):
        images = read_gif(input_path, n_frames)[:n_frames]
    elif is_video_file and input_path.endswith(".mp4"):
        images = read_mp4(input_path, n_frames)[:n_frames]
    else:
        print(f"Loading {len(all_img_paths)} video frames...")
        images = [Image.open(img_path) for img_path in all_img_paths]

    if len(images) < n_frames:
        images = (images + images[::-1])[:n_frames]
    if len(images) != n_frames:
        raise ValueError(f"Input video contains fewer than {n_frames} frames.")

    images_v0 = []

    for image in images:
        image = ToTensor()(image).unsqueeze(0).to(device)
        images_v0.append(image * 2.0 - 1.0)
    return images_v0


def preprocess_video(
    input_path,
    remove_bg=False,
    n_frames=21,
    W=576,
    H=576,
    output_folder=None,
    image_frame_ratio=0.917,
    base_count=0,
):
    print(f"preprocess {input_path}")
    if output_folder is None:
        output_folder = os.path.dirname(input_path)
    path = Path(input_path)
    is_video_file = False
    all_img_paths = []
    if path.is_file():
        if any([input_path.endswith(x) for x in [".gif", ".mp4"]]):
            is_video_file = True
        else:
            raise ValueError("Path is not a valid video file.")
    elif path.is_dir():
        all_img_paths = sorted(
            [
                f
                for f in path.iterdir()
                if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]
            ]
        )[:n_frames]
    elif "*" in input_path:
        all_img_paths = sorted(glob(input_path))[:n_frames]
    else:
        raise ValueError

    if is_video_file and input_path.endswith(".gif"):
        images = read_gif(input_path, n_frames)[:n_frames]
    elif is_video_file and input_path.endswith(".mp4"):
        images = read_mp4(input_path, n_frames)[:n_frames]
    else:
        print(f"Loading {len(all_img_paths)} video frames...")
        images = [Image.open(img_path) for img_path in all_img_paths]

    if len(images) != n_frames:
        raise ValueError(
            f"Input video contains {len(images)} frames, fewer than {n_frames} frames."
        )

    # Remove background
    for i, image in enumerate(images):
        if remove_bg:
            if image.mode == "RGBA":
                pass
            else:
                # image.thumbnail([W, H], Image.Resampling.LANCZOS)
                image = remove(image.convert("RGBA"), alpha_matting=True)
            images[i] = image

    # Crop video frames, assume the object is already in the center of the image
    white_thresh = 250
    images_v0 = []
    box_coord = [np.inf, np.inf, 0, 0]
    for image in images:
        image_arr = np.array(image)
        in_w, in_h = image_arr.shape[:2]
        original_center = (in_w // 2, in_h // 2)
        if image.mode == "RGBA":
            ret, mask = cv2.threshold(
                np.array(image.split()[-1]), 0, 255, cv2.THRESH_BINARY
            )
        else:
            # assume the input image has white background
            ret, mask = cv2.threshold(
                (np.array(image).mean(-1) <= white_thresh).astype(np.uint8) * 255,
                0,
                255,
                cv2.THRESH_BINARY,
            )

        x, y, w, h = cv2.boundingRect(mask)
        box_coord[0] = min(box_coord[0], x)
        box_coord[1] = min(box_coord[1], y)
        box_coord[2] = max(box_coord[2], x + w)
        box_coord[3] = max(box_coord[3], y + h)
    box_square = max(
        original_center[0] - box_coord[0], original_center[1] - box_coord[1]
    )
    box_square = max(box_square, box_coord[2] - original_center[0])
    box_square = max(box_square, box_coord[3] - original_center[1])
    x, y = max(0, original_center[0] - box_square), max(
        0, original_center[1] - box_square
    )
    w, h = min(image_arr.shape[0], 2 * box_square), min(
        image_arr.shape[1], 2 * box_square
    )
    box_size = box_square * 2

    for image in images:
        if image.mode == "RGB":
            image = image.convert("RGBA")
        image_arr = np.array(image)
        side_len = (
            int(box_size / image_frame_ratio) if image_frame_ratio is not None else in_w
        )
        padded_image = np.zeros((side_len, side_len, 4), dtype=np.uint8)
        center = side_len // 2
        box_size_w = min(w, box_size)
        box_size_h = min(h, box_size)
        padded_image[
            center - box_size_w // 2 : center - box_size_w // 2 + box_size_w,
            center - box_size_h // 2 : center - box_size_h // 2 + box_size_h,
        ] = image_arr[x : x + w, y : y + h]

        rgba = Image.fromarray(padded_image).resize((W, H), Image.LANCZOS)
        # rgba = image.resize((W, H), Image.LANCZOS)
        rgba_arr = np.array(rgba) / 255.0
        rgb = rgba_arr[..., :3] * rgba_arr[..., -1:] + (1 - rgba_arr[..., -1:])
        image = (rgb * 255).astype(np.uint8)

        images_v0.append(image)

    processed_file = os.path.join(output_folder, f"{base_count:06d}_process_input.mp4")
    imageio.mimwrite(processed_file, images_v0, fps=10)
    return processed_file


def sample_sv3d(
    image,
    num_frames: Optional[int] = None,  # 21 for SV3D
    num_steps: Optional[int] = None,
    version: str = "sv3d_u",
    fps_id: int = 6,
    motion_bucket_id: int = 127,
    cond_aug: float = 0.02,
    decoding_t: int = 14,  # Number of frames decoded at a time! This eats most VRAM. Reduce if necessary.
    device: str = "cuda",
    polar_rad: Optional[Union[float, List[float]]] = None,
    azim_rad: Optional[List[float]] = None,
    verbose: Optional[bool] = False,
    sv3d_model=None,
):
    """
    Simple script to generate a single sample conditioned on an image `input_path` or multiple images, one for each
    image file in folder `input_path`. If you run out of VRAM, try decreasing `decoding_t`.
    """

    if sv3d_model is None:
        if version == "sv3d_u":
            model_config = "scripts/sampling/configs/sv3d_u.yaml"
        elif version == "sv3d_p":
            model_config = "scripts/sampling/configs/sv3d_p.yaml"
        else:
            raise ValueError(f"Version {version} does not exist.")

        model, filter = load_model(
            model_config,
            device,
            num_frames,
            num_steps,
            verbose,
        )
    else:
        model = sv3d_model

    load_module_gpu(model)

    H, W = image.shape[2:]
    F = 8
    C = 4
    shape = (num_frames, C, H // F, W // F)

    value_dict = {}
    value_dict["cond_frames_without_noise"] = image
    value_dict["motion_bucket_id"] = motion_bucket_id
    value_dict["fps_id"] = fps_id
    value_dict["cond_aug"] = cond_aug
    value_dict["cond_frames"] = image + cond_aug * torch.randn_like(image)
    if "sv3d_p" in version:
        value_dict["polars_rad"] = polar_rad
        value_dict["azimuths_rad"] = azim_rad

    with torch.no_grad():
        with torch.autocast(device):
            load_module_gpu(model.conditioner)
            batch, batch_uc = get_batch_sv3d(
                get_unique_embedder_keys_from_conditioner(model.conditioner),
                value_dict,
                [1, num_frames],
                T=num_frames,
                device=device,
            )
            c, uc = model.conditioner.get_unconditional_conditioning(
                batch,
                batch_uc=batch_uc,
                force_uc_zero_embeddings=[
                    "cond_frames",
                    "cond_frames_without_noise",
                ],
            )
            unload_module_gpu(model.conditioner)

            for k in ["crossattn", "concat"]:
                uc[k] = repeat(uc[k], "b ... -> b t ...", t=num_frames)
                uc[k] = rearrange(uc[k], "b t ... -> (b t) ...", t=num_frames)
                c[k] = repeat(c[k], "b ... -> b t ...", t=num_frames)
                c[k] = rearrange(c[k], "b t ... -> (b t) ...", t=num_frames)

            randn = torch.randn(shape, device=device)

            additional_model_inputs = {}
            additional_model_inputs["image_only_indicator"] = torch.zeros(
                2, num_frames
            ).to(device)
            additional_model_inputs["num_video_frames"] = batch["num_video_frames"]

            def denoiser(input, sigma, c):
                return model.denoiser(
                    model.model, input, sigma, c, **additional_model_inputs
                )

            load_module_gpu(model.model)
            load_module_gpu(model.denoiser)
            samples_z = model.sampler(denoiser, randn, cond=c, uc=uc)
            unload_module_gpu(model.denoiser)
            unload_module_gpu(model.model)

            load_module_gpu(model.first_stage_model)
            model.en_and_decode_n_samples_a_time = decoding_t
            samples_x = model.decode_first_stage(samples_z)
            unload_module_gpu(model.first_stage_model)

            samples_x[-1:] = value_dict["cond_frames_without_noise"]
            samples = torch.clamp(samples_x, min=-1.0, max=1.0)

    unload_module_gpu(model)
    return samples


def decode_latents(
    model, samples_z, img_matrix, frame_indices, view_indices, timesteps
):
    load_module_gpu(model.first_stage_model)
    for t in frame_indices:
        for v in view_indices:
            if True:  # t != 0 and v != 0:
                if isinstance(model.first_stage_model.decoder, VideoDecoder):
                    samples_x = model.decode_first_stage(
                        samples_z[t, v][None], timesteps=timesteps
                    )
                else:
                    samples_x = model.decode_first_stage(samples_z[t, v][None])
                samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)
                img_matrix[t][v] = samples * 2 - 1
    unload_module_gpu(model.first_stage_model)
    return img_matrix


def init_embedder_options_no_st(keys, init_dict, prompt=None, negative_prompt=None):
    # Hardcoded demo settings; might undergo some changes in the future

    value_dict = {}
    for key in keys:
        if key == "txt":
            if prompt is None:
                prompt = "A professional photograph of an astronaut riding a pig"
            if negative_prompt is None:
                negative_prompt = ""

            value_dict["prompt"] = prompt
            value_dict["negative_prompt"] = negative_prompt

        if key == "original_size_as_tuple":
            orig_width = init_dict["orig_width"]
            orig_height = init_dict["orig_height"]

            value_dict["orig_width"] = orig_width
            value_dict["orig_height"] = orig_height

        if key == "crop_coords_top_left":
            crop_coord_top = 0
            crop_coord_left = 0

            value_dict["crop_coords_top"] = crop_coord_top
            value_dict["crop_coords_left"] = crop_coord_left

        if key == "aesthetic_score":
            value_dict["aesthetic_score"] = 6.0
            value_dict["negative_aesthetic_score"] = 2.5

        if key == "target_size_as_tuple":
            value_dict["target_width"] = init_dict["target_width"]
            value_dict["target_height"] = init_dict["target_height"]

        if key in ["fps_id", "fps"]:
            fps = 6

            value_dict["fps"] = fps
            value_dict["fps_id"] = fps - 1

        if key == "motion_bucket_id":
            mb_id = 127
            value_dict["motion_bucket_id"] = mb_id

        if key == "noise_level":
            value_dict["noise_level"] = 0

    return value_dict


def get_discretization_no_st(discretization, options, key=1):
    if discretization == "LegacyDDPMDiscretization":
        discretization_config = {
            "target": "sgm.modules.diffusionmodules.discretizer.LegacyDDPMDiscretization",
        }
    elif discretization == "EDMDiscretization":
        sigma_min = options.get("sigma_min", 0.03)
        sigma_max = options.get("sigma_max", 14.61)
        rho = options.get("rho", 3.0)
        discretization_config = {
            "target": "sgm.modules.diffusionmodules.discretizer.EDMDiscretization",
            "params": {
                "sigma_min": sigma_min,
                "sigma_max": sigma_max,
                "rho": rho,
            },
        }
    return discretization_config


def get_guider_no_st(options, key):
    guider = [
        "VanillaCFG",
        "IdentityGuider",
        "LinearPredictionGuider",
        "TrianglePredictionGuider",
        "TrapezoidPredictionGuider",
        "SpatiotemporalPredictionGuider",
    ][options.get("guider", 2)]

    additional_guider_kwargs = (
        options["additional_guider_kwargs"]
        if "additional_guider_kwargs" in options
        else {}
    )

    if guider == "IdentityGuider":
        guider_config = {
            "target": "sgm.modules.diffusionmodules.guiders.IdentityGuider"
        }
    elif guider == "VanillaCFG":
        scale_schedule = "Identity"

        if scale_schedule == "Identity":
            scale = options.get("cfg", 5.0)

            scale_schedule_config = {
                "target": "sgm.modules.diffusionmodules.guiders.IdentitySchedule",
                "params": {"scale": scale},
            }

        elif scale_schedule == "Oscillating":
            small_scale = 4.0
            large_scale = 16.0
            sigma_cutoff = 1.0

            scale_schedule_config = {
                "target": "sgm.modules.diffusionmodules.guiders.OscillatingSchedule",
                "params": {
                    "small_scale": small_scale,
                    "large_scale": large_scale,
                    "sigma_cutoff": sigma_cutoff,
                },
            }
        else:
            raise NotImplementedError

        guider_config = {
            "target": "sgm.modules.diffusionmodules.guiders.VanillaCFG",
            "params": {
                "scale_schedule_config": scale_schedule_config,
                **additional_guider_kwargs,
            },
        }
    elif guider == "LinearPredictionGuider":
        max_scale = options.get("cfg", 1.5)

        guider_config = {
            "target": "sgm.modules.diffusionmodules.guiders.LinearPredictionGuider",
            "params": {
                "max_scale": max_scale,
                "num_frames": options["num_frames"],
                **additional_guider_kwargs,
            },
        }
    elif guider == "TrianglePredictionGuider":
        max_scale = options.get("cfg", 1.5)
        period = options.get("period", 1.0)
        period_fusing = options.get("period_fusing", "max")

        guider_config = {
            "target": "sgm.modules.diffusionmodules.guiders.TrianglePredictionGuider",
            "params": {
                "max_scale": max_scale,
                "num_frames": options["num_frames"],
                "period": period,
                "period_fusing": period_fusing,
                **additional_guider_kwargs,
            },
        }
    elif guider == "TrapezoidPredictionGuider":
        max_scale = options.get("cfg", 1.5)
        edge_perc = options.get("edge_perc", 0.1)

        guider_config = {
            "target": "sgm.modules.diffusionmodules.guiders.TrapezoidPredictionGuider",
            "params": {
                "max_scale": max_scale,
                "num_frames": options["num_frames"],
                "edge_perc": edge_perc,
                **additional_guider_kwargs,
            },
        }
    elif guider == "SpatiotemporalPredictionGuider":
        max_scale = options.get("cfg", 1.5)
        min_scale = options.get("min_cfg", 1.0)

        guider_config = {
            "target": "sgm.modules.diffusionmodules.guiders.SpatiotemporalPredictionGuider",
            "params": {
                "max_scale": max_scale,
                "min_scale": min_scale,
                "num_frames": options["num_frames"],
                "num_views": options["num_views"],
                **additional_guider_kwargs,
            },
        }
    else:
        raise NotImplementedError
    return guider_config


def get_sampler_no_st(sampler_name, steps, discretization_config, guider_config, key=1):
    if sampler_name == "EulerEDMSampler" or sampler_name == "HeunEDMSampler":
        s_churn = 0.0
        s_tmin = 0.0
        s_tmax = 999.0
        s_noise = 1.0

        if sampler_name == "EulerEDMSampler":
            sampler = EulerEDMSampler(
                num_steps=steps,
                discretization_config=discretization_config,
                guider_config=guider_config,
                s_churn=s_churn,
                s_tmin=s_tmin,
                s_tmax=s_tmax,
                s_noise=s_noise,
                verbose=False,
            )
        elif sampler_name == "HeunEDMSampler":
            sampler = HeunEDMSampler(
                num_steps=steps,
                discretization_config=discretization_config,
                guider_config=guider_config,
                s_churn=s_churn,
                s_tmin=s_tmin,
                s_tmax=s_tmax,
                s_noise=s_noise,
                verbose=False,
            )
    elif (
        sampler_name == "EulerAncestralSampler"
        or sampler_name == "DPMPP2SAncestralSampler"
    ):
        s_noise = 1.0
        eta = 1.0

        if sampler_name == "EulerAncestralSampler":
            sampler = EulerAncestralSampler(
                num_steps=steps,
                discretization_config=discretization_config,
                guider_config=guider_config,
                eta=eta,
                s_noise=s_noise,
                verbose=False,
            )
        elif sampler_name == "DPMPP2SAncestralSampler":
            sampler = DPMPP2SAncestralSampler(
                num_steps=steps,
                discretization_config=discretization_config,
                guider_config=guider_config,
                eta=eta,
                s_noise=s_noise,
                verbose=False,
            )
    elif sampler_name == "DPMPP2MSampler":
        sampler = DPMPP2MSampler(
            num_steps=steps,
            discretization_config=discretization_config,
            guider_config=guider_config,
            verbose=False,
        )
    elif sampler_name == "LinearMultistepSampler":
        order = 4
        sampler = LinearMultistepSampler(
            num_steps=steps,
            discretization_config=discretization_config,
            guider_config=guider_config,
            order=order,
            verbose=False,
        )
    else:
        raise ValueError(f"unknown sampler {sampler_name}!")

    return sampler


def init_sampling_no_st(
    key=1,
    options: Optional[Dict[str, int]] = None,
):
    options = {} if options is None else options

    num_rows, num_cols = 1, 1
    steps = options.get("num_steps", 50)
    sampler = [
        "EulerEDMSampler",
        "HeunEDMSampler",
        "EulerAncestralSampler",
        "DPMPP2SAncestralSampler",
        "DPMPP2MSampler",
        "LinearMultistepSampler",
    ][options.get("sampler", 0)]
    discretization = [
        "LegacyDDPMDiscretization",
        "EDMDiscretization",
    ][options.get("discretization", 1)]

    discretization_config = get_discretization_no_st(
        discretization, options=options, key=key
    )

    guider_config = get_guider_no_st(options=options, key=key)

    sampler = get_sampler_no_st(
        sampler, steps, discretization_config, guider_config, key=key
    )
    return sampler, num_rows, num_cols


def run_img2vid(
    version_dict,
    model,
    image,
    seed=23,
    polar_rad=[10] * 21,
    azim_rad=np.linspace(0, 360, 21 + 1)[1:],
    cond_motion=None,
    cond_view=None,
    decoding_t=None,
    cond_mv=True,
):
    options = version_dict["options"]
    H = version_dict["H"]
    W = version_dict["W"]
    T = version_dict["T"]
    C = version_dict["C"]
    F = version_dict["f"]
    init_dict = {
        "orig_width": 576,
        "orig_height": 576,
        "target_width": W,
        "target_height": H,
    }
    ukeys = set(get_unique_embedder_keys_from_conditioner(model.conditioner))

    value_dict = init_embedder_options_no_st(
        ukeys,
        init_dict,
        negative_prompt=options.get("negative_promt", ""),
        prompt="A 3D model.",
    )
    if "fps" not in ukeys:
        value_dict["fps"] = 6

    value_dict["is_image"] = 0
    value_dict["is_webvid"] = 0
    if cond_mv:
        value_dict["image_only_indicator"] = 1.0
    else:
        value_dict["image_only_indicator"] = 0.0

    cond_aug = 0.00
    if cond_motion is not None:
        value_dict["cond_frames_without_noise"] = cond_motion
        value_dict["cond_frames"] = (
            cond_motion[:, None].repeat(1, cond_view.shape[0], 1, 1, 1).flatten(0, 1)
        )
    else:
        value_dict["cond_frames_without_noise"] = image
        value_dict["cond_frames"] = image + cond_aug * torch.randn_like(image)
    value_dict["cond_aug"] = cond_aug
    value_dict["polar_rad"] = polar_rad
    value_dict["azimuth_rad"] = azim_rad
    value_dict["rotated"] = False
    value_dict["cond_motion"] = cond_motion
    value_dict["cond_view"] = cond_view

    # seed_everything(seed)

    options["num_frames"] = T
    sampler, num_rows, num_cols = init_sampling_no_st(options=options)
    num_samples = num_rows * num_cols

    samples = do_sample(
        model,
        sampler,
        value_dict,
        num_samples,
        H,
        W,
        C,
        F,
        T=T,
        batch2model_input=["num_video_frames", "image_only_indicator"],
        force_uc_zero_embeddings=options.get("force_uc_zero_embeddings", None),
        force_cond_zero_embeddings=options.get("force_cond_zero_embeddings", None),
        return_latents=False,
        decoding_t=decoding_t,
    )

    return samples


def prepare_inputs_forward_backward(
    img_matrix,
    view_indices,
    frame_indices,
    v0,
    t0,
    t1,
    model,
    version_dict,
    seed,
    polars,
    azims,
):
    # forward sampling
    forward_frame_indices = frame_indices.copy()
    image = img_matrix[t0][v0]
    cond_motion = torch.cat([img_matrix[t][v0] for t in forward_frame_indices], 0)
    cond_view = torch.cat([img_matrix[t0][v] for v in view_indices], 0)
    forward_inputs = prepare_sampling(
        version_dict,
        model,
        image,
        seed,
        polars,
        azims,
        cond_motion,
        cond_view,
    )

    # backward sampling
    backward_frame_indices = frame_indices[::-1].copy()
    image = img_matrix[t1][v0]
    cond_motion = torch.cat([img_matrix[t][v0] for t in backward_frame_indices], 0)
    cond_view = torch.cat([img_matrix[t1][v] for v in view_indices], 0)
    backward_inputs = prepare_sampling(
        version_dict,
        model,
        image,
        seed,
        polars,
        azims,
        cond_motion,
        cond_view,
    )
    return (
        forward_inputs,
        forward_frame_indices,
        backward_inputs,
        backward_frame_indices,
    )


def prepare_inputs(
    frame_indices,
    img_matrix,
    v0,
    view_indices,
    model,
    version_dict,
    seed,
    polars,
    azims,
):
    load_module_gpu(model.conditioner)
    # forward sampling
    forward_frame_indices = frame_indices.copy()
    t0 = forward_frame_indices[0]
    image = img_matrix[t0][v0]
    cond_motion = torch.cat([img_matrix[t][v0] for t in forward_frame_indices], 0)
    cond_view = torch.cat([img_matrix[t0][v] for v in view_indices], 0)
    forward_inputs = prepare_sampling(
        version_dict,
        model,
        image,
        seed,
        polars,
        azims,
        cond_motion,
        cond_view,
    )

    # backward sampling
    backward_frame_indices = frame_indices[::-1].copy()
    t0 = backward_frame_indices[0]
    image = img_matrix[t0][v0]
    cond_motion = torch.cat([img_matrix[t][v0] for t in backward_frame_indices], 0)
    cond_view = torch.cat([img_matrix[t0][v] for v in view_indices], 0)
    backward_inputs = prepare_sampling(
        version_dict,
        model,
        image,
        seed,
        polars,
        azims,
        cond_motion,
        cond_view,
    )

    unload_module_gpu(model.conditioner)
    return (
        forward_inputs,
        forward_frame_indices,
        backward_inputs,
        backward_frame_indices,
    )


def do_sample(
    model,
    sampler,
    value_dict,
    num_samples,
    H,
    W,
    C,
    F,
    force_uc_zero_embeddings: Optional[List] = None,
    force_cond_zero_embeddings: Optional[List] = None,
    batch2model_input: List = None,
    return_latents=False,
    filter=None,
    T=None,
    additional_batch_uc_fields=None,
    decoding_t=None,
):
    force_uc_zero_embeddings = default(force_uc_zero_embeddings, [])
    batch2model_input = default(batch2model_input, [])
    additional_batch_uc_fields = default(additional_batch_uc_fields, [])

    precision_scope = autocast
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                if T is not None:
                    num_samples = [num_samples, T]
                else:
                    num_samples = [num_samples]

                load_module_gpu(model.conditioner)
                batch, batch_uc = get_batch(
                    get_unique_embedder_keys_from_conditioner(model.conditioner),
                    value_dict,
                    num_samples,
                    T=T,
                    additional_batch_uc_fields=additional_batch_uc_fields,
                )
                c, uc = model.conditioner.get_unconditional_conditioning(
                    batch,
                    batch_uc=batch_uc,
                    force_uc_zero_embeddings=force_uc_zero_embeddings,
                    force_cond_zero_embeddings=force_cond_zero_embeddings,
                )
                unload_module_gpu(model.conditioner)

                for k in c:
                    if not k == "crossattn":
                        c[k], uc[k] = map(
                            lambda y: y[k][: math.prod(num_samples)].to("cuda"), (c, uc)
                        )

                if value_dict["image_only_indicator"] == 0:
                    c["cond_view"] *= 0
                    uc["cond_view"] *= 0

                additional_model_inputs = {}
                for k in batch2model_input:
                    if k == "image_only_indicator":
                        assert T is not None

                        if isinstance(
                            sampler.guider,
                            (
                                VanillaCFG,
                                LinearPredictionGuider,
                                TrianglePredictionGuider,
                                TrapezoidPredictionGuider,
                                SpatiotemporalPredictionGuider,
                            ),
                        ):
                            additional_model_inputs[k] = (
                                torch.zeros(num_samples[0] * 2, num_samples[1]).to(
                                    "cuda"
                                )
                                + value_dict["image_only_indicator"]
                            )
                        else:
                            additional_model_inputs[k] = torch.zeros(num_samples).to(
                                "cuda"
                            )
                    else:
                        additional_model_inputs[k] = batch[k]

                shape = (math.prod(num_samples), C, H // F, W // F)
                randn = torch.randn(shape).to("cuda")

                def denoiser(input, sigma, c):
                    return model.denoiser(
                        model.model, input, sigma, c, **additional_model_inputs
                    )

                load_module_gpu(model.model)
                load_module_gpu(model.denoiser)
                samples_z = sampler(denoiser, randn, cond=c, uc=uc)
                unload_module_gpu(model.denoiser)
                unload_module_gpu(model.model)

                load_module_gpu(model.first_stage_model)
                if isinstance(model.first_stage_model.decoder, VideoDecoder):
                    samples_x = model.decode_first_stage(
                        samples_z, timesteps=default(decoding_t, T)
                    )
                else:
                    samples_x = model.decode_first_stage(samples_z)
                samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)
                unload_module_gpu(model.first_stage_model)

                if filter is not None:
                    samples = filter(samples)

                if return_latents:
                    return samples, samples_z

                return samples


def prepare_sampling_(
    model,
    sampler,
    value_dict,
    num_samples,
    force_uc_zero_embeddings: Optional[List] = None,
    force_cond_zero_embeddings: Optional[List] = None,
    batch2model_input: List = None,
    T=None,
    additional_batch_uc_fields=None,
):
    force_uc_zero_embeddings = default(force_uc_zero_embeddings, [])
    batch2model_input = default(batch2model_input, [])
    additional_batch_uc_fields = default(additional_batch_uc_fields, [])

    precision_scope = autocast
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                if T is not None:
                    num_samples = [num_samples, T]
                else:
                    num_samples = [num_samples]
                load_module_gpu(model.conditioner)
                batch, batch_uc = get_batch(
                    get_unique_embedder_keys_from_conditioner(model.conditioner),
                    value_dict,
                    num_samples,
                    T=T,
                    additional_batch_uc_fields=additional_batch_uc_fields,
                )
                c, uc = model.conditioner.get_unconditional_conditioning(
                    batch,
                    batch_uc=batch_uc,
                    force_uc_zero_embeddings=force_uc_zero_embeddings,
                    force_cond_zero_embeddings=force_cond_zero_embeddings,
                )
                unload_module_gpu(model.conditioner)

                for k in c:
                    if not k == "crossattn":
                        c[k], uc[k] = map(
                            lambda y: y[k][: math.prod(num_samples)].to("cuda"), (c, uc)
                        )

                additional_model_inputs = {}
                for k in batch2model_input:
                    if k == "image_only_indicator":
                        assert T is not None

                        if isinstance(
                            sampler.guider,
                            (
                                VanillaCFG,
                                LinearPredictionGuider,
                                TrianglePredictionGuider,
                                TrapezoidPredictionGuider,
                                SpatiotemporalPredictionGuider,
                            ),
                        ):
                            additional_model_inputs[k] = (
                                torch.zeros(num_samples[0] * 2, num_samples[1]).to(
                                    "cuda"
                                )
                                + value_dict["image_only_indicator"]
                            )
                        else:
                            additional_model_inputs[k] = torch.zeros(num_samples).to(
                                "cuda"
                            )
                    else:
                        additional_model_inputs[k] = batch[k]

    return c, uc, additional_model_inputs


def do_sample_per_step(
    model, sampler, noisy_latents, c, uc, step, additional_model_inputs
):
    precision_scope = autocast
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                noisy_latents_scaled, s_in, sigmas, num_sigmas, _, _ = (
                    sampler.prepare_sampling_loop(
                        noisy_latents.clone(), c, uc, sampler.num_steps
                    )
                )

                if step == 0:
                    latents = noisy_latents_scaled
                else:
                    latents = noisy_latents

                def denoiser(input, sigma, c):
                    return model.denoiser(
                        model.model, input, sigma, c, **additional_model_inputs
                    )

                gamma = (
                    min(sampler.s_churn / (num_sigmas - 1), 2**0.5 - 1)
                    if sampler.s_tmin <= sigmas[step] <= sampler.s_tmax
                    else 0.0
                )

                load_module_gpu(model.model)
                load_module_gpu(model.denoiser)
                samples_z = sampler.sampler_step(
                    s_in * sigmas[step],
                    s_in * sigmas[step + 1],
                    denoiser,
                    latents,
                    c,
                    uc,
                    gamma,
                )
                unload_module_gpu(model.denoiser)
                unload_module_gpu(model.model)
    return samples_z


def prepare_sampling(
    version_dict,
    model,
    image,
    seed=23,
    polar_rad=[10] * 21,
    azim_rad=np.linspace(0, 360, 21 + 1)[1:],
    cond_motion=None,
    cond_view=None,
):
    options = version_dict["options"]
    H = version_dict["H"]
    W = version_dict["W"]
    T = version_dict["T"]
    C = version_dict["C"]
    F = version_dict["f"]
    init_dict = {
        "orig_width": 576,
        "orig_height": 576,
        "target_width": W,
        "target_height": H,
    }
    ukeys = set(get_unique_embedder_keys_from_conditioner(model.conditioner))

    value_dict = init_embedder_options_no_st(
        ukeys,
        init_dict,
        negative_prompt=options.get("negative_promt", ""),
        prompt="A 3D model.",
    )
    if "fps" not in ukeys:
        value_dict["fps"] = 6

    value_dict["is_image"] = 0
    value_dict["is_webvid"] = 0
    value_dict["image_only_indicator"] = 1.0

    cond_aug = 0.00
    if cond_motion is not None:
        value_dict["cond_frames_without_noise"] = cond_motion
        value_dict["cond_frames"] = (
            cond_motion[:, None].repeat(1, cond_view.shape[0], 1, 1, 1).flatten(0, 1)
        )
    else:
        value_dict["cond_frames_without_noise"] = image
        value_dict["cond_frames"] = image + cond_aug * torch.randn_like(image)
    value_dict["cond_aug"] = cond_aug
    value_dict["polar_rad"] = polar_rad
    value_dict["azimuth_rad"] = azim_rad
    value_dict["rotated"] = False
    value_dict["cond_motion"] = cond_motion
    value_dict["cond_view"] = cond_view

    options["num_frames"] = T
    sampler, num_rows, num_cols = init_sampling_no_st(options=options)
    num_samples = num_rows * num_cols

    c, uc, additional_model_inputs = prepare_sampling_(
        model,
        sampler,
        value_dict,
        num_samples,
        force_uc_zero_embeddings=options.get("force_uc_zero_embeddings", None),
        force_cond_zero_embeddings=options.get("force_cond_zero_embeddings", None),
        batch2model_input=["num_video_frames", "image_only_indicator"],
        T=T,
    )

    return c, uc, additional_model_inputs, sampler


def get_unique_embedder_keys_from_conditioner(conditioner):
    return list(set([x.input_key for x in conditioner.embedders]))


def get_batch_sv3d(keys, value_dict, N, T, device):
    batch = {}
    batch_uc = {}

    for key in keys:
        if key == "fps_id":
            batch[key] = (
                torch.tensor([value_dict["fps_id"]])
                .to(device)
                .repeat(int(math.prod(N)))
            )
        elif key == "motion_bucket_id":
            batch[key] = (
                torch.tensor([value_dict["motion_bucket_id"]])
                .to(device)
                .repeat(int(math.prod(N)))
            )
        elif key == "cond_aug":
            batch[key] = repeat(
                torch.tensor([value_dict["cond_aug"]]).to(device),
                "1 -> b",
                b=math.prod(N),
            )
        elif key == "cond_frames" or key == "cond_frames_without_noise":
            batch[key] = repeat(value_dict[key], "1 ... -> b ...", b=N[0])
        elif key == "polars_rad" or key == "azimuths_rad":
            batch[key] = torch.tensor(value_dict[key]).to(device).repeat(N[0])
        else:
            batch[key] = value_dict[key]

    if T is not None:
        batch["num_video_frames"] = T

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc


def get_batch(
    keys,
    value_dict: dict,
    N: Union[List, ListConfig],
    device: str = "cuda",
    T: int = None,
    additional_batch_uc_fields: List[str] = [],
):
    batch = {}
    batch_uc = {}

    for key in keys:
        if key == "txt":
            batch["txt"] = [value_dict["prompt"]] * math.prod(N)
            batch_uc["txt"] = [value_dict["negative_prompt"]] * math.prod(N)

        elif key == "original_size_as_tuple":
            batch["original_size_as_tuple"] = (
                torch.tensor([value_dict["orig_height"], value_dict["orig_width"]])
                .to(device)
                .repeat(math.prod(N), 1)
            )
        elif key == "crop_coords_top_left":
            batch["crop_coords_top_left"] = (
                torch.tensor(
                    [value_dict["crop_coords_top"], value_dict["crop_coords_left"]]
                )
                .to(device)
                .repeat(math.prod(N), 1)
            )
        elif key == "aesthetic_score":
            batch["aesthetic_score"] = (
                torch.tensor([value_dict["aesthetic_score"]])
                .to(device)
                .repeat(math.prod(N), 1)
            )
            batch_uc["aesthetic_score"] = (
                torch.tensor([value_dict["negative_aesthetic_score"]])
                .to(device)
                .repeat(math.prod(N), 1)
            )

        elif key == "target_size_as_tuple":
            batch["target_size_as_tuple"] = (
                torch.tensor([value_dict["target_height"], value_dict["target_width"]])
                .to(device)
                .repeat(math.prod(N), 1)
            )
        elif key == "fps":
            batch[key] = (
                torch.tensor([value_dict["fps"]]).to(device).repeat(math.prod(N))
            )
        elif key == "fps_id":
            batch[key] = (
                torch.tensor([value_dict["fps_id"]]).to(device).repeat(math.prod(N))
            )
        elif key == "motion_bucket_id":
            batch[key] = (
                torch.tensor([value_dict["motion_bucket_id"]])
                .to(device)
                .repeat(math.prod(N))
            )
        elif key == "pool_image":
            batch[key] = repeat(value_dict[key], "1 ... -> b ...", b=math.prod(N)).to(
                device, dtype=torch.half
            )
        elif key == "is_image":
            batch[key] = (
                torch.tensor([value_dict["is_image"]])
                .to(device)
                .repeat(math.prod(N))
                .long()
            )
        elif key == "is_webvid":
            batch[key] = (
                torch.tensor([value_dict["is_webvid"]])
                .to(device)
                .repeat(math.prod(N))
                .long()
            )
        elif key == "cond_aug":
            batch[key] = repeat(
                torch.tensor([value_dict["cond_aug"]]).to("cuda"),
                "1 -> b",
                b=math.prod(N),
            )
        elif (
            key == "cond_frames"
            or key == "cond_frames_without_noise"
            or key == "back_frames"
        ):
            # batch[key] = repeat(value_dict[key], "1 ... -> b ...", b=N[0])
            batch[key] = value_dict[key]

        elif key == "interpolation_context":
            batch[key] = repeat(
                value_dict["interpolation_context"], "b ... -> (b n) ...", n=N[1]
            )

        elif key == "start_frame":
            assert T is not None
            batch[key] = repeat(value_dict[key], "b ... -> (b t) ...", t=T)

        elif key == "polar_rad" or key == "azimuth_rad":
            batch[key] = (
                torch.tensor(value_dict[key]).to(device).repeat(math.prod(N) // T)
            )

        elif key == "rotated":
            batch[key] = (
                torch.tensor([value_dict["rotated"]]).to(device).repeat(math.prod(N))
            )

        else:
            batch[key] = value_dict[key]

    if T is not None:
        batch["num_video_frames"] = T

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
        elif key in additional_batch_uc_fields and key not in batch_uc:
            batch_uc[key] = copy.copy(batch[key])
    return batch, batch_uc


def load_model(
    config: str,
    device: str,
    num_frames: int,
    num_steps: int,
    verbose: bool = False,
    ckpt_path: str = None,
):
    config = OmegaConf.load(config)
    if device == "cuda":
        config.model.params.conditioner_config.params.emb_models[
            0
        ].params.open_clip_embedding_config.params.init_device = device

    config.model.params.sampler_config.params.verbose = verbose
    config.model.params.sampler_config.params.num_steps = num_steps
    config.model.params.sampler_config.params.guider_config.params.num_frames = (
        num_frames
    )
    if ckpt_path is not None:
        config.model.params.ckpt_path = ckpt_path
    if device == "cuda":
        with torch.device(device):
            model = instantiate_from_config(config.model).to(device).eval()
    else:
        model = instantiate_from_config(config.model).to(device).eval()

    filter = DeepFloydDataFiltering(verbose=False, device=device)
    return model, filter
