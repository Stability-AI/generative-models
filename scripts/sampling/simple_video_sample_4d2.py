import os
import sys
from glob import glob
from typing import List, Optional, Union

from tqdm import tqdm

sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), "../../")))
import numpy as np
import torch
from fire import Fire

from sgm.modules.encoders.modules import VideoPredictionEmbedderWithEncoder
from scripts.demo.sv4d_helpers import (
    decode_latents,
    load_model,
    initial_model_load,
    read_video,
    run_img2vid,
    prepare_sampling,
    do_sample_per_step,
    save_video,
    preprocess_video,
)


def sample(
    input_path: str = "assets/sv4d_videos/camel.gif",  # Can either be image file or folder with image files
    output_folder: Optional[str] = "outputs/sv4d2",
    num_steps: Optional[int] = 50,
    img_size: int = 576, # image resolution
    n_frames: int = 21, # number of input and output video frames
    fps_id: int = 6,
    motion_bucket_id: int = 127,
    cond_aug: float = 1e-5,
    seed: int = 23,
    encoding_t: int = 8,  # Number of frames encoded at a time! This eats most VRAM. Reduce if necessary.
    decoding_t: int = 4,  # Number of frames decoded at a time! This eats most VRAM. Reduce if necessary.
    device: str = "cuda",
    elevations_deg: Optional[List[float]] = 0.0,
    azimuths_deg: Optional[List[float]] = None,
    image_frame_ratio: Optional[float] = 0.9,
    verbose: Optional[bool] = False,
    remove_bg: bool = False,
):
    """
    Simple script to generate multiple novel-view videos conditioned on a video `input_path` or multiple frames, one for each
    image file in folder `input_path`. If you run out of VRAM, try decreasing `decoding_t` and `encoding_t`.
    """
    # Set model config
    T = 12  # number of frames per sample
    V = 4  # number of views per sample
    F = 8  # vae factor to downsize image->latent
    C = 4
    H, W = img_size, img_size
    n_views = V + 1  # number of output video views (1 input view + 8 novel views)
    subsampled_views = np.arange(n_views)

    model_config = "scripts/sampling/configs/sv4d2.yaml"
    version_dict = {
        "T": T * V,
        "H": H,
        "W": W,
        "C": C,
        "f": F,
        "options": {
            "discretization": 1,
            "cfg": 2.0,
            "min_cfg": 2.0,
            "num_views": V,
            "sigma_min": 0.002,
            "sigma_max": 700.0,
            "rho": 7.0,
            "guider": 2,
            "num_steps": num_steps,
            "force_uc_zero_embeddings": [
                "cond_frames",
                "cond_frames_without_noise",
                "cond_view",
                "cond_motion",
            ],
            "additional_guider_kwargs": {
                "additional_cond_keys": ["cond_view", "cond_motion"]
            },
        },
    }

    torch.manual_seed(seed)
    os.makedirs(output_folder, exist_ok=True)

    # Read input video frames i.e. images at view 0
    print(f"Reading {input_path}")
    base_count = len(glob(os.path.join(output_folder, "*.mp4"))) // n_views
    processed_input_path = preprocess_video(
        input_path,
        remove_bg=remove_bg,
        n_frames=n_frames,
        W=W,
        H=H,
        output_folder=output_folder,
        image_frame_ratio=image_frame_ratio,
        base_count=base_count,
    )
    images_v0 = read_video(processed_input_path, n_frames=n_frames, device=device)
    images_t0 = torch.zeros(n_views, 3, H, W).float().to(device)

    # Get camera viewpoints
    if isinstance(elevations_deg, float) or isinstance(elevations_deg, int):
        elevations_deg = [elevations_deg] * n_views
    assert (
        len(elevations_deg) == n_views
    ), f"Please provide 1 value, or a list of {n_views} values for elevations_deg! Given {len(elevations_deg)}"
    if azimuths_deg is None:
        # azimuths_deg = np.linspace(0, 360, n_views + 1)[1:] % 360
        azimuths_deg = np.array([0, 60, 120, 180, 240])
    assert (
        len(azimuths_deg) == n_views
    ), f"Please provide a list of {n_views} values for azimuths_deg! Given {len(azimuths_deg)}"
    polars_rad = np.array([np.deg2rad(90 - e) for e in elevations_deg])
    azimuths_rad = np.array(
        [np.deg2rad((a - azimuths_deg[-1]) % 360) for a in azimuths_deg]
    )

    # Initialize image matrix
    img_matrix = [[None] * n_views for _ in range(n_frames)]
    for i, v in enumerate(subsampled_views):
        img_matrix[0][i] = images_t0[v].unsqueeze(0)
    for t in range(n_frames):
        img_matrix[t][0] = images_v0[t]

    # Load SV4D++ model
    model, _ = load_model(
        model_config,
        device,
        version_dict["T"],
        num_steps,
        verbose,
    )
    model.en_and_decode_n_samples_a_time = decoding_t
    for emb in model.conditioner.embedders:
        if isinstance(emb, VideoPredictionEmbedderWithEncoder):
            emb.en_and_decode_n_samples_a_time = encoding_t

    # Sampling novel-view videos
    v0 = 0
    view_indices = np.arange(V) + 1
    for t0 in tqdm(range(0, n_frames, T)):
        if t0 + T > n_frames:
            t0 = n_frames - T
        frame_indices = t0 + np.arange(T)
        print(f"Sampling frames {frame_indices}")
        image = img_matrix[t0][v0]
        cond_motion = torch.cat([img_matrix[t][v0] for t in frame_indices], 0)
        cond_view = torch.cat([img_matrix[t0][v] for v in view_indices], 0)
        polars = polars_rad[subsampled_views[1:]][None].repeat(T, 0).flatten()
        azims = azimuths_rad[subsampled_views[1:]][None].repeat(T, 0).flatten()
        polars = (polars - polars_rad[v0] + torch.pi/2) % (torch.pi * 2)
        azims = (azims - azimuths_rad[v0]) % (torch.pi * 2)
        cond_mv = False if t0 == 0 else True
        samples = run_img2vid(
            version_dict, model, image, seed, polars, azims, cond_motion, cond_view, decoding_t, cond_mv=cond_mv
        )
        samples = samples.view(T, V, 3, H, W)

        for i, t in enumerate(frame_indices):
            for j, v in enumerate(view_indices):
                img_matrix[t][v] = samples[i, j][None] * 2 - 1

    # Save output videos
    for v in view_indices:
        vid_file = os.path.join(output_folder, f"{base_count:06d}_v{v:03d}.mp4")
        print(f"Saving {vid_file}")
        save_video(vid_file, [img_matrix[t][v] for t in range(n_frames) if img_matrix[t][v] is not None])



if __name__ == "__main__":
    Fire(sample)
