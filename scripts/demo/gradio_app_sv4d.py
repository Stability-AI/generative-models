# Adding this at the very top of app.py to make 'generative-models' directory discoverable
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "generative-models"))

from glob import glob
from typing import Optional

import gradio as gr
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from typing import List, Optional, Union
import torchvision

from sgm.modules.encoders.modules import VideoPredictionEmbedderWithEncoder
from scripts.demo.sv4d_helpers import (
    decode_latents,
    load_model,
    initial_model_load,
    read_video,
    run_img2vid,
    prepare_inputs,
    do_sample_per_step,
    sample_sv3d,
    save_video,
    preprocess_video,
)


# the tmp path, if /tmp/gradio is not writable, change it to a writable path
# os.environ["GRADIO_TEMP_DIR"] = "gradio_tmp"

version = "sv4d"  # replace with 'sv3d_p' or 'sv3d_u' for other models

# Define the repo, local directory and filename
repo_id = "stabilityai/sv4d"
filename = f"{version}.safetensors"  # replace with "sv3d_u.safetensors" or "sv3d_p.safetensors"
local_dir = "checkpoints"
local_ckpt_path = os.path.join(local_dir, filename)

# Check if the file already exists
if not os.path.exists(local_ckpt_path):
    # If the file doesn't exist, download it
    hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)
    print("File downloaded. (sv4d)")
else:
    print("File already exists. No need to download. (sv4d)")

device = "cuda"
max_64_bit_int = 2**63 - 1

num_frames = 21
num_steps = 20
model_config = f"scripts/sampling/configs/{version}.yaml"

# Set model config
T = 5  # number of frames per sample
V = 8  # number of views per sample
F = 8  # vae factor to downsize image->latent
C = 4
H, W = 576, 576
n_frames = 21  # number of input and output video frames
n_views = V + 1  # number of output video views (1 input view + 8 novel views)
n_views_sv3d = 21
subsampled_views = np.array(
    [0, 2, 5, 7, 9, 12, 14, 16, 19]
)  # subsample (V+1=)9 (uniform) views from 21 SV3D views

version_dict = {
    "T": T * V,
    "H": H,
    "W": W,
    "C": C,
    "f": F,
    "options": {
        "discretization": 1,
        "cfg": 3,
        "sigma_min": 0.002,
        "sigma_max": 700.0,
        "rho": 7.0,
        "guider": 5,
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

# Load SV4D model
model, filter = load_model(
    model_config,
    device,
    version_dict["T"],
    num_steps,
)
model = initial_model_load(model)

# -----------sv3d config and model loading----------------
# if version == "sv3d_u":
sv3d_model_config = "scripts/sampling/configs/sv3d_u.yaml"
# elif version == "sv3d_p":
#     sv3d_model_config = "scripts/sampling/configs/sv3d_p.yaml"
# else:
#     raise ValueError(f"Version {version} does not exist.")

# Define the repo, local directory and filename
repo_id = "stabilityai/sv3d"
filename = f"sv3d_u.safetensors"  # replace with "sv3d_u.safetensors" or "sv3d_p.safetensors"
local_dir = "checkpoints"
local_ckpt_path = os.path.join(local_dir, filename)

# Check if the file already exists
if not os.path.exists(local_ckpt_path):
    # If the file doesn't exist, download it
    hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)
    print("File downloaded. (sv3d)")
else:
    print("File already exists. No need to download. (sv3d)")

# load sv3d model
sv3d_model, filter = load_model(
    sv3d_model_config,
    device,
    21,
    num_steps,
    verbose=False,
)
sv3d_model = initial_model_load(sv3d_model)
# ------------------

def sample_anchor(
    input_path: str = "assets/test_image.png",  # Can either be image file or folder with image files
    seed: Optional[int] = None,
    encoding_t: int = 8,  # Number of frames encoded at a time! This eats most VRAM. Reduce if necessary.
    decoding_t: int = 4,  # Number of frames decoded at a time! This eats most VRAM. Reduce if necessary.
    num_steps: int = 20,
    sv3d_version: str = "sv3d_u",  # sv3d_u or sv3d_p
    fps_id: int = 6,
    motion_bucket_id: int = 127,
    cond_aug: float = 1e-5,
    device: str = "cuda",
    elevations_deg: Optional[Union[float, List[float]]] = 10.0,
    azimuths_deg: Optional[List[float]] = None,
    verbose: Optional[bool] = False,
):
    """
    Simple script to generate multiple novel-view videos conditioned on a video `input_path` or multiple frames, one for each
    image file in folder `input_path`. If you run out of VRAM, try decreasing `decoding_t`.
    """
    output_folder = os.path.dirname(input_path)

    torch.manual_seed(seed)
    os.makedirs(output_folder, exist_ok=True)

    # Read input video frames i.e. images at view 0
    print(f"Reading {input_path}")
    images_v0 = read_video(
        input_path,
        n_frames=n_frames,
        device=device,
    )

    # Get camera viewpoints
    if isinstance(elevations_deg, float) or isinstance(elevations_deg, int):
        elevations_deg = [elevations_deg] * n_views_sv3d
    assert (
        len(elevations_deg) == n_views_sv3d
    ), f"Please provide 1 value, or a list of {n_views_sv3d} values for elevations_deg! Given {len(elevations_deg)}"
    if azimuths_deg is None:
        azimuths_deg = np.linspace(0, 360, n_views_sv3d + 1)[1:] % 360
    assert (
        len(azimuths_deg) == n_views_sv3d
    ), f"Please provide a list of {n_views_sv3d} values for azimuths_deg! Given {len(azimuths_deg)}"
    polars_rad = np.array([np.deg2rad(90 - e) for e in elevations_deg])
    azimuths_rad = np.array(
        [np.deg2rad((a - azimuths_deg[-1]) % 360) for a in azimuths_deg]
    )

    # Sample multi-view images of the first frame using SV3D i.e. images at time 0
    sv3d_model.sampler.num_steps = num_steps
    print("sv3d_model.sampler.num_steps", sv3d_model.sampler.num_steps)
    images_t0 = sample_sv3d(
        images_v0[0],
        n_views_sv3d,
        num_steps,
        sv3d_version,
        fps_id,
        motion_bucket_id,
        cond_aug,
        decoding_t,
        device,
        polars_rad,
        azimuths_rad,
        verbose,
        sv3d_model,
    )
    images_t0 = torch.roll(images_t0, 1, 0)  # move conditioning image to first frame

    sv3d_file = os.path.join(output_folder, "t000.mp4")
    save_video(sv3d_file, images_t0.unsqueeze(1))
    
    for emb in model.conditioner.embedders:
        if isinstance(emb, VideoPredictionEmbedderWithEncoder):
            emb.en_and_decode_n_samples_a_time = encoding_t
    model.en_and_decode_n_samples_a_time = decoding_t
    # Initialize image matrix
    img_matrix = [[None] * n_views for _ in range(n_frames)]
    for i, v in enumerate(subsampled_views):
        img_matrix[0][i] = images_t0[v].unsqueeze(0)
    for t in range(n_frames):
        img_matrix[t][0] = images_v0[t]

    # Interleaved sampling for anchor frames
    t0, v0 = 0, 0
    frame_indices = np.arange(T - 1, n_frames, T - 1)  # [4, 8, 12, 16, 20]
    view_indices = np.arange(V) + 1
    print(f"Sampling anchor frames {frame_indices}")
    image = img_matrix[t0][v0]
    cond_motion = torch.cat([img_matrix[t][v0] for t in frame_indices], 0)
    cond_view = torch.cat([img_matrix[t0][v] for v in view_indices], 0)
    polars = polars_rad[subsampled_views[1:]][None].repeat(T, 0).flatten()
    azims = azimuths_rad[subsampled_views[1:]][None].repeat(T, 0).flatten()
    azims = (azims - azimuths_rad[v0]) % (torch.pi * 2)
    model.sampler.num_steps = num_steps
    version_dict["options"]["num_steps"] = num_steps
    samples = run_img2vid(
        version_dict, model, image, seed, polars, azims, cond_motion, cond_view, decoding_t
    )
    samples = samples.view(T, V, 3, H, W)
    for i, t in enumerate(frame_indices):
        for j, v in enumerate(view_indices):
            if img_matrix[t][v] is None:
                img_matrix[t][v] = samples[i, j][None] * 2 - 1

    # concat video
    grid_list = []
    for t in frame_indices:
        imgs_view = torch.cat(img_matrix[t])
        grid_list.append(torchvision.utils.make_grid(imgs_view, nrow=3).unsqueeze(0))
    # save output videos
    anchor_vis_file = os.path.join(output_folder, "anchor_vis.mp4")
    save_video(anchor_vis_file, grid_list, fps=3)
    anchor_file = os.path.join(output_folder, "anchor.mp4")
    image_list = samples.view(T*V, 3, H, W).unsqueeze(1) * 2 - 1
    save_video(anchor_file, image_list)

    return sv3d_file, anchor_vis_file, anchor_file


def sample_all(
    input_path: str = "inputs/test_video1.mp4",  # Can either be video file or folder with image files
    sv3d_path: str = "outputs/sv4d/000000_t000.mp4",
    anchor_path: str = "outputs/sv4d/000000_anchor.mp4",
    seed: Optional[int] = None,
    num_steps: int = 20,
    device: str = "cuda",
    elevations_deg: Optional[Union[float, List[float]]] = 10.0,
    azimuths_deg: Optional[List[float]] = None,
):
    """
    Simple script to generate multiple novel-view videos conditioned on a video `input_path` or multiple frames, one for each
    image file in folder `input_path`. If you run out of VRAM, try decreasing `decoding_t`.
    """
    output_folder = os.path.dirname(input_path)
    torch.manual_seed(seed)
    os.makedirs(output_folder, exist_ok=True)

    # Read input video frames i.e. images at view 0
    print(f"Reading {input_path}")
    images_v0 = read_video(
        input_path,
        n_frames=n_frames,
        device=device,
    )

    images_t0 = read_video(
        sv3d_path,
        n_frames=n_views_sv3d,
        device=device,
    )

    # Get camera viewpoints
    if isinstance(elevations_deg, float) or isinstance(elevations_deg, int):
        elevations_deg = [elevations_deg] * n_views_sv3d
    assert (
        len(elevations_deg) == n_views_sv3d
    ), f"Please provide 1 value, or a list of {n_views_sv3d} values for elevations_deg! Given {len(elevations_deg)}"
    if azimuths_deg is None:
        azimuths_deg = np.linspace(0, 360, n_views_sv3d + 1)[1:] % 360
    assert (
        len(azimuths_deg) == n_views_sv3d
    ), f"Please provide a list of {n_views_sv3d} values for azimuths_deg! Given {len(azimuths_deg)}"
    polars_rad = np.array([np.deg2rad(90 - e) for e in elevations_deg])
    azimuths_rad = np.array(
        [np.deg2rad((a - azimuths_deg[-1]) % 360) for a in azimuths_deg]
    )

    # Initialize image matrix
    img_matrix = [[None] * n_views for _ in range(n_frames)]
    for i, v in enumerate(subsampled_views):
        img_matrix[0][i] = images_t0[v]
    for t in range(n_frames):
        img_matrix[t][0] = images_v0[t]

    # load interleaved sampling for anchor frames
    t0, v0 = 0, 0
    frame_indices = np.arange(T - 1, n_frames, T - 1)  # [4, 8, 12, 16, 20]
    view_indices = np.arange(V) + 1

    anchor_frames = read_video(
        anchor_path,
        n_frames=T * V,
        device=device,
    )
    anchor_frames = torch.cat(anchor_frames).view(T, V, 3, H, W)
    for i, t in enumerate(frame_indices):
        for j, v in enumerate(view_indices):
            if img_matrix[t][v] is None:
                img_matrix[t][v] = anchor_frames[i, j][None]

    # Dense sampling for the rest
    print(f"Sampling dense frames:")
    for t0 in np.arange(0, n_frames - 1, T - 1):  # [0, 4, 8, 12, 16]
        frame_indices = t0 + np.arange(T)
        print(f"Sampling dense frames {frame_indices}")
        latent_matrix = torch.randn(n_frames, n_views, C, H // F, W // F).to("cuda")

        polars = polars_rad[subsampled_views[1:]][None].repeat(T, 0).flatten()
        azims = azimuths_rad[subsampled_views[1:]][None].repeat(T, 0).flatten()
        azims = (azims - azimuths_rad[v0]) % (torch.pi * 2)
        
        # alternate between forward and backward conditioning
        forward_inputs, forward_frame_indices, backward_inputs, backward_frame_indices = prepare_inputs(
            frame_indices, 
            img_matrix, 
            v0, 
            view_indices, 
            model, 
            version_dict, 
            seed, 
            polars, 
            azims
        )
        
        for step in range(num_steps):
            if step % 2 == 1:
                c, uc, additional_model_inputs, sampler = forward_inputs
                frame_indices = forward_frame_indices
            else:
                c, uc, additional_model_inputs, sampler = backward_inputs
                frame_indices = backward_frame_indices
            noisy_latents = latent_matrix[frame_indices][:, view_indices].flatten(0, 1)
                
            samples = do_sample_per_step(
                model,
                sampler,
                noisy_latents,
                c,
                uc,
                step,
                additional_model_inputs,
            )
            samples = samples.view(T, V, C, H // F, W // F)
            for i, t in enumerate(frame_indices):
                for j, v in enumerate(view_indices):
                    latent_matrix[t, v] = samples[i, j]

        img_matrix = decode_latents(model, latent_matrix, img_matrix, frame_indices, view_indices, T)


    # concat video
    grid_list = []
    for t in range(n_frames):
        imgs_view = torch.cat(img_matrix[t])
        grid_list.append(torchvision.utils.make_grid(imgs_view, nrow=3).unsqueeze(0))
    # save output videos
    vid_file = os.path.join(output_folder, "sv4d_final.mp4")
    save_video(vid_file, grid_list)

    return vid_file, seed


with gr.Blocks() as demo:
    gr.Markdown(
        """# Demo for SV4D from Stability AI ([model](https://huggingface.co/stabilityai/sv4d), [news](https://stability.ai/news/stable-video-4d))
#### Research release ([_non-commercial_](https://huggingface.co/stabilityai/sv4d/blob/main/LICENSE.md)): generate 8 novel view videos from a single-view video (with white background).
#### It takes ~45s to generate anchor frames and another ~160s to generate full results (21 frames).  
#### Hints for improving performance:  
- Use a white background; 
- Make the object in the center of the image; 
- The SV4D process the first 21 frames of the uploaded video. Gradio provides a nice option of trimming the uploaded video if needed.  
  """
    )
    with gr.Row():
        with gr.Column():
            input_video = gr.Video(label="Upload your video")
            generate_btn = gr.Button("Step 1: generate 8 novel view videos (5 anchor frames each)")
            interpolate_btn = gr.Button("Step 2: Extend novel view videos to 21 frames")
        with gr.Column():
            anchor_video = gr.Video(label="SV4D outputs (anchor frames)")
            sv3d_video = gr.Video(label="SV3D outputs", interactive=False)
        with gr.Column():
            sv4d_interpolated_video = gr.Video(label="SV4D outputs (21 frames)")

    with gr.Accordion("Advanced options", open=False):
        seed = gr.Slider(
            label="Seed",
            value=23,
            # randomize=True,
            minimum=0,
            maximum=100,
            step=1,
        )
        encoding_t = gr.Slider(
            label="Encode n frames at a time",
            info="Number of frames encoded at a time! This eats most VRAM. Reduce if necessary.",
            value=8,
            minimum=1,
            maximum=40,
        )
        decoding_t = gr.Slider(
            label="Decode n frames at a time",
            info="Number of frames decoded at a time! This eats most VRAM. Reduce if necessary.",
            value=4,
            minimum=1,
            maximum=14,
        )
        denoising_steps = gr.Slider(
            label="Number of denoising steps",
            info="Increase will improve the performance but needs more time.",
            value=20,
            minimum=10,
            maximum=50,
            step=1,
        )
        remove_bg = gr.Checkbox(
            label="Remove background",
            info="We use rembg. Users can check the alternative way: SAM2 (https://github.com/facebookresearch/segment-anything-2)",
        )

    input_video.upload(fn=preprocess_video, inputs=[input_video, remove_bg], outputs=input_video, queue=False)

    with gr.Row(visible=False):
        anchor_frames = gr.Video()

    generate_btn.click(
        fn=sample_anchor,
        inputs=[input_video, seed, encoding_t, decoding_t, denoising_steps],
        outputs=[sv3d_video, anchor_video, anchor_frames],
        api_name="SV4D output (5 frames)",
    )

    interpolate_btn.click(
        fn=sample_all,
        inputs=[input_video, sv3d_video, anchor_frames, seed, denoising_steps],
        outputs=[sv4d_interpolated_video, seed],
        api_name="SV4D interpolation (21 frames)",
    )

    examples = gr.Examples(
        fn=preprocess_video,
        examples=[
            "./assets/sv4d_videos/test_video1.mp4",
            "./assets/sv4d_videos/test_video2.mp4",
            "./assets/sv4d_videos/green_robot.mp4",
            "./assets/sv4d_videos/dolphin.mp4",
            "./assets/sv4d_videos/lucia_v000.mp4",
            "./assets/sv4d_videos/snowboard_v000.mp4",
            "./assets/sv4d_videos/stroller_v000.mp4",
            "./assets/sv4d_videos/human5.mp4",
            "./assets/sv4d_videos/bunnyman.mp4",
            "./assets/sv4d_videos/hiphop_parrot.mp4",
            "./assets/sv4d_videos/guppie_v0.mp4",
            "./assets/sv4d_videos/wave_hello.mp4",
            "./assets/sv4d_videos/pistol_v0.mp4",
            "./assets/sv4d_videos/human7.mp4",
            "./assets/sv4d_videos/monkey.mp4",
            "./assets/sv4d_videos/train_v0.mp4",
        ],
        inputs=[input_video],
        run_on_click=True,
        outputs=[input_video],
    )

if __name__ == "__main__":
    demo.queue(max_size=20)
    demo.launch(share=True)
 