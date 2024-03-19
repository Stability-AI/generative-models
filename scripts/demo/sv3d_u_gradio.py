# Adding this at the very top of app.py to make 'generative-models' directory discoverable
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import random
from glob import glob
from pathlib import Path
from typing import Optional

import cv2
import gradio as gr
import imageio
import numpy as np
import torch
from einops import rearrange, repeat
from huggingface_hub import hf_hub_download
from PIL import Image
from rembg import remove
from scripts.sampling.simple_video_sample import (
    get_batch,
    get_unique_embedder_keys_from_conditioner,
    load_model,
)
from sgm.inference.helpers import embed_watermark
from torchvision.transforms import ToTensor

version = "sv3d_u"  # replace with 'sv3d_p' or 'sv3d_u' for other models

# Define the repo, local directory and filename
repo_id = "stabilityai/sv3d"
filename = f"{version}.safetensors"  # replace with "sv3d_u.safetensors" or "sv3d_p.safetensors"
local_dir = "checkpoints"
local_ckpt_path = os.path.join(local_dir, filename)

# Check if the file already exists
if not os.path.exists(local_ckpt_path):
    # If the file doesn't exist, download it
    hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)
    print("File downloaded.")
else:
    print("File already exists. No need to download.")

device = "cuda"
max_64_bit_int = 2**63 - 1

num_frames = 21
num_steps = 50
model_config = f"scripts/sampling/configs/{version}.yaml"

model, filter = load_model(
    model_config,
    device,
    num_frames,
    num_steps,
)


def sample(
    input_path: str = "assets/test_image.png",  # Can either be image file or folder with image files
    seed: Optional[int] = None,
    randomize_seed: bool = True,
    decoding_t: int = 7,  # Number of frames decoded at a time! This eats most VRAM. Reduce if necessary.
    device: str = "cuda",
    output_folder: str = None,
    image_frame_ratio: Optional[float] = None,
):
    """
    Simple script to generate a single sample conditioned on an image `input_path` or multiple images, one for each
    image file in folder `input_path`. If you run out of VRAM, try decreasing `decoding_t`.
    """
    if randomize_seed:
        seed = random.randint(0, max_64_bit_int)

    torch.manual_seed(seed)

    path = Path(input_path)
    all_img_paths = []
    if path.is_file():
        if any([input_path.endswith(x) for x in ["jpg", "jpeg", "png"]]):
            all_img_paths = [input_path]
        else:
            raise ValueError("Path is not valid image file.")
    elif path.is_dir():
        all_img_paths = sorted(
            [
                f
                for f in path.iterdir()
                if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]
            ]
        )
        if len(all_img_paths) == 0:
            raise ValueError("Folder does not contain any images.")
    else:
        raise ValueError

    for input_img_path in all_img_paths:

        image = Image.open(input_img_path)
        if image.mode == "RGBA":
            pass
        else:
            # remove bg
            image.thumbnail([768, 768], Image.Resampling.LANCZOS)
            image = remove(image.convert("RGBA"), alpha_matting=True)

        # resize object in frame
        image_arr = np.array(image)
        in_w, in_h = image_arr.shape[:2]
        ret, mask = cv2.threshold(
            np.array(image.split()[-1]), 0, 255, cv2.THRESH_BINARY
        )
        x, y, w, h = cv2.boundingRect(mask)
        max_size = max(w, h)
        side_len = (
            int(max_size / image_frame_ratio) if image_frame_ratio is not None else in_w
        )
        padded_image = np.zeros((side_len, side_len, 4), dtype=np.uint8)
        center = side_len // 2
        padded_image[
            center - h // 2 : center - h // 2 + h,
            center - w // 2 : center - w // 2 + w,
        ] = image_arr[y : y + h, x : x + w]
        # resize frame to 576x576
        rgba = Image.fromarray(padded_image).resize((576, 576), Image.LANCZOS)
        # white bg
        rgba_arr = np.array(rgba) / 255.0
        rgb = rgba_arr[..., :3] * rgba_arr[..., -1:] + (1 - rgba_arr[..., -1:])
        input_image = Image.fromarray((rgb * 255).astype(np.uint8))

        image = ToTensor()(input_image)
        image = image * 2.0 - 1.0

        image = image.unsqueeze(0).to(device)
        H, W = image.shape[2:]
        assert image.shape[1] == 3
        F = 8
        C = 4
        shape = (num_frames, C, H // F, W // F)
        if (H, W) != (576, 576) and "sv3d" in version:
            print(
                "WARNING: The conditioning frame you provided is not 576x576. This leads to suboptimal performance as model was only trained on 576x576."
            )

        cond_aug = 1e-5

        value_dict = {}
        value_dict["cond_aug"] = cond_aug
        value_dict["cond_frames_without_noise"] = image
        value_dict["cond_frames"] = image + cond_aug * torch.randn_like(image)
        value_dict["cond_aug"] = cond_aug

        output_folder = output_folder or f"outputs/gradio/{version}"
        cond_aug = 1e-5

        with torch.no_grad():
            with torch.autocast(device):
                batch, batch_uc = get_batch(
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

                samples_z = model.sampler(denoiser, randn, cond=c, uc=uc)
                model.en_and_decode_n_samples_a_time = decoding_t
                samples_x = model.decode_first_stage(samples_z)
                samples_x[-1:] = value_dict["cond_frames_without_noise"]
                samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)

                os.makedirs(output_folder, exist_ok=True)
                base_count = len(glob(os.path.join(output_folder, "*.mp4")))

                imageio.imwrite(
                    os.path.join(output_folder, f"{base_count:06d}.jpg"), input_image
                )

                samples = embed_watermark(samples)
                samples = filter(samples)
                vid = (
                    (rearrange(samples, "t c h w -> t h w c") * 255)
                    .cpu()
                    .numpy()
                    .astype(np.uint8)
                )
                video_path = os.path.join(output_folder, f"{base_count:06d}.mp4")
                imageio.mimwrite(video_path, vid)

        return video_path, seed


def resize_image(image_path, output_size=(576, 576)):
    image = Image.open(image_path)
    # Calculate aspect ratios
    target_aspect = output_size[0] / output_size[1]  # Aspect ratio of the desired size
    image_aspect = image.width / image.height  # Aspect ratio of the original image

    # Resize then crop if the original image is larger
    if image_aspect > target_aspect:
        # Resize the image to match the target height, maintaining aspect ratio
        new_height = output_size[1]
        new_width = int(new_height * image_aspect)
        resized_image = image.resize((new_width, new_height), Image.LANCZOS)
        # Calculate coordinates for cropping
        left = (new_width - output_size[0]) / 2
        top = 0
        right = (new_width + output_size[0]) / 2
        bottom = output_size[1]
    else:
        # Resize the image to match the target width, maintaining aspect ratio
        new_width = output_size[0]
        new_height = int(new_width / image_aspect)
        resized_image = image.resize((new_width, new_height), Image.LANCZOS)
        # Calculate coordinates for cropping
        left = 0
        top = (new_height - output_size[1]) / 2
        right = output_size[0]
        bottom = (new_height + output_size[1]) / 2

    # Crop the image
    cropped_image = resized_image.crop((left, top, right, bottom))

    return cropped_image


with gr.Blocks() as demo:
    gr.Markdown(
        """# Demo for SV3D_u from Stability AI ([model](https://huggingface.co/stabilityai/sv3d), [news](https://stability.ai/news/introducing-stable-video-3d))
#### Research release ([_non-commercial_](https://huggingface.co/stabilityai/sv3d/blob/main/LICENSE)): generate 21 frames orbital video from a single image, at the same elevation.
Generation takes ~40s (for 50 steps) in an A100.
  """
    )
    with gr.Row():
        with gr.Column():
            image = gr.Image(label="Upload your image", type="filepath")
            generate_btn = gr.Button("Generate")
        video = gr.Video()
    with gr.Accordion("Advanced options", open=False):
        seed = gr.Slider(
            label="Seed",
            value=23,
            randomize=True,
            minimum=0,
            maximum=max_64_bit_int,
            step=1,
        )
        randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
        decoding_t = gr.Slider(
            label="Decode n frames at a time",
            info="Number of frames decoded at a time! This eats most VRAM. Reduce if necessary.",
            value=7,
            minimum=1,
            maximum=14,
        )

    image.upload(fn=resize_image, inputs=image, outputs=image, queue=False)
    generate_btn.click(
        fn=sample,
        inputs=[image, seed, randomize_seed, decoding_t],
        outputs=[video, seed],
        api_name="video",
    )

if __name__ == "__main__":
    demo.queue(max_size=20)
    demo.launch(share=True)
