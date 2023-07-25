"""
This is a very minimal txt2img example for SD-XL only.
"""
import argparse
import logging
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
import einops
import omegaconf
import pytorch_lightning
from sgm.modules.diffusionmodules.sampling import EulerEDMSampler
from sgm.util import load_model_from_config, get_default_device_name


def run_txt2img(
    *,
    model,
    prompt: str,
    steps: int = 10,
    width: int = 1024,
    height: int = 1024,
    cfg_scale=5.0,
    num_samples=1,
    seed: int,
    device: str,
):
    discretization_config = {
        "target": "sgm.modules.diffusionmodules.discretizer.LegacyDDPMDiscretization",
    }
    guider_config = {
        "target": "sgm.modules.diffusionmodules.guiders.VanillaCFG",
        "params": {
            "scale": cfg_scale,
            "dyn_thresh_config": {
                "target": "sgm.modules.diffusionmodules.sampling_utils.NoDynamicThresholding",
            },
        },
    }
    sampler = EulerEDMSampler(
        num_steps=steps,
        discretization_config=discretization_config,
        guider_config=guider_config,
        s_churn=0,
        s_tmin=0,
        s_tmax=999,
        s_noise=1.0,
        verbose=False,
    )
    C = 4  # SD-XL value
    F = 8  # SD-XL value

    with torch.no_grad(), model.ema_scope():
        pytorch_lightning.seed_everything(seed)
        batch = {
            "txt": [prompt] * num_samples,
            "crop_coords_top_left": torch.tensor([0, 0])
            .to(device)
            .repeat(num_samples, 1),
            "original_size_as_tuple": torch.tensor([1024, 1024])
            .to(device)
            .repeat(num_samples, 1),  # SD-XL values
            "target_size_as_tuple": torch.tensor([width, width])
            .to(device)
            .repeat(num_samples, 1),
        }
        c, uc = model.conditioner.get_unconditional_conditioning(
            batch, force_uc_zero_embeddings=["txt"]
        )
        for k in c:
            if k != "crossattn":
                c[k] = c[k][:num_samples].to(device)
                uc[k] = uc[k][:num_samples].to(device)

        shape = (num_samples, C, height // F, width // F)
        initial_latent = torch.randn(shape).to(device)

        def denoiser(input, sigma, c):
            return model.denoiser(model.model, input, sigma, c)

        latent_samples = sampler(denoiser, initial_latent, cond=c, uc=uc)
        decoded_samples = model.decode_first_stage(latent_samples)
        samples = torch.clamp((decoded_samples + 1.0) / 2.0, min=0.0, max=1.0)
        return samples


@torch.no_grad()
def fast_load(*, config, ckpt, device):
    config = omegaconf.OmegaConf.load(config)
    # This patch is borrowed from AUTOMATIC1111's stable-diffusion-webui;
    # we don't need to initialize the weights just for them to be overwritten
    # by the checkpoint.
    with (
        patch.object(torch.nn.init, "kaiming_uniform_"),
        patch.object(torch.nn.init, "_no_grad_normal_"),
        patch.object(torch.nn.init, "_no_grad_uniform_"),
    ):
        model = load_model_from_config(
            config, ckpt=ckpt, device="cpu", freeze=True, verbose=False
        )
    model.to(device)
    model.eval()
    return model


def main():
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")
    # Quiesce some uninformative CLIP and attention logging.
    logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
    logging.getLogger("sgm.modules.attention").setLevel(logging.ERROR)

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=get_default_device_name())
    ap.add_argument(
        "--prompt",
        default="Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--cfg-scale", type=float, default=5.0)
    ap.add_argument("--num-samples", type=int, default=1)
    args = ap.parse_args()
    model = fast_load(
        config="configs/inference/sd_xl_base.yaml",
        ckpt="checkpoints/sd_xl_base_0.9.safetensors",
        device=args.device,
    )

    samples = run_txt2img(
        model=model,
        prompt=args.prompt,
        steps=args.steps,
        width=args.width,
        height=args.height,
        cfg_scale=args.cfg_scale,
        num_samples=args.num_samples,
        device=args.device,
        seed=args.seed,
    )

    out_path = Path("output")
    out_path.mkdir(exist_ok=True)

    prefix = int(time.time())

    for i, sample in enumerate(samples, 1):
        filename = out_path / f"{prefix}-{i:04}.png"
        print(f"Saving {i}/{len(samples)}: {filename}")
        sample = 255.0 * einops.rearrange(sample, "c h w -> h w c")
        Image.fromarray(sample.cpu().numpy().astype(np.uint8)).save(filename)


if __name__ == "__main__":
    main()
