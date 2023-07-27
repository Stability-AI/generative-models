"""
This is a very minimal txt2img example using `sgm.inference.api`.
"""
from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
import einops
import omegaconf
import pytorch_lightning

from sgm import get_configs_path
from sgm.inference.api import (
    model_specs,
    ModelArchitecture,
    SamplingParams,
    SamplingSpec,
    get_sampler_config,
    Discretization,
)
from sgm.inference.helpers import do_sample
from sgm.util import load_model_from_config, get_default_device_name

logger = logging.getLogger("txt2img")


def run_txt2img(
    *,
    model,
    spec: SamplingSpec,
    prompt: str,
    steps: int,
    width: int | None,
    height: int | None,
    scale: float | None,
    num_samples=1,
    seed: int,
    device: str,
):
    params = SamplingParams(
        discretization=Discretization.EDM,
        height=(height or spec.height),
        rho=7,
        steps=steps,
        width=(width or spec.width),
    )
    if scale:
        params.scale = scale

    with torch.no_grad(), model.ema_scope():
        pytorch_lightning.seed_everything(seed)
        sampler = get_sampler_config(params)
        value_dict = {
            **dataclasses.asdict(params),
            "prompt": prompt,
            "negative_prompt": "",
            "target_width": params.width,
            "target_height": params.height,
        }
        logger.info("Starting sampling with %s", params)
        return do_sample(
            model,
            sampler,
            value_dict,
            num_samples,
            params.height,
            params.width,
            spec.channels,
            spec.factor,
            force_uc_zero_embeddings=["txt"] if not spec.is_legacy else [],
            return_latents=False,
            filter=None,
            device=device,
        )


@torch.no_grad()
def fast_load(*, config, ckpt, device):
    config = omegaconf.OmegaConf.load(config)
    logger.info("Loading model")
    # This patch is borrowed from AUTOMATIC1111's stable-diffusion-webui;
    # we don't need to initialize the weights just for them to be overwritten
    # by the checkpoint.
    with (
        patch.object(torch.nn.init, "kaiming_uniform_"),
        patch.object(torch.nn.init, "_no_grad_normal_"),
        patch.object(torch.nn.init, "_no_grad_uniform_"),
    ):
        model = load_model_from_config(
            config,
            ckpt=ckpt,
            device="cpu",
            freeze=True,
            verbose=False,
        )
    logger.info("Moving model to device")
    model.to(device)
    model.eval()
    return model


def main():
    logging.basicConfig(
        level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s"
    )
    # Quiesce some uninformative CLIP and attention logging.
    logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
    logging.getLogger("sgm.modules.attention").setLevel(logging.ERROR)

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--spec",
        default=ModelArchitecture.SDXL_V1_BASE.value,
        choices=[s.value for s in ModelArchitecture],
    )
    ap.add_argument("--device", default=get_default_device_name())
    ap.add_argument(
        "--prompt",
        default="Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--width", type=int)
    ap.add_argument("--height", type=int)
    ap.add_argument("--scale", type=float)
    ap.add_argument("--num-samples", type=int, default=1)
    args = ap.parse_args()
    spec = model_specs[ModelArchitecture(args.spec)]
    logger.info(f"Using model spec: {spec}")
    model = fast_load(
        config=os.path.join(get_configs_path(), "inference", spec.config),
        ckpt=os.path.join("checkpoints", spec.ckpt),
        device=args.device,
    )

    samples = run_txt2img(
        model=model,
        spec=spec,
        prompt=args.prompt,
        steps=args.steps,
        width=args.width,
        height=args.height,
        scale=args.scale,
        num_samples=args.num_samples,
        device=args.device,
        seed=args.seed,
    )

    out_path = Path("outputs")
    out_path.mkdir(exist_ok=True)

    prefix = int(time.time())

    for i, sample in enumerate(samples, 1):
        filename = out_path / f"{prefix}-{i:04}.png"
        print(f"Saving {i}/{len(samples)}: {filename}")
        sample = 255.0 * einops.rearrange(sample, "c h w -> h w c")
        Image.fromarray(sample.cpu().numpy().astype(np.uint8)).save(filename)


if __name__ == "__main__":
    main()
