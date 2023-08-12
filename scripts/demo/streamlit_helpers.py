import os

import numpy as np
import streamlit as st
import torch
from einops import rearrange, repeat
from PIL import Image
from torchvision import transforms
from typing import Optional, Tuple, Dict, Any


from scripts.util.detection.nsfw_and_watermark_dectection import DeepFloydDataFiltering

from sgm.inference.api import (
    Discretization,
    Guider,
    Sampler,
    SamplingParams,
    SamplingSpec,
    SamplingPipeline,
    Thresholder,
)
from sgm.inference.helpers import embed_watermark, CudaModelManager


@st.cache_resource()
def init_st(
    spec: SamplingSpec,
    load_ckpt=True,
    load_filter=True,
    use_fp16=True,
    enable_swap=True,
) -> Dict[str, Any]:
    state: Dict[str, Any] = dict()
    config = spec.config
    ckpt = spec.ckpt

    if enable_swap:
        pipeline = SamplingPipeline(
            model_spec=spec,
            use_fp16=use_fp16,
            device=CudaModelManager(device="cuda", swap_device="cpu"),
        )
    else:
        pipeline = SamplingPipeline(model_spec=spec, use_fp16=use_fp16)

    state["spec"] = spec
    state["model"] = pipeline
    state["ckpt"] = ckpt if load_ckpt else None
    state["config"] = config
    state["params"] = spec.default_params
    if load_filter:
        state["filter"] = DeepFloydDataFiltering(verbose=False)
    else:
        state["filter"] = None
    return state


def get_unique_embedder_keys_from_conditioner(conditioner):
    return list(set([x.input_key for x in conditioner.embedders]))


def init_embedder_options(
    keys, params: SamplingParams, prompt=None, negative_prompt=None
) -> SamplingParams:
    for key in keys:
        if key == "txt":
            if prompt is None:
                prompt = st.text_input(
                    "Prompt", "A professional photograph of an astronaut riding a pig"
                )
            if negative_prompt is None:
                negative_prompt = st.text_input("Negative prompt", "")

        if key == "original_size_as_tuple":
            orig_width = st.number_input(
                "orig_width",
                value=params.orig_width,
                min_value=16,
            )
            orig_height = st.number_input(
                "orig_height",
                value=params.orig_height,
                min_value=16,
            )

            params.orig_width = int(orig_width)
            params.orig_height = int(orig_height)

        if key == "crop_coords_top_left":
            crop_coord_top = st.number_input(
                "crop_coords_top", value=params.crop_coords_top, min_value=0
            )
            crop_coord_left = st.number_input(
                "crop_coords_left", value=params.crop_coords_left, min_value=0
            )

            params.crop_coords_top = int(crop_coord_top)
            params.crop_coords_left = int(crop_coord_left)
    return params


def perform_save_locally(save_path, samples):
    os.makedirs(os.path.join(save_path), exist_ok=True)
    base_count = len(os.listdir(os.path.join(save_path)))
    samples = embed_watermark(samples)
    for sample in samples:
        sample = 255.0 * rearrange(sample.cpu().numpy(), "c h w -> h w c")
        Image.fromarray(sample.astype(np.uint8)).save(
            os.path.join(save_path, f"{base_count:09}.png")
        )
        base_count += 1


def init_save_locally(_dir, init_value: bool = False):
    save_locally = st.sidebar.checkbox("Save images locally", value=init_value)
    if save_locally:
        save_path = st.text_input("Save path", value=os.path.join(_dir, "samples"))
    else:
        save_path = None

    return save_locally, save_path


def show_samples(samples, outputs):
    if isinstance(samples, tuple):
        samples, _ = samples
    grid = embed_watermark(torch.stack([samples]))
    grid = rearrange(grid, "n b c h w -> (n h) (b w) c")
    outputs.image(grid.cpu().numpy())


def get_guider(params: SamplingParams, key=1) -> SamplingParams:
    params.guider = Guider(
        st.sidebar.selectbox(
            f"Discretization #{key}", [member.value for member in Guider]
        )
    )

    if params.guider == Guider.VANILLA:
        scale = st.number_input(
            f"cfg-scale #{key}", value=params.scale, min_value=0.0, max_value=100.0
        )
        params.scale = scale
        thresholder = st.sidebar.selectbox(
            f"Thresholder #{key}",
            [
                "None",
            ],
        )

        if thresholder == "None":
            params.thresholder = Thresholder.NONE
        else:
            raise NotImplementedError
    return params


def init_sampling(
    params: SamplingParams,
    key=1,
    specify_num_samples=True,
) -> Tuple[SamplingParams, int, int]:
    num_rows, num_cols = 1, 1
    if specify_num_samples:
        num_cols = st.number_input(
            f"num cols #{key}", value=2, min_value=1, max_value=10
        )

    params.steps = int(
        st.sidebar.number_input(
            f"steps #{key}", value=params.steps, min_value=1, max_value=1000
        )
    )

    params.sampler = Sampler(
        st.sidebar.selectbox(
            f"Sampler #{key}",
            [member.value for member in Sampler],
            0,
        )
    )
    params.discretization = Discretization(
        st.sidebar.selectbox(
            f"Discretization #{key}",
            [member.value for member in Discretization],
        )
    )

    params = get_discretization(params=params, key=key)
    params = get_guider(params=params, key=key)
    params = get_sampler(params=params, key=key)

    return params, num_rows, num_cols


def get_discretization(params: SamplingParams, key=1) -> SamplingParams:
    if params.discretization == Discretization.EDM:
        params.sigma_min = st.number_input(f"sigma_min #{key}", value=params.sigma_min)
        params.sigma_max = st.number_input(f"sigma_max #{key}", value=params.sigma_max)
        params.rho = st.number_input(f"rho #{key}", value=params.rho)
    return params


def get_sampler(params: SamplingParams, key=1) -> SamplingParams:
    if params.sampler in (Sampler.EULER_EDM, Sampler.HEUN_EDM):
        params.s_churn = st.sidebar.number_input(
            f"s_churn #{key}", value=params.s_churn, min_value=0.0
        )
        params.s_tmin = st.sidebar.number_input(
            f"s_tmin #{key}", value=params.s_tmin, min_value=0.0
        )
        params.s_tmax = st.sidebar.number_input(
            f"s_tmax #{key}", value=params.s_tmax, min_value=0.0
        )
        params.s_noise = st.sidebar.number_input(
            f"s_noise #{key}", value=params.s_noise, min_value=0.0
        )

    elif params.sampler in (Sampler.EULER_ANCESTRAL, Sampler.DPMPP2S_ANCESTRAL):
        params.s_noise = st.sidebar.number_input(
            "s_noise", value=params.s_noise, min_value=0.0
        )
        params.eta = st.sidebar.number_input("eta", value=params.eta, min_value=0.0)

    elif params.sampler == Sampler.LINEAR_MULTISTEP:
        params.order = int(
            st.sidebar.number_input("order", value=params.order, min_value=1)
        )
    return params


def get_interactive_image(key=None) -> Optional[Image.Image]:
    image = st.file_uploader("Input", type=["jpg", "JPEG", "png"], key=key)
    if image is not None:
        image = Image.open(image)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        return image
    return None


def load_img(display=True, key=None) -> Optional[torch.Tensor]:
    image = get_interactive_image(key=key)
    if image is None:
        return None
    if display:
        st.image(image)
    w, h = image.size
    print(f"loaded input image of size ({w}, {h})")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2.0 - 1.0),
        ]
    )
    img = transform(image)[None, ...]
    st.text(f"input min/max/mean: {img.min():.3f}/{img.max():.3f}/{img.mean():.3f}")
    return img


def get_init_img(batch_size=1, key=None):
    init_image = load_img(key=key).cuda()
    init_image = repeat(init_image, "1 ... -> b ...", b=batch_size)
    return init_image
