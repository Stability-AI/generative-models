import os
import streamlit as st
import torch
from PIL import Image
from einops import rearrange, repeat
from omegaconf import OmegaConf
from torchvision import transforms


from sgm.modules.diffusionmodules.sampling import (
    DPMPP2MSampler,
    DPMPP2SAncestralSampler,
    EulerAncestralSampler,
    EulerEDMSampler,
    HeunEDMSampler,
    LinearMultistepSampler,
)
from sgm.inference.helpers import Img2ImgDiscretizationWrapper, embed_watermark
from sgm.util import load_model_from_config


@st.cache_resource()
def init_st(version_dict, load_ckpt=True):
    state = dict()
    if not "model" in state:
        config = version_dict["config"]
        ckpt = version_dict["ckpt"]

        config = OmegaConf.load(config)
        model = load_model_from_config(config, ckpt if load_ckpt else None)
        model = model.to("cuda")
        model.conditioner.half()
        model.model.half()

        state["model"] = model
        state["ckpt"] = ckpt if load_ckpt else None
        state["config"] = config
    return state


def init_embedder_options(keys, init_dict, prompt=None, negative_prompt=None):
    # Hardcoded demo settings; might undergo some changes in the future

    value_dict = {}
    for key in keys:
        if key == "txt":
            if prompt is None:
                prompt = st.text_input(
                    "Prompt", "A professional photograph of an astronaut riding a pig"
                )
            if negative_prompt is None:
                negative_prompt = st.text_input("Negative prompt", "")

            value_dict["prompt"] = prompt
            value_dict["negative_prompt"] = negative_prompt

        if key == "original_size_as_tuple":
            orig_width = st.number_input(
                "orig_width",
                value=init_dict["orig_width"],
                min_value=16,
            )
            orig_height = st.number_input(
                "orig_height",
                value=init_dict["orig_height"],
                min_value=16,
            )

            value_dict["orig_width"] = orig_width
            value_dict["orig_height"] = orig_height

        if key == "crop_coords_top_left":
            crop_coord_top = st.number_input("crop_coords_top", value=0, min_value=0)
            crop_coord_left = st.number_input("crop_coords_left", value=0, min_value=0)

            value_dict["crop_coords_top"] = crop_coord_top
            value_dict["crop_coords_left"] = crop_coord_left

        if key == "aesthetic_score":
            value_dict["aesthetic_score"] = 6.0
            value_dict["negative_aesthetic_score"] = 2.5

        if key == "target_size_as_tuple":
            target_width = st.number_input(
                "target_width",
                value=init_dict["target_width"],
                min_value=16,
            )
            target_height = st.number_input(
                "target_height",
                value=init_dict["target_height"],
                min_value=16,
            )

            value_dict["target_width"] = target_width
            value_dict["target_height"] = target_height

    return value_dict


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


def get_guider(key):
    guider = st.sidebar.selectbox(
        f"Discretization #{key}",
        [
            "VanillaCFG",
            "IdentityGuider",
        ],
    )

    if guider == "IdentityGuider":
        guider_config = {
            "target": "sgm.modules.diffusionmodules.guiders.IdentityGuider"
        }
    elif guider == "VanillaCFG":
        scale = st.number_input(
            f"cfg-scale #{key}", value=5.0, min_value=0.0, max_value=100.0
        )

        thresholder = st.sidebar.selectbox(
            f"Thresholder #{key}",
            [
                "None",
            ],
        )

        if thresholder == "None":
            dyn_thresh_config = {
                "target": "sgm.modules.diffusionmodules.sampling_utils.NoDynamicThresholding"
            }
        else:
            raise NotImplementedError

        guider_config = {
            "target": "sgm.modules.diffusionmodules.guiders.VanillaCFG",
            "params": {"scale": scale, "dyn_thresh_config": dyn_thresh_config},
        }
    else:
        raise NotImplementedError
    return guider_config


def init_sampling(
    key=1, img2img_strength=1.0, use_identity_guider=False, get_num_samples=True
):
    if get_num_samples:
        num_rows = 1
        num_cols = st.number_input(
            f"num cols #{key}", value=2, min_value=1, max_value=10
        )

    steps = st.sidebar.number_input(
        f"steps #{key}", value=50, min_value=1, max_value=1000
    )
    sampler = st.sidebar.selectbox(
        f"Sampler #{key}",
        [
            "EulerEDMSampler",
            "HeunEDMSampler",
            "EulerAncestralSampler",
            "DPMPP2SAncestralSampler",
            "DPMPP2MSampler",
            "LinearMultistepSampler",
        ],
        0,
    )
    discretization = st.sidebar.selectbox(
        f"Discretization #{key}",
        [
            "LegacyDDPMDiscretization",
            "EDMDiscretization",
        ],
    )

    discretization_config = get_discretization(discretization, key=key)

    guider_config = get_guider(key=key)

    sampler = get_sampler(sampler, steps, discretization_config, guider_config, key=key)
    if img2img_strength < 1.0:
        st.warning(
            f"Wrapping {sampler.__class__.__name__} with Img2ImgDiscretizationWrapper"
        )
        sampler.discretization = Img2ImgDiscretizationWrapper(
            sampler.discretization, strength=img2img_strength
        )
    if get_num_samples:
        return num_rows, num_cols, sampler
    return sampler


def get_discretization(discretization, key=1):
    if discretization == "LegacyDDPMDiscretization":
        discretization_config = {
            "target": "sgm.modules.diffusionmodules.discretizer.LegacyDDPMDiscretization",
        }
    elif discretization == "EDMDiscretization":
        sigma_min = st.number_input(f"sigma_min #{key}", value=0.03)  # 0.0292
        sigma_max = st.number_input(f"sigma_max #{key}", value=14.61)  # 14.6146
        rho = st.number_input(f"rho #{key}", value=3.0)
        discretization_config = {
            "target": "sgm.modules.diffusionmodules.discretizer.EDMDiscretization",
            "params": {
                "sigma_min": sigma_min,
                "sigma_max": sigma_max,
                "rho": rho,
            },
        }

    return discretization_config


def get_sampler(sampler_name, steps, discretization_config, guider_config, key=1):
    if sampler_name == "EulerEDMSampler" or sampler_name == "HeunEDMSampler":
        s_churn = st.sidebar.number_input(f"s_churn #{key}", value=0.0, min_value=0.0)
        s_tmin = st.sidebar.number_input(f"s_tmin #{key}", value=0.0, min_value=0.0)
        s_tmax = st.sidebar.number_input(f"s_tmax #{key}", value=999.0, min_value=0.0)
        s_noise = st.sidebar.number_input(f"s_noise #{key}", value=1.0, min_value=0.0)

        if sampler_name == "EulerEDMSampler":
            sampler = EulerEDMSampler(
                num_steps=steps,
                discretization_config=discretization_config,
                guider_config=guider_config,
                s_churn=s_churn,
                s_tmin=s_tmin,
                s_tmax=s_tmax,
                s_noise=s_noise,
                verbose=True,
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
                verbose=True,
            )
    elif (
        sampler_name == "EulerAncestralSampler"
        or sampler_name == "DPMPP2SAncestralSampler"
    ):
        s_noise = st.sidebar.number_input("s_noise", value=1.0, min_value=0.0)
        eta = st.sidebar.number_input("eta", value=1.0, min_value=0.0)

        if sampler_name == "EulerAncestralSampler":
            sampler = EulerAncestralSampler(
                num_steps=steps,
                discretization_config=discretization_config,
                guider_config=guider_config,
                eta=eta,
                s_noise=s_noise,
                verbose=True,
            )
        elif sampler_name == "DPMPP2SAncestralSampler":
            sampler = DPMPP2SAncestralSampler(
                num_steps=steps,
                discretization_config=discretization_config,
                guider_config=guider_config,
                eta=eta,
                s_noise=s_noise,
                verbose=True,
            )
    elif sampler_name == "DPMPP2MSampler":
        sampler = DPMPP2MSampler(
            num_steps=steps,
            discretization_config=discretization_config,
            guider_config=guider_config,
            verbose=True,
        )
    elif sampler_name == "LinearMultistepSampler":
        order = st.sidebar.number_input("order", value=4, min_value=1)
        sampler = LinearMultistepSampler(
            num_steps=steps,
            discretization_config=discretization_config,
            guider_config=guider_config,
            order=order,
            verbose=True,
        )
    else:
        raise ValueError(f"unknown sampler {sampler_name}!")

    return sampler


def get_interactive_image(key=None) -> Image.Image:
    image = st.file_uploader("Input", type=["jpg", "JPEG", "png"], key=key)
    if image is not None:
        image = Image.open(image)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        return image


def load_img(display=True, key=None):
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
