import os

import numpy as np
import streamlit as st
import torch
from einops import repeat
from pytorch_lightning import seed_everything

from sgm.inference.api import (
    SamplingSpec,
    SamplingParams,
    ModelArchitecture,
    SamplingPipeline,
    model_specs,
)
from sgm.inference.helpers import (
    get_unique_embedder_keys_from_conditioner,
    perform_save_locally,
)
from scripts.demo.streamlit_helpers import (
    get_interactive_image,
    init_embedder_options,
    init_sampling,
    init_save_locally,
    init_st,
    show_samples,
)

SAVE_PATH = "outputs/demo/txt2img/"

SD_XL_BASE_RATIOS = {
    "0.5": (704, 1408),
    "0.52": (704, 1344),
    "0.57": (768, 1344),
    "0.6": (768, 1280),
    "0.68": (832, 1216),
    "0.72": (832, 1152),
    "0.78": (896, 1152),
    "0.82": (896, 1088),
    "0.88": (960, 1088),
    "0.94": (960, 1024),
    "1.0": (1024, 1024),
    "1.07": (1024, 960),
    "1.13": (1088, 960),
    "1.21": (1088, 896),
    "1.29": (1152, 896),
    "1.38": (1152, 832),
    "1.46": (1216, 832),
    "1.67": (1280, 768),
    "1.75": (1344, 768),
    "1.91": (1344, 704),
    "2.0": (1408, 704),
    "2.09": (1472, 704),
    "2.4": (1536, 640),
    "2.5": (1600, 640),
    "2.89": (1664, 576),
    "3.0": (1728, 576),
}


def load_img(display=True, key=None, device="cuda"):
    image = get_interactive_image(key=key)
    if image is None:
        return None
    if display:
        st.image(image)
    w, h = image.size
    print(f"loaded input image of size ({w}, {h})")
    width, height = map(
        lambda x: x - x % 64, (w, h)
    )  # resize to integer multiple of 64
    image = image.resize((width, height))
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0
    return image.to(device)


def run_txt2img(
    state,
    model_id: ModelArchitecture,
    prompt: str,
    negative_prompt: str,
    return_latents=False,
    stage2strength=None,
):
    model: SamplingPipeline = state["model"]
    params: SamplingParams = state["params"]
    if model_id in sdxl_base_model_list:
        width, height = st.selectbox(
            "Resolution:", list(SD_XL_BASE_RATIOS.values()), 10
        )
    else:
        height = int(
            st.number_input("H", value=params.height, min_value=64, max_value=2048)
        )
        width = int(
            st.number_input("W", value=params.width, min_value=64, max_value=2048)
        )

    params = init_embedder_options(
        get_unique_embedder_keys_from_conditioner(model.model.conditioner),
        params=params,
        prompt=prompt,
        negative_prompt=negative_prompt,
    )
    params, num_rows, num_cols = init_sampling(params=params)
    num_samples = num_rows * num_cols
    params.height = height
    params.width = width

    if st.button("Sample"):
        st.write(f"**Model I:** {version}")
        outputs = st.empty()
        st.text("Sampling")
        out = model.text_to_image(
            params=params,
            prompt=prompt,
            negative_prompt=negative_prompt,
            samples=int(num_samples),
            return_latents=return_latents,
            noise_strength=stage2strength,
            filter=state["filter"],
        )

        show_samples(out, outputs)

        return out


def run_img2img(
    state,
    prompt: str,
    negative_prompt: str,
    return_latents=False,
    stage2strength=None,
):
    model: SamplingPipeline = state["model"]
    params: SamplingParams = state["params"]

    img = load_img()
    if img is None:
        return None
    params.height, params.width = img.shape[2], img.shape[3]

    params = init_embedder_options(
        get_unique_embedder_keys_from_conditioner(model.model.conditioner),
        params=params,
        prompt=prompt,
        negative_prompt=negative_prompt,
    )
    params.img2img_strength = st.number_input(
        "**Img2Img Strength**", value=0.75, min_value=0.0, max_value=1.0
    )
    params, num_rows, num_cols = init_sampling(params=params)
    num_samples = num_rows * num_cols

    if st.button("Sample"):
        outputs = st.empty()
        st.text("Sampling")
        out = model.image_to_image(
            image=repeat(img, "1 ... -> n ...", n=num_samples),
            params=params,
            prompt=prompt,
            negative_prompt=negative_prompt,
            samples=int(num_samples),
            return_latents=return_latents,
            noise_strength=stage2strength,
            filter=state["filter"],
        )

        show_samples(out, outputs)
        return out


def apply_refiner(
    input,
    state,
    num_samples: int,
    prompt: str,
    negative_prompt: str,
    finish_denoising=False,
):
    model: SamplingPipeline = state["model"]
    params: SamplingParams = state["params"]

    params.orig_width = input.shape[3] * 8
    params.orig_height = input.shape[2] * 8
    params.width = input.shape[3] * 8
    params.height = input.shape[2] * 8

    st.warning(f"refiner input shape: {input.shape}")

    samples = model.refiner(
        image=input,
        params=params,
        prompt=prompt,
        negative_prompt=negative_prompt,
        samples=num_samples,
        return_latents=False,
        filter=state["filter"],
        add_noise=not finish_denoising,
    )

    return samples


sdxl_base_model_list = [
    ModelArchitecture.SDXL_V1_0_BASE,
    ModelArchitecture.SDXL_V0_9_BASE,
]

sdxl_refiner_model_list = [
    ModelArchitecture.SDXL_V1_0_REFINER,
    ModelArchitecture.SDXL_V0_9_REFINER,
]

if __name__ == "__main__":
    st.title("Stable Diffusion")
    version = st.selectbox(
        "Model Version",
        [member.value for member in ModelArchitecture],
        0,
    )
    version_enum = ModelArchitecture(version)
    specs = model_specs[version_enum]
    mode = st.radio("Mode", ("txt2img", "img2img"), 0)
    st.write("__________________________")

    st.write("**Performance Options:**")
    use_fp16 = st.checkbox("Use fp16 (Saves VRAM)", True)
    enable_swap = st.checkbox("Swap models to CPU (Saves VRAM, uses RAM)", True)
    st.write("__________________________")

    if version_enum in sdxl_base_model_list:
        add_pipeline = st.checkbox("Load SDXL-refiner?", False)
        st.write("__________________________")
    else:
        add_pipeline = False

    seed = int(
        st.sidebar.number_input("seed", value=42, min_value=0, max_value=int(1e9))
    )
    seed_everything(seed)

    save_locally, save_path = init_save_locally(os.path.join(SAVE_PATH, str(version)))
    state = init_st(
        model_specs[version_enum],
        load_filter=True,
        use_fp16=use_fp16,
        enable_swap=enable_swap,
    )
    model = state["model"]

    is_legacy = specs.is_legacy

    prompt = st.text_input(
        "prompt",
        "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
    )
    if is_legacy:
        negative_prompt = st.text_input("negative prompt", "")
    else:
        negative_prompt = ""  # which is unused

    stage2strength = None
    finish_denoising = False

    if add_pipeline:
        st.write("__________________________")
        version2 = ModelArchitecture(
            st.selectbox(
                "Refiner:",
                [member.value for member in sdxl_refiner_model_list],
            )
        )
        st.warning(
            f"Running with {version2} as the second stage model. Make sure to provide (V)RAM :) "
        )
        st.write("**Refiner Options:**")

        specs2 = model_specs[version2]
        state2 = init_st(
            specs2, load_filter=False, use_fp16=use_fp16, enable_swap=enable_swap
        )
        params2 = state2["params"]

        params2.img2img_strength = st.number_input(
            "**Refinement strength**", value=0.15, min_value=0.0, max_value=1.0
        )

        params2, *_ = init_sampling(
            params=state2["params"],
            key=2,
            specify_num_samples=False,
        )
        st.write("__________________________")
        finish_denoising = st.checkbox("Finish denoising with refiner.", True)
        if finish_denoising:
            stage2strength = params2.img2img_strength
        else:
            stage2strength = None
    else:
        state2 = None
        params2 = None
        stage2strength = None

    if mode == "txt2img":
        out = run_txt2img(
            state=state,
            model_id=version_enum,
            prompt=prompt,
            negative_prompt=negative_prompt,
            return_latents=add_pipeline,
            stage2strength=stage2strength,
        )
    elif mode == "img2img":
        out = run_img2img(
            state=state,
            prompt=prompt,
            negative_prompt=negative_prompt,
            return_latents=add_pipeline,
            stage2strength=stage2strength,
        )
    else:
        raise ValueError(f"unknown mode {mode}")
    if isinstance(out, (tuple, list)):
        samples, samples_z = out
    else:
        samples = out
        samples_z = None

    if add_pipeline and samples_z is not None:
        outputs = st.empty()
        st.write("**Running Refinement Stage**")
        samples = apply_refiner(
            input=samples_z,
            state=state2,
            num_samples=samples_z.shape[0],
            prompt=prompt,
            negative_prompt=negative_prompt if is_legacy else "",
            finish_denoising=finish_denoising,
        )
        show_samples(samples, outputs)

    if save_locally and samples is not None:
        perform_save_locally(save_path, samples)
