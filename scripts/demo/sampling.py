from dataclasses import asdict
from pytorch_lightning import seed_everything

from sgm.inference.api import (
    SamplingParams,
    ModelArchitecture,
    Sampler,
    SamplingPipeline,
    model_specs,
)
from sgm.inference.helpers import (
    do_img2img,
    do_sample,
    get_unique_embedder_keys_from_conditioner,
    perform_save_locally,
)
from scripts.demo.streamlit_helpers import *

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

VERSION2SPECS = {
    "SDXL-base-1.0": {
        "H": 1024,
        "W": 1024,
        "C": 4,
        "f": 8,
        "is_legacy": False,
        "config": "configs/inference/sd_xl_base.yaml",
        "ckpt": "checkpoints/sd_xl_base_1.0.safetensors",
    },
    "SDXL-base-0.9": {
        "H": 1024,
        "W": 1024,
        "C": 4,
        "f": 8,
        "is_legacy": False,
        "config": "configs/inference/sd_xl_base.yaml",
        "ckpt": "checkpoints/sd_xl_base_0.9.safetensors",
    },
    "SD-2.1": {
        "H": 512,
        "W": 512,
        "C": 4,
        "f": 8,
        "is_legacy": True,
        "config": "configs/inference/sd_2_1.yaml",
        "ckpt": "checkpoints/v2-1_512-ema-pruned.safetensors",
    },
    "SD-2.1-768": {
        "H": 768,
        "W": 768,
        "C": 4,
        "f": 8,
        "is_legacy": True,
        "config": "configs/inference/sd_2_1_768.yaml",
        "ckpt": "checkpoints/v2-1_768-ema-pruned.safetensors",
    },
    "SDXL-refiner-0.9": {
        "H": 1024,
        "W": 1024,
        "C": 4,
        "f": 8,
        "is_legacy": True,
        "config": "configs/inference/sd_xl_refiner.yaml",
        "ckpt": "checkpoints/sd_xl_refiner_0.9.safetensors",
    },
    "SDXL-refiner-1.0": {
        "H": 1024,
        "W": 1024,
        "C": 4,
        "f": 8,
        "is_legacy": True,
        "config": "configs/inference/sd_xl_refiner.yaml",
        "ckpt": "checkpoints/sd_xl_refiner_1.0.safetensors",
    },
}


def load_img(display=True, key=None, device="cuda"):
    image = get_interactive_image(key=key)
    if image is None:
        return None
    if display:
        st.image(image)
    w, h = image.size
    print(f"loaded input image of size ({w}, {h})")
    width, height = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 64
    image = image.resize((width, height))
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0
    return image.to(device)


def run_txt2img(
    state,
    version: str,
    prompt: str,
    negative_prompt: str,
    return_latents=False,
    stage2strength=None,
):
    spec: SamplingSpec = state.get("spec")
    model: SamplingPipeline = state.get("model")
    params: SamplingParams = state.get("params")
    if version.startswith("stable-diffusion-xl") and version.endswith("-base"):
        params.width, params.height = st.selectbox(
            "Resolution:", list(SD_XL_BASE_RATIOS.values()), 10
        )
    else:
        params.height = int(st.number_input("H", value=spec.height, min_value=64, max_value=2048))
        params.width = int(st.number_input("W", value=spec.width, min_value=64, max_value=2048))

    init_embedder_options(
        get_unique_embedder_keys_from_conditioner(state["model"].conditioner),
        params=params,
        prompt=prompt,
        negative_prompt=negative_prompt,
    )
    num_rows, num_cols = init_sampling(params=params)
    num_samples = num_rows * num_cols

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
            filter=state.get("filter"),
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
    model: SamplingPipeline = state.get("model")
    params: SamplingParams = state.get("params")

    img = load_img()
    if img is None:
        return None
    params.height, params.width = img.shape[2], img.shape[3]

    init_embedder_options(
        get_unique_embedder_keys_from_conditioner(state["model"].conditioner),
        params=params,
        prompt=prompt,
        negative_prompt=negative_prompt,
    )
    params.img2img_strength = st.number_input(
        "**Img2Img Strength**", value=0.75, min_value=0.0, max_value=1.0
    )
    num_rows, num_cols = init_sampling(params=params)
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
            filter=state.get("filter"),
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
    model: SamplingPipeline = state.get("model")
    params: SamplingParams = state.get("params")

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
        filter=state.get("filter"),
        add_noise=not finish_denoising,
    )

    return samples


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

    set_lowvram_mode(st.checkbox("Low vram mode", True))

    if str(version).startswith("stable-diffusion-xl"):
        add_pipeline = st.checkbox("Load SDXL-refiner?", False)
        st.write("__________________________")
    else:
        add_pipeline = False

    seed = int(st.sidebar.number_input("seed", value=42, min_value=0, max_value=int(1e9)))
    seed_everything(seed)

    save_locally, save_path = init_save_locally(os.path.join(SAVE_PATH, str(version)))
    state = init_st(model_specs[version_enum], load_filter=True)
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
                [
                    ModelArchitecture.SDXL_V1_REFINER,
                    ModelArchitecture.SDXL_V0_9_REFINER,
                ],
            )
        )
        st.warning(
            f"Running with {version2} as the second stage model. Make sure to provide (V)RAM :) "
        )
        st.write("**Refiner Options:**")

        specs_2 = model_specs[version2]
        state2 = init_st(specs_2, load_filter=False)

        stage2strength = st.number_input(
            "**Refinement strength**", value=0.15, min_value=0.0, max_value=1.0
        )

        sampler2, *_ = init_sampling(
            key=2,
            params=state2["params"],
            specify_num_samples=False,
        )
        st.write("__________________________")
        finish_denoising = st.checkbox("Finish denoising with refiner.", True)
        if not finish_denoising:
            stage2strength = None
    else:
        state2 = None

    if mode == "txt2img":
        out = run_txt2img(
            state=state,
            version=str(version),
            prompt=prompt,
            negative_prompt=negative_prompt,
            return_latents=add_pipeline,
            stage2strength=stage2strength,
        )
    elif mode == "img2img":
        out = state["model"].image_to_image(
            params=state["params"],
            prompt=prompt,
            negative_prompt=negative_prompt,
            return_latents=add_pipeline,
            noise_strength=stage2strength,
            filter=state.get("filter"),
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
