from pytorch_lightning import seed_everything

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
    version,
    version_dict,
    is_legacy=False,
    return_latents=False,
    filter=None,
    stage2strength=None,
):
    if version.startswith("SDXL-base"):
        W, H = st.selectbox("Resolution:", list(SD_XL_BASE_RATIOS.values()), 10)
    else:
        H = st.number_input("H", value=version_dict["H"], min_value=64, max_value=2048)
        W = st.number_input("W", value=version_dict["W"], min_value=64, max_value=2048)
    C = version_dict["C"]
    F = version_dict["f"]

    init_dict = {
        "orig_width": W,
        "orig_height": H,
        "target_width": W,
        "target_height": H,
    }
    value_dict = init_embedder_options(
        get_unique_embedder_keys_from_conditioner(state["model"].conditioner),
        init_dict,
        prompt=prompt,
        negative_prompt=negative_prompt,
    )
    sampler, num_rows, num_cols = init_sampling(stage2strength=stage2strength)
    num_samples = num_rows * num_cols

    if st.button("Sample"):
        st.write(f"**Model I:** {version}")
        out = do_sample(
            state["model"],
            sampler,
            value_dict,
            num_samples,
            H,
            W,
            C,
            F,
            force_uc_zero_embeddings=["txt"] if not is_legacy else [],
            return_latents=return_latents,
            filter=filter,
        )
        return out


def run_img2img(
    state,
    version_dict,
    is_legacy=False,
    return_latents=False,
    filter=None,
    stage2strength=None,
):
    img = load_img()
    if img is None:
        return None
    H, W = img.shape[2], img.shape[3]

    init_dict = {
        "orig_width": W,
        "orig_height": H,
        "target_width": W,
        "target_height": H,
    }
    value_dict = init_embedder_options(
        get_unique_embedder_keys_from_conditioner(state["model"].conditioner),
        init_dict,
        prompt=prompt,
        negative_prompt=negative_prompt,
    )
    strength = st.number_input(
        "**Img2Img Strength**", value=0.75, min_value=0.0, max_value=1.0
    )
    sampler, num_rows, num_cols = init_sampling(
        img2img_strength=strength,
        stage2strength=stage2strength,
    )
    num_samples = num_rows * num_cols

    if st.button("Sample"):
        out = do_img2img(
            repeat(img, "1 ... -> n ...", n=num_samples),
            state["model"],
            sampler,
            value_dict,
            num_samples,
            force_uc_zero_embeddings=["txt"] if not is_legacy else [],
            return_latents=return_latents,
            filter=filter,
        )
        return out


def apply_refiner(
    input,
    state,
    sampler,
    num_samples,
    prompt,
    negative_prompt,
    filter=None,
    finish_denoising=False,
):
    init_dict = {
        "orig_width": input.shape[3] * 8,
        "orig_height": input.shape[2] * 8,
        "target_width": input.shape[3] * 8,
        "target_height": input.shape[2] * 8,
    }

    value_dict = init_dict
    value_dict["prompt"] = prompt
    value_dict["negative_prompt"] = negative_prompt

    value_dict["crop_coords_top"] = 0
    value_dict["crop_coords_left"] = 0

    value_dict["aesthetic_score"] = 6.0
    value_dict["negative_aesthetic_score"] = 2.5

    st.warning(f"refiner input shape: {input.shape}")
    samples = do_img2img(
        input,
        state["model"],
        sampler,
        value_dict,
        num_samples,
        skip_encode=True,
        filter=filter,
        add_noise=not finish_denoising,
    )

    return samples


if __name__ == "__main__":
    st.title("Stable Diffusion")
    version = st.selectbox("Model Version", list(VERSION2SPECS.keys()), 0)
    version_dict = VERSION2SPECS[version]
    mode = st.radio("Mode", ("txt2img", "img2img"), 0)
    st.write("__________________________")

    set_lowvram_mode(st.checkbox("Low vram mode", True))

    if version.startswith("SDXL-base"):
        add_pipeline = st.checkbox("Load SDXL-refiner?", False)
        st.write("__________________________")
    else:
        add_pipeline = False

    seed = st.sidebar.number_input("seed", value=42, min_value=0, max_value=int(1e9))
    seed_everything(seed)

    save_locally, save_path = init_save_locally(os.path.join(SAVE_PATH, version))

    state = init_st(version_dict, load_filter=True)
    if state["msg"]:
        st.info(state["msg"])
    model = state["model"]

    is_legacy = version_dict["is_legacy"]

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
        version2 = st.selectbox("Refiner:", ["SDXL-refiner-1.0", "SDXL-refiner-0.9"])
        st.warning(
            f"Running with {version2} as the second stage model. Make sure to provide (V)RAM :) "
        )
        st.write("**Refiner Options:**")

        version_dict2 = VERSION2SPECS[version2]
        state2 = init_st(version_dict2, load_filter=False)
        st.info(state2["msg"])

        stage2strength = st.number_input(
            "**Refinement strength**", value=0.15, min_value=0.0, max_value=1.0
        )

        sampler2, *_ = init_sampling(
            key=2,
            img2img_strength=stage2strength,
            specify_num_samples=False,
        )
        st.write("__________________________")
        finish_denoising = st.checkbox("Finish denoising with refiner.", True)
        if not finish_denoising:
            stage2strength = None

    if mode == "txt2img":
        out = run_txt2img(
            state,
            version,
            version_dict,
            is_legacy=is_legacy,
            return_latents=add_pipeline,
            filter=state.get("filter"),
            stage2strength=stage2strength,
        )
    elif mode == "img2img":
        out = run_img2img(
            state,
            version_dict,
            is_legacy=is_legacy,
            return_latents=add_pipeline,
            filter=state.get("filter"),
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
        st.write("**Running Refinement Stage**")
        samples = apply_refiner(
            samples_z,
            state2,
            sampler2,
            samples_z.shape[0],
            prompt=prompt,
            negative_prompt=negative_prompt if is_legacy else "",
            filter=state.get("filter"),
            finish_denoising=finish_denoising,
        )

    if save_locally and samples is not None:
        perform_save_locally(save_path, samples)
