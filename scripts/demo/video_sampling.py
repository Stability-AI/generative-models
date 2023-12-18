import os

from pytorch_lightning import seed_everything

from scripts.demo.streamlit_helpers import *

SAVE_PATH = "outputs/demo/vid/"

VERSION2SPECS = {
    "svd": {
        "T": 14,
        "H": 576,
        "W": 1024,
        "C": 4,
        "f": 8,
        "config": "configs/inference/svd.yaml",
        "ckpt": "checkpoints/svd.safetensors",
        "options": {
            "discretization": 1,
            "cfg": 2.5,
            "sigma_min": 0.002,
            "sigma_max": 700.0,
            "rho": 7.0,
            "guider": 2,
            "force_uc_zero_embeddings": ["cond_frames", "cond_frames_without_noise"],
            "num_steps": 25,
        },
    },
    "svd_image_decoder": {
        "T": 14,
        "H": 576,
        "W": 1024,
        "C": 4,
        "f": 8,
        "config": "configs/inference/svd_image_decoder.yaml",
        "ckpt": "checkpoints/svd_image_decoder.safetensors",
        "options": {
            "discretization": 1,
            "cfg": 2.5,
            "sigma_min": 0.002,
            "sigma_max": 700.0,
            "rho": 7.0,
            "guider": 2,
            "force_uc_zero_embeddings": ["cond_frames", "cond_frames_without_noise"],
            "num_steps": 25,
        },
    },
    "svd_xt": {
        "T": 25,
        "H": 576,
        "W": 1024,
        "C": 4,
        "f": 8,
        "config": "configs/inference/svd.yaml",
        "ckpt": "checkpoints/svd_xt.safetensors",
        "options": {
            "discretization": 1,
            "cfg": 3.0,
            "min_cfg": 1.5,
            "sigma_min": 0.002,
            "sigma_max": 700.0,
            "rho": 7.0,
            "guider": 2,
            "force_uc_zero_embeddings": ["cond_frames", "cond_frames_without_noise"],
            "num_steps": 30,
            "decoding_t": 14,
        },
    },
    "svd_xt_image_decoder": {
        "T": 25,
        "H": 576,
        "W": 1024,
        "C": 4,
        "f": 8,
        "config": "configs/inference/svd_image_decoder.yaml",
        "ckpt": "checkpoints/svd_xt_image_decoder.safetensors",
        "options": {
            "discretization": 1,
            "cfg": 3.0,
            "min_cfg": 1.5,
            "sigma_min": 0.002,
            "sigma_max": 700.0,
            "rho": 7.0,
            "guider": 2,
            "force_uc_zero_embeddings": ["cond_frames", "cond_frames_without_noise"],
            "num_steps": 30,
            "decoding_t": 14,
        },
    },
}


if __name__ == "__main__":
    st.title("Stable Video Diffusion")
    version = st.selectbox(
        "Model Version",
        [k for k in VERSION2SPECS.keys()],
        0,
    )
    version_dict = VERSION2SPECS[version]
    if st.checkbox("Load Model"):
        mode = "img2vid"
    else:
        mode = "skip"

    H = st.sidebar.number_input(
        "H", value=version_dict["H"], min_value=64, max_value=2048
    )
    W = st.sidebar.number_input(
        "W", value=version_dict["W"], min_value=64, max_value=2048
    )
    T = st.sidebar.number_input(
        "T", value=version_dict["T"], min_value=0, max_value=128
    )
    C = version_dict["C"]
    F = version_dict["f"]
    options = version_dict["options"]

    if mode != "skip":
        state = init_st(version_dict, load_filter=True)
        if state["msg"]:
            st.info(state["msg"])
        model = state["model"]

        ukeys = set(
            get_unique_embedder_keys_from_conditioner(state["model"].conditioner)
        )

        value_dict = init_embedder_options(
            ukeys,
            {},
        )

        value_dict["image_only_indicator"] = 0

        if mode == "img2vid":
            img = load_img_for_prediction(W, H)
            cond_aug = st.number_input(
                "Conditioning augmentation:", value=0.02, min_value=0.0
            )
            value_dict["cond_frames_without_noise"] = img
            value_dict["cond_frames"] = img + cond_aug * torch.randn_like(img)
            value_dict["cond_aug"] = cond_aug

        seed = st.sidebar.number_input(
            "seed", value=23, min_value=0, max_value=int(1e9)
        )
        seed_everything(seed)

        save_locally, save_path = init_save_locally(
            os.path.join(SAVE_PATH, version), init_value=True
        )

        options["num_frames"] = T

        sampler, num_rows, num_cols = init_sampling(options=options)
        num_samples = num_rows * num_cols

        decoding_t = st.number_input(
            "Decode t frames at a time (set small if you are low on VRAM)",
            value=options.get("decoding_t", T),
            min_value=1,
            max_value=int(1e9),
        )

        if st.checkbox("Overwrite fps in mp4 generator", False):
            saving_fps = st.number_input(
                f"saving video at fps:", value=value_dict["fps"], min_value=1
            )
        else:
            saving_fps = value_dict["fps"]

        if st.button("Sample"):
            out = do_sample(
                model,
                sampler,
                value_dict,
                num_samples,
                H,
                W,
                C,
                F,
                T=T,
                batch2model_input=["num_video_frames", "image_only_indicator"],
                force_uc_zero_embeddings=options.get("force_uc_zero_embeddings", None),
                force_cond_zero_embeddings=options.get(
                    "force_cond_zero_embeddings", None
                ),
                return_latents=False,
                decoding_t=decoding_t,
            )

            if isinstance(out, (tuple, list)):
                samples, samples_z = out
            else:
                samples = out
                samples_z = None

            if save_locally:
                save_video_as_grid_and_mp4(samples, save_path, T, fps=saving_fps)
