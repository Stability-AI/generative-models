from dataclasses import dataclass, asdict
from enum import Enum
from omegaconf import OmegaConf
import os
import pathlib
from sgm.inference.helpers import (
    do_sample,
    do_img2img,
    Img2ImgDiscretizationWrapper,
    Txt2NoisyDiscretizationWrapper,
)
from sgm.modules.diffusionmodules.sampling import (
    EulerEDMSampler,
    HeunEDMSampler,
    EulerAncestralSampler,
    DPMPP2SAncestralSampler,
    DPMPP2MSampler,
    LinearMultistepSampler,
)
from sgm.util import load_model_from_config
from typing import Optional


class ModelArchitecture(str, Enum):
    SDXL_V1_BASE = "stable-diffusion-xl-v1-base"
    SDXL_V1_REFINER = "stable-diffusion-xl-v1-refiner"
    SDXL_V0_9_BASE = "stable-diffusion-xl-v0-9-base"
    SDXL_V0_9_REFINER = "stable-diffusion-xl-v0-9-refiner"
    SD_2_1 = "stable-diffusion-v2-1"
    SD_2_1_768 = "stable-diffusion-v2-1-768"


class Sampler(str, Enum):
    EULER_EDM = "EulerEDMSampler"
    HEUN_EDM = "HeunEDMSampler"
    EULER_ANCESTRAL = "EulerAncestralSampler"
    DPMPP2S_ANCESTRAL = "DPMPP2SAncestralSampler"
    DPMPP2M = "DPMPP2MSampler"
    LINEAR_MULTISTEP = "LinearMultistepSampler"


class Discretization(str, Enum):
    LEGACY_DDPM = "LegacyDDPMDiscretization"
    EDM = "EDMDiscretization"


class Guider(str, Enum):
    VANILLA = "VanillaCFG"
    IDENTITY = "IdentityGuider"


class Thresholder(str, Enum):
    NONE = "None"


@dataclass
class SamplingParams:
    width: int = 1024
    height: int = 1024
    steps: int = 40
    sampler: Sampler = Sampler.EULER_EDM
    discretization: Discretization = Discretization.LEGACY_DDPM
    guider: Guider = Guider.VANILLA
    thresholder: Thresholder = Thresholder.NONE
    scale: float = 5.0
    aesthetic_score: float = 6.0
    negative_aesthetic_score: float = 2.5
    img2img_strength: float = 1.0
    orig_width: int = 1024
    orig_height: int = 1024
    crop_coords_top: int = 0
    crop_coords_left: int = 0
    sigma_min: float = 0.0292
    sigma_max: float = 14.6146
    rho: float = 3.0
    s_churn: float = 0.0
    s_tmin: float = 0.0
    s_tmax: float = 999.0
    s_noise: float = 1.0
    eta: float = 1.0
    order: int = 4


@dataclass
class SamplingSpec:
    width: int
    height: int
    channels: int
    factor: int
    is_legacy: bool
    config: str
    ckpt: str
    is_guided: bool


model_specs = {
    ModelArchitecture.SD_2_1: SamplingSpec(
        height=512,
        width=512,
        channels=4,
        factor=8,
        is_legacy=True,
        config="sd_2_1.yaml",
        ckpt="v2-1_512-ema-pruned.safetensors",
        is_guided=True,
    ),
    ModelArchitecture.SD_2_1_768: SamplingSpec(
        height=768,
        width=768,
        channels=4,
        factor=8,
        is_legacy=True,
        config="sd_2_1_768.yaml",
        ckpt="v2-1_768-ema-pruned.safetensors",
        is_guided=True,
    ),
    ModelArchitecture.SDXL_V0_9_BASE: SamplingSpec(
        height=1024,
        width=1024,
        channels=4,
        factor=8,
        is_legacy=False,
        config="sd_xl_base.yaml",
        ckpt="sd_xl_base_0.9.safetensors",
        is_guided=True,
    ),
    ModelArchitecture.SDXL_V0_9_REFINER: SamplingSpec(
        height=1024,
        width=1024,
        channels=4,
        factor=8,
        is_legacy=True,
        config="sd_xl_refiner.yaml",
        ckpt="sd_xl_refiner_0.9.safetensors",
        is_guided=True,
    ),
    ModelArchitecture.SDXL_V1_BASE: SamplingSpec(
        height=1024,
        width=1024,
        channels=4,
        factor=8,
        is_legacy=False,
        config="sd_xl_base.yaml",
        ckpt="sd_xl_base_1.0.safetensors",
        is_guided=True,
    ),
    ModelArchitecture.SDXL_V1_REFINER: SamplingSpec(
        height=1024,
        width=1024,
        channels=4,
        factor=8,
        is_legacy=True,
        config="sd_xl_refiner.yaml",
        ckpt="sd_xl_refiner_1.0.safetensors",
        is_guided=True,
    ),
}


class SamplingPipeline:
    def __init__(
        self,
        model_id: Optional[ModelArchitecture] = None,
        model_spec: Optional[SamplingSpec] = None,
        model_path=None,
        config_path=None,
        device="cuda",
        use_fp16=True,
    ) -> None:
        self.model_id = model_id
        if model_spec is not None:
            self.specs = model_spec
        elif model_id is not None:
            if model_id not in model_specs:
                raise ValueError(f"Model {model_id} not supported")
            self.specs = model_specs[model_id]
        else:
            raise ValueError("Either model_id or model_spec should be provided")

        if model_path is None:
            model_path = pathlib.Path(__file__).parent.parent.resolve() / "checkpoints"
            if not os.path.exists(model_path):
                # This supports development installs where checkpoints is root level of the repo
                model_path = (
                    pathlib.Path(__file__).parent.parent.parent.resolve()
                    / "checkpoints"
                )
        if config_path is None:
            config_path = (
                pathlib.Path(__file__).parent.parent.resolve() / "configs/inference"
            )
            if not os.path.exists(config_path):
                # This supports development installs where configs is root level of the repo
                config_path = (
                    pathlib.Path(__file__).parent.parent.parent.resolve()
                    / "configs/inference"
                )
        self.config = str(pathlib.Path(config_path) / self.specs.config)
        self.ckpt = str(pathlib.Path(model_path) / self.specs.ckpt)
        if not os.path.exists(self.config):
            raise ValueError(
                f"Config {self.config} not found, check model spec or config_path"
            )
        if not os.path.exists(self.ckpt):
            raise ValueError(
                f"Checkpoint {self.ckpt} not found, check model spec or config_path"
            )
        self.device = device
        self.model = self._load_model(device=device, use_fp16=use_fp16)

    def _load_model(self, device="cuda", use_fp16=True):
        config = OmegaConf.load(self.config)
        model = load_model_from_config(config, self.ckpt)
        if model is None:
            raise ValueError(f"Model {self.model_id} could not be loaded")
        model.to(device)
        if use_fp16:
            model.conditioner.half()
            model.model.half()
        return model

    def text_to_image(
        self,
        params: SamplingParams,
        prompt: str,
        negative_prompt: str = "",
        samples: int = 1,
        return_latents: bool = False,
        noise_strength=None,
        filter=None,
    ):
        sampler = get_sampler_config(params)

        sampler.discretization = self.wrap_discretization(
            sampler.discretization,
            image_strength=None,
            noise_strength=noise_strength,
            steps=params.steps,
        )

        value_dict = asdict(params)
        value_dict["prompt"] = prompt
        value_dict["negative_prompt"] = negative_prompt
        value_dict["target_width"] = params.width
        value_dict["target_height"] = params.height
        return do_sample(
            self.model,
            sampler,
            value_dict,
            samples,
            params.height,
            params.width,
            self.specs.channels,
            self.specs.factor,
            force_uc_zero_embeddings=["txt"] if not self.specs.is_legacy else [],
            return_latents=return_latents,
            filter=filter,
        )

    def image_to_image(
        self,
        params: SamplingParams,
        image,
        prompt: str,
        negative_prompt: str = "",
        samples: int = 1,
        return_latents: bool = False,
        noise_strength=None,
        filter=None,
    ):
        sampler = get_sampler_config(params)

        sampler.discretization = self.wrap_discretization(
            sampler.discretization,
            image_strength=params.img2img_strength,
            noise_strength=noise_strength,
            steps=params.steps,
        )

        height, width = image.shape[2], image.shape[3]
        value_dict = asdict(params)
        value_dict["prompt"] = prompt
        value_dict["negative_prompt"] = negative_prompt
        value_dict["target_width"] = width
        value_dict["target_height"] = height
        return do_img2img(
            image,
            self.model,
            sampler,
            value_dict,
            samples,
            force_uc_zero_embeddings=["txt"] if not self.specs.is_legacy else [],
            return_latents=return_latents,
            filter=filter,
        )

    def wrap_discretization(
        self, discretization, image_strength=None, noise_strength=None, steps=None
    ):
        if isinstance(discretization, Img2ImgDiscretizationWrapper) or isinstance(
            discretization, Txt2NoisyDiscretizationWrapper
        ):
            return discretization  # Already wrapped
        if image_strength is not None and image_strength < 1.0 and image_strength > 0.0:
            discretization = Img2ImgDiscretizationWrapper(
                discretization, strength=image_strength
            )

        if (
            noise_strength is not None
            and noise_strength < 1.0
            and noise_strength > 0.0
            and steps is not None
        ):
            discretization = Txt2NoisyDiscretizationWrapper(
                discretization,
                strength=noise_strength,
                original_steps=steps,
            )
        return discretization

    def refiner(
        self,
        image,
        prompt: str,
        negative_prompt: str = "",
        params: SamplingParams = SamplingParams(
            sampler=Sampler.EULER_EDM, steps=40, img2img_strength=0.15
        ),
        samples: int = 1,
        return_latents: bool = False,
        filter=None,
        add_noise=False,
    ):
        sampler = get_sampler_config(params)
        value_dict = {
            "orig_width": image.shape[3] * 8,
            "orig_height": image.shape[2] * 8,
            "target_width": image.shape[3] * 8,
            "target_height": image.shape[2] * 8,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "crop_coords_top": 0,
            "crop_coords_left": 0,
            "aesthetic_score": 6.0,
            "negative_aesthetic_score": 2.5,
        }

        sampler.discretization = self.wrap_discretization(
            sampler.discretization, image_strength=params.img2img_strength
        )

        return do_img2img(
            image,
            self.model,
            sampler,
            value_dict,
            samples,
            skip_encode=True,
            return_latents=return_latents,
            filter=filter,
            add_noise=add_noise,
        )


def get_guider_config(params: SamplingParams):
    if params.guider == Guider.IDENTITY:
        guider_config = {
            "target": "sgm.modules.diffusionmodules.guiders.IdentityGuider"
        }
    elif params.guider == Guider.VANILLA:
        scale = params.scale

        thresholder = params.thresholder

        if thresholder == Thresholder.NONE:
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


def get_discretization_config(params: SamplingParams):
    if params.discretization == Discretization.LEGACY_DDPM:
        discretization_config = {
            "target": "sgm.modules.diffusionmodules.discretizer.LegacyDDPMDiscretization",
        }
    elif params.discretization == Discretization.EDM:
        discretization_config = {
            "target": "sgm.modules.diffusionmodules.discretizer.EDMDiscretization",
            "params": {
                "sigma_min": params.sigma_min,
                "sigma_max": params.sigma_max,
                "rho": params.rho,
            },
        }
    else:
        raise ValueError(f"unknown discretization {params.discretization}")
    return discretization_config


def get_sampler_config(params: SamplingParams):
    discretization_config = get_discretization_config(params)
    guider_config = get_guider_config(params)
    sampler = None
    if params.sampler == Sampler.EULER_EDM:
        return EulerEDMSampler(
            num_steps=params.steps,
            discretization_config=discretization_config,
            guider_config=guider_config,
            s_churn=params.s_churn,
            s_tmin=params.s_tmin,
            s_tmax=params.s_tmax,
            s_noise=params.s_noise,
            verbose=True,
        )
    if params.sampler == Sampler.HEUN_EDM:
        return HeunEDMSampler(
            num_steps=params.steps,
            discretization_config=discretization_config,
            guider_config=guider_config,
            s_churn=params.s_churn,
            s_tmin=params.s_tmin,
            s_tmax=params.s_tmax,
            s_noise=params.s_noise,
            verbose=True,
        )
    if params.sampler == Sampler.EULER_ANCESTRAL:
        return EulerAncestralSampler(
            num_steps=params.steps,
            discretization_config=discretization_config,
            guider_config=guider_config,
            eta=params.eta,
            s_noise=params.s_noise,
            verbose=True,
        )
    if params.sampler == Sampler.DPMPP2S_ANCESTRAL:
        return DPMPP2SAncestralSampler(
            num_steps=params.steps,
            discretization_config=discretization_config,
            guider_config=guider_config,
            eta=params.eta,
            s_noise=params.s_noise,
            verbose=True,
        )
    if params.sampler == Sampler.DPMPP2M:
        return DPMPP2MSampler(
            num_steps=params.steps,
            discretization_config=discretization_config,
            guider_config=guider_config,
            verbose=True,
        )
    if params.sampler == Sampler.LINEAR_MULTISTEP:
        return LinearMultistepSampler(
            num_steps=params.steps,
            discretization_config=discretization_config,
            guider_config=guider_config,
            order=params.order,
            verbose=True,
        )

    raise ValueError(f"unknown sampler {params.sampler}!")
