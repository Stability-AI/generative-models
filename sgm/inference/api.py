from dataclasses import dataclass, asdict
from enum import Enum
from omegaconf import OmegaConf
import os
from sgm.inference.helpers import (
    do_sample,
    do_img2img,
    DeviceModelManager,
    get_model_manager,
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
from sgm.util import load_model_from_config, get_configs_path, get_checkpoints_path
import torch
from typing import Optional, Dict, Any, Union


class ModelArchitecture(str, Enum):
    SDXL_V1_0_BASE = "stable-diffusion-xl-v1-base"
    SDXL_V1_0_REFINER = "stable-diffusion-xl-v1-refiner"
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
    """
    Parameters for sampling.
    """

    width: Optional[int] = None
    height: Optional[int] = None
    steps: Optional[int] = None
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
    default_params: SamplingParams


# The defaults here are derived from user preference testing.
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
        default_params=SamplingParams(
            width=512,
            height=512,
            steps=40,
            scale=7.0,
        ),
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
        default_params=SamplingParams(
            width=768,
            height=768,
            steps=40,
            scale=7.0,
        ),
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
        default_params=SamplingParams(width=1024, height=1024, steps=40, scale=5.0),
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
        default_params=SamplingParams(
            width=1024, height=1024, steps=40, scale=5.0, img2img_strength=0.15
        ),
    ),
    ModelArchitecture.SDXL_V1_0_BASE: SamplingSpec(
        height=1024,
        width=1024,
        channels=4,
        factor=8,
        is_legacy=False,
        config="sd_xl_base.yaml",
        ckpt="sd_xl_base_1.0.safetensors",
        is_guided=True,
        default_params=SamplingParams(width=1024, height=1024, steps=40, scale=5.0),
    ),
    ModelArchitecture.SDXL_V1_0_REFINER: SamplingSpec(
        height=1024,
        width=1024,
        channels=4,
        factor=8,
        is_legacy=True,
        config="sd_xl_refiner.yaml",
        ckpt="sd_xl_refiner_1.0.safetensors",
        is_guided=True,
        default_params=SamplingParams(
            width=1024, height=1024, steps=40, scale=5.0, img2img_strength=0.15
        ),
    ),
}


def wrap_discretization(
    discretization, image_strength=None, noise_strength=None, steps=None
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


class SamplingPipeline:
    def __init__(
        self,
        model_id: Optional[ModelArchitecture] = None,
        model_spec: Optional[SamplingSpec] = None,
        model_path: Optional[str] = None,
        config_path: Optional[str] = None,
        use_fp16: bool = True,
        device: Optional[Union[DeviceModelManager, str, torch.device]] = None,
    ) -> None:
        """
        Sampling pipeline for generating images from a model.

        @param model_id: Model architecture to use. If not specified, model_spec must be specified.
        @param model_spec: Model specification to use. If not specified, model_id must be specified.
        @param model_path: Path to model checkpoints folder.
        @param config_path: Path to model config folder.
        @param use_fp16: Whether to use fp16 for sampling.
        @param device: Device manager to use with this pipeline. If a string or torch.device is passed, a device  manager will be created based on device type if possible.
        """

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
            model_path = get_checkpoints_path()
        if config_path is None:
            config_path = get_configs_path()
        self.config = os.path.join(config_path, "inference", self.specs.config)
        self.ckpt = os.path.join(model_path, self.specs.ckpt)
        if not os.path.exists(self.config):
            raise ValueError(
                f"Config {self.config} not found, check model spec or config_path"
            )
        if not os.path.exists(self.ckpt):
            raise ValueError(
                f"Checkpoint {self.ckpt} not found, check model spec or config_path"
            )

        self.device_manager = get_model_manager(device)

        self.model = self._load_model(
            device_manager=self.device_manager, use_fp16=use_fp16
        )

    def _load_model(self, device_manager: DeviceModelManager, use_fp16=True):
        config = OmegaConf.load(self.config)
        model = load_model_from_config(config, self.ckpt)
        if model is None:
            raise ValueError(f"Model {self.model_id} could not be loaded")
        device_manager.load(model)
        if use_fp16:
            model.conditioner.half()
            model.model.half()
        return model

    def text_to_image(
        self,
        prompt: str,
        params: Optional[SamplingParams] = None,
        negative_prompt: str = "",
        samples: int = 1,
        return_latents: bool = False,
        noise_strength: Optional[float] = None,
        filter=None,
    ):
        if params is None:
            params = self.specs.default_params
        else:
            # Set defaults if optional params are not specified
            if params.width is None:
                params.width = self.specs.default_params.width
            if params.height is None:
                params.height = self.specs.default_params.height
            if params.steps is None:
                params.steps = self.specs.default_params.steps

        sampler = get_sampler_config(params)

        sampler.discretization = wrap_discretization(
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
            device=self.device_manager,
        )

    def image_to_image(
        self,
        image: torch.Tensor,
        prompt: str,
        params: Optional[SamplingParams] = None,
        negative_prompt: str = "",
        samples: int = 1,
        return_latents: bool = False,
        noise_strength: Optional[float] = None,
        filter=None,
    ):
        if params is None:
            params = self.specs.default_params
        sampler = get_sampler_config(params)

        sampler.discretization = wrap_discretization(
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
        value_dict["orig_width"] = width
        value_dict["orig_height"] = height
        return do_img2img(
            image,
            self.model,
            sampler,
            value_dict,
            samples,
            force_uc_zero_embeddings=["txt"] if not self.specs.is_legacy else [],
            return_latents=return_latents,
            filter=filter,
            device=self.device_manager,
        )

    def refiner(
        self,
        image: torch.Tensor,
        prompt: str,
        negative_prompt: str = "",
        params: Optional[SamplingParams] = None,
        samples: int = 1,
        return_latents: bool = False,
        filter: Any = None,
        add_noise: bool = False,
    ):
        if params is None:
            params = self.specs.default_params

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

        sampler.discretization = wrap_discretization(
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
            device=self.device_manager,
        )


def get_guider_config(params: SamplingParams) -> Dict[str, Any]:
    guider_config: Dict[str, Any]
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


def get_discretization_config(params: SamplingParams) -> Dict[str, Any]:
    discretization_config: Dict[str, Any]
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
