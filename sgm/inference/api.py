from enum import Enum
from omegaconf import OmegaConf
from pydantic import BaseModel
import pathlib
from sgm.inference.helpers import (    
    do_sample,
    do_img2img,
    Img2ImgDiscretizationWrapper,
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


class ModelIdentifier(str, Enum):
    SD_2_1 = "StableDiffusion2.1"
    SD_2_1_768 = "StableDiffusion2.1-768"
    SDXL_BASE = "SDXLBase"
    SDXL_REFINER = "SDXLRefiner"


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


class SamplingParams(BaseModel):
    width: int = 1024
    height: int = 1024
    steps: int = 50
    sampler: Sampler = Sampler.DPMPP2M
    discretization: Discretization = Discretization.LEGACY_DDPM
    guider: Guider = Guider.VANILLA
    thresholder: Thresholder = Thresholder.NONE
    scale: float = 6.0
    aesthetic_score: float = 5.0
    negative_aesthetic_score: float = 5.0
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


class SamplingSpec(BaseModel):
    width: int
    height: int
    channels: int
    factor: int
    is_legacy: bool
    config: str
    ckpt: str
    is_guided: bool


model_specs = {
    ModelIdentifier.SD_2_1: SamplingSpec(
        height=512,
        width=512,
        channels=4,
        factor=8,
        is_legacy=True,
        config="sd_2_1.yaml",
        ckpt="v2-1_512-ema-pruned.safetensors",
        is_guided=True,
    ),
    ModelIdentifier.SD_2_1_768: SamplingSpec(
        height=768,
        width=768,
        channels=4,
        factor=8,
        is_legacy=True,
        config="sd_2_1_768.yaml",
        ckpt="v2-1_768-ema-pruned.safetensors",
        is_guided=True,
    ),
    ModelIdentifier.SDXL_BASE: SamplingSpec(
        height=1024,
        width=1024,
        channels=4,
        factor=8,
        is_legacy=False,
        config="sd_xl_base.yaml",
        ckpt="sd_xl_base_0.9.safetensors",
        is_guided=True,
    ),
    ModelIdentifier.SDXL_REFINER: SamplingSpec(
        height=1024,
        width=1024,
        channels=4,
        factor=8,
        is_legacy=True,
        config="sd_xl_refiner.yaml",
        ckpt="sd_xl_refiner_0.9.safetensors",
        is_guided=True,
    ),
}


class SamplingPipeline:
    def __init__(
        self,
        model_id: ModelIdentifier,
        model_path="checkpoints",
        config_path="configs/inference",
        device="cuda",
    ) -> None:
        if model_id not in model_specs:
            raise ValueError(f"Model {model_id} not supported")
        self.model_id = model_id
        self.specs = model_specs[self.model_id]
        self.specs.config = str(pathlib.Path(config_path, self.specs.config))
        self.specs.ckpt = str(pathlib.Path(model_path, self.specs.ckpt))
        self.device = device
        self.model = self._load_model()

    def _load_model(self, device="cuda"):
        config = OmegaConf.load(self.specs.config)
        model = load_model_from_config(config, self.specs.ckpt)
        model.to(device)
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
    ):
        sampler = get_sampler_config(params)
        value_dict = dict(params)
        value_dict["prompt"] = prompt
        value_dict["negative_prompt"] = negative_prompt
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
            filter=None,
        )

    def image_to_image(
        self,
        params: SamplingParams,
        image,
        prompt: str,
        negative_prompt: str = "",
        samples: int = 1,
        return_latents: bool = False,
    ):
        sampler = get_sampler_config(params)

        if params.img2img_strength < 1.0:
            sampler.discretization = Img2ImgDiscretizationWrapper(
                sampler.discretization,
                strength=params.img2img_strength,
            )

        value_dict = dict(params)
        value_dict["prompt"] = prompt
        value_dict["negative_prompt"] = negative_prompt

        return do_img2img(
            image,
            self.model,
            sampler,
            value_dict,
            samples,
            force_uc_zero_embeddings=["txt"] if not self.specs.is_legacy else [],
            return_latents=return_latents,
            filter=None,
        )

    def refiner(
        self,
        params: SamplingParams,
        image,
        prompt: str,
        negative_prompt: str = None,
        samples: int = 1,
        return_latents: bool = False,
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

        return do_img2img(
            image,
            self.model,
            sampler,
            value_dict,
            samples,
            skip_encode=True,
            return_latents=return_latents,
            filter=None,
        )

def get_guider_config(params:SamplingParams):
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
        raise ValueError(f"unknown discertization {params.discretization}")
    return discretization_config


def get_sampler_config(params: SamplingParams):
    discretization_config = get_discretization_config(params)
    guider_config = get_guider_config(params)
    if params.sampler == Sampler.EULER_EDM or params.sampler == Sampler.HEUN_EDM:
        if params.sampler == Sampler.EULER_EDM:
            sampler = EulerEDMSampler(
                num_steps=params.steps,
                discretization_config=discretization_config,
                guider_config=guider_config,
                s_churn=params.s_churn,
                s_tmin=params.s_tmin,
                s_tmax=params.s_tmax,
                s_noise=params.s_noise,
                verbose=True,
            )
        elif params.sampler == Sampler.HEUN_EDM:
            sampler = HeunEDMSampler(
                num_steps=params.steps,
                discretization_config=discretization_config,
                guider_config=guider_config,
                s_churn=params.s_churn,
                s_tmin=params.s_tmin,
                s_tmax=params.s_tmax,
                s_noise=params.s_noise,
                verbose=True,
            )
    elif (
        params.sampler == Sampler.EULER_ANCESTRAL or params.sampler == Sampler.DPMPP2S_ANCESTRAL
    ):        
        if params.sampler == Sampler.EULER_ANCESTRAL:
            sampler = EulerAncestralSampler(
                num_steps=params.steps,
                discretization_config=discretization_config,
                guider_config=guider_config,
                eta=params.eta,
                s_noise=params.s_noise,
                verbose=True,
            )
        elif params.sampler == Sampler.DPMPP2S_ANCESTRAL:
            sampler = DPMPP2SAncestralSampler(
                num_steps=params.steps,
                discretization_config=discretization_config,
                guider_config=guider_config,
                eta=params.eta,
                s_noise=params.s_noise,
                verbose=True,
            )
    elif params.sampler == Sampler.DPMPP2M:
        sampler = DPMPP2MSampler(
            num_steps=params.steps,
            discretization_config=discretization_config,
            guider_config=guider_config,
            verbose=True,
        )
    elif params.sampler == Sampler.LINEAR_MULTISTEP:        
        sampler = LinearMultistepSampler(
            num_steps=params.steps,
            discretization_config=discretization_config,
            guider_config=guider_config,
            order=params.order,
            verbose=True,
        )
    else:
        raise ValueError(f"unknown sampler {sampler}!")

    return sampler
