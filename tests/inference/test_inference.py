import numpy
from PIL import Image
import pytest
from pytest import fixture
from omegaconf import OmegaConf
import torch

from sgm.util import load_model_from_config
import sgm.inference.helpers as helpers

VERSION2SPECS = {
    "SD-XL base": {
        "H": 1024,
        "W": 1024,
        "C": 4,
        "f": 8,
        "is_legacy": False,
        "config": "configs/inference/sd_xl_base.yaml",
        "ckpt": "checkpoints/sd_xl_base_0.9.safetensors",
        "is_guided": True,
    },
    "sd-2.1": {
        "H": 512,
        "W": 512,
        "C": 4,
        "f": 8,
        "is_legacy": True,
        "config": "configs/inference/sd_2_1.yaml",
        "ckpt": "checkpoints/v2-1_512-ema-pruned.safetensors",
        "is_guided": True,
    },
    "sd-2.1-768": {
        "H": 768,
        "W": 768,
        "C": 4,
        "f": 8,
        "is_legacy": True,
        "config": "configs/inference/sd_2_1_768.yaml",
        "ckpt": "checkpoints/v2-1_768-ema-pruned.safetensors",
    },
    "SDXL-Refiner": {
        "H": 1024,
        "W": 1024,
        "C": 4,
        "f": 8,
        "is_legacy": True,
        "config": "configs/inference/sd_xl_refiner.yaml",
        "ckpt": "checkpoints/sd_xl_refiner_0.9.safetensors",
        "is_guided": True,
    },
}

samplers = [
    "EulerEDMSampler",
    "HeunEDMSampler",
    "EulerAncestralSampler",
    "DPMPP2SAncestralSampler",
    "DPMPP2MSampler",
    "LinearMultistepSampler",
]


@pytest.mark.inference
class TestInference:
    @fixture(
        scope="class", params=["SD-XL base", "sd-2.1", "sd-2.1-768", "SDXL-Refiner"]
    )
    def model(self, request):
        specs = VERSION2SPECS[request.param]
        config = OmegaConf.load(specs["config"])
        model, _ = load_model_from_config(config, specs["ckpt"])
        model.conditioner.half()
        model.model.half()
        yield model, specs
        del model
        torch.cuda.empty_cache()

    def create_init_image(self, h, w):
        image_array = numpy.random.rand(h, w, 3) * 255
        image = Image.fromarray(image_array.astype("uint8")).convert("RGB")
        return helpers.get_input_image_tensor(image)

    @pytest.mark.parametrize("sampler_name", samplers)
    def test_txt2img(self, model, sampler_name):
        specs = model[1]
        model = model[0]
        value_dict = {
            "prompt": "A professional photograph of an astronaut riding a pig",
            "negative_prompt": "",
            "aesthetic_score": 6.0,
            "negative_aesthetic_score": 2.5,
            "orig_height": specs["H"],
            "orig_width": specs["W"],
            "target_height": specs["H"],
            "target_width": specs["W"],
            "crop_coords_top": 0,
            "crop_coords_left": 0,
        }
        sampler = helpers.get_sampler(
            sampler_name=sampler_name,
            steps=10,
            discretization_config=helpers.get_discretization(
                "LegacyDDPMDiscretization"
            ),
            guider_config=helpers.get_guider(guider="VanillaCFG", scale=7.0),
        )
        output = helpers.do_sample(
            model=model,
            sampler=sampler,
            value_dict=value_dict,
            num_samples=1,
            H=specs["H"],
            W=specs["W"],
            C=specs["C"],
            F=specs["f"],
        )

        assert output is not None

    @pytest.mark.parametrize("sampler_name", samplers)
    def test_img2img(self, model, sampler_name):
        specs = model[1]
        model = model[0]
        init_image = self.create_init_image(specs["H"], specs["W"]).to("cuda")
        value_dict = {
            "prompt": "A professional photograph of an astronaut riding a pig",
            "negative_prompt": "",
            "aesthetic_score": 6.0,
            "negative_aesthetic_score": 2.5,
            "orig_height": specs["H"],
            "orig_width": specs["W"],
            "target_height": specs["H"],
            "target_width": specs["W"],
            "crop_coords_top": 0,
            "crop_coords_left": 0,
        }

        sampler = helpers.get_sampler(
            sampler_name=sampler_name,
            steps=10,
            discretization_config=helpers.get_discretization(
                "LegacyDDPMDiscretization"
            ),
            guider_config=helpers.get_guider(guider="VanillaCFG", scale=7.0),
        )

        output = helpers.do_img2img(
            img=init_image,
            model=model,
            sampler=sampler,
            value_dict=value_dict,
            num_samples=1,
        )
