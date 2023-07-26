import numpy
from PIL import Image
import pytest
from pytest import fixture
from omegaconf import OmegaConf
import torch

from sgm.util import load_model_from_config
from sgm.inference.api import model_specs, SamplingParams, SamplingPipeline, Sampler
import sgm.inference.helpers as helpers


@pytest.mark.inference
class TestInference:
    @fixture(scope="class", params=model_specs.keys())
    def pipeline(self, request) -> SamplingPipeline:
        pipeline = SamplingPipeline(request.param)
        yield pipeline
        del pipeline
        torch.cuda.empty_cache()

    def create_init_image(self, h, w):
        image_array = numpy.random.rand(h, w, 3) * 255
        image = Image.fromarray(image_array.astype("uint8")).convert("RGB")
        return helpers.get_input_image_tensor(image)

    @pytest.mark.parametrize("sampler_enum", Sampler)
    def test_txt2img(self, pipeline: SamplingPipeline, sampler_enum):
        output = pipeline.text_to_image(
            params=SamplingParams(sampler=sampler_enum.value, steps=10),
            prompt="A professional photograph of an astronaut riding a pig",
            negative_prompt="",
            samples=1,
        )

        assert output is not None

    @pytest.mark.parametrize("sampler_enum", Sampler)
    def test_img2img(self, pipeline: SamplingPipeline, sampler_enum):
        output = pipeline.image_to_image(
            params=SamplingParams(sampler=sampler_enum.value, steps=10),
            image=self.create_init_image(pipeline.specs.height, pipeline.specs.width),
            prompt="A professional photograph of an astronaut riding a pig",
            negative_prompt="",
            samples=1,
        )
        assert output is not None
