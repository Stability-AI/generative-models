import numpy
from PIL import Image
import pytest
from pytest import fixture
import torch
from typing import Tuple

from sgm.inference.api import (
    model_specs,
    SamplingParams,
    SamplingPipeline,
    Sampler,
    ModelArchitecture,
)
import sgm.inference.helpers as helpers


@pytest.mark.inference
class TestInference:
    @fixture(scope="class", params=model_specs.keys())
    def pipeline(self, request) -> SamplingPipeline:
        pipeline = SamplingPipeline(request.param)
        yield pipeline
        del pipeline
        torch.cuda.empty_cache()

    @fixture(
        scope="class",
        params=[
            [ModelArchitecture.SDXL_V1_BASE, ModelArchitecture.SDXL_V1_REFINER],
            [ModelArchitecture.SDXL_V0_9_BASE, ModelArchitecture.SDXL_V0_9_REFINER],
        ],
        ids=["SDXL_V1", "SDXL_V0_9"],
    )
    def sdxl_pipelines(self, request) -> Tuple[SamplingPipeline, SamplingPipeline]:
        base_pipeline = SamplingPipeline(request.param[0])
        refiner_pipeline = SamplingPipeline(request.param[1])
        yield base_pipeline, refiner_pipeline
        del base_pipeline
        del refiner_pipeline
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

    @pytest.mark.parametrize("sampler_enum", Sampler)
    @pytest.mark.parametrize(
        "use_init_image", [True, False], ids=["img2img", "txt2img"]
    )
    def test_sdxl_with_refiner(
        self,
        sdxl_pipelines: Tuple[SamplingPipeline, SamplingPipeline],
        sampler_enum,
        use_init_image,
    ):
        base_pipeline, refiner_pipeline = sdxl_pipelines
        if use_init_image:
            output = base_pipeline.image_to_image(
                params=SamplingParams(sampler=sampler_enum.value, steps=10),
                image=self.create_init_image(
                    base_pipeline.specs.height, base_pipeline.specs.width
                ),
                prompt="A professional photograph of an astronaut riding a pig",
                negative_prompt="",
                samples=1,
                return_latents=True,
            )
        else:
            output = base_pipeline.text_to_image(
                params=SamplingParams(sampler=sampler_enum.value, steps=10),
                prompt="A professional photograph of an astronaut riding a pig",
                negative_prompt="",
                samples=1,
                return_latents=True,
            )

        assert isinstance(output, (tuple, list))
        samples, samples_z = output
        assert samples is not None
        assert samples_z is not None
        refiner_pipeline.refiner(
            params=SamplingParams(sampler=sampler_enum.value, steps=10),
            image=samples_z,
            prompt="A professional photograph of an astronaut riding a pig",
            negative_prompt="",
            samples=1,
        )
