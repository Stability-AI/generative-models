import numpy
from PIL import Image
import pytest
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
    @classmethod
    def setup_class(cls):
        cls.pipeline_objects = {}
        cls.sdxl_pipelines_objects = {}
        for model_name in model_specs.keys():
            cls.pipeline_objects[model_name] = SamplingPipeline(model_name)

        for arch_pair in [
            [ModelArchitecture.SDXL_V1_BASE, ModelArchitecture.SDXL_V1_REFINER],
            [ModelArchitecture.SDXL_V0_9_BASE, ModelArchitecture.SDXL_V0_9_REFINER],
        ]:
            cls.sdxl_pipelines_objects[tuple(arch_pair)] = (
                SamplingPipeline(arch_pair[0]),
                SamplingPipeline(arch_pair[1]),
            )

    @classmethod
    def teardown_class(cls):
        for pipeline in cls.pipeline_objects.values():
            del pipeline
        for base_pipeline, refiner_pipeline in cls.sdxl_pipelines_objects.values():
            del base_pipeline
            del refiner_pipeline
        torch.cuda.empty_cache()

    def create_init_image(self, h, w):
        image_array = numpy.random.rand(h, w, 3) * 255
        image = Image.fromarray(image_array.astype("uint8")).convert("RGB")
        return helpers.get_input_image_tensor(image)

    @pytest.mark.parametrize("sampler_enum", Sampler)
    def test_txt2img(self, sampler_enum):
        pipeline = self.pipeline_objects[sampler_enum.value]
        output = pipeline.text_to_image(
            params=SamplingParams(sampler=sampler_enum.value, steps=10),
            prompt="A professional photograph of an astronaut riding a pig",
            negative_prompt="",
            samples=1,
        )
        assert output is not None

    @pytest.mark.parametrize("sampler_enum", Sampler)
    def test_img2img(self, sampler_enum):
        pipeline = self.pipeline_objects[sampler_enum.value]
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
    def test_sdxl_with_refiner(self, sampler_enum, use_init_image):
        base_pipeline, refiner_pipeline = self.sdxl_pipelines_objects[
            (ModelArchitecture(base_pipeline.specs.architecture), ModelArchitecture(refiner_pipeline.specs.architecture))
        ]
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
