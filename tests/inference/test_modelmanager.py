import pytest
import torch

from sgm.inference.api import (    
    SamplingPipeline,    
    ModelArchitecture,    
)
import sgm.inference.helpers as helpers

def get_torch_device(model: torch.nn.Module) -> torch.device:
    param = next(model.parameters(), None)
    if param is not None:
        return param.device
    else:
        buf = next(model.buffers(), None)
        if buf is not None:
            return buf.device
        else:                
            raise TypeError("Could not determine device of input model")
    

@pytest.mark.inference
def test_default_loading():
    pipeline = SamplingPipeline(model_id=ModelArchitecture.SD_2_1)
    assert get_torch_device(pipeline.model.model).type == "cuda"
    assert get_torch_device(pipeline.model.conditioner).type == "cuda"
    with pipeline.device_manager.use(pipeline.model.model):
        assert get_torch_device(pipeline.model.model).type == "cuda"
    assert get_torch_device(pipeline.model.model).type == "cuda"
    with pipeline.device_manager.use(pipeline.model.conditioner):
        assert get_torch_device(pipeline.model.conditioner).type == "cuda"
    assert get_torch_device(pipeline.model.conditioner).type == "cuda"

@pytest.mark.inference
def test_model_swapping():
    pipeline = SamplingPipeline(model_id=ModelArchitecture.SD_2_1, device=helpers.CudaModelManager(device="cuda", swap_device="cpu"))
    assert get_torch_device(pipeline.model.model).type == "cpu"
    assert get_torch_device(pipeline.model.conditioner).type == "cpu"
    with pipeline.device_manager.use(pipeline.model.model):
        assert get_torch_device(pipeline.model.model).type == "cuda"
    assert get_torch_device(pipeline.model.model).type == "cpu"
    with pipeline.device_manager.use(pipeline.model.conditioner):
        assert get_torch_device(pipeline.model.conditioner).type == "cuda"
    assert get_torch_device(pipeline.model.conditioner).type == "cpu"