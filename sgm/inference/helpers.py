import contextlib
import os
from typing import Union, List, Optional

import math
import numpy as np
import torch
from PIL import Image
from einops import rearrange
from imwatermark import WatermarkEncoder
from omegaconf import ListConfig

from sgm.util import append_dims


class WatermarkEmbedder:
    def __init__(self, watermark):
        self.watermark = watermark
        self.num_bits = len(WATERMARK_BITS)
        self.encoder = WatermarkEncoder()
        self.encoder.set_watermark("bits", self.watermark)

    def __call__(self, image: torch.Tensor):
        """
        Adds a predefined watermark to the input image

        Args:
            image: ([N,] B, C, H, W) in range [0, 1]

        Returns:
            same as input but watermarked
        """
        # watermarking libary expects input as cv2 BGR format
        squeeze = len(image.shape) == 4
        if squeeze:
            image = image[None, ...]
        n = image.shape[0]
        image_np = rearrange(
            (255 * image).detach().cpu(), "n b c h w -> (n b) h w c"
        ).numpy()[:, :, :, ::-1]
        # torch (b, c, h, w) in [0, 1] -> numpy (b, h, w, c) [0, 255]
        for k in range(image_np.shape[0]):
            image_np[k] = self.encoder.encode(image_np[k], "dwtDct")
        image = torch.from_numpy(
            rearrange(image_np[:, :, :, ::-1], "(n b) h w c -> n b c h w", n=n)
        ).to(image.device)
        image = torch.clamp(image / 255, min=0.0, max=1.0)
        if squeeze:
            image = image[0]
        return image


# A fixed 48-bit message that was choosen at random
# WATERMARK_MESSAGE = 0xB3EC907BB19E
WATERMARK_MESSAGE = 0b101100111110110010010000011110111011000110011110
# bin(x)[2:] gives bits of x as str, use int to convert them to 0/1
WATERMARK_BITS = [int(bit) for bit in bin(WATERMARK_MESSAGE)[2:]]
embed_watermark = WatermarkEmbedder(WATERMARK_BITS)


class DeviceModelManager(object):
    """
    Default model loading class, should work for all device classes.
    """

    def __init__(
        self,
        device: Union[torch.device, str],
        swap_device: Optional[Union[torch.device, str]] = None,
    ) -> None:
        """
        Args:
            device (Union[torch.device, str]): The device to use for the model.
        """
        self.device = torch.device(device)
        self.swap_device = (
            torch.device(swap_device) if swap_device is not None else self.device
        )

    def load(self, model: torch.nn.Module) -> None:
        """
        Loads a model to the (swap) device.
        """
        model.to(self.swap_device)

    def autocast(self):
        """
        Context manager that enables autocast for the device if supported.
        """
        if self.device.type not in ("cuda", "cpu"):
            return contextlib.nullcontext()
        return torch.autocast(self.device.type)

    @contextlib.contextmanager
    def use(self, model: torch.nn.Module):
        """
        Context manager that ensures a model is on the correct device during use.
        The default model loader does not perform any swapping, so the model will
        stay on device.
        """
        try:
            model.to(self.device)
            yield
        finally:
            if self.device != self.swap_device:
                model.to(self.swap_device)


class CudaModelManager(DeviceModelManager):
    """
    Device manager that loads a model to a CUDA device, optionally swapping to CPU when not in use.
    """

    @contextlib.contextmanager
    def use(self, model: Union[torch.nn.Module, torch.Tensor]):
        """
        Context manager that ensures a model is on the correct device during use.
        If a swap device was provided, this will move the model to it after use and clear cache.
        """
        model.to(self.device)
        yield
        if self.device != self.swap_device:
            model.to(self.swap_device)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def get_unique_embedder_keys_from_conditioner(conditioner):
    return list({x.input_key for x in conditioner.embedders})


def perform_save_locally(save_path, samples):
    os.makedirs(os.path.join(save_path), exist_ok=True)
    base_count = len(os.listdir(os.path.join(save_path)))
    samples = embed_watermark(samples)
    for sample in samples:
        sample = 255.0 * rearrange(sample.cpu().numpy(), "c h w -> h w c")
        Image.fromarray(sample.astype(np.uint8)).save(
            os.path.join(save_path, f"{base_count:09}.png")
        )
        base_count += 1


def get_model_manager(
    device: Optional[Union[DeviceModelManager, str, torch.device]]
) -> DeviceModelManager:
    if isinstance(device, DeviceModelManager):
        return device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    if device.type == "cuda":
        return CudaModelManager(device=device)
    else:
        return DeviceModelManager(device=device)


class Img2ImgDiscretizationWrapper:
    """
    wraps a discretizer, and prunes the sigmas
    params:
        strength: float between 0.0 and 1.0. 1.0 means full sampling (all sigmas are returned)
    """

    def __init__(self, discretization, strength: float = 1.0):
        self.discretization = discretization
        self.strength = strength
        assert 0.0 <= self.strength <= 1.0

    def __call__(self, *args, **kwargs):
        # sigmas start large first, and decrease then
        sigmas = self.discretization(*args, **kwargs)
        print(f"sigmas after discretization, before pruning img2img: ", sigmas)
        sigmas = torch.flip(sigmas, (0,))
        sigmas = sigmas[: max(int(self.strength * len(sigmas)), 1)]
        print("prune index:", max(int(self.strength * len(sigmas)), 1))
        sigmas = torch.flip(sigmas, (0,))
        print(f"sigmas after pruning: ", sigmas)
        return sigmas


class Txt2NoisyDiscretizationWrapper:
    """
    wraps a discretizer, and prunes the sigmas
    params:
        strength: float between 0.0 and 1.0. 0.0 means full sampling (all sigmas are returned)
    """

    def __init__(self, discretization, strength: float = 0.0, original_steps=None):
        self.discretization = discretization
        self.strength = strength
        self.original_steps = original_steps
        assert 0.0 <= self.strength <= 1.0

    def __call__(self, *args, **kwargs):
        # sigmas start large first, and decrease then
        sigmas = self.discretization(*args, **kwargs)
        print(f"sigmas after discretization, before pruning img2img: ", sigmas)
        sigmas = torch.flip(sigmas, (0,))
        if self.original_steps is None:
            steps = len(sigmas)
        else:
            steps = self.original_steps + 1
        prune_index = max(min(int(self.strength * steps) - 1, steps - 1), 0)
        sigmas = sigmas[prune_index:]
        print("prune index:", prune_index)
        sigmas = torch.flip(sigmas, (0,))
        print(f"sigmas after pruning: ", sigmas)
        return sigmas


def do_sample(
    model,
    sampler,
    value_dict,
    num_samples,
    H,
    W,
    C,
    F,
    force_uc_zero_embeddings: Optional[List] = None,
    batch2model_input: Optional[List] = None,
    return_latents=False,
    filter=None,
    device: Optional[Union[DeviceModelManager, str, torch.device]] = None,
):
    if force_uc_zero_embeddings is None:
        force_uc_zero_embeddings = []
    if batch2model_input is None:
        batch2model_input = []

    device_manager = get_model_manager(device=device)

    with torch.no_grad():
        with device_manager.autocast():
            with model.ema_scope():
                num_samples = [num_samples]
                with device_manager.use(model.conditioner):
                    batch, batch_uc = get_batch(
                        get_unique_embedder_keys_from_conditioner(model.conditioner),
                        value_dict,
                        num_samples,
                    )
                    for key in batch:
                        if isinstance(batch[key], torch.Tensor):
                            print(key, batch[key].shape)
                        elif isinstance(batch[key], list):
                            print(key, [len(l) for l in batch[key]])
                        else:
                            print(key, batch[key])
                    c, uc = model.conditioner.get_unconditional_conditioning(
                        batch,
                        batch_uc=batch_uc,
                        force_uc_zero_embeddings=force_uc_zero_embeddings,
                    )

                for k in c:
                    if not k == "crossattn":
                        c[k], uc[k] = map(
                            lambda y: y[k][: math.prod(num_samples)].to(
                                device_manager.device
                            ),
                            (c, uc),
                        )

                additional_model_inputs = {}
                for k in batch2model_input:
                    additional_model_inputs[k] = batch[k]

                shape = (math.prod(num_samples), C, H // F, W // F)
                randn = torch.randn(shape).to(device_manager.device)

                def denoiser(input, sigma, c):
                    return model.denoiser(
                        model.model, input, sigma, c, **additional_model_inputs
                    )

                with device_manager.use(model.denoiser):
                    with device_manager.use(model.model):
                        samples_z = sampler(denoiser, randn, cond=c, uc=uc)

                with device_manager.use(model.first_stage_model):
                    samples_x = model.decode_first_stage(samples_z)
                    samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)

                if filter is not None:
                    samples = filter(samples)

                if return_latents:
                    return samples, samples_z
                return samples


def get_batch(keys, value_dict, N: Union[List, ListConfig], device="cuda"):
    # Hardcoded demo setups; might undergo some changes in the future

    batch = {}
    batch_uc = {}

    for key in keys:
        if key == "txt":
            batch["txt"] = (
                np.repeat([value_dict["prompt"]], repeats=math.prod(N))
                .reshape(N)
                .tolist()
            )
            batch_uc["txt"] = (
                np.repeat([value_dict["negative_prompt"]], repeats=math.prod(N))
                .reshape(N)
                .tolist()
            )
        elif key == "original_size_as_tuple":
            batch["original_size_as_tuple"] = (
                torch.tensor([value_dict["orig_height"], value_dict["orig_width"]])
                .to(device)
                .repeat(*N, 1)
            )
        elif key == "crop_coords_top_left":
            batch["crop_coords_top_left"] = (
                torch.tensor(
                    [value_dict["crop_coords_top"], value_dict["crop_coords_left"]]
                )
                .to(device)
                .repeat(*N, 1)
            )
        elif key == "aesthetic_score":
            batch["aesthetic_score"] = (
                torch.tensor([value_dict["aesthetic_score"]]).to(device).repeat(*N, 1)
            )
            batch_uc["aesthetic_score"] = (
                torch.tensor([value_dict["negative_aesthetic_score"]])
                .to(device)
                .repeat(*N, 1)
            )

        elif key == "target_size_as_tuple":
            batch["target_size_as_tuple"] = (
                torch.tensor([value_dict["target_height"], value_dict["target_width"]])
                .to(device)
                .repeat(*N, 1)
            )
        else:
            batch[key] = value_dict[key]

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc


def get_input_image_tensor(image: Image.Image, device="cuda"):
    w, h = image.size
    print(f"loaded input image of size ({w}, {h})")
    width, height = map(
        lambda x: x - x % 64, (w, h)
    )  # resize to integer multiple of 64
    image = image.resize((width, height))
    image_array = np.array(image.convert("RGB"))
    image_array = image_array[None].transpose(0, 3, 1, 2)
    image_tensor = torch.from_numpy(image_array).to(dtype=torch.float32) / 127.5 - 1.0
    return image_tensor.to(device)


def do_img2img(
    img,
    model,
    sampler,
    value_dict,
    num_samples,
    force_uc_zero_embeddings=[],
    additional_kwargs={},
    offset_noise_level: float = 0.0,
    return_latents=False,
    skip_encode=False,
    filter=None,
    add_noise=True,
    device: Optional[Union[DeviceModelManager, str, torch.device]] = None,
):
    device_manager = get_model_manager(device)
    with torch.no_grad():
        with device_manager.autocast():
            with model.ema_scope():
                with device_manager.use(model.conditioner):
                    batch, batch_uc = get_batch(
                        get_unique_embedder_keys_from_conditioner(model.conditioner),
                        value_dict,
                        [num_samples],
                    )
                    c, uc = model.conditioner.get_unconditional_conditioning(
                        batch,
                        batch_uc=batch_uc,
                        force_uc_zero_embeddings=force_uc_zero_embeddings,
                    )

                for k in c:
                    c[k], uc[k] = map(
                        lambda y: y[k][:num_samples].to(device_manager.device), (c, uc)
                    )

                for k in additional_kwargs:
                    c[k] = uc[k] = additional_kwargs[k]
                if skip_encode:
                    z = img
                else:
                    with device_manager.use(model.first_stage_model):
                        z = model.encode_first_stage(img)

                noise = torch.randn_like(z)

                sigmas = sampler.discretization(sampler.num_steps)
                sigma = sigmas[0].to(z.device)

                if offset_noise_level > 0.0:
                    noise = noise + offset_noise_level * append_dims(
                        torch.randn(z.shape[0], device=z.device), z.ndim
                    )
                if add_noise:
                    noised_z = z + noise * append_dims(sigma, z.ndim).cuda()
                    noised_z = noised_z / torch.sqrt(
                        1.0 + sigmas[0] ** 2.0
                    )  # Note: hardcoded to DDPM-like scaling. need to generalize later.
                else:
                    noised_z = z / torch.sqrt(1.0 + sigmas[0] ** 2.0)

                def denoiser(x, sigma, c):
                    return model.denoiser(model.model, x, sigma, c)

                with device_manager.use(model.denoiser):
                    with device_manager.use(model.model):
                        samples_z = sampler(denoiser, noised_z, cond=c, uc=uc)

                with device_manager.use(model.first_stage_model):
                    samples_x = model.decode_first_stage(samples_z)
                    samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)

                if filter is not None:
                    samples = filter(samples)

                if return_latents:
                    return samples, samples_z
                return samples
