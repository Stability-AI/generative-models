import torch
import numpy as np
from functools import partial

from ldm.util import append_zero
from ldm.modules.diffusionmodules.util import make_beta_schedule


class Discretization:
    def __call__(self, n, do_append_zero=True, device="cuda", flip=False):
        sigmas = self.get_sigmas(n, device)
        sigmas = append_zero(sigmas) if do_append_zero else sigmas
        return sigmas if not flip else torch.flip(sigmas, (0,))


class EDMDiscretization(Discretization):
    def __init__(self, sigma_min=0.02, sigma_max=80.0, rho=7.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho

    def get_sigmas(self, n, device):
        ramp = torch.linspace(0, 1, n, device=device)
        min_inv_rho = self.sigma_min ** (1 / self.rho)
        max_inv_rho = self.sigma_max ** (1 / self.rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** self.rho
        return sigmas


class LegacyDDPMDiscretization(Discretization):
    def __init__(
        self,
        linear_start=0.00085,
        linear_end=0.0120,
        num_timesteps=1000,
        legacy_range=True,
    ):
        self.num_timesteps = num_timesteps
        betas = make_beta_schedule(
            "linear", num_timesteps, linear_start=linear_start, linear_end=linear_end
        )
        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.to_torch = partial(torch.tensor, dtype=torch.float32)
        self.legacy_range = legacy_range

    def get_sigmas(self, n, device):
        if n < self.num_timesteps:
            c = self.num_timesteps // n

            if self.legacy_range:
                timesteps = np.asarray(list(range(0, self.num_timesteps, c)))
                timesteps += 1  # Legacy LDM Hack
            else:
                timesteps = np.asarray(list(range(0, self.num_timesteps + 1, c)))
                timesteps -= 1
                timesteps = timesteps[1:]

            alphas_cumprod = self.alphas_cumprod[timesteps]
        else:
            alphas_cumprod = self.alphas_cumprod

        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        sigmas = to_torch((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
        return torch.flip(sigmas, (0,))
