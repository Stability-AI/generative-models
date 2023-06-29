import torch
import numpy as np
from functools import partial

from ...util import append_zero
from ...modules.diffusionmodules.util import make_beta_schedule


def generate_roughly_equally_spaced_steps(n, m):
    # 0, ..., m - 1
    m -= 1

    # We are getting rid of leading 0 later, so increase steps
    n += 1

    # Calculate the step size
    step = m / (n - 1)

    # Generate the list
    steps_reversed = [int(m - i * step) for i in range(n)]
    steps = steps_reversed[::-1]

    # Get rid of leading 0
    steps = steps[1:]

    return np.array(steps)


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
    ):
        self.num_timesteps = num_timesteps
        betas = make_beta_schedule(
            "linear", num_timesteps, linear_start=linear_start, linear_end=linear_end
        )
        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.to_torch = partial(torch.tensor, dtype=torch.float32)

    def get_sigmas(self, n, device):
        if n < self.num_timesteps:
            timesteps = generate_roughly_equally_spaced_steps(n, self.num_timesteps)
            alphas_cumprod = self.alphas_cumprod[timesteps]
        elif n == self.num_timesteps:
            alphas_cumprod = self.alphas_cumprod
        else:
            raise ValueError

        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        sigmas = to_torch((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
        return torch.flip(sigmas, (0,))
