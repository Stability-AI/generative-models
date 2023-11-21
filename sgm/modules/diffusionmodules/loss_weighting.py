from abc import ABC, abstractmethod

import torch


class DiffusionLossWeighting(ABC):
    @abstractmethod
    def __call__(self, sigma: torch.Tensor) -> torch.Tensor:
        pass


class UnitWeighting(DiffusionLossWeighting):
    def __call__(self, sigma: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(sigma, device=sigma.device)


class EDMWeighting(DiffusionLossWeighting):
    def __init__(self, sigma_data: float = 0.5):
        self.sigma_data = sigma_data

    def __call__(self, sigma: torch.Tensor) -> torch.Tensor:
        return (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2


class VWeighting(EDMWeighting):
    def __init__(self):
        super().__init__(sigma_data=1.0)


class EpsWeighting(DiffusionLossWeighting):
    def __call__(self, sigma: torch.Tensor) -> torch.Tensor:
        return sigma**-2.0
