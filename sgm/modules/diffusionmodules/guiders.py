import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Literal, Optional, Tuple, Union

import torch
from einops import rearrange, repeat

from ...util import append_dims, default

logpy = logging.getLogger(__name__)


class Guider(ABC):
    @abstractmethod
    def __call__(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
        pass

    def prepare_inputs(
        self, x: torch.Tensor, s: float, c: Dict, uc: Dict
    ) -> Tuple[torch.Tensor, float, Dict]:
        pass


class VanillaCFG(Guider):
    def __init__(self, scale: float):
        self.scale = scale

    def __call__(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        x_u, x_c = x.chunk(2)
        x_pred = x_u + self.scale * (x_c - x_u)
        return x_pred

    def prepare_inputs(self, x, s, c, uc):
        c_out = dict()

        for k in c:
            if k in ["vector", "crossattn", "concat"]:
                c_out[k] = torch.cat((uc[k], c[k]), 0)
            else:
                assert c[k] == uc[k]
                c_out[k] = c[k]
        return torch.cat([x] * 2), torch.cat([s] * 2), c_out


class IdentityGuider(Guider):
    def __call__(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
        return x

    def prepare_inputs(
        self, x: torch.Tensor, s: float, c: Dict, uc: Dict
    ) -> Tuple[torch.Tensor, float, Dict]:
        c_out = dict()

        for k in c:
            c_out[k] = c[k]

        return x, s, c_out


class LinearPredictionGuider(Guider):
    def __init__(
        self,
        max_scale: float,
        num_frames: int,
        min_scale: float = 1.0,
        additional_cond_keys: Optional[Union[List[str], str]] = None,
    ):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.num_frames = num_frames
        self.scale = torch.linspace(min_scale, max_scale, num_frames).unsqueeze(0)

        additional_cond_keys = default(additional_cond_keys, [])
        if isinstance(additional_cond_keys, str):
            additional_cond_keys = [additional_cond_keys]
        self.additional_cond_keys = additional_cond_keys

    def __call__(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        x_u, x_c = x.chunk(2)

        x_u = rearrange(x_u, "(b t) ... -> b t ...", t=self.num_frames)
        x_c = rearrange(x_c, "(b t) ... -> b t ...", t=self.num_frames)
        scale = repeat(self.scale, "1 t -> b t", b=x_u.shape[0])
        scale = append_dims(scale, x_u.ndim).to(x_u.device)

        return rearrange(x_u + scale * (x_c - x_u), "b t ... -> (b t) ...")

    def prepare_inputs(
        self, x: torch.Tensor, s: torch.Tensor, c: dict, uc: dict
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        c_out = dict()

        for k in c:
            if k in ["vector", "crossattn", "concat"] + self.additional_cond_keys:
                c_out[k] = torch.cat((uc[k], c[k]), 0)
            else:
                assert c[k] == uc[k]
                c_out[k] = c[k]
        return torch.cat([x] * 2), torch.cat([s] * 2), c_out


class TrianglePredictionGuider(LinearPredictionGuider):
    def __init__(
        self,
        max_scale: float,
        num_frames: int,
        min_scale: float = 1.0,
        period: float | List[float] = 1.0,
        period_fusing: Literal["mean", "multiply", "max"] = "max",
        additional_cond_keys: Optional[Union[List[str], str]] = None,
    ):
        super().__init__(max_scale, num_frames, min_scale, additional_cond_keys)
        values = torch.linspace(0, 1, num_frames)
        # Constructs a triangle wave
        if isinstance(period, float):
            period = [period]

        scales = []
        for p in period:
            scales.append(self.triangle_wave(values, p))

        if period_fusing == "mean":
            scale = sum(scales) / len(period)
        elif period_fusing == "multiply":
            scale = torch.prod(torch.stack(scales), dim=0)
        elif period_fusing == "max":
            scale = torch.max(torch.stack(scales), dim=0).values
        self.scale = (scale * (max_scale - min_scale) + min_scale).unsqueeze(0)

    def triangle_wave(self, values: torch.Tensor, period) -> torch.Tensor:
        return 2 * (values / period - torch.floor(values / period + 0.5)).abs()
