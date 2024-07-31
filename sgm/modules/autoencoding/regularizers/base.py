from abc import abstractmethod
from typing import Any, Tuple

import torch
import torch.nn.functional as F
from torch import nn


class AbstractRegularizer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        raise NotImplementedError()

    @abstractmethod
    def get_trainable_parameters(self) -> Any:
        raise NotImplementedError()


class IdentityRegularizer(AbstractRegularizer):
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        return z, dict()

    def get_trainable_parameters(self) -> Any:
        yield from ()


def measure_perplexity(
    predicted_indices: torch.Tensor, num_centroids: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    # src: https://github.com/karpathy/deep-vector-quantization/blob/main/model.py
    # eval cluster perplexity. when perplexity == num_embeddings then all clusters are used exactly equally
    encodings = (
        F.one_hot(predicted_indices, num_centroids).float().reshape(-1, num_centroids)
    )
    avg_probs = encodings.mean(0)
    perplexity = (-(avg_probs * torch.log(avg_probs + 1e-10)).sum()).exp()
    cluster_use = torch.sum(avg_probs > 0)
    return perplexity, cluster_use
