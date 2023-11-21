import logging
from abc import abstractmethod
from typing import Dict, Iterator, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import einsum

from .base import AbstractRegularizer, measure_perplexity

logpy = logging.getLogger(__name__)


class AbstractQuantizer(AbstractRegularizer):
    def __init__(self):
        super().__init__()
        # Define these in your init
        # shape (N,)
        self.used: Optional[torch.Tensor]
        self.re_embed: int
        self.unknown_index: Union[Literal["random"], int]

    def remap_to_used(self, inds: torch.Tensor) -> torch.Tensor:
        assert self.used is not None, "You need to define used indices for remap"
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        match = (inds[:, :, None] == used[None, None, ...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2) < 1
        if self.unknown_index == "random":
            new[unknown] = torch.randint(0, self.re_embed, size=new[unknown].shape).to(
                device=new.device
            )
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds: torch.Tensor) -> torch.Tensor:
        assert self.used is not None, "You need to define used indices for remap"
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]:  # extra token
            inds[inds >= self.used.shape[0]] = 0  # simply set to zero
        back = torch.gather(used[None, :][inds.shape[0] * [0], :], 1, inds)
        return back.reshape(ishape)

    @abstractmethod
    def get_codebook_entry(
        self, indices: torch.Tensor, shape: Optional[Tuple[int, ...]] = None
    ) -> torch.Tensor:
        raise NotImplementedError()

    def get_trainable_parameters(self) -> Iterator[torch.nn.Parameter]:
        yield from self.parameters()


class GumbelQuantizer(AbstractQuantizer):
    """
    credit to @karpathy:
    https://github.com/karpathy/deep-vector-quantization/blob/main/model.py (thanks!)
    Gumbel Softmax trick quantizer
    Categorical Reparameterization with Gumbel-Softmax, Jang et al. 2016
    https://arxiv.org/abs/1611.01144
    """

    def __init__(
        self,
        num_hiddens: int,
        embedding_dim: int,
        n_embed: int,
        straight_through: bool = True,
        kl_weight: float = 5e-4,
        temp_init: float = 1.0,
        remap: Optional[str] = None,
        unknown_index: str = "random",
        loss_key: str = "loss/vq",
    ) -> None:
        super().__init__()

        self.loss_key = loss_key
        self.embedding_dim = embedding_dim
        self.n_embed = n_embed

        self.straight_through = straight_through
        self.temperature = temp_init
        self.kl_weight = kl_weight

        self.proj = nn.Conv2d(num_hiddens, n_embed, 1)
        self.embed = nn.Embedding(n_embed, embedding_dim)

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
        else:
            self.used = None
            self.re_embed = n_embed
        if unknown_index == "extra":
            self.unknown_index = self.re_embed
            self.re_embed = self.re_embed + 1
        else:
            assert unknown_index == "random" or isinstance(
                unknown_index, int
            ), "unknown index needs to be 'random', 'extra' or any integer"
            self.unknown_index = unknown_index  # "random" or "extra" or integer
        if self.remap is not None:
            logpy.info(
                f"Remapping {self.n_embed} indices to {self.re_embed} indices. "
                f"Using {self.unknown_index} for unknown indices."
            )

    def forward(
        self, z: torch.Tensor, temp: Optional[float] = None, return_logits: bool = False
    ) -> Tuple[torch.Tensor, Dict]:
        # force hard = True when we are in eval mode, as we must quantize.
        # actually, always true seems to work
        hard = self.straight_through if self.training else True
        temp = self.temperature if temp is None else temp
        out_dict = {}
        logits = self.proj(z)
        if self.remap is not None:
            # continue only with used logits
            full_zeros = torch.zeros_like(logits)
            logits = logits[:, self.used, ...]

        soft_one_hot = F.gumbel_softmax(logits, tau=temp, dim=1, hard=hard)
        if self.remap is not None:
            # go back to all entries but unused set to zero
            full_zeros[:, self.used, ...] = soft_one_hot
            soft_one_hot = full_zeros
        z_q = einsum("b n h w, n d -> b d h w", soft_one_hot, self.embed.weight)

        # + kl divergence to the prior loss
        qy = F.softmax(logits, dim=1)
        diff = (
            self.kl_weight
            * torch.sum(qy * torch.log(qy * self.n_embed + 1e-10), dim=1).mean()
        )
        out_dict[self.loss_key] = diff

        ind = soft_one_hot.argmax(dim=1)
        out_dict["indices"] = ind
        if self.remap is not None:
            ind = self.remap_to_used(ind)

        if return_logits:
            out_dict["logits"] = logits

        return z_q, out_dict

    def get_codebook_entry(self, indices, shape):
        # TODO: shape not yet optional
        b, h, w, c = shape
        assert b * h * w == indices.shape[0]
        indices = rearrange(indices, "(b h w) -> b h w", b=b, h=h, w=w)
        if self.remap is not None:
            indices = self.unmap_to_all(indices)
        one_hot = (
            F.one_hot(indices, num_classes=self.n_embed).permute(0, 3, 1, 2).float()
        )
        z_q = einsum("b n h w, n d -> b d h w", one_hot, self.embed.weight)
        return z_q


class VectorQuantizer(AbstractQuantizer):
    """
    ____________________________________________
    Discretization bottleneck part of the VQ-VAE.
    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term,
        beta * ||z_e(x)-sg[e]||^2
    _____________________________________________
    """

    def __init__(
        self,
        n_e: int,
        e_dim: int,
        beta: float = 0.25,
        remap: Optional[str] = None,
        unknown_index: str = "random",
        sane_index_shape: bool = False,
        log_perplexity: bool = False,
        embedding_weight_norm: bool = False,
        loss_key: str = "loss/vq",
    ):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.loss_key = loss_key

        if not embedding_weight_norm:
            self.embedding = nn.Embedding(self.n_e, self.e_dim)
            self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        else:
            self.embedding = torch.nn.utils.weight_norm(
                nn.Embedding(self.n_e, self.e_dim), dim=1
            )

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
        else:
            self.used = None
            self.re_embed = n_e
        if unknown_index == "extra":
            self.unknown_index = self.re_embed
            self.re_embed = self.re_embed + 1
        else:
            assert unknown_index == "random" or isinstance(
                unknown_index, int
            ), "unknown index needs to be 'random', 'extra' or any integer"
            self.unknown_index = unknown_index  # "random" or "extra" or integer
        if self.remap is not None:
            logpy.info(
                f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                f"Using {self.unknown_index} for unknown indices."
            )

        self.sane_index_shape = sane_index_shape
        self.log_perplexity = log_perplexity

    def forward(
        self,
        z: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        do_reshape = z.ndim == 4
        if do_reshape:
            #     # reshape z -> (batch, height, width, channel) and flatten
            z = rearrange(z, "b c h w -> b h w c").contiguous()

        else:
            assert z.ndim < 4, "No reshaping strategy for inputs > 4 dimensions defined"
            z = z.contiguous()

        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2
            * torch.einsum(
                "bd,dn->bn", z_flattened, rearrange(self.embedding.weight, "n d -> d n")
            )
        )

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        loss_dict = {}
        if self.log_perplexity:
            perplexity, cluster_usage = measure_perplexity(
                min_encoding_indices.detach(), self.n_e
            )
            loss_dict.update({"perplexity": perplexity, "cluster_usage": cluster_usage})

        # compute loss for embedding
        loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + torch.mean(
            (z_q - z.detach()) ** 2
        )
        loss_dict[self.loss_key] = loss

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        if do_reshape:
            z_q = rearrange(z_q, "b h w c -> b c h w").contiguous()

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(
                z.shape[0], -1
            )  # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1, 1)  # flatten

        if self.sane_index_shape:
            if do_reshape:
                min_encoding_indices = min_encoding_indices.reshape(
                    z_q.shape[0], z_q.shape[2], z_q.shape[3]
                )
            else:
                min_encoding_indices = rearrange(
                    min_encoding_indices, "(b s) 1 -> b s", b=z_q.shape[0]
                )

        loss_dict["min_encoding_indices"] = min_encoding_indices

        return z_q, loss_dict

    def get_codebook_entry(
        self, indices: torch.Tensor, shape: Optional[Tuple[int, ...]] = None
    ) -> torch.Tensor:
        # shape specifying (batch, height, width, channel)
        if self.remap is not None:
            assert shape is not None, "Need to give shape for remap"
            indices = indices.reshape(shape[0], -1)  # add batch axis
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1)  # flatten again

        # get quantized latent vectors
        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q


class EmbeddingEMA(nn.Module):
    def __init__(self, num_tokens, codebook_dim, decay=0.99, eps=1e-5):
        super().__init__()
        self.decay = decay
        self.eps = eps
        weight = torch.randn(num_tokens, codebook_dim)
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.cluster_size = nn.Parameter(torch.zeros(num_tokens), requires_grad=False)
        self.embed_avg = nn.Parameter(weight.clone(), requires_grad=False)
        self.update = True

    def forward(self, embed_id):
        return F.embedding(embed_id, self.weight)

    def cluster_size_ema_update(self, new_cluster_size):
        self.cluster_size.data.mul_(self.decay).add_(
            new_cluster_size, alpha=1 - self.decay
        )

    def embed_avg_ema_update(self, new_embed_avg):
        self.embed_avg.data.mul_(self.decay).add_(new_embed_avg, alpha=1 - self.decay)

    def weight_update(self, num_tokens):
        n = self.cluster_size.sum()
        smoothed_cluster_size = (
            (self.cluster_size + self.eps) / (n + num_tokens * self.eps) * n
        )
        # normalize embedding average with smoothed cluster size
        embed_normalized = self.embed_avg / smoothed_cluster_size.unsqueeze(1)
        self.weight.data.copy_(embed_normalized)


class EMAVectorQuantizer(AbstractQuantizer):
    def __init__(
        self,
        n_embed: int,
        embedding_dim: int,
        beta: float,
        decay: float = 0.99,
        eps: float = 1e-5,
        remap: Optional[str] = None,
        unknown_index: str = "random",
        loss_key: str = "loss/vq",
    ):
        super().__init__()
        self.codebook_dim = embedding_dim
        self.num_tokens = n_embed
        self.beta = beta
        self.loss_key = loss_key

        self.embedding = EmbeddingEMA(self.num_tokens, self.codebook_dim, decay, eps)

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
        else:
            self.used = None
            self.re_embed = n_embed
        if unknown_index == "extra":
            self.unknown_index = self.re_embed
            self.re_embed = self.re_embed + 1
        else:
            assert unknown_index == "random" or isinstance(
                unknown_index, int
            ), "unknown index needs to be 'random', 'extra' or any integer"
            self.unknown_index = unknown_index  # "random" or "extra" or integer
        if self.remap is not None:
            logpy.info(
                f"Remapping {self.n_embed} indices to {self.re_embed} indices. "
                f"Using {self.unknown_index} for unknown indices."
            )

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        # reshape z -> (batch, height, width, channel) and flatten
        # z, 'b c h w -> b h w c'
        z = rearrange(z, "b c h w -> b h w c")
        z_flattened = z.reshape(-1, self.codebook_dim)

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = (
            z_flattened.pow(2).sum(dim=1, keepdim=True)
            + self.embedding.weight.pow(2).sum(dim=1)
            - 2 * torch.einsum("bd,nd->bn", z_flattened, self.embedding.weight)
        )  # 'n d -> d n'

        encoding_indices = torch.argmin(d, dim=1)

        z_q = self.embedding(encoding_indices).view(z.shape)
        encodings = F.one_hot(encoding_indices, self.num_tokens).type(z.dtype)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        if self.training and self.embedding.update:
            # EMA cluster size
            encodings_sum = encodings.sum(0)
            self.embedding.cluster_size_ema_update(encodings_sum)
            # EMA embedding average
            embed_sum = encodings.transpose(0, 1) @ z_flattened
            self.embedding.embed_avg_ema_update(embed_sum)
            # normalize embed_avg and update weight
            self.embedding.weight_update(self.num_tokens)

        # compute loss for embedding
        loss = self.beta * F.mse_loss(z_q.detach(), z)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        # z_q, 'b h w c -> b c h w'
        z_q = rearrange(z_q, "b h w c -> b c h w")

        out_dict = {
            self.loss_key: loss,
            "encodings": encodings,
            "encoding_indices": encoding_indices,
            "perplexity": perplexity,
        }

        return z_q, out_dict


class VectorQuantizerWithInputProjection(VectorQuantizer):
    def __init__(
        self,
        input_dim: int,
        n_codes: int,
        codebook_dim: int,
        beta: float = 1.0,
        output_dim: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(n_codes, codebook_dim, beta, **kwargs)
        self.proj_in = nn.Linear(input_dim, codebook_dim)
        self.output_dim = output_dim
        if output_dim is not None:
            self.proj_out = nn.Linear(codebook_dim, output_dim)
        else:
            self.proj_out = nn.Identity()

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        rearr = False
        in_shape = z.shape

        if z.ndim > 3:
            rearr = self.output_dim is not None
            z = rearrange(z, "b c ... -> b (...) c")
        z = self.proj_in(z)
        z_q, loss_dict = super().forward(z)

        z_q = self.proj_out(z_q)
        if rearr:
            if len(in_shape) == 4:
                z_q = rearrange(z_q, "b (h w) c -> b c h w ", w=in_shape[-1])
            elif len(in_shape) == 5:
                z_q = rearrange(
                    z_q, "b (t h w) c -> b c t h w ", w=in_shape[-1], h=in_shape[-2]
                )
            else:
                raise NotImplementedError(
                    f"rearranging not available for {len(in_shape)}-dimensional input."
                )

        return z_q, loss_dict
