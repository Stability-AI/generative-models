import torch
import torch.nn as nn

from ....util import default, instantiate_from_config
from ..lpips.loss.lpips import LPIPS


class LatentLPIPS(nn.Module):
    def __init__(
        self,
        decoder_config,
        perceptual_weight=1.0,
        latent_weight=1.0,
        scale_input_to_tgt_size=False,
        scale_tgt_to_input_size=False,
        perceptual_weight_on_inputs=0.0,
    ):
        super().__init__()
        self.scale_input_to_tgt_size = scale_input_to_tgt_size
        self.scale_tgt_to_input_size = scale_tgt_to_input_size
        self.init_decoder(decoder_config)
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        self.latent_weight = latent_weight
        self.perceptual_weight_on_inputs = perceptual_weight_on_inputs

    def init_decoder(self, config):
        self.decoder = instantiate_from_config(config)
        if hasattr(self.decoder, "encoder"):
            del self.decoder.encoder

    def forward(self, latent_inputs, latent_predictions, image_inputs, split="train"):
        log = dict()
        loss = (latent_inputs - latent_predictions) ** 2
        log[f"{split}/latent_l2_loss"] = loss.mean().detach()
        image_reconstructions = None
        if self.perceptual_weight > 0.0:
            image_reconstructions = self.decoder.decode(latent_predictions)
            image_targets = self.decoder.decode(latent_inputs)
            perceptual_loss = self.perceptual_loss(
                image_targets.contiguous(), image_reconstructions.contiguous()
            )
            loss = (
                self.latent_weight * loss.mean()
                + self.perceptual_weight * perceptual_loss.mean()
            )
            log[f"{split}/perceptual_loss"] = perceptual_loss.mean().detach()

        if self.perceptual_weight_on_inputs > 0.0:
            image_reconstructions = default(
                image_reconstructions, self.decoder.decode(latent_predictions)
            )
            if self.scale_input_to_tgt_size:
                image_inputs = torch.nn.functional.interpolate(
                    image_inputs,
                    image_reconstructions.shape[2:],
                    mode="bicubic",
                    antialias=True,
                )
            elif self.scale_tgt_to_input_size:
                image_reconstructions = torch.nn.functional.interpolate(
                    image_reconstructions,
                    image_inputs.shape[2:],
                    mode="bicubic",
                    antialias=True,
                )

            perceptual_loss2 = self.perceptual_loss(
                image_inputs.contiguous(), image_reconstructions.contiguous()
            )
            loss = loss + self.perceptual_weight_on_inputs * perceptual_loss2.mean()
            log[f"{split}/perceptual_loss_on_inputs"] = perceptual_loss2.mean().detach()
        return loss, log
