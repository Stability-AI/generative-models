from functools import partial
from typing import List, Optional, Union

from einops import rearrange

from ...modules.diffusionmodules.openaimodel import *
from ...modules.video_attention import SpatialVideoTransformer
from ...modules.spacetime_attention import (
    BasicTransformerTimeMixBlock,
    PostHocSpatialTransformerWithTimeMixing,
    PostHocSpatialTransformerWithTimeMixingAndMotion,
)
from ...util import default
from .util import AlphaBlender, get_alpha


class VideoResBlock(ResBlock):
    def __init__(
        self,
        channels: int,
        emb_channels: int,
        dropout: float,
        video_kernel_size: Union[int, List[int]] = 3,
        merge_strategy: str = "fixed",
        merge_factor: float = 0.5,
        out_channels: Optional[int] = None,
        use_conv: bool = False,
        use_scale_shift_norm: bool = False,
        dims: int = 2,
        use_checkpoint: bool = False,
        up: bool = False,
        down: bool = False,
    ):
        super().__init__(
            channels,
            emb_channels,
            dropout,
            out_channels=out_channels,
            use_conv=use_conv,
            use_scale_shift_norm=use_scale_shift_norm,
            dims=dims,
            use_checkpoint=use_checkpoint,
            up=up,
            down=down,
        )

        self.time_stack = ResBlock(
            default(out_channels, channels),
            emb_channels,
            dropout=dropout,
            dims=3,
            out_channels=default(out_channels, channels),
            use_scale_shift_norm=False,
            use_conv=False,
            up=False,
            down=False,
            kernel_size=video_kernel_size,
            use_checkpoint=use_checkpoint,
            exchange_temb_dims=True,
        )
        self.time_mixer = AlphaBlender(
            alpha=merge_factor,
            merge_strategy=merge_strategy,
            rearrange_pattern="b t -> b 1 t 1 1",
        )

    def forward(
        self,
        x: th.Tensor,
        emb: th.Tensor,
        num_video_frames: int,
        image_only_indicator: Optional[th.Tensor] = None,
    ) -> th.Tensor:
        x = super().forward(x, emb)

        x_mix = rearrange(x, "(b t) c h w -> b c t h w", t=num_video_frames)
        x = rearrange(x, "(b t) c h w -> b c t h w", t=num_video_frames)

        x = self.time_stack(
            x, rearrange(emb, "(b t) ... -> b t ...", t=num_video_frames)
        )
        x = self.time_mixer(
            x_spatial=x_mix, x_temporal=x, image_only_indicator=image_only_indicator
        )
        x = rearrange(x, "b c t h w -> (b t) c h w")
        return x


class VideoUNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        out_channels: int,
        num_res_blocks: int,
        attention_resolutions: int,
        dropout: float = 0.0,
        channel_mult: List[int] = (1, 2, 4, 8),
        conv_resample: bool = True,
        dims: int = 2,
        num_classes: Optional[int] = None,
        use_checkpoint: bool = False,
        num_heads: int = -1,
        num_head_channels: int = -1,
        num_heads_upsample: int = -1,
        use_scale_shift_norm: bool = False,
        resblock_updown: bool = False,
        transformer_depth: Union[List[int], int] = 1,
        transformer_depth_middle: Optional[int] = None,
        context_dim: Optional[int] = None,
        time_downup: bool = False,
        time_context_dim: Optional[int] = None,
        extra_ff_mix_layer: bool = False,
        use_spatial_context: bool = False,
        merge_strategy: str = "fixed",
        merge_factor: float = 0.5,
        spatial_transformer_attn_type: str = "softmax",
        video_kernel_size: Union[int, List[int]] = 3,
        use_linear_in_transformer: bool = False,
        adm_in_channels: Optional[int] = None,
        disable_temporal_crossattention: bool = False,
        max_ddpm_temb_period: int = 10000,
    ):
        super().__init__()
        assert context_dim is not None

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1

        if num_head_channels == -1:
            assert num_heads != -1

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        if isinstance(transformer_depth, int):
            transformer_depth = len(channel_mult) * [transformer_depth]
        transformer_depth_middle = default(
            transformer_depth_middle, transformer_depth[-1]
        )

        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            if isinstance(self.num_classes, int):
                self.label_emb = nn.Embedding(num_classes, time_embed_dim)
            elif self.num_classes == "continuous":
                print("setting up linear c_adm embedding layer")
                self.label_emb = nn.Linear(1, time_embed_dim)
            elif self.num_classes == "timestep":
                self.label_emb = nn.Sequential(
                    Timestep(model_channels),
                    nn.Sequential(
                        linear(model_channels, time_embed_dim),
                        nn.SiLU(),
                        linear(time_embed_dim, time_embed_dim),
                    ),
                )

            elif self.num_classes == "sequential":
                assert adm_in_channels is not None
                self.label_emb = nn.Sequential(
                    nn.Sequential(
                        linear(adm_in_channels, time_embed_dim),
                        nn.SiLU(),
                        linear(time_embed_dim, time_embed_dim),
                    )
                )
            else:
                raise ValueError()

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1

        def get_attention_layer(
            ch,
            num_heads,
            dim_head,
            depth=1,
            context_dim=None,
            use_checkpoint=False,
            disabled_sa=False,
        ):
            return SpatialVideoTransformer(
                ch,
                num_heads,
                dim_head,
                depth=depth,
                context_dim=context_dim,
                time_context_dim=time_context_dim,
                dropout=dropout,
                ff_in=extra_ff_mix_layer,
                use_spatial_context=use_spatial_context,
                merge_strategy=merge_strategy,
                merge_factor=merge_factor,
                checkpoint=use_checkpoint,
                use_linear=use_linear_in_transformer,
                attn_mode=spatial_transformer_attn_type,
                disable_self_attn=disabled_sa,
                disable_temporal_crossattention=disable_temporal_crossattention,
                max_time_embed_period=max_ddpm_temb_period,
            )

        def get_resblock(
            merge_factor,
            merge_strategy,
            video_kernel_size,
            ch,
            time_embed_dim,
            dropout,
            out_ch,
            dims,
            use_checkpoint,
            use_scale_shift_norm,
            down=False,
            up=False,
        ):
            return VideoResBlock(
                merge_factor=merge_factor,
                merge_strategy=merge_strategy,
                video_kernel_size=video_kernel_size,
                channels=ch,
                emb_channels=time_embed_dim,
                dropout=dropout,
                out_channels=out_ch,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                down=down,
                up=up,
            )

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    get_resblock(
                        merge_factor=merge_factor,
                        merge_strategy=merge_strategy,
                        video_kernel_size=video_kernel_size,
                        ch=ch,
                        time_embed_dim=time_embed_dim,
                        dropout=dropout,
                        out_ch=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels

                    layers.append(
                        get_attention_layer(
                            ch,
                            num_heads,
                            dim_head,
                            depth=transformer_depth[level],
                            context_dim=context_dim,
                            use_checkpoint=use_checkpoint,
                            disabled_sa=False,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                ds *= 2
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        get_resblock(
                            merge_factor=merge_factor,
                            merge_strategy=merge_strategy,
                            video_kernel_size=video_kernel_size,
                            ch=ch,
                            time_embed_dim=time_embed_dim,
                            dropout=dropout,
                            out_ch=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch,
                            conv_resample,
                            dims=dims,
                            out_channels=out_ch,
                            third_down=time_downup,
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)

                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels

        self.middle_block = TimestepEmbedSequential(
            get_resblock(
                merge_factor=merge_factor,
                merge_strategy=merge_strategy,
                video_kernel_size=video_kernel_size,
                ch=ch,
                time_embed_dim=time_embed_dim,
                out_ch=None,
                dropout=dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            get_attention_layer(
                ch,
                num_heads,
                dim_head,
                depth=transformer_depth_middle,
                context_dim=context_dim,
                use_checkpoint=use_checkpoint,
            ),
            get_resblock(
                merge_factor=merge_factor,
                merge_strategy=merge_strategy,
                video_kernel_size=video_kernel_size,
                ch=ch,
                out_ch=None,
                time_embed_dim=time_embed_dim,
                dropout=dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    get_resblock(
                        merge_factor=merge_factor,
                        merge_strategy=merge_strategy,
                        video_kernel_size=video_kernel_size,
                        ch=ch + ich,
                        time_embed_dim=time_embed_dim,
                        dropout=dropout,
                        out_ch=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels

                    layers.append(
                        get_attention_layer(
                            ch,
                            num_heads,
                            dim_head,
                            depth=transformer_depth[level],
                            context_dim=context_dim,
                            use_checkpoint=use_checkpoint,
                            disabled_sa=False,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    ds //= 2
                    layers.append(
                        get_resblock(
                            merge_factor=merge_factor,
                            merge_strategy=merge_strategy,
                            video_kernel_size=video_kernel_size,
                            ch=ch,
                            time_embed_dim=time_embed_dim,
                            dropout=dropout,
                            out_ch=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(
                            ch,
                            conv_resample,
                            dims=dims,
                            out_channels=out_ch,
                            third_up=time_downup,
                        )
                    )

                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )

    def forward(
        self,
        x: th.Tensor,
        timesteps: th.Tensor,
        context: Optional[th.Tensor] = None,
        y: Optional[th.Tensor] = None,
        time_context: Optional[th.Tensor] = None,
        num_video_frames: Optional[int] = None,
        image_only_indicator: Optional[th.Tensor] = None,
    ):
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional -> no, relax this TODO"
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            assert y.shape[0] == x.shape[0]
            emb = emb + self.label_emb(y)

        h = x
        for module in self.input_blocks:
            h = module(
                h,
                emb,
                context=context,
                image_only_indicator=image_only_indicator,
                time_context=time_context,
                num_video_frames=num_video_frames,
            )
            hs.append(h)
        h = self.middle_block(
            h,
            emb,
            context=context,
            image_only_indicator=image_only_indicator,
            time_context=time_context,
            num_video_frames=num_video_frames,
        )
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(
                h,
                emb,
                context=context,
                image_only_indicator=image_only_indicator,
                time_context=time_context,
                num_video_frames=num_video_frames,
            )
        h = h.type(x.dtype)
        return self.out(h)


class PostHocAttentionBlockWithTimeMixing(AttentionBlock):
    def __init__(
        self,
        in_channels: int,
        n_heads: int,
        d_head: int,
        use_checkpoint: bool = False,
        use_new_attention_order: bool = False,
        dropout: float = 0.0,
        use_spatial_context: bool = False,
        merge_strategy: bool = "fixed",
        merge_factor: float = 0.5,
        apply_sigmoid_to_merge: bool = True,
        ff_in: bool = False,
        attn_mode: str = "softmax",
        disable_temporal_crossattention: bool = False,
    ):
        super().__init__(
            in_channels,
            n_heads,
            d_head,
            use_checkpoint=use_checkpoint,
            use_new_attention_order=use_new_attention_order,
        )
        inner_dim = n_heads * d_head

        self.time_mix_blocks = nn.ModuleList(
            [
                BasicTransformerTimeMixBlock(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    checkpoint=use_checkpoint,
                    ff_in=ff_in,
                    attn_mode=attn_mode,
                    disable_temporal_crossattention=disable_temporal_crossattention,
                )
            ]
        )
        self.in_channels = in_channels

        time_embed_dim = self.in_channels * 4
        self.time_mix_time_embed = nn.Sequential(
            linear(self.in_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, self.in_channels),
        )

        self.use_spatial_context = use_spatial_context

        if merge_strategy == "fixed":
            self.register_buffer("mix_factor", th.Tensor([merge_factor]))
        elif merge_strategy == "learned" or merge_strategy == "learned_with_images":
            self.register_parameter(
                "mix_factor", th.nn.Parameter(th.Tensor([merge_factor]))
            )
        elif merge_strategy == "fixed_with_images":
            self.mix_factor = None
        else:
            raise ValueError(f"unknown merge strategy {merge_strategy}")

        self.get_alpha_fn = functools.partial(
            get_alpha,
            merge_strategy,
            self.mix_factor,
            apply_sigmoid=apply_sigmoid_to_merge,
        )

    def forward(
        self,
        x: th.Tensor,
        context: Optional[th.Tensor] = None,
        # cam: Optional[th.Tensor] = None,
        time_context: Optional[th.Tensor] = None,
        timesteps: Optional[int] = None,
        image_only_indicator: Optional[th.Tensor] = None,
        conv_view: Optional[th.Tensor] = None,
        conv_motion: Optional[th.Tensor] = None,
    ):
        if time_context is not None:
            raise NotImplementedError

        _, _, h, w = x.shape
        if exists(context):
            context = rearrange(context, "b t ... -> (b t) ...")
        if self.use_spatial_context:
            time_context = repeat(context[:, 0], "b ... -> (b n) ...", n=h * w)

        x = super().forward(
            x,
        )

        x = rearrange(x, "b c h w -> b (h w) c")
        x_mix = x

        num_frames = th.arange(timesteps, device=x.device)
        num_frames = repeat(num_frames, "t -> b t", b=x.shape[0] // timesteps)
        num_frames = rearrange(num_frames, "b t -> (b t)")
        t_emb = timestep_embedding(num_frames, self.in_channels, repeat_only=False)
        emb = self.time_mix_time_embed(t_emb)
        emb = emb[:, None, :]
        x_mix = x_mix + emb

        x_mix = self.time_mix_blocks[0](
            x_mix, context=time_context, timesteps=timesteps
        )

        alpha = self.get_alpha_fn(image_only_indicator=image_only_indicator)
        x = alpha * x + (1.0 - alpha) * x_mix
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        return x


class PostHocResBlockWithTime(ResBlock):
    def __init__(
        self,
        channels: int,
        emb_channels: int,
        dropout: float,
        time_kernel_size: Union[int, List[int]] = 3,
        merge_strategy: bool = "fixed",
        merge_factor: float = 0.5,
        apply_sigmoid_to_merge: bool = True,
        out_channels: Optional[int] = None,
        use_conv: bool = False,
        use_scale_shift_norm: bool = False,
        dims: int = 2,
        use_checkpoint: bool = False,
        up: bool = False,
        down: bool = False,
        time_mix_legacy: bool = True,
        replicate_bug: bool = False,
    ):
        super().__init__(
            channels,
            emb_channels,
            dropout,
            out_channels=out_channels,
            use_conv=use_conv,
            use_scale_shift_norm=use_scale_shift_norm,
            dims=dims,
            use_checkpoint=use_checkpoint,
            up=up,
            down=down,
        )

        self.time_mix_blocks = ResBlock(
            default(out_channels, channels),
            emb_channels,
            dropout=dropout,
            dims=3,
            out_channels=default(out_channels, channels),
            use_scale_shift_norm=False,
            use_conv=False,
            up=False,
            down=False,
            kernel_size=time_kernel_size,
            use_checkpoint=use_checkpoint,
            exchange_temb_dims=True,
        )
        self.time_mix_legacy = time_mix_legacy
        if self.time_mix_legacy:
            if merge_strategy == "fixed":
                self.register_buffer("mix_factor", th.Tensor([merge_factor]))
            elif merge_strategy == "learned" or merge_strategy == "learned_with_images":
                self.register_parameter(
                    "mix_factor", th.nn.Parameter(th.Tensor([merge_factor]))
                )
            elif merge_strategy == "fixed_with_images":
                self.mix_factor = None
            else:
                raise ValueError(f"unknown merge strategy {merge_strategy}")

            self.get_alpha_fn = functools.partial(
                get_alpha,
                merge_strategy,
                self.mix_factor,
                apply_sigmoid=apply_sigmoid_to_merge,
            )
        else:
            if False: # replicate_bug:
                logpy.warning(
                    "*****************************************************************************************\n"
                    "GRAVE WARNING: YOU'RE USING THE BUGGY LEGACY ALPHABLENDER!!! ARE YOU SURE YOU WANT THIS?!\n"
                    "*****************************************************************************************"
                )
                self.time_mixer = LegacyAlphaBlenderWithBug(
                    alpha=merge_factor,
                    merge_strategy=merge_strategy,
                    rearrange_pattern="b t -> b 1 t 1 1",
                )
            else:
                self.time_mixer = AlphaBlender(
                    alpha=merge_factor,
                    merge_strategy=merge_strategy,
                    rearrange_pattern="b t -> b 1 t 1 1",
                )

    def forward(
        self,
        x: th.Tensor,
        emb: th.Tensor,
        num_video_frames: int,
        image_only_indicator: Optional[th.Tensor] = None,
        cond_view: Optional[th.Tensor] = None,
        cond_motion: Optional[th.Tensor] = None,
    ) -> th.Tensor:
        x = super().forward(x, emb)

        x_mix = rearrange(x, "(b t) c h w -> b c t h w", t=num_video_frames)
        x = rearrange(x, "(b t) c h w -> b c t h w", t=num_video_frames)

        x = self.time_mix_blocks(
            x, rearrange(emb, "(b t) ... -> b t ...", t=num_video_frames)
        )

        if self.time_mix_legacy:
            alpha = self.get_alpha_fn(image_only_indicator=image_only_indicator*0.0)
            x = alpha.to(x.dtype) * x + (1.0 - alpha).to(x.dtype) * x_mix
        else:
            x = self.time_mixer(
                x_spatial=x_mix, x_temporal=x, image_only_indicator=image_only_indicator*0.0
            )
        x = rearrange(x, "b c t h w -> (b t) c h w")
        return x


class SpatialUNetModelWithTime(nn.Module):
    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        out_channels: int,
        num_res_blocks: int,
        attention_resolutions: int,
        dropout: float = 0.0,
        channel_mult: List[int] = (1, 2, 4, 8),
        conv_resample: bool = True,
        dims: int = 2,
        num_classes: Optional[int] = None,
        use_checkpoint: bool = False,
        num_heads: int = -1,
        num_head_channels: int = -1,
        num_heads_upsample: int = -1,
        use_scale_shift_norm: bool = False,
        resblock_updown: bool = False,
        use_new_attention_order: bool = False,
        use_spatial_transformer: bool = False,
        transformer_depth: Union[List[int], int] = 1,
        transformer_depth_middle: Optional[int] = None,
        context_dim: Optional[int] = None,
        time_downup: bool = False,
        time_context_dim: Optional[int] = None,
        view_context_dim: Optional[int] = None,
        motion_context_dim: Optional[int] = None,
        extra_ff_mix_layer: bool = False,
        use_spatial_context: bool = False,
        time_block_merge_strategy: str = "fixed",
        time_block_merge_factor: float = 0.5,
        view_block_merge_factor: float = 0.5,
        motion_block_merge_factor: float = 0.5,
        spatial_transformer_attn_type: str = "softmax",
        time_kernel_size: Union[int, List[int]] = 3,
        use_linear_in_transformer: bool = False,
        legacy: bool = True,
        adm_in_channels: Optional[int] = None,
        use_temporal_resblock: bool = True,
        disable_temporal_crossattention: bool = False,
        time_mix_legacy: bool = True,
        max_ddpm_temb_period: int = 10000,
        replicate_time_mix_bug: bool = False,
        use_motion_attention: bool = False,
        use_camera_emb: bool = False,
        use_3d_attention: bool = False,
        separate_motion_merge_factor: bool = False,
    ):
        super().__init__()

        if use_spatial_transformer:
            assert context_dim is not None

        if context_dim is not None:
            assert use_spatial_transformer

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1

        if num_head_channels == -1:
            assert num_heads != -1

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        if isinstance(transformer_depth, int):
            transformer_depth = len(channel_mult) * [transformer_depth]
        transformer_depth_middle = default(
            transformer_depth_middle, transformer_depth[-1]
        )

        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.use_temporal_resblocks = use_temporal_resblock

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            if isinstance(self.num_classes, int):
                self.label_emb = nn.Embedding(num_classes, time_embed_dim)
            elif self.num_classes == "continuous":
                print("setting up linear c_adm embedding layer")
                self.label_emb = nn.Linear(1, time_embed_dim)
            elif self.num_classes == "timestep":
                self.label_emb = nn.Sequential(
                    Timestep(model_channels),
                    nn.Sequential(
                        linear(model_channels, time_embed_dim),
                        nn.SiLU(),
                        linear(time_embed_dim, time_embed_dim),
                    ),
                )

            elif self.num_classes == "sequential":
                assert adm_in_channels is not None
                self.label_emb = nn.Sequential(
                    nn.Sequential(
                        linear(adm_in_channels, time_embed_dim),
                        nn.SiLU(),
                        linear(time_embed_dim, time_embed_dim),
                    )
                )
            else:
                raise ValueError()

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1

        def get_attention_layer(
            ch,
            num_heads,
            dim_head,
            depth=1,
            context_dim=None,
            use_checkpoint=False,
            disabled_sa=False,
        ):
            if not use_spatial_transformer:
                return PostHocAttentionBlockWithTimeMixing(
                    ch,
                    num_heads,
                    dim_head,
                    use_checkpoint=use_checkpoint,
                    use_new_attention_order=use_new_attention_order,
                    dropout=dropout,
                    ff_in=extra_ff_mix_layer,
                    use_spatial_context=use_spatial_context,
                    merge_strategy=time_block_merge_strategy,
                    merge_factor=time_block_merge_factor,
                    attn_mode=spatial_transformer_attn_type,
                    disable_temporal_crossattention=disable_temporal_crossattention,
                )

            elif use_motion_attention:
                return PostHocSpatialTransformerWithTimeMixingAndMotion(
                    ch,
                    num_heads,
                    dim_head,
                    depth=depth,
                    context_dim=context_dim,
                    time_context_dim=time_context_dim,
                    motion_context_dim=motion_context_dim,
                    dropout=dropout,
                    ff_in=extra_ff_mix_layer,
                    use_spatial_context=use_spatial_context,
                    use_camera_emb=use_camera_emb,
                    use_3d_attention=use_3d_attention,
                    separate_motion_merge_factor=separate_motion_merge_factor,
                    adm_in_channels=adm_in_channels,
                    merge_strategy=time_block_merge_strategy,
                    merge_factor=time_block_merge_factor,
                    merge_factor_motion=motion_block_merge_factor,
                    checkpoint=use_checkpoint,
                    use_linear=use_linear_in_transformer,
                    attn_mode=spatial_transformer_attn_type,
                    disable_self_attn=disabled_sa,
                    disable_temporal_crossattention=disable_temporal_crossattention,
                    time_mix_legacy=time_mix_legacy,
                    max_time_embed_period=max_ddpm_temb_period,
                )

            else:
                return PostHocSpatialTransformerWithTimeMixing(
                    ch,
                    num_heads,
                    dim_head,
                    depth=depth,
                    context_dim=context_dim,
                    time_context_dim=time_context_dim,
                    dropout=dropout,
                    ff_in=extra_ff_mix_layer,
                    use_spatial_context=use_spatial_context,
                    merge_strategy=time_block_merge_strategy,
                    merge_factor=time_block_merge_factor,
                    checkpoint=use_checkpoint,
                    use_linear=use_linear_in_transformer,
                    attn_mode=spatial_transformer_attn_type,
                    disable_self_attn=disabled_sa,
                    disable_temporal_crossattention=disable_temporal_crossattention,
                    time_mix_legacy=time_mix_legacy,
                    max_time_embed_period=max_ddpm_temb_period,
                )

        def get_resblock(
            time_block_merge_factor,
            time_block_merge_strategy,
            time_kernel_size,
            ch,
            time_embed_dim,
            dropout,
            out_ch,
            dims,
            use_checkpoint,
            use_scale_shift_norm,
            down=False,
            up=False,
        ):
            if self.use_temporal_resblocks:
                return PostHocResBlockWithTime(
                    merge_factor=time_block_merge_factor,
                    merge_strategy=time_block_merge_strategy,
                    time_kernel_size=time_kernel_size,
                    channels=ch,
                    emb_channels=time_embed_dim,
                    dropout=dropout,
                    out_channels=out_ch,
                    dims=dims,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm,
                    down=down,
                    up=up,
                    time_mix_legacy=time_mix_legacy,
                    replicate_bug=replicate_time_mix_bug,
                )
            else:
                return ResBlock(
                    channels=ch,
                    emb_channels=time_embed_dim,
                    dropout=dropout,
                    out_channels=out_ch,
                    use_checkpoint=use_checkpoint,
                    dims=dims,
                    use_scale_shift_norm=use_scale_shift_norm,
                    down=down,
                    up=up,
                )

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    get_resblock(
                        time_block_merge_factor=time_block_merge_factor,
                        time_block_merge_strategy=time_block_merge_strategy,
                        time_kernel_size=time_kernel_size,
                        ch=ch,
                        time_embed_dim=time_embed_dim,
                        dropout=dropout,
                        out_ch=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        dim_head = (
                            ch // num_heads
                            if use_spatial_transformer
                            else num_head_channels
                        )

                    layers.append(
                        get_attention_layer(
                            ch,
                            num_heads,
                            dim_head,
                            depth=transformer_depth[level],
                            context_dim=context_dim,
                            use_checkpoint=use_checkpoint,
                            disabled_sa=False,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                ds *= 2
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        get_resblock(
                            time_block_merge_factor=time_block_merge_factor,
                            time_block_merge_strategy=time_block_merge_strategy,
                            time_kernel_size=time_kernel_size,
                            ch=ch,
                            time_embed_dim=time_embed_dim,
                            dropout=dropout,
                            out_ch=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch,
                            conv_resample,
                            dims=dims,
                            out_channels=out_ch,
                            third_down=time_downup,
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)

                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels

        self.middle_block = TimestepEmbedSequential(
            get_resblock(
                time_block_merge_factor=time_block_merge_factor,
                time_block_merge_strategy=time_block_merge_strategy,
                time_kernel_size=time_kernel_size,
                ch=ch,
                time_embed_dim=time_embed_dim,
                out_ch=None,
                dropout=dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            get_attention_layer(
                ch,
                num_heads,
                dim_head,
                depth=transformer_depth_middle,
                context_dim=context_dim,
                use_checkpoint=use_checkpoint,
            ),
            get_resblock(
                time_block_merge_factor=time_block_merge_factor,
                time_block_merge_strategy=time_block_merge_strategy,
                time_kernel_size=time_kernel_size,
                ch=ch,
                out_ch=None,
                time_embed_dim=time_embed_dim,
                dropout=dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    get_resblock(
                        time_block_merge_factor=time_block_merge_factor,
                        time_block_merge_strategy=time_block_merge_strategy,
                        time_kernel_size=time_kernel_size,
                        ch=ch + ich,
                        time_embed_dim=time_embed_dim,
                        dropout=dropout,
                        out_ch=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        dim_head = (
                            ch // num_heads
                            if use_spatial_transformer
                            else num_head_channels
                        )

                    layers.append(
                        get_attention_layer(
                            ch,
                            num_heads,
                            dim_head,
                            depth=transformer_depth[level],
                            context_dim=context_dim,
                            use_checkpoint=use_checkpoint,
                            disabled_sa=False,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    ds //= 2
                    layers.append(
                        get_resblock(
                            time_block_merge_factor=time_block_merge_factor,
                            time_block_merge_strategy=time_block_merge_strategy,
                            time_kernel_size=time_kernel_size,
                            ch=ch,
                            time_embed_dim=time_embed_dim,
                            dropout=dropout,
                            out_ch=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(
                            ch,
                            conv_resample,
                            dims=dims,
                            out_channels=out_ch,
                            third_up=time_downup,
                        )
                    )

                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )

    def forward(
        self,
        x: th.Tensor,
        timesteps: th.Tensor,
        context: Optional[th.Tensor] = None,
        y: Optional[th.Tensor] = None,
        cam: Optional[th.Tensor] = None,
        time_context: Optional[th.Tensor] = None,
        num_video_frames: Optional[int] = None,
        image_only_indicator: Optional[th.Tensor] = None,
        cond_view: Optional[th.Tensor] = None,
        cond_motion: Optional[th.Tensor] = None,
        time_step: Optional[int] = None,
    ):
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional -> no, relax this TODO"
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False) # 21 x 320
        emb = self.time_embed(t_emb) # 21 x 1280
        time = str(timesteps[0].data.cpu().numpy())

        if self.num_classes is not None:
            assert y.shape[0] == x.shape[0]
            emb = emb + self.label_emb(y) # 21 x 1280

        h = x # 21 x 8 x 64 x 64
        for i, module in enumerate(self.input_blocks):
            h = module(
                h,
                emb,
                context=context,
                cam=cam,
                image_only_indicator=image_only_indicator,
                cond_view=cond_view,
                cond_motion=cond_motion,
                time_context=time_context,
                num_video_frames=num_video_frames,
                time_step=time_step,
                name='encoder_{}_{}'.format(time, i)
            )
            hs.append(h)
        h = self.middle_block(
            h,
            emb,
            context=context,
            cam=cam,
            image_only_indicator=image_only_indicator,
            cond_view=cond_view,
            cond_motion=cond_motion,
            time_context=time_context,
            num_video_frames=num_video_frames,
            time_step=time_step,
            name='middle_{}_0'.format(time, i)
        )
        for i, module in enumerate(self.output_blocks):
            h = th.cat([h, hs.pop()], dim=1)
            h = module(
                h,
                emb,
                context=context,
                cam=cam,
                image_only_indicator=image_only_indicator,
                cond_view=cond_view,
                cond_motion=cond_motion,
                time_context=time_context,
                num_video_frames=num_video_frames,
                time_step=time_step,
                name='decoder_{}_{}'.format(time, i)
            )
        h = h.type(x.dtype)
        return self.out(h)
