from functools import partial

import torch

from ..modules.attention import *
from ..modules.diffusionmodules.util import (
    AlphaBlender,
    get_alpha,
    linear,
    mixed_checkpoint,
    timestep_embedding,
)


class TimeMixSequential(nn.Sequential):
    def forward(self, x, context=None, timesteps=None):
        for layer in self:
            x = layer(x, context, timesteps)

        return x


class BasicTransformerTimeMixBlock(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention,
        "softmax-xformers": MemoryEfficientCrossAttention,
    }

    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0.0,
        context_dim=None,
        gated_ff=True,
        checkpoint=True,
        timesteps=None,
        ff_in=False,
        inner_dim=None,
        attn_mode="softmax",
        disable_self_attn=False,
        disable_temporal_crossattention=False,
        switch_temporal_ca_to_sa=False,
    ):
        super().__init__()

        attn_cls = self.ATTENTION_MODES[attn_mode]

        self.ff_in = ff_in or inner_dim is not None
        if inner_dim is None:
            inner_dim = dim

        assert int(n_heads * d_head) == inner_dim

        self.is_res = inner_dim == dim

        if self.ff_in:
            self.norm_in = nn.LayerNorm(dim)
            self.ff_in = FeedForward(
                dim, dim_out=inner_dim, dropout=dropout, glu=gated_ff
            )

        self.timesteps = timesteps
        self.disable_self_attn = disable_self_attn
        if self.disable_self_attn:
            self.attn1 = attn_cls(
                query_dim=inner_dim,
                heads=n_heads,
                dim_head=d_head,
                context_dim=context_dim,
                dropout=dropout,
            )  # is a cross-attention
        else:
            self.attn1 = attn_cls(
                query_dim=inner_dim, heads=n_heads, dim_head=d_head, dropout=dropout
            )  # is a self-attention

        self.ff = FeedForward(inner_dim, dim_out=dim, dropout=dropout, glu=gated_ff)

        if disable_temporal_crossattention:
            if switch_temporal_ca_to_sa:
                raise ValueError
            else:
                self.attn2 = None
        else:
            self.norm2 = nn.LayerNorm(inner_dim)
            if switch_temporal_ca_to_sa:
                self.attn2 = attn_cls(
                    query_dim=inner_dim, heads=n_heads, dim_head=d_head, dropout=dropout
                )  # is a self-attention
            else:
                self.attn2 = attn_cls(
                    query_dim=inner_dim,
                    context_dim=context_dim,
                    heads=n_heads,
                    dim_head=d_head,
                    dropout=dropout,
                )  # is self-attn if context is none

        self.norm1 = nn.LayerNorm(inner_dim)
        self.norm3 = nn.LayerNorm(inner_dim)
        self.switch_temporal_ca_to_sa = switch_temporal_ca_to_sa

        self.checkpoint = checkpoint
        if self.checkpoint:
            logpy.info(f"{self.__class__.__name__} is using checkpointing")

    def forward(
        self, x: torch.Tensor, context: torch.Tensor = None, timesteps: int = None
    ) -> torch.Tensor:
        if self.checkpoint:
            return checkpoint(self._forward, x, context, timesteps)
        else:
            return self._forward(x, context, timesteps=timesteps)

    def _forward(self, x, context=None, timesteps=None):
        assert self.timesteps or timesteps
        assert not (self.timesteps and timesteps) or self.timesteps == timesteps
        timesteps = self.timesteps or timesteps
        B, S, C = x.shape
        x = rearrange(x, "(b t) s c -> (b s) t c", t=timesteps)

        if self.ff_in:
            x_skip = x
            x = self.ff_in(self.norm_in(x))
            if self.is_res:
                x += x_skip

        if self.disable_self_attn:
            x = self.attn1(self.norm1(x), context=context) + x
        else:
            x = self.attn1(self.norm1(x)) + x

        if self.attn2 is not None:
            if self.switch_temporal_ca_to_sa:
                x = self.attn2(self.norm2(x)) + x
            else:
                x = self.attn2(self.norm2(x), context=context) + x
        x_skip = x
        x = self.ff(self.norm3(x))
        if self.is_res:
            x += x_skip

        x = rearrange(
            x, "(b s) t c -> (b t) s c", s=S, b=B // timesteps, c=C, t=timesteps
        )
        return x

    def get_last_layer(self):
        return self.ff.net[-1].weight


class PostHocSpatialTransformerWithTimeMixing(SpatialTransformer):
    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        depth=1,
        dropout=0.0,
        use_linear=False,
        context_dim=None,
        use_spatial_context=False,
        timesteps=None,
        merge_strategy: str = "fixed",
        merge_factor: float = 0.5,
        apply_sigmoid_to_merge: bool = True,
        time_context_dim=None,
        ff_in=False,
        checkpoint=False,
        time_depth=1,
        attn_mode="softmax",
        disable_self_attn=False,
        disable_temporal_crossattention=False,
        time_mix_legacy: bool = True,
        max_time_embed_period: int = 10000,
    ):
        super().__init__(
            in_channels,
            n_heads,
            d_head,
            depth=depth,
            dropout=dropout,
            attn_type=attn_mode,
            use_checkpoint=checkpoint,
            context_dim=context_dim,
            use_linear=use_linear,
            disable_self_attn=disable_self_attn,
        )
        self.time_depth = time_depth
        self.depth = depth
        self.max_time_embed_period = max_time_embed_period

        time_mix_d_head = d_head
        n_time_mix_heads = n_heads

        time_mix_inner_dim = int(time_mix_d_head * n_time_mix_heads)

        inner_dim = n_heads * d_head
        if use_spatial_context:
            time_context_dim = context_dim

        self.time_mix_blocks = nn.ModuleList(
            [
                BasicTransformerTimeMixBlock(
                    inner_dim,
                    n_time_mix_heads,
                    time_mix_d_head,
                    dropout=dropout,
                    context_dim=time_context_dim,
                    timesteps=timesteps,
                    checkpoint=checkpoint,
                    ff_in=ff_in,
                    inner_dim=time_mix_inner_dim,
                    attn_mode=attn_mode,
                    disable_self_attn=disable_self_attn,
                    disable_temporal_crossattention=disable_temporal_crossattention,
                )
                for _ in range(self.depth)
            ]
        )

        assert len(self.time_mix_blocks) == len(self.transformer_blocks)

        self.use_spatial_context = use_spatial_context
        self.in_channels = in_channels

        time_embed_dim = self.in_channels * 4
        self.time_mix_time_embed = nn.Sequential(
            linear(self.in_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, self.in_channels),
        )

        self.time_mix_legacy = time_mix_legacy
        if self.time_mix_legacy:
            if merge_strategy == "fixed":
                self.register_buffer("mix_factor", torch.Tensor([merge_factor]))
            elif merge_strategy == "learned" or merge_strategy == "learned_with_images":
                self.register_parameter(
                    "mix_factor", torch.nn.Parameter(torch.Tensor([merge_factor]))
                )
            elif merge_strategy == "fixed_with_images":
                self.mix_factor = None
            else:
                raise ValueError(f"unknown merge strategy {merge_strategy}")

            self.get_alpha_fn = partial(
                get_alpha,
                merge_strategy,
                self.mix_factor,
                apply_sigmoid=apply_sigmoid_to_merge,
                is_attn=True,
            )
        else:
            self.time_mixer = AlphaBlender(
                alpha=merge_factor, merge_strategy=merge_strategy
            )

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        # cam: Optional[torch.Tensor] = None,
        time_context: Optional[torch.Tensor] = None,
        timesteps: Optional[int] = None,
        image_only_indicator: Optional[torch.Tensor] = None,
        cond_view: Optional[torch.Tensor] = None,
        cond_motion: Optional[torch.Tensor] = None,
        time_step: Optional[int] = None,
        name: Optional[str] = None,
    ) -> torch.Tensor:
        _, _, h, w = x.shape
        x_in = x
        spatial_context = None
        if exists(context):
            spatial_context = context

        if self.use_spatial_context:
            assert (
                context.ndim == 3
            ), f"n dims of spatial context should be 3 but are {context.ndim}"

            time_context = context
            time_context_first_timestep = time_context[::timesteps]
            time_context = repeat(
                time_context_first_timestep, "b ... -> (b n) ...", n=h * w
            )
        elif time_context is not None and not self.use_spatial_context:
            time_context = repeat(time_context, "b ... -> (b n) ...", n=h * w)
            if time_context.ndim == 2:
                time_context = rearrange(time_context, "b c -> b 1 c")

        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        if self.use_linear:
            x = self.proj_in(x)

        if self.time_mix_legacy:
            alpha = self.get_alpha_fn(image_only_indicator=image_only_indicator)

        num_frames = torch.arange(timesteps, device=x.device)
        num_frames = repeat(num_frames, "t -> b t", b=x.shape[0] // timesteps)
        num_frames = rearrange(num_frames, "b t -> (b t)")
        t_emb = timestep_embedding(
            num_frames,
            self.in_channels,
            repeat_only=False,
            max_period=self.max_time_embed_period,
        )
        emb = self.time_mix_time_embed(t_emb)
        emb = emb[:, None, :]

        for it_, (block, mix_block) in enumerate(
            zip(self.transformer_blocks, self.time_mix_blocks)
        ):
            # spatial attention
            x = block(
                x,
                context=spatial_context,
                time_step=time_step, 
                name=name + '_' + str(it_)
            )

            x_mix = x
            x_mix = x_mix + emb

            # temporal attention
            x_mix = mix_block(x_mix, context=time_context, timesteps=timesteps)
            if self.time_mix_legacy:
                x = alpha.to(x.dtype) * x + (1.0 - alpha).to(x.dtype) * x_mix
            else:
                x = self.time_mixer(
                    x_spatial=x,
                    x_temporal=x_mix,
                    image_only_indicator=image_only_indicator,
                )

        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        if not self.use_linear:
            x = self.proj_out(x)
        out = x + x_in
        return out
    

class PostHocSpatialTransformerWithTimeMixingAndMotion(SpatialTransformer):
    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        depth=1,
        dropout=0.0,
        use_linear=False,
        context_dim=None,
        use_spatial_context=False,
        timesteps=None,
        merge_strategy: str = "fixed",
        merge_factor: float = 0.5,
        apply_sigmoid_to_merge: bool = True,
        time_context_dim=None,
        ff_in=False,
        checkpoint=False,
        time_depth=1,
        attn_mode="softmax",
        disable_self_attn=False,
        disable_temporal_crossattention=False,
        time_mix_legacy: bool = True,
        max_time_embed_period: int = 10000,
    ):
        super().__init__(
            in_channels,
            n_heads,
            d_head,
            depth=depth,
            dropout=dropout,
            attn_type=attn_mode,
            use_checkpoint=checkpoint,
            context_dim=context_dim,
            use_linear=use_linear,
            disable_self_attn=disable_self_attn,
        )
        self.time_depth = time_depth
        self.depth = depth
        self.max_time_embed_period = max_time_embed_period

        time_mix_d_head = d_head
        n_time_mix_heads = n_heads

        time_mix_inner_dim = int(time_mix_d_head * n_time_mix_heads)

        inner_dim = n_heads * d_head
        if use_spatial_context:
            time_context_dim = context_dim

        camera_context_dim = time_context_dim
        motion_context_dim = 4 # time_context_dim

        # Camera attention layer
        self.time_mix_blocks = nn.ModuleList(
            [
                BasicTransformerTimeMixBlock(
                    inner_dim,
                    n_time_mix_heads,
                    time_mix_d_head,
                    dropout=dropout,
                    context_dim=camera_context_dim,
                    timesteps=timesteps,
                    checkpoint=checkpoint,
                    ff_in=ff_in,
                    inner_dim=time_mix_inner_dim,
                    attn_mode=attn_mode,
                    disable_self_attn=disable_self_attn,
                    disable_temporal_crossattention=disable_temporal_crossattention,
                )
                for _ in range(self.depth)
            ]
        )

        # Motion attention layer
        self.motion_blocks = nn.ModuleList(
            [
                BasicTransformerTimeMixBlock(
                    inner_dim,
                    n_time_mix_heads,
                    time_mix_d_head,
                    dropout=dropout,
                    context_dim=motion_context_dim,
                    timesteps=timesteps,
                    checkpoint=checkpoint,
                    ff_in=ff_in,
                    inner_dim=time_mix_inner_dim,
                    attn_mode=attn_mode,
                    disable_self_attn=disable_self_attn,
                    disable_temporal_crossattention=disable_temporal_crossattention,
                )
                for _ in range(self.depth)
            ]
        )

        assert len(self.time_mix_blocks) == len(self.transformer_blocks)

        self.use_spatial_context = use_spatial_context
        self.in_channels = in_channels

        time_embed_dim = self.in_channels * 4
        # Camera view embedding
        self.time_mix_time_embed = nn.Sequential(
            linear(self.in_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, self.in_channels),
        )
        # Motion time embedding
        self.time_mix_motion_embed = nn.Sequential(
            linear(self.in_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, self.in_channels),
        )

        self.time_mix_legacy = time_mix_legacy
        if self.time_mix_legacy:
            if merge_strategy == "fixed":
                self.register_buffer("mix_factor", torch.Tensor([merge_factor]))
            elif merge_strategy == "learned" or merge_strategy == "learned_with_images":
                self.register_parameter(
                    "mix_factor", torch.nn.Parameter(torch.Tensor([merge_factor]))
                )
            elif merge_strategy == "fixed_with_images":
                self.mix_factor = None
            else:
                raise ValueError(f"unknown merge strategy {merge_strategy}")

            self.get_alpha_fn = partial(
                get_alpha,
                merge_strategy,
                self.mix_factor,
                apply_sigmoid=apply_sigmoid_to_merge,
                is_attn=True,
            )
        else:
            self.time_mixer = AlphaBlender(
                alpha=merge_factor, merge_strategy=merge_strategy
            )

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        # cam: Optional[torch.Tensor] = None,
        time_context: Optional[torch.Tensor] = None,
        timesteps: Optional[int] = None,
        image_only_indicator: Optional[torch.Tensor] = None,
        cond_view: Optional[torch.Tensor] = None,
        cond_motion: Optional[torch.Tensor] = None,
        time_step: Optional[int] = None,
        name: Optional[str] = None,
    ) -> torch.Tensor:
        _, _, h, w = x.shape
        x_in = x
        spatial_context = None
        if exists(context):
            spatial_context = context

        # cond_view: b v 4 h w
        # cond_motion: b t 4 h w
        b, t, d1 = context.shape # CLIP
        v, d2 = cond_view.shape[0]//b, cond_view.shape[1] # VAE
        cond_view = torch.nn.functional.interpolate(cond_view, size=(h,w), mode="bilinear") # b*v d h w
        spatial_context = context[:,:,None].repeat(1,1,v,1).reshape(-1,1,d1) # (b*t*v) 1 d1
        camera_context = context[:,:,None].repeat(1,1,h*w,1).reshape(-1,1,d1) # (b*t*h*w) 1 d1
        motion_context = cond_view.permute(0,2,3,1).reshape(-1,1,d2) # (b*v*h*w) 1 d2

        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, "b c h w -> b (h w) c") # 21 x 4096 x 320
        if self.use_linear:
            x = self.proj_in(x)
        c = x.shape[-1]

        if self.time_mix_legacy:
            alpha = self.get_alpha_fn(image_only_indicator=image_only_indicator)

        num_frames = torch.arange(t, device=x.device)
        num_frames = repeat(num_frames, "t -> b t", b=b)
        num_frames = rearrange(num_frames, "b t -> (b t)")
        t_emb = timestep_embedding(
            num_frames,
            self.in_channels,
            repeat_only=False,
            max_period=self.max_time_embed_period,
        )
        emb_time = self.time_mix_motion_embed(t_emb)
        emb_time = emb_time[:, None, :] # b*t x 1 x 320

        num_views = torch.arange(v, device=x.device)
        num_views = repeat(num_views, "t -> b t", b=b)
        num_views = rearrange(num_views, "b t -> (b t)")
        v_emb = timestep_embedding(
            num_views,
            self.in_channels,
            repeat_only=False,
            max_period=self.max_time_embed_period,
        )
        emb_view = self.time_mix_time_embed(v_emb)
        emb_view = emb_view[:, None, :] # b*v x 1 x 320

        for it_, (block, time_block, mot_block) in enumerate(
            zip(self.transformer_blocks, self.time_mix_blocks, self.motion_blocks)
        ):
            # Spatial attention
            x = block(
                x,
                context=spatial_context,
            )

            # Camera attention
            x = x.view(b, t, v, h*w, c).permute(0,2,1,3,4).reshape(b*v,-1,c) # b*v t*h*w c
            x_mix = x + emb_view
            x_mix = time_block(x_mix, context=camera_context, timesteps=v)
            if self.time_mix_legacy:
                x = alpha.to(x.dtype) * x + (1.0 - alpha).to(x.dtype) * x_mix
            else:
                x = self.time_mixer(
                    x_spatial=x,
                    x_temporal=x_mix,
                    image_only_indicator=image_only_indicator[:,:v],
                )

            # Motion attention
            x = x.view(b, v, t, h*w, c).permute(0,2,1,3,4).reshape(b*t,-1,c) # b*t v*h*w c
            x_mix = x + emb_time
            x_mix = mot_block(x_mix, context=motion_context, timesteps=t)
            if self.time_mix_legacy:
                x = alpha.to(x.dtype) * x + (1.0 - alpha).to(x.dtype) * x_mix
            else:
                x = self.time_mixer(
                    x_spatial=x,
                    x_temporal=x_mix,
                    image_only_indicator=image_only_indicator[:,:t],
                )

            x = x.view(b, t, v, h*w, c).reshape(-1,h*w,c) # b*t*v h*w c

        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        if not self.use_linear:
            x = self.proj_out(x)
        out = x + x_in
        return out